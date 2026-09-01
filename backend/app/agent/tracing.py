"""LangSmith tracing.

Off unless ``LANGSMITH_TRACING=true`` and a key is present, and configured by
setting the environment variables the LangSmith SDK reads. Doing it here rather
than expecting them in the shell keeps a single source of truth — the same .env
that configures everything else — and means a missing key degrades to "no
traces" rather than to a crash on first extraction.

What the traces are actually for: token cost per ingestion and extraction
latency, both of which need to be watched once real postings start flowing
through, and neither of which is visible from the API response alone.
"""

import logging
import os
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast

from app.core.config import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


async def configure_tracing() -> bool:
    """Wire up tracing, verifying the credential first. Returns whether it is on.

    The credential is checked at startup because the SDK fails *per run*, not at
    configuration time: a rejected key produces an identical warning on every
    single extraction, buried among request logs, while the API still returns
    200 and nothing looks wrong. One clear line at boot beats that.

    Only an explicit 401/403 disables tracing. A timeout or connection error
    leaves it enabled — LangSmith being briefly unreachable is not a reason to
    silently stop tracing for the life of the process.
    """
    if not settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty; tracing disabled.")
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    verdict = await _check_credential(settings.langsmith_api_key, settings.langsmith_endpoint)
    if verdict is False:
        # Almost always a region mismatch rather than a bad key. LangSmith runs
        # separate US, EU and APAC deployments, and a key issued in one is
        # refused by another with a plain 403 that says nothing about
        # geography — indistinguishable from an invalid credential.
        logger.error(
            "LangSmith rejected the API key against %s, so tracing is disabled. "
            "Most often this is the wrong regional endpoint rather than a bad "
            "key: US is https://api.smith.langchain.com, EU is "
            "https://eu.api.smith.langchain.com, APAC is "
            "https://apac.api.smith.langchain.com. Copy LANGSMITH_ENDPOINT from "
            "the same setup screen as the key.",
            settings.langsmith_endpoint,
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    logger.info(
        "LangSmith tracing enabled (project=%r, endpoint=%s, key %s)",
        settings.langsmith_project,
        settings.langsmith_endpoint,
        "verified" if verdict else "unverified — LangSmith unreachable",
    )
    return True


async def _check_credential(api_key: str, endpoint: str) -> bool | None:
    """True if accepted, False if rejected, None if it could not be determined.

    Goes through the SDK's ``runs.query`` rather than a hand-rolled
    ``GET /sessions``. That was the legacy query API, which LangSmith is
    removing on 31 January 2027 and already warns about in the console — and
    since this ran on every boot, it was the only thing keeping the warning lit.
    Using the SDK means their migration is inherited rather than re-discovered.

    Only 401 and 403 count as rejection. The query is deliberately incomplete —
    it names no project — so a *valid* key answers 400: authentication is
    checked before parameters, which makes "not 403" the signal. Anything else,
    including an unreachable LangSmith, leaves tracing on; a brief outage is no
    reason to stop tracing for the life of the process.
    """
    from langsmith import Client

    try:
        async for _ in Client(api_key=api_key, api_url=endpoint).runs.query(page_size=1):
            break
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            return False
        if status is None:
            logger.warning("Could not reach LangSmith to verify the key: %s", exc)
            return None
    return True


def hide(*names: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Drop arguments that would make a trace useless or enormous.

    ``self`` and ``session`` serialise as ``<sqlalchemy.ext.asyncio.AsyncSession
    object at 0x...>``, which is what the ingestion runs already show in the
    input column — a repr nobody can read, in the field you scan to find the run
    you want. ``tools`` is worse: the schema block is ~3,300 tokens of JSON,
    identical on every round, uploaded each time.
    """

    def process(inputs: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in inputs.items() if k not in names}

    return process


def record_model(model: str, provider: str) -> None:
    """Tag the current LLM run with what answered it.

    LangSmith prices a run from ``ls_model_name`` and ``ls_provider`` in its
    metadata. A LangChain chat model sets these itself, which is why token
    counts and cost appear for free there and were blank here — the client is
    hand-written, so nothing was filling them in. Set at call time rather than
    on the decorator because the model is configuration, not a constant.
    """
    try:
        from langsmith.run_helpers import get_current_run_tree
    except ImportError:  # pragma: no cover - langsmith is a hard dependency
        return

    run = get_current_run_tree()
    if run is not None:
        run.metadata.update({"ls_provider": provider, "ls_model_name": model})


def as_llm_run(messages: list[Any], usage: Any) -> dict[str, Any]:
    """The output shape LangSmith reads token counts out of.

    ``usage_metadata`` is the documented contract; anything else is stored as
    an opaque blob and the run shows no tokens and no cost. ``cache_read`` is
    included because it is most of the prompt here — without it the cost shown
    would be the undiscounted one, which is roughly double what is billed.
    """
    return {
        "messages": messages,
        "usage_metadata": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_token_details": {"cache_read": usage.cached_tokens},
        },
    }


RunType = Literal["tool", "chain", "llm", "retriever", "embedding", "prompt", "parser"]


def traced(
    name: str,
    run_type: RunType = "chain",
    *,
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    process_outputs: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[F], F]:
    """Mark a function as a step in a LangSmith trace.

    The ingestion pipeline traces itself because LangGraph does it; the
    assistant is a hand-written loop over httpx and so was completely invisible
    — the run list showed job_ingestion and nothing else, while the part with
    twenty-two tools and a confirmation flow left no record at all.

    A no-op when tracing is off, which includes the whole test suite.
    """

    def decorate(func: F) -> F:
        try:
            from langsmith import traceable
        except ImportError:  # pragma: no cover - langsmith is a hard dependency
            return func
        decorated = traceable(
            run_type,
            name=name,
            process_inputs=process_inputs,
            process_outputs=process_outputs,
        )(func)
        return cast(F, decorated)

    return decorate


def run_metadata(
    user_id: str | None = None, **extra: str | int | float | None
) -> dict[str, object]:
    """Tags for one graph run.

    ``user_id`` is included so a trace can be tied back to whose ingestion it
    was when debugging a specific complaint. No posting text and no email — the
    trace is enough to find the run, not to reconstruct its contents.
    """
    metadata: dict[str, object] = {
        "prompt_version": None,
        "model": settings.extraction_model,
        "environment": settings.environment,
    }
    if user_id:
        metadata["user_id"] = user_id
    metadata.update({k: v for k, v in extra.items() if v is not None})
    return metadata
