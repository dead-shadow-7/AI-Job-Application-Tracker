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

from app.core.config import settings

logger = logging.getLogger(__name__)


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
    """True if accepted, False if rejected, None if it could not be determined."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{endpoint.rstrip('/')}/sessions",
                headers={"x-api-key": api_key},
                params={"limit": 1},
            )
    except httpx.RequestError as exc:
        logger.warning("Could not reach LangSmith to verify the key: %s", exc)
        return None

    return response.status_code not in (401, 403)


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
