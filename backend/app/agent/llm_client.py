"""A thin, typed Groq client.

Groq's API is OpenAI-compatible, so this is a small amount of code rather than a
dependency. It is written directly instead of through a LangChain chat wrapper
for one reason: strict schema adherence is the entire value of this call, and a
wrapper's ``with_structured_output`` may emit function-calling or loose JSON
mode depending on version. Here the exact ``response_format`` sent is visible
and pinned.
"""

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.agent.tracing import as_llm_run, hide, record_model, traced
from app.core.config import settings
from app.core.exceptions import DomainError
from app.schemas.extraction import to_strict_json_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Retried: transient. 429 is the request-rate limit, which a backfill will hit;
# 5xx is Groq's problem, not the request's.
#
# 413 is included because Groq returns it for *token*-per-minute exhaustion,
# not only for oversized payloads — the body carries code `rate_limit_exceeded`.
# Reading it as "request too big" sends you optimising the prompt when the fix
# is to wait or reduce max_completion_tokens.
RETRYABLE_STATUS = {408, 409, 413, 429, 500, 502, 503, 504}

# One pooled client per event loop. Building an AsyncClient per call discarded
# the connection with it, so every round paid a fresh TCP and TLS handshake to a
# remote host — six of them in an assistant turn, for nothing.
_http_client: httpx.AsyncClient | None = None
_http_loop: asyncio.AbstractEventLoop | None = None


def get_http_client() -> httpx.AsyncClient:
    """The shared client, rebuilt if the running loop has changed.

    Keyed on the loop rather than on ``is_closed``: a pooled keep-alive
    connection is bound to the loop that opened it, and after that loop closes
    the client still reports ``is_closed == False``. Reusing it then raises
    ``RuntimeError: Event loop is closed`` from deep inside httpcore. This is
    the same hazard the test suite disposes the DB engine for — see the
    docstring in tests/conftest.py — and it would otherwise bite the first
    caller to run on a second loop (a script, or the scheduled sweep).
    """
    global _http_client, _http_loop
    loop = asyncio.get_running_loop()
    if _http_client is None or _http_client.is_closed or _http_loop is not loop:
        _http_client = httpx.AsyncClient(
            # Split rather than one scalar: a connect that hangs is a dead host
            # and should fail fast, where a 60s read is just a long generation.
            # One 90s number for both made them indistinguishable.
            timeout=httpx.Timeout(settings.llm_timeout_seconds, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        _http_loop = loop
    return _http_client


async def close_http_client() -> None:
    global _http_client, _http_loop
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
        _http_loop = None


class LLMError(DomainError):
    """The model could not be reached, or returned something unusable."""


class LLMUsage(BaseModel):
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    # How much of the prompt was served from the provider's cache. Worth
    # recording because it is most of the bill: the system prompt and tool
    # schemas are ~3,100 tokens of identical prefix on every round, and a hit
    # halves what they cost. Nothing here enables it — caching is automatic
    # above 1,024 tokens — but a hit rate that quietly collapses after a prompt
    # edit would otherwise show up only as a larger invoice.
    cached_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


class StructuredResult[TModel: BaseModel](BaseModel):
    data: TModel
    usage: LLMUsage


@dataclass(slots=True)
class TextDelta:
    """A fragment of the answer, handed over as it is generated."""

    text: str


@dataclass(slots=True)
class TurnComplete:
    """The assembled turn, once the stream ends.

    Deliberately the same shape a non-streamed completion returns — role,
    content, reassembled ``tool_calls`` — so the loop reading it never has to
    know that the message arrived in a few hundred pieces.
    """

    message: dict[str, Any]
    usage: LLMUsage


StreamEvent = TextDelta | TurnComplete


def _stream_run(outputs: Any) -> dict[str, Any]:
    """Reshape a streamed turn into what LangSmith prices a run from.

    A traced async generator hands over everything it yielded, which here is a
    long tail of text fragments followed by one assembled turn. Only the last
    carries the message and the token counts, so the fragments are dropped
    rather than uploaded as several hundred rows of one word each.

    Written defensively, like its sibling below: a trace that fails to render
    is annoying, a request that fails because tracing raised is not acceptable.
    """
    value = outputs.get("output") if isinstance(outputs, dict) else outputs
    events = value if isinstance(value, list) else [value]
    finished = next((e for e in reversed(events) if isinstance(e, TurnComplete)), None)
    if finished is None:
        return outputs if isinstance(outputs, dict) else {"output": outputs}

    return as_llm_run([finished.message], finished.usage)


def _extract_run(outputs: Any) -> dict[str, Any]:
    """The same, for the structured-extraction call."""
    value = outputs.get("output") if isinstance(outputs, dict) else outputs
    if not isinstance(value, StructuredResult):
        return outputs if isinstance(outputs, dict) else {"output": outputs}

    return as_llm_run([{"role": "assistant", "content": value.data.model_dump_json()}], value.usage)


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    @traced(
        "extract",
        run_type="llm",
        process_inputs=hide("self", "schema"),
        process_outputs=_extract_run,
    )
    async def extract(
        self,
        *,
        schema: type[T],
        system: str,
        user: str,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> StructuredResult[T]:
        """Call the model and return an instance of ``schema``.

        Strict mode makes the response schema-valid by construction, so this
        does not parse-and-pray. A ValidationError here means the schema and the
        model disagree — a bug worth surfacing, not something to paper over.
        """
        if not self.is_configured:
            raise LLMError("GROQ_API_KEY is not set.")

        target_model = model or settings.extraction_model
        payload: dict[str, Any] = {
            "model": target_model,
            "temperature": temperature,
            # Counted against the tokens-per-minute budget *before* generation,
            # so an optimistic ceiling here fails the request outright on the
            # free tier rather than merely allowing a long answer. A full
            # extraction runs well under 3k.
            "max_completion_tokens": max_tokens or settings.llm_max_output_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": to_strict_json_schema(schema),
                },
            },
        }

        body, usage = await self._post(payload, target_model)
        record_model(usage.model, settings.llm_provider)
        content = body["choices"][0]["message"]["content"]

        try:
            return StructuredResult(data=schema.model_validate_json(content), usage=usage)
        except ValidationError as exc:
            logger.error("Model returned schema-invalid JSON for %s: %s", schema.__name__, exc)
            raise LLMError(
                f"{target_model} returned JSON that does not match {schema.__name__}."
            ) from exc

    @traced(
        "chat",
        run_type="llm",
        process_inputs=hide("self", "tools"),
        process_outputs=_stream_run,
    )
    async def stream_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """One turn of a tool-calling conversation, delivered as it is written.

        Yields a ``TextDelta`` per fragment of prose and exactly one
        ``TurnComplete`` at the end, carrying the same assistant message a
        non-streamed call would have returned: either ``tool_calls`` to
        execute, or ``content`` and the loop is done. So the loop above drives
        identically whether or not anyone is watching it type.

        Deliberately not wrapped in a framework — the loop is a dozen lines and
        keeping it visible means the stopping condition and the tool-result
        plumbing are inspectable rather than inherited.
        """
        if not self.is_configured:
            raise LLMError("No LLM API key is configured.")

        payload: dict[str, Any] = {
            "model": model or settings.extraction_model,
            "temperature": temperature,
            # The chat ceiling, not the extraction one: rule 7 of the system
            # prompt asks for one or two sentences, and this budget is debited
            # before generation rather than after it.
            "max_completion_tokens": max_tokens or settings.llm_chat_output_tokens,
            "messages": messages,
            "stream": True,
            # Usage is omitted from a streamed response unless it is asked for.
            # Without this every turn reports zero tokens: the cache hit rate
            # the loop logs goes to nothing and LangSmith prices the run at
            # zero — both silently, and exactly as the traffic moves here.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        record_model(payload["model"], settings.llm_provider)
        async for event in self._stream(payload, payload["model"]):
            yield event

    async def _stream(self, payload: dict[str, Any], model: str) -> AsyncIterator[StreamEvent]:
        """Open the stream, retrying only while nothing has been delivered.

        ``_post``'s retry loop cannot simply be reused. Once a fragment has
        been handed to the caller it is already on someone's screen, and
        starting the request again would repeat it from the beginning — so a
        failure after the first fragment is reported rather than retried.
        """
        last_error: Exception | None = None
        final_attempt = settings.llm_max_retries - 1

        for attempt in range(settings.llm_max_retries):
            started = perf_counter()
            delivered = False
            try:
                # Fetched per attempt for the same reason as _post: a shutdown
                # during the seconds this loop can sleep would otherwise leave
                # us holding a closed client.
                async with get_http_client().stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        # A streamed response arrives with its body unread, and
                        # `_error_message` reads it — without this the provider's
                        # own explanation is replaced by a ResponseNotRead.
                        await response.aread()
                        message = self._error_message(response)

                        if response.status_code in RETRYABLE_STATUS:
                            last_error = LLMError(message)
                            logger.warning(
                                "Groq %s (attempt %d/%d): %s",
                                response.status_code,
                                attempt + 1,
                                settings.llm_max_retries,
                                message,
                            )
                            if attempt < final_attempt:
                                await self._backoff(attempt, response.headers.get("retry-after"))
                            continue

                        logger.error(
                            "Groq %s rejected the request: %s", response.status_code, message
                        )
                        raise LLMError(message)

                    async for event in self._read_stream(response, model, started):
                        delivered = True
                        yield event
                    return
            except (httpx.RequestError, RuntimeError) as exc:
                if delivered:
                    # Half an answer is on screen. Retrying would append a
                    # second attempt to the first; saying so is the only honest
                    # option left.
                    raise LLMError(
                        f"The connection to {model} dropped part-way through the answer."
                    ) from exc
                last_error = exc
                if attempt < final_attempt:
                    await self._backoff(attempt)
                continue

        raise LLMError(f"Groq unreachable after {settings.llm_max_retries} attempts: {last_error}")

    @staticmethod
    async def _read_stream(
        response: httpx.Response, model: str, started: float
    ) -> AsyncIterator[StreamEvent]:
        """Server-sent chunks into text fragments and one assembled turn.

        Tool calls are the fiddly part: the name arrives in one chunk and the
        JSON arguments dribble in over the next several, keyed by their index
        in the call list. They are rebuilt here so that nothing above this line
        has to know the difference.
        """
        content: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        raw_usage: dict[str, Any] = {}

        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Ignoring an unparseable stream chunk from %s", model)
                continue

            # Arrives as a final chunk carrying no choices, because of
            # stream_options above.
            raw_usage = chunk.get("usage") or raw_usage

            for choice in chunk.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}

                if text := delta.get("content"):
                    content.append(text)
                    yield TextDelta(text)

                for call in delta.get("tool_calls") or []:
                    slot = tool_calls.setdefault(
                        call.get("index", 0),
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fragment = call.get("function") or {}
                    if fragment.get("name"):
                        slot["function"]["name"] = fragment["name"]
                    # Concatenated rather than assigned: this is one JSON object
                    # arriving a few characters at a time.
                    slot["function"]["arguments"] += fragment.get("arguments") or ""

        # The same guard the non-streamed path carries, and streaming does not
        # retire it. A truncated sentence is at least visible; a tool call cut
        # mid-JSON either fails to parse or parses into arguments nobody asked
        # for, and neither announces itself.
        if finish_reason == "length":
            raise LLMError(
                f"{model} hit the output ceiling before finishing. "
                "Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS."
            )

        message: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]

        cached = (raw_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        yield TurnComplete(
            message=message,
            usage=LLMUsage(
                model=model,
                prompt_tokens=raw_usage.get("prompt_tokens", 0),
                completion_tokens=raw_usage.get("completion_tokens", 0),
                total_tokens=raw_usage.get("total_tokens", 0),
                cached_tokens=cached,
                # `response.elapsed` is only populated once a streamed response
                # is closed, which is after this line runs.
                latency_ms=int((perf_counter() - started) * 1000),
            ),
        )

    async def _post(self, payload: dict[str, Any], model: str) -> tuple[dict[str, Any], LLMUsage]:
        last_error: Exception | None = None
        final_attempt = settings.llm_max_retries - 1

        for attempt in range(settings.llm_max_retries):
            try:
                # Fetched per attempt, not once above the loop: a shutdown
                # during the seconds this loop can sleep would otherwise leave
                # us holding a closed client and raising RuntimeError.
                response = await get_http_client().post(
                    f"{self._base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            except (httpx.RequestError, RuntimeError) as exc:
                last_error = exc
                # Sleeping after the last attempt delays only the exception.
                if attempt < final_attempt:
                    await self._backoff(attempt)
                continue

            if response.status_code == 200:
                body = response.json()
                # A truncated answer is otherwise completely silent: the choice
                # comes back with empty content and no tool_calls, the assistant
                # loop reads that as "nothing to say", and the user gets a
                # shrug instead of an error while their turn is still recorded.
                if (body.get("choices") or [{}])[0].get("finish_reason") == "length":
                    raise LLMError(
                        f"{model} hit the output ceiling before finishing. "
                        "Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS."
                    )
                raw = body.get("usage", {})
                return body, LLMUsage(
                    model=model,
                    prompt_tokens=raw.get("prompt_tokens", 0),
                    completion_tokens=raw.get("completion_tokens", 0),
                    total_tokens=raw.get("total_tokens", 0),
                    cached_tokens=(raw.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                    latency_ms=int(response.elapsed.total_seconds() * 1000),
                )

            message = self._error_message(response)

            if response.status_code in RETRYABLE_STATUS:
                last_error = LLMError(message)
                logger.warning(
                    "Groq %s (attempt %d/%d): %s",
                    response.status_code,
                    attempt + 1,
                    settings.llm_max_retries,
                    message,
                )
                # Honour Retry-After when the limiter supplies it; guessing
                # shorter just burns the next attempt against the same window.
                if attempt < final_attempt:
                    await self._backoff(attempt, response.headers.get("retry-after"))
                continue

            logger.error("Groq %s rejected the request: %s", response.status_code, message)
            raise LLMError(message)

        raise LLMError(f"Groq unreachable after {settings.llm_max_retries} attempts: {last_error}")

    @staticmethod
    def _error_message(response: httpx.Response) -> str:
        """Surface Groq's own message rather than a bare status code.

        The status alone is actively misleading here: a 413 usually means
        "payload too large", but Groq also returns it for tokens-per-minute
        exhaustion, and its body says so precisely — including how many tokens
        were requested against what limit. Swallowing that sends whoever is
        debugging to shrink a prompt that was never the problem.
        """
        try:
            error = response.json().get("error", {})
            detail = error.get("message") or response.text[:300]
            if error.get("code") == "rate_limit_exceeded":
                return f"Groq rate limit reached. {detail}"
            return f"Groq returned {response.status_code}: {detail}"
        except ValueError:
            return f"Groq returned {response.status_code}: {response.text[:300]}"

    @staticmethod
    async def _backoff(attempt: int, retry_after: str | None = None) -> None:
        """Wait before retrying, jittered on both paths.

        The jitter matters most on the Retry-After branch, not least: Groq
        sends that header on every 429, so under exactly the rate-limit
        contention worth spreading out, an un-jittered sleep re-aligns every
        caller on the same instant. Only ever added to the delay, so a
        server-supplied floor is still honoured.
        """
        if retry_after:
            try:
                delay = min(float(retry_after), 30.0)
            except ValueError:
                delay = min(2**attempt, 8)
            await asyncio.sleep(delay * random.uniform(1.0, 1.3))
            return

        await asyncio.sleep(min(2**attempt, 8) * random.uniform(0.5, 1.5))


llm_client = LLMClient()
