"""The typed façade over every model call.

LangChain carries the transport — HTTP, server-sent events, chunk accumulation,
the OpenAI dialect all three providers speak. What stays here is what the
framework has no way to know about this deployment, and each of the three is a
correction rather than a preference.

**The schema is built here and bound as a dict.** Not handed over as a Pydantic
class, and not through ``with_structured_output``. A dict already shaped as a
json_schema response format is forwarded untouched, so what
``to_strict_json_schema`` produced is exactly what Groq receives — including the
``$ref`` inlining its validator requires and which nothing in LangChain
performs. The earlier objection recorded here, that a wrapper "may emit
function-calling or loose JSON mode depending on version", is no longer true of
``ChatOpenAI``: it defaults to strict json_schema and raises rather than
degrading. It is still true of ``ChatGroq``, which is why Groq is reached
through its OpenAI-compatible host instead. The reason to keep the dict is the
inlining, not the doubt.

**The retry policy is ours.** The SDK's set omits 413, which is the status Groq
returns for token-per-minute exhaustion rather than for an oversized request,
and running both loops would multiply the attempts inside a deadline sized for
one of them. Its retries are switched off in ``agent/models.py``.

**Truncation raises.** A reply cut at the ceiling comes back as an ordinary
message with a field set, and the assistant loop reads empty content and no tool
calls as "nothing to say" — answering with a shrug while the user's turn is
already recorded.

What this module deliberately does not do is orchestrate. There is no agent and
no graph here; see ``agent/assistant.py`` for why the loop above it is written
out rather than inherited.
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar, cast

import httpx
import openai
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import BaseModel, ValidationError

from app.agent.models import ChatModel, build_chat_model, chat_model_config
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


def _usage_of(message: AIMessage, model: str, latency_ms: int) -> LLMUsage:
    """Token counts, translated out of LangChain's provider-neutral shape.

    ``cache_read`` is the one worth naming. Groq reports it as
    ``prompt_tokens_details.cached_tokens`` and LangChain normalises it here; if
    that normalisation ever stopped applying to a non-OpenAI host, every cost
    figure would roughly double without anything failing.
    """
    usage: dict[str, Any] = dict(message.usage_metadata or {})
    details = usage.get("input_token_details") or {}
    return LLMUsage(
        model=model,
        prompt_tokens=usage.get("input_tokens", 0),
        completion_tokens=usage.get("output_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
        cached_tokens=details.get("cache_read", 0),
        latency_ms=latency_ms,
    )


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
        chat = build_chat_model(
            model=target_model,
            # Counted against the tokens-per-minute budget *before* generation,
            # so an optimistic ceiling here fails the request outright on the
            # free tier rather than merely allowing a long answer. A full
            # extraction runs well under 3k.
            max_output_tokens=max_tokens or settings.llm_max_output_tokens,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature,
        ).bind(
            # Bound as a dict rather than handed the Pydantic class, and this is
            # the load-bearing choice in the whole file. A dict already shaped as
            # a json_schema response format is forwarded untouched, so what
            # `to_strict_json_schema` produced is what Groq receives. Passing the
            # class instead would let the SDK render the schema itself — losing
            # the `$ref` inlining that Groq's validator requires, on a path no
            # test can see because the extraction stub replaces this method.
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": to_strict_json_schema(schema),
                },
            }
        )

        message, latency_ms = await self._invoke(
            chat, [SystemMessage(system), HumanMessage(user)], target_model
        )
        usage = _usage_of(message, target_model, latency_ms)
        record_model(usage.model, settings.llm_provider)

        try:
            return StructuredResult(data=schema.model_validate_json(message.text), usage=usage)
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

        target_model = model or settings.extraction_model
        chat = build_chat_model(
            model=target_model,
            # The chat ceiling, not the extraction one: rule 7 of the system
            # prompt asks for one or two sentences, and this budget is debited
            # before generation rather than after it.
            max_output_tokens=max_tokens or settings.llm_chat_output_tokens,
            api_key=self._api_key,
            base_url=self._base_url,
            temperature=temperature,
        )
        if tools:
            # Bound as raw dicts rather than through ``bind_tools``, which would
            # convert them. The conversion is what would be lost: every optional
            # parameter is widened to ``["string", "null"]`` because Groq
            # validates tool arguments and rejects the whole message with a 400
            # when a model passes null for an optional it has nothing to say
            # about. See the schema builders in agent/tools.py.
            chat = chat.bind(tools=tools, tool_choice="auto")

        record_model(target_model, settings.llm_provider)
        async for event in self._stream(chat, messages, target_model):
            yield event

    async def _stream(
        self, chat: ChatModel, messages: list[dict[str, Any]], model: str
    ) -> AsyncIterator[StreamEvent]:
        """Open the stream, retrying only while nothing has been delivered.

        ``_invoke``'s retry loop cannot simply be reused. Once a fragment has
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
                # Chunks are folded together as they arrive rather than parsed
                # by hand. Addition is what reassembles a tool call: the name
                # lands in one chunk and its JSON arguments dribble in over the
                # next several, keyed by position in the call list.
                full: AIMessageChunk | None = None
                async for streamed in chat.astream(messages, config=chat_model_config(model)):
                    # A streaming chat model yields AIMessageChunk; the Runnable
                    # signature is the generic one and cannot say so.
                    chunk = cast(AIMessageChunk, streamed)
                    full = chunk if full is None else full + chunk
                    if chunk.text:
                        delivered = True
                        yield TextDelta(chunk.text)

                yield self._assembled(full, model, started)
                return
            except openai.APIStatusError as exc:
                # The SDK reads an error body before raising, so the provider's
                # own explanation of which field it objected to survives — the
                # part worth having, and the part a bare status code drops.
                detail = self._error_message(exc.response)
                if exc.status_code in RETRYABLE_STATUS:
                    last_error = LLMError(detail)
                    logger.warning(
                        "Groq %s (attempt %d/%d): %s",
                        exc.status_code,
                        attempt + 1,
                        settings.llm_max_retries,
                        detail,
                    )
                    if attempt < final_attempt:
                        await self._backoff(attempt, exc.response.headers.get("retry-after"))
                    continue

                logger.error("Groq %s rejected the request: %s", exc.status_code, detail)
                raise LLMError(detail) from exc
            except (openai.APIError, httpx.RequestError, RuntimeError) as exc:
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
    def _assembled(full: AIMessageChunk | None, model: str, started: float) -> TurnComplete:
        """The accumulated stream, back in the shape a plain completion returns.

        Translated rather than passed through, so that nothing above this line
        has to know a turn arrived in a few hundred pieces — or which library
        delivered it. Three details are each a silent bug if taken from the
        convenient field instead:

        ``content`` must be ``None`` and not ``""`` on a tool-calls-only turn.
        The accumulator's default is the empty string, and the chat stub the
        assistant tests run against emits ``None`` — a divergence there would let
        the loop depend on a shape the real client never sends.

        ``arguments`` must be the raw accumulated string, taken from
        ``tool_call_chunks`` rather than re-serialised from the parsed
        ``tool_calls``. The parser turns arguments cut mid-JSON into an empty
        object, which reads exactly like a call that legitimately had none; the
        loop's own decoder is what is supposed to notice, and it can only notice
        the original text.

        Truncation raises here, after the stream has drained and before the turn
        is handed over, because a tool call cut mid-JSON either fails to parse or
        parses into arguments nobody asked for.
        """
        if full is None:
            # An empty stream. Reported as an empty turn, as it always was: the
            # loop stops, and the caller answers that it found nothing to say.
            full = AIMessageChunk(content="")

        if full.response_metadata.get("finish_reason") == "length":
            raise LLMError(
                f"{model} hit the output ceiling before finishing. "
                "Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS."
            )

        message: dict[str, Any] = {"role": "assistant", "content": full.text or None}
        if full.tool_call_chunks:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or "",
                    "type": "function",
                    "function": {
                        "name": call.get("name") or "",
                        "arguments": call.get("args") or "",
                    },
                }
                for call in sorted(full.tool_call_chunks, key=lambda c: c.get("index") or 0)
            ]

        return TurnComplete(
            message=message,
            # `perf_counter` rather than the response's own elapsed time, which
            # is only populated once a streamed response has closed — after the
            # last chunk this method is assembling.
            usage=_usage_of(full, model, int((perf_counter() - started) * 1000)),
        )

    async def _invoke(
        self, chat: ChatModel, messages: list[BaseMessage], model: str
    ) -> tuple[AIMessage, int]:
        """One completion, retried on the statuses this provider means by them.

        The SDK has a retry loop of its own and it is switched off in
        ``build_chat_model``, for two reasons. Its set omits 413 — the status
        Groq returns for token-rate exhaustion — and leaving both on would
        multiply the attempts inside a request deadline sized for one of them.
        """
        last_error: Exception | None = None
        final_attempt = settings.llm_max_retries - 1

        for attempt in range(settings.llm_max_retries):
            started = perf_counter()
            try:
                message = await chat.ainvoke(messages, config=chat_model_config(model))
            except openai.LengthFinishReasonError as exc:
                # The SDK notices truncation first when a response_format is set,
                # and raises something that is deliberately *not* an APIError —
                # so it would sail past every handler below and out of this
                # module as a 500, taking /chat's 422 and matching's graceful
                # degrade with it. Retrying is pointless: the ceiling is the
                # same next time. Reworded to say which knob to turn.
                raise LLMError(
                    f"{model} hit the output ceiling before finishing. "
                    "Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS."
                ) from exc
            except openai.APIStatusError as exc:
                detail = self._error_message(exc.response)
                if exc.status_code in RETRYABLE_STATUS:
                    last_error = LLMError(detail)
                    logger.warning(
                        "Groq %s (attempt %d/%d): %s",
                        exc.status_code,
                        attempt + 1,
                        settings.llm_max_retries,
                        detail,
                    )
                    # Honour Retry-After when the limiter supplies it; guessing
                    # shorter just burns the next attempt against the same window.
                    if attempt < final_attempt:
                        await self._backoff(attempt, exc.response.headers.get("retry-after"))
                    continue

                logger.error("Groq %s rejected the request: %s", exc.status_code, detail)
                raise LLMError(detail) from exc
            except (openai.APIError, httpx.RequestError, RuntimeError) as exc:
                # Timeouts and connection failures, plus the bare RuntimeError a
                # pooled connection raises once its event loop has closed.
                last_error = exc
                # Sleeping after the last attempt delays only the exception.
                if attempt < final_attempt:
                    await self._backoff(attempt)
                continue

            # A truncated answer is otherwise completely silent: the choice comes
            # back with empty content and no tool_calls, the assistant loop reads
            # that as "nothing to say", and the user gets a shrug instead of an
            # error while their turn is still recorded. Nothing in LangChain
            # raises on this; it is an ordinary message with a field set.
            if message.response_metadata.get("finish_reason") == "length":
                raise LLMError(
                    f"{model} hit the output ceiling before finishing. "
                    "Ask for less at once, or raise LLM_CHAT_OUTPUT_TOKENS."
                )
            # A chat model always answers with an AIMessage; the Runnable
            # signature is the generic one and cannot say so.
            return cast(AIMessage, message), int((perf_counter() - started) * 1000)

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
