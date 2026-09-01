"""A thin, typed Groq client.

Groq's API is OpenAI-compatible, so this is a small amount of code rather than a
dependency. It is written directly instead of through a LangChain chat wrapper
for one reason: strict schema adherence is the entire value of this call, and a
wrapper's ``with_structured_output`` may emit function-calling or loose JSON
mode depending on version. Here the exact ``response_format`` sent is visible
and pinned.
"""

import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.agent.tracing import hide, traced
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


class LLMClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._api_key = api_key if api_key is not None else settings.llm_api_key
        self._base_url = (base_url or settings.llm_base_url).rstrip("/")

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

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
        content = body["choices"][0]["message"]["content"]

        try:
            return StructuredResult(data=schema.model_validate_json(content), usage=usage)
        except ValidationError as exc:
            logger.error("Model returned schema-invalid JSON for %s: %s", schema.__name__, exc)
            raise LLMError(
                f"{target_model} returned JSON that does not match {schema.__name__}."
            ) from exc

    @traced("chat", run_type="llm", process_inputs=hide("self", "tools"))
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], LLMUsage]:
        """One turn of a tool-calling conversation.

        Returns the raw assistant message so the caller can drive the loop:
        either it carries ``tool_calls`` to execute, or it carries ``content``
        and the loop is done. Deliberately not wrapped in a framework — the
        loop is a dozen lines and keeping it visible means the stopping
        condition and the tool-result plumbing are inspectable rather than
        inherited.
        """
        if not self.is_configured:
            raise LLMError("No LLM API key is configured.")

        payload: dict[str, Any] = {
            "model": model or settings.extraction_model,
            "temperature": temperature,
            "max_completion_tokens": max_tokens or settings.llm_max_output_tokens,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body, usage = await self._post(payload, payload["model"])
        return body["choices"][0]["message"], usage

    async def _post(self, payload: dict[str, Any], model: str) -> tuple[dict[str, Any], LLMUsage]:
        last_error: Exception | None = None

        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            for attempt in range(settings.llm_max_retries):
                try:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=payload,
                    )
                except httpx.RequestError as exc:
                    last_error = exc
                    await self._backoff(attempt)
                    continue

                if response.status_code == 200:
                    body = response.json()
                    raw = body.get("usage", {})
                    return body, LLMUsage(
                        model=model,
                        prompt_tokens=raw.get("prompt_tokens", 0),
                        completion_tokens=raw.get("completion_tokens", 0),
                        total_tokens=raw.get("total_tokens", 0),
                        cached_tokens=(raw.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        ),
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
        import asyncio

        if retry_after:
            try:
                await asyncio.sleep(min(float(retry_after), 30.0))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(2**attempt, 8))


llm_client = LLMClient()
