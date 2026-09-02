"""Builds the chat model every LLM call goes through.

One ``ChatOpenAI`` pointed at three base URLs, matching the provider table in
core/config.py. Not ``ChatGroq``, and not one class per provider: ``ChatGroq``
defaults to function-calling for structured output and silently downgrades
``strict=True`` to nothing for every model but two, which is exactly the failure
the pinned ``response_format`` in llm_client.py exists to prevent. ``ChatOpenAI``
pointed at Groq's own OpenAI-compatible host keeps the strict path and raises
rather than degrading when a host will not honour it.

Every keyword below is load-bearing, and three of them are corrections to a
default rather than a preference. They are listed in the tests next to what
breaks without them, because none of the three fails loudly.
"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.agent.http_client import get_http_client, get_sync_http_client
from app.core.config import settings

ChatModel = Runnable[LanguageModelInput, BaseMessage]


def build_chat_model(
    *,
    model: str,
    max_output_tokens: int,
    api_key: str,
    base_url: str,
    temperature: float = 0.0,
) -> ChatModel:
    """A model bound to its output ceiling, ready to invoke or stream.

    Built per call rather than cached. The async client it is handed belongs to
    one event loop, and holding a model across loops would freeze a stale pool
    inside it — reintroducing precisely the ``Event loop is closed`` failure
    ``get_http_client`` is keyed to avoid. Construction is cheap once both HTTP
    clients are supplied; the handshake was the expense, and that is still
    pooled.
    """
    chat = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        # The pool, the split timeout and the loop keying all arrive with these.
        http_async_client=get_http_client(),
        http_client=get_sync_http_client(),
        # Off by default whenever a base_url is set *or* a client is injected,
        # and this does both. Without it a streamed turn reports no tokens at
        # all: the cache-hit rate the assistant logs goes to zero and LangSmith
        # prices the run at nothing, neither of which raises.
        stream_usage=True,
        # Retries stay with LLMClient. The SDK's set is {408, 409, 429, >=500}
        # and omits 413 — which is the one Groq returns for token-rate
        # exhaustion. Leaving both layers on would also multiply the attempts
        # inside a request deadline sized for one of them.
        max_retries=0,
        timeout=settings.llm_timeout_seconds,
        # `max_tokens` maps to the deprecated wire field. Groq debits
        # `max_completion_tokens` against the tokens-per-minute budget *before*
        # generating, so the name is a rate-limit decision rather than a synonym.
        max_tokens=None,
    )
    return chat.bind(max_completion_tokens=max_output_tokens)


def chat_model_config(model: str) -> dict[str, Any]:
    """Per-invocation config naming the provider LangSmith should price against.

    ``ChatOpenAI`` reports ``ls_provider="openai"`` for all three hosts, because
    that is the dialect it speaks. Every run this project has recorded says
    groq, gemini or aicredits, and cost is attributed from exactly these two
    keys — so leaving the default in place would split the history and quietly
    reprice it against the wrong catalogue.
    """
    return {
        "metadata": {"ls_provider": settings.llm_provider, "ls_model_name": model},
    }
