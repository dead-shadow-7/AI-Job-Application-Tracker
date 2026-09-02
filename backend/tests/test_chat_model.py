"""The chat model's construction, which is almost entirely overrides.

Four of the keywords in ``build_chat_model`` correct a LangChain or OpenAI-SDK
default rather than express a preference, and not one of them fails loudly when
dropped. That is what this file is for: each test names the default it is
holding back and what goes wrong when it wins.

The pool tests are the other half. The model is built per call and handed a
client that belongs to one event loop, so "is it the same client" and "does a
new loop get a new one" are the two questions that decide whether the pool works
or whether it raises from inside httpcore some time later.
"""

import asyncio
import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from app.agent import http_client
from app.agent.models import build_chat_model, chat_model_config
from app.core.config import Settings, settings


@pytest.fixture(autouse=True)
def _fresh_pool() -> None:
    """The pool is a process singleton; these tests build and discard several."""
    http_client._async_client = None
    http_client._async_loop = None
    http_client._sync_client = None


def model(**overrides: object) -> object:
    defaults = {
        "model": "test-model",
        "max_output_tokens": 1024,
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
    }
    return build_chat_model(**{**defaults, **overrides})  # type: ignore[arg-type]


def chat(bound: object) -> object:
    """The ChatOpenAI underneath the ceiling binding."""
    return bound.bound  # type: ignore[attr-defined]


# --- The defaults being held back -----------------------------------------


async def test_streamed_usage_is_asked_for_explicitly() -> None:
    """Off by default whenever a base_url is set or a client is injected.

    This does both, so the default would be off, and off means every streamed
    turn reports zero tokens — the cache-hit log goes quiet and LangSmith prices
    the run at nothing. Three silent failures from one omitted keyword.
    """
    assert chat(model()).stream_usage is True


async def test_the_sdk_does_not_retry_underneath_our_retry_loop() -> None:
    """Its set omits 413, which is the one Groq means token-rate-limit by.

    Two retry layers would also multiply: three SDK attempts inside three of
    ours is nine, inside a request deadline sized for three.
    """
    assert chat(model()).max_retries == 0


async def test_the_ceiling_is_spent_as_max_completion_tokens() -> None:
    """`max_tokens` is the deprecated wire field and is not a synonym here.

    Groq debits `max_completion_tokens` against the tokens-per-minute budget
    before generating, so the wrong name spends the budget it was meant to cap.
    """
    bound = model(max_output_tokens=777)

    assert chat(bound).max_tokens is None
    assert bound.kwargs["max_completion_tokens"] == 777  # type: ignore[attr-defined]


async def test_the_two_ceilings_stay_distinct() -> None:
    """Extraction gets three thousand tokens; a chat turn gets one.

    Reserving the extraction ceiling on chat spent a third of the free-tier
    budget on output that never arrived, on every one of up to six rounds.
    """
    extraction = model(max_output_tokens=settings.llm_max_output_tokens)
    conversation = model(max_output_tokens=settings.llm_chat_output_tokens)

    assert (
        conversation.kwargs["max_completion_tokens"]  # type: ignore[attr-defined]
        < extraction.kwargs["max_completion_tokens"]  # type: ignore[attr-defined]
    )


# --- The pool -------------------------------------------------------------


async def test_every_model_on_one_loop_shares_one_connection_pool() -> None:
    """Building a client per call discarded the keep-alive with it.

    That cost a TCP and TLS handshake per round — six in an assistant turn — so
    the model may be rebuilt freely but the pool underneath may not.
    """
    first, second = model(), model()

    assert chat(first).root_async_client._client is chat(second).root_async_client._client


async def test_the_sync_client_the_sdk_insists_on_is_shared_too() -> None:
    """Nothing sends through it; the SDK builds one anyway.

    Left to itself it builds a fresh one per model, and the model is built per
    request — so the leak is one pool per LLM call, never closed.
    """
    first, second = model(), model()

    assert chat(first).root_client._client is chat(second).root_client._client


def test_a_second_event_loop_gets_its_own_pool() -> None:
    """A keep-alive connection belongs to the loop that opened it.

    After that loop closes the client still reports `is_closed == False`, so
    reusing it raises `Event loop is closed` from inside httpcore rather than
    reconnecting. The scheduled sweep and any script run on a second loop.
    """
    clients = []

    async def build() -> None:
        clients.append(chat(model()).root_async_client._client)

    asyncio.run(build())
    asyncio.run(build())

    assert clients[0] is not clients[1]


async def test_the_pool_splits_connect_from_read() -> None:
    """One scalar made a dead host and a long generation indistinguishable."""
    timeout = http_client.get_http_client().timeout

    assert timeout.connect == 10.0
    assert timeout.read == settings.llm_timeout_seconds


# --- The provider table ---------------------------------------------------


@pytest.mark.parametrize("provider", ["groq", "gemini", "aicredits"])
async def test_each_provider_is_the_same_class_at_a_different_host(provider: str) -> None:
    """One ChatOpenAI, three base URLs — not three provider classes.

    ChatGroq would be the obvious choice for Groq and is the wrong one: it
    defaults to function-calling for structured output and silently ignores
    strict mode for every model but two.
    """
    configured = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider=provider,
        groq_api_key="groq-key",
        gemini_api_key="gemini-key",
        aicredits_api_key="aicredits-key",
    )

    built = chat(model(api_key=configured.llm_api_key, base_url=configured.llm_base_url))

    assert built.openai_api_key.get_secret_value() == f"{provider}-key"
    assert str(built.openai_api_base) == configured.llm_base_url


def test_the_trace_names_the_real_provider_not_the_dialect() -> None:
    """ChatOpenAI reports `openai` for all three hosts, because that is the
    dialect it speaks. LangSmith prices from exactly these two keys, and every
    run recorded so far names the actual provider — so the default would split
    the history and reprice it against the wrong catalogue."""
    metadata = chat_model_config("some-model")["metadata"]

    assert metadata["ls_provider"] == settings.llm_provider
    assert metadata["ls_provider"] != "openai"
    assert metadata["ls_model_name"] == "some-model"


# --- What actually reaches the wire ---------------------------------------


async def test_the_pinned_response_format_survives_the_model_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single assumption the whole migration rests on.

    A dict already shaped as a json_schema response format is passed through
    rather than rebuilt, so `strict: true` and the inlined schema arrive exactly
    as written. Were it rebuilt from a Pydantic model instead, the `$ref`
    inlining would be lost and every real extraction would 400.
    """
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "{}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    monkeypatch.setattr(
        "app.agent.models.get_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    pinned = {
        "type": "json_schema",
        "json_schema": {
            "name": "extractedjob",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
                "additionalProperties": False,
            },
        },
    }

    await model().bind(response_format=pinned).ainvoke([HumanMessage("u")])  # type: ignore[attr-defined]

    assert sent[0]["response_format"] == pinned


async def test_cached_prompt_tokens_survive_the_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discount is most of the bill and is reported in a provider-specific
    field. LangChain normalises `prompt_tokens_details.cached_tokens` into
    `input_token_details.cache_read`; if that mapping ever stopped applying to a
    non-OpenAI host, the displayed cost would silently double."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 900},
                },
            },
        )

    monkeypatch.setattr(
        "app.agent.models.get_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    message = await model().ainvoke([HumanMessage("u")])  # type: ignore[attr-defined]

    assert message.usage_metadata["input_token_details"]["cache_read"] == 900
    assert message.usage_metadata["input_tokens"] == 1000
