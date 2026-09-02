"""Assumptions about the installed packages that the code silently relies on.

These are not version-number assertions — a floor in pyproject.toml already does
that, and a test restating it would only ever fail for the same reason. What is
tested here is the thing a version bump would break without saying so.

The httpx one is the sharp one. `openai` 3.x is built on the separate `httpx2`
distribution, and an object handed to the SDK client has to come from whichever
httpx the SDK itself uses. The pooled AsyncClient this application builds is a
plain `httpx.AsyncClient`, so the day `openai>=3` resolves, injecting it stops
working — and the failure surfaces as a type error deep inside the SDK, not as a
dependency conflict.
"""

import httpx
import openai
import pytest
from langchain_openai import ChatOpenAI


def test_the_openai_sdk_and_this_application_share_one_httpx() -> None:
    """The precondition for injecting our own connection pool.

    If this fails, `openai>=3` has resolved and `http_async_client` no longer
    accepts a plain `httpx.AsyncClient`. The fix is not to remove this test: it
    is to rebuild the pool from `langchain_openai._compat.httpx` and re-point
    the transport-level tests at that class.
    """
    from openai import _base_client

    assert _base_client.httpx is httpx
    assert openai.version.VERSION.startswith("2.")


def test_a_pooled_client_reaches_the_sdk_unchanged() -> None:
    """Injection is what carries the pool, the split timeout and the loop keying.

    Asserted through the object identity rather than by making a request,
    because what matters is that the SDK uses *this* client — a copy would keep
    its own connections and its own timeouts.
    """
    pool = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
    )

    model = ChatOpenAI(
        model="test", api_key="k", base_url="https://example.invalid/v1", http_async_client=pool
    )

    assert model.root_async_client._client is pool


def test_streaming_usage_is_off_until_it_is_asked_for() -> None:
    """The default this codebase must override, pinned so the override survives.

    ChatOpenAI turns `stream_usage` on only for the stock OpenAI host with the
    stock client. Setting a base_url or injecting a client — both of which this
    application does — leaves it unset, and an unset value means a streamed turn
    reports no tokens at all: the cache-hit log goes quiet and LangSmith prices
    the run at zero, neither of which raises.
    """
    model = ChatOpenAI(
        model="test",
        api_key="k",
        base_url="https://example.invalid/v1",
        http_async_client=httpx.AsyncClient(),
    )

    assert not model.stream_usage


def test_a_missing_key_fails_at_construction_not_at_the_call() -> None:
    """Why the model is built lazily, after `is_configured` has been checked.

    Four routes answer "no LLM is configured" with a friendly message. Building
    a chat model eagerly — at import, or before that guard — would replace it
    with an OpenAIError from inside the SDK.
    """
    with pytest.raises(openai.OpenAIError, match="Missing credentials"):
        ChatOpenAI(model="test", api_key="", base_url="https://example.invalid/v1")
