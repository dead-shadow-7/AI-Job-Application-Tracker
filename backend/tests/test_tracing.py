"""The token and cost path, end to end, with nothing uploaded.

Tracing is switched off across the suite, so what is testable here is the
*translation* — provider usage into LLMUsage into the shape LangSmith reads —
and that chain had no coverage at all. It is worth some, because every link in
it fails by reporting a plausible wrong number rather than by raising.

The cache discount is the reason. The system prompt and the tool schemas are
about three thousand tokens of identical prefix on every round, a cache hit
halves what they cost, and a broken link anywhere along here shows up only as an
invoice roughly twice the size of the dashboard.
"""

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.agent.llm_client import LLMUsage, _usage_of
from app.agent.models import chat_model_config
from app.agent.tracing import as_llm_run, hide
from app.core.config import settings

USAGE = {
    "input_tokens": 3200,
    "output_tokens": 23,
    "total_tokens": 3223,
    "input_token_details": {"cache_read": 3100},
    "output_token_details": {},
}


def test_a_cached_prompt_reaches_llm_usage() -> None:
    """The first of the three links: LangChain's shape into ours."""
    usage = _usage_of(AIMessage(content="ok", usage_metadata=USAGE), "some-model", 1200)

    assert usage.prompt_tokens == 3200
    assert usage.completion_tokens == 23
    assert usage.total_tokens == 3223
    assert usage.cached_tokens == 3100
    assert usage.latency_ms == 1200
    assert usage.model == "some-model"


def test_the_cache_hit_rate_is_computed_from_the_discounted_share() -> None:
    usage = _usage_of(AIMessage(content="ok", usage_metadata=USAGE), "m", 0)

    assert usage.cache_hit_rate == pytest.approx(3100 / 3200)


def test_a_turn_with_no_usage_reports_zero_rather_than_raising() -> None:
    """A provider that omits usage should cost nothing, not fail the request.

    Streaming is where this happens: usage arrives in a trailing chunk that only
    appears because it was asked for, and a host that ignores the request sends
    the answer perfectly well without it.
    """
    usage = _usage_of(AIMessageChunk(content="ok"), "m", 5)

    assert usage.total_tokens == 0
    assert usage.cached_tokens == 0
    assert usage.cache_hit_rate == 0.0


def test_the_cache_read_survives_into_what_langsmith_prices() -> None:
    """The last link: our shape into the one the trace is costed from.

    `usage_metadata` is the documented contract — anything else is stored as an
    opaque blob and the run shows no tokens and no cost at all.
    """
    usage = LLMUsage(
        model="m", prompt_tokens=3200, completion_tokens=23, total_tokens=3223, cached_tokens=3100
    )

    run = as_llm_run([{"role": "assistant", "content": "ok"}], usage)

    assert run["usage_metadata"] == {
        "input_tokens": 3200,
        "output_tokens": 23,
        "total_tokens": 3223,
        "input_token_details": {"cache_read": 3100},
    }


def test_the_run_is_attributed_to_the_provider_that_answered() -> None:
    """ChatOpenAI would report `openai` for all three hosts, being the dialect
    it speaks. Cost is attributed from exactly these two keys, and every run
    recorded so far names the real provider, so accepting the default would
    split the history and reprice it against the wrong catalogue."""
    metadata = chat_model_config(settings.extraction_model)["metadata"]

    assert metadata["ls_provider"] == settings.llm_provider
    assert metadata["ls_model_name"] == settings.extraction_model


def test_the_tool_block_is_kept_out_of_the_trace_input() -> None:
    """Around three thousand tokens of identical JSON, on every round of every
    turn. Readable traces are the smaller half of the reason; the upload volume
    is the larger one."""
    process = hide("self", "tools")

    kept = process({"self": object(), "tools": [{"a": 1}], "messages": [{"role": "user"}]})

    assert kept == {"messages": [{"role": "user"}]}


async def test_a_streamed_turn_reports_its_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole chain over a real stream, because the pieces above pass even
    when the trailing usage chunk is never requested — which is the default
    whenever a base_url or an HTTP client is supplied, and this does both."""
    from app.agent.llm_client import LLMClient, TurnComplete

    body = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Two open."}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        'data: {"choices":[],"usage":{"prompt_tokens":3200,"completion_tokens":23,'
        '"total_tokens":3223,"prompt_tokens_details":{"cached_tokens":3100}}}\n\n'
        "data: [DONE]\n\n"
    )
    monkeypatch.setattr(
        "app.agent.models.get_http_client",
        lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, content=body.encode(), headers={"content-type": "text/event-stream"}
                )
            )
        ),
    )
    client = LLMClient(api_key="k", base_url="https://example.invalid/v1")

    events = [
        event async for event in client.stream_chat(messages=[{"role": "user", "content": "hi"}])
    ]

    finished = events[-1]
    assert isinstance(finished, TurnComplete)
    assert finished.usage.total_tokens == 3223
    assert finished.usage.cached_tokens == 3100
