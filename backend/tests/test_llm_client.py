"""What the client does with a reply the model streamed badly.

Truncation is the failure worth a test here because it does not look like one.
The provider returns HTTP 200 with `finish_reason='length'`, empty content and
no `tool_calls`; the assistant loop reads that as "the model had nothing to
say" and answers "I did not find anything to say about that", while the user's
turn is already committed to the transcript. Nothing logs, nothing raises, and
the only clue is a reply that quietly ignored the question.

That margin is thinner than it was — the chat ceiling is now 1,024 tokens
rather than the 3,000 an extraction gets — so the guard matters more than it
did when the ceiling was three times the largest plausible answer.

The second thing tested here is reassembly. A streamed tool call arrives as a
name in one chunk and its JSON arguments spread over the next several, and
putting them back together wrongly produces arguments that parse — into
something nobody asked for.
"""

import json
from datetime import timedelta
from typing import Any

import httpx
import pytest

from app.agent.llm_client import LLMClient, LLMError, TextDelta, TurnComplete
from app.schemas.extraction import ExtractedJob


def install(monkeypatch: pytest.MonkeyPatch, *chunks: dict[str, Any]) -> LLMClient:
    """An LLMClient whose HTTP layer streams `chunks` back as server-sent events.

    Patches `get_http_client` rather than the module global, so the test says
    nothing about how the client is cached and keeps working if that changes.
    """
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content=body.encode(), headers={"content-type": "text/event-stream"}
            )
        )
    )
    monkeypatch.setattr("app.agent.llm_client.get_http_client", lambda: stub)
    return LLMClient(api_key="test-key", base_url="https://example.invalid/v1")


def delta(finish_reason: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": fields, "finish_reason": finish_reason}]}


def spent(**counts: int) -> dict[str, Any]:
    """The usage-only chunk that closes a stream when include_usage is on."""
    return {"choices": [], "usage": counts}


async def collect(client: LLMClient, message: str) -> tuple[str, list[Any]]:
    """Drive one turn, returning the streamed text and every event in order."""
    events = [
        event async for event in client.stream_chat(messages=[{"role": "user", "content": message}])
    ]
    return "".join(e.text for e in events if isinstance(e, TextDelta)), events


async def test_a_truncated_reply_raises_rather_than_reading_as_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install(monkeypatch, delta(content="Two are"), delta("length"))

    with pytest.raises(LLMError, match="output ceiling"):
        await collect(client, "which ones are open?")


async def test_a_truncated_tool_call_raises_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous shape: arguments cut mid-JSON.

    Worse than a truncated sentence, because a half-written argument either
    fails to parse or parses into something the user never asked for.
    """
    client = install(
        monkeypatch,
        delta(
            tool_calls=[
                {
                    "index": 0,
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "propose_status_change", "arguments": '{"applicat'},
                }
            ]
        ),
        delta("length"),
    )

    with pytest.raises(LLMError, match="output ceiling"):
        await collect(client, "withdraw the Amazon one")


async def test_a_finished_reply_arrives_in_fragments_and_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install(
        monkeypatch,
        delta(role="assistant", content="Two "),
        delta(content="are still "),
        delta(content="open."),
        delta("stop"),
        spent(prompt_tokens=100, completion_tokens=10, total_tokens=110),
    )

    streamed, events = await collect(client, "which ones are open?")

    assert streamed == "Two are still open."
    finished = events[-1]
    assert isinstance(finished, TurnComplete)
    assert finished.message["content"] == "Two are still open."
    assert finished.usage.total_tokens == 110


async def test_a_tool_call_split_across_chunks_is_put_back_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The name arrives once; the arguments dribble in and must be concatenated.

    Assigning rather than appending here would leave `{"query": ` as the whole
    argument object — which fails to parse, and the loop treats an unparseable
    argument list as an empty one.
    """
    client = install(
        monkeypatch,
        delta(
            tool_calls=[
                {"index": 0, "id": "call_0", "function": {"name": "propose_event"}},
            ]
        ),
        delta(tool_calls=[{"index": 0, "function": {"arguments": '{"query"'}}]),
        delta(tool_calls=[{"index": 0, "function": {"arguments": ': "Amazon"}'}}]),
        delta("tool_calls"),
    )

    _, events = await collect(client, "mark Amazon rejected")

    finished = events[-1]
    assert isinstance(finished, TurnComplete)
    assert finished.message["content"] is None
    call = finished.message["tool_calls"][0]
    assert call["id"] == "call_0"
    assert call["function"]["name"] == "propose_event"
    assert json.loads(call["function"]["arguments"]) == {"query": "Amazon"}


async def test_the_provider_s_own_error_survives_a_streamed_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed response arrives with its body unread.

    Reading the status alone would report "Groq returned 400" and drop the
    sentence saying which field it objected to — the only part worth having.
    """
    stub = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(400, json={"error": {"message": "stream_options unsupported"}})
        )
    )
    monkeypatch.setattr("app.agent.llm_client.get_http_client", lambda: stub)
    client = LLMClient(api_key="test-key", base_url="https://example.invalid/v1")

    with pytest.raises(LLMError, match="stream_options unsupported"):
        await collect(client, "hello")


async def test_a_truncated_extraction_raises_on_the_unstreamed_path_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same guard, on the call that has no stream to watch.

    Structured extraction is where truncation is most deceptive: the reply is
    valid JSON right up to the point it stops, so a cut-off response can parse
    into a job with half its fields missing and be saved as a real extraction.
    Nothing in the model layer objects — `finish_reason` is a field on an
    otherwise ordinary message — so this has to be checked deliberately.
    """

    def truncated(_: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"company_name": "Ac'},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 900, "completion_tokens": 3000, "total_tokens": 3900},
            },
        )
        response.elapsed = timedelta(milliseconds=10)
        return response

    monkeypatch.setattr(
        "app.agent.models.get_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(truncated)),
    )
    client = LLMClient(api_key="test-key", base_url="https://example.invalid/v1")

    with pytest.raises(LLMError, match="output ceiling"):
        await client.extract(schema=ExtractedJob, system="s", user="u")
