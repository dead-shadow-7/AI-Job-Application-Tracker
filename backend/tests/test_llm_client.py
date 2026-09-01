"""What the client does with a reply the model did not finish.

Truncation is the failure worth a test here because it does not look like one.
The provider returns HTTP 200 with `finish_reason='length'`, empty content and
no `tool_calls`; the assistant loop reads that as "the model had nothing to
say" and answers "I did not find anything to say about that", while the user's
turn is already committed to the transcript. Nothing logs, nothing raises, and
the only clue is a reply that quietly ignored the question.

That margin is thinner than it was — the chat ceiling is now 1,024 tokens
rather than the 3,000 an extraction gets — so the guard matters more than it
did when the ceiling was three times the largest plausible answer.
"""

from datetime import timedelta
from typing import Any

import httpx
import pytest

from app.agent.llm_client import LLMClient, LLMError

ELAPSED = timedelta(milliseconds=12)


def install(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any]) -> LLMClient:
    """An LLMClient whose HTTP layer answers every request with `body`.

    Patches `get_http_client` rather than the module global, so the test says
    nothing about how the client is cached and keeps working if that changes.
    """

    def respond(_: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, json=body)
        # MockTransport does not time the request, and reading `.elapsed` on an
        # untimed response raises. Real responses always have it by the time
        # _post looks, so this restores the shape rather than working around a
        # production problem.
        response._elapsed = ELAPSED
        return response

    stub = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("app.agent.llm_client.get_http_client", lambda: stub)
    return LLMClient(api_key="test-key", base_url="https://example.invalid/v1")


def completion(finish_reason: str, **message: Any) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


async def test_a_truncated_reply_raises_rather_than_reading_as_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = install(monkeypatch, completion("length", content=""))

    with pytest.raises(LLMError, match="output ceiling"):
        await client.chat(messages=[{"role": "user", "content": "which ones are open?"}])


async def test_a_truncated_tool_call_raises_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dangerous shape: arguments cut mid-JSON.

    Worse than a truncated sentence, because a half-written argument either
    fails to parse or parses into something the user never asked for.
    """
    client = install(
        monkeypatch,
        completion(
            "length",
            content=None,
            tool_calls=[
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "propose_status_change", "arguments": '{"applicat'},
                }
            ],
        ),
    )

    with pytest.raises(LLMError, match="output ceiling"):
        await client.chat(messages=[{"role": "user", "content": "withdraw the Amazon one"}])


async def test_a_finished_reply_is_returned_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    client = install(monkeypatch, completion("stop", content="Two are still open."))

    message, usage = await client.chat(
        messages=[{"role": "user", "content": "which ones are open?"}]
    )

    assert message["content"] == "Two are still open."
    assert usage.total_tokens == 110
    assert usage.latency_ms == 12
