"""The exact JSON that leaves this process, pinned field by field.

Every assertion here is a cost, a correctness or a compatibility guarantee that
is invisible from inside the application:

* ``response_format`` is what makes extraction schema-valid by construction. It
  is built by hand rather than by a library because ``to_strict_json_schema``
  does a ``$ref`` inlining pass no library does — see test_extraction_schema.py.
* ``max_completion_tokens`` is not a synonym for ``max_tokens``. Groq debits the
  former against the tokens-per-minute budget *before* generating, so the field
  name is a rate-limit decision, not a stylistic one.
* ``stream_options.include_usage`` is the only reason a streamed turn reports
  any tokens at all. Losing it zeroes the cache-hit log and prices every
  LangSmith run at nothing, silently and simultaneously.
* The ``["string", "null"]`` widening on optional tool parameters exists because
  Groq validates tool arguments and rejects the entire request with a 400 when
  a model passes ``null`` for an optional it has nothing to say about.

The seam is httpx's transport, so these hold regardless of what builds the
request above it. The two patch targets are separate on purpose: the extraction
and chat paths are independently movable.
"""

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import httpx
import pytest

from app.agent.llm_client import LLMClient
from app.agent.tools import TOOL_SCHEMAS
from app.core.config import settings
from app.schemas.extraction import ExtractedJob, to_strict_json_schema

# The accessor each path fetches its pooled client from. One line each, so a
# path that moves to a different builder re-points its own tests and nothing
# else.
EXTRACT_HTTP = "app.agent.models.get_http_client"
CHAT_HTTP = "app.agent.models.get_http_client"

BASE_URL = "https://example.invalid/v1"

EXTRACTED = {
    "company_name": "Acme",
    "title": "Backend Engineer",
    "seniority": "mid",
    "employment_type": "full_time",
    "work_mode": "remote",
    "location": "Pune, India",
    "salary": {
        "raw_text": None,
        "min_amount": None,
        "max_amount": None,
        "currency": None,
        "period": None,
    },
    "years_experience_min": 3,
    "years_experience_max": 5,
    "responsibilities": "Build things.",
    "requirements": [{"text": "Python", "kind": "must"}],
    "skills": ["Python"],
    "confidence": 0.9,
}


class Recorder:
    """Captures the outgoing request and replays a freshly built reply."""

    def __init__(self, build: Callable[[], httpx.Response]) -> None:
        self.build = build
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.build()

    @property
    def sent(self) -> dict[str, Any]:
        assert self.requests, "nothing was sent"
        return json.loads(self.requests[-1].content)


def install(
    monkeypatch: pytest.MonkeyPatch, target: str, build: Callable[[], httpx.Response]
) -> Recorder:
    recorder = Recorder(build)
    stub = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    monkeypatch.setattr(target, lambda: stub)
    return recorder


def completion(content: dict[str, Any]) -> Callable[[], httpx.Response]:
    def build() -> httpx.Response:
        response = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(content)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620},
            },
        )
        # httpx populates `elapsed` from the response stream, and a body handed
        # to MockTransport whole never streams — so the non-streamed path's
        # `response.elapsed.total_seconds()` would raise here and nowhere else.
        response.elapsed = timedelta(milliseconds=12)
        return response

    return build


def sse(*chunks: dict[str, Any]) -> Callable[[], httpx.Response]:
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"

    def build() -> httpx.Response:
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )

    return build


def client() -> LLMClient:
    return LLMClient(api_key="test-key", base_url=BASE_URL)


# --- Extraction -----------------------------------------------------------


async def test_the_extraction_request_pins_the_strict_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole dict, compared literally.

    A weaker assertion — "response_format is present" — would pass on
    ``{"type": "json_object"}``, which is loose JSON mode: the model returns
    valid JSON of whatever shape it likes and the failure surfaces as a
    ValidationError two layers up.
    """
    recorder = install(monkeypatch, EXTRACT_HTTP, completion(EXTRACTED))

    await client().extract(schema=ExtractedJob, system="s", user="u")

    assert recorder.sent["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "extractedjob",
            "strict": True,
            "schema": to_strict_json_schema(ExtractedJob),
        },
    }


async def test_the_extraction_request_spends_the_extraction_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = install(monkeypatch, EXTRACT_HTTP, completion(EXTRACTED))

    await client().extract(schema=ExtractedJob, system="s", user="u")
    sent = recorder.sent

    assert sent["max_completion_tokens"] == settings.llm_max_output_tokens
    # The deprecated field would not be debited before generation, which is the
    # only reason the ceiling is set at all.
    assert "max_tokens" not in sent
    assert sent["temperature"] == 0.0
    assert sent["model"] == settings.extraction_model


async def test_extraction_does_not_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structured output and token streaming are not combined anywhere.

    Recorded rather than assumed: on the chat-completions API a request that
    carries ``response_format`` is sent unstreamed, so anyone who later wants
    both should find this test rather than discover the interaction.

    The field is now sent explicitly as false where it used to be omitted —
    the same request, stated rather than implied.
    """
    recorder = install(monkeypatch, EXTRACT_HTTP, completion(EXTRACTED))

    await client().extract(schema=ExtractedJob, system="s", user="u")

    assert recorder.sent["stream"] is False
    assert "stream_options" not in recorder.sent


async def test_the_request_goes_to_the_configured_host_with_a_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = install(monkeypatch, EXTRACT_HTTP, completion(EXTRACTED))

    await client().extract(schema=ExtractedJob, system="s", user="u")
    request = recorder.requests[-1]

    assert str(request.url) == f"{BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"


# --- Chat -----------------------------------------------------------------


async def drain(tools: list[dict[str, Any]] | None) -> None:
    async for _ in client().stream_chat(messages=[{"role": "user", "content": "hi"}], tools=tools):
        pass


async def test_the_chat_request_asks_for_usage_with_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without this the turn reports zero tokens and nothing says so."""
    recorder = install(
        monkeypatch, CHAT_HTTP, sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    )

    await drain(None)
    sent = recorder.sent

    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


async def test_the_chat_request_spends_the_smaller_chat_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = install(
        monkeypatch, CHAT_HTTP, sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    )

    await drain(None)
    sent = recorder.sent

    assert sent["max_completion_tokens"] == settings.llm_chat_output_tokens
    assert sent["max_completion_tokens"] < settings.llm_max_output_tokens
    assert "max_tokens" not in sent
    # Loose JSON mode on a tool-calling turn would suppress the tool calls.
    assert "response_format" not in sent


async def test_the_tool_schemas_are_sent_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identical, and only when there are tools to send.

    Any conversion layer between ``TOOL_SCHEMAS`` and the wire is a place the
    null widening below can be normalised away.
    """
    recorder = install(
        monkeypatch, CHAT_HTTP, sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    )

    await drain(TOOL_SCHEMAS)
    sent = recorder.sent

    assert sent["tools"] == TOOL_SCHEMAS
    assert sent["tool_choice"] == "auto"


async def test_no_tool_choice_is_sent_when_there_are_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = install(
        monkeypatch, CHAT_HTTP, sse({"choices": [{"index": 0, "delta": {"content": "hi"}}]})
    )

    await drain(None)

    assert "tools" not in recorder.sent
    assert "tool_choice" not in recorder.sent


# --- The tool schemas themselves ------------------------------------------


def parameters() -> list[dict[str, Any]]:
    return [tool["function"]["parameters"] for tool in TOOL_SCHEMAS]


def test_optional_tool_parameters_accept_null() -> None:
    """A bare ``{"type": "string"}`` on an optional argument is a 400 waiting.

    Models treat "omit it" and "pass null" as the same intent. Groq does not,
    and rejects the whole message rather than the one argument.
    """
    widened = [
        (name, schema)
        for params in parameters()
        for name, schema in params["properties"].items()
        if name not in params["required"]
    ]

    assert widened, "expected at least one optional tool parameter"
    for name, schema in widened:
        assert isinstance(schema["type"], list), name
        assert "null" in schema["type"], name
        if "enum" in schema:
            assert None in schema["enum"], name


def test_required_tool_parameters_stay_strict() -> None:
    """Null in a required argument is a real mistake and should be caught."""
    for params in parameters():
        for name in params["required"]:
            assert params["properties"][name]["type"] == "string" or isinstance(
                params["properties"][name]["type"], str
            ), name
