"""The streamed turn.

`/agent/chat/stream` is the same loop as `/agent/chat` — same tools, same
proposal, same transcript — delivered as it happens rather than at the end. So
what is worth testing is not that the assistant works, which test_agent.py
covers, but that nothing was lost or gained in the delivery: the fragments
reassemble into the answer that was saved, a proposal still has to be confirmed
separately, and a failure part-way through arrives as an event rather than as a
200 that merely stops.
"""

import json
from typing import Any

from httpx import AsyncClient
from sqlalchemy import text

from app.agent.llm_client import LLMError
from app.db.session import open_user_session
from tests.conftest import calls, says
from tests.factories import Session


async def stream(user: Session, message: str) -> list[dict[str, Any]]:
    """Drive one streamed turn and return its events in order."""
    events: list[dict[str, Any]] = []
    async with user.client.stream(
        "POST",
        "/api/v1/agent/chat/stream",
        json={"message": message},
        headers=user.headers,
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:") :]))
    return events


def texts(events: list[dict[str, Any]]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "delta")


def final(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in reversed(events) if e["type"] in {"done", "error"})


async def test_the_answer_arrives_in_fragments_that_rebuild_it(client: AsyncClient, llm) -> None:
    llm(says("Amazon is at screening."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    events = await stream(user, "How is Amazon going?")

    assert events[0] == {"type": "start"}
    # More than one, or nothing was actually streamed.
    assert len([e for e in events if e["type"] == "delta"]) > 1
    assert texts(events) == "Amazon is at screening."
    assert final(events)["message"] == "Amazon is at screening."


async def test_tools_are_named_as_they_run(client: AsyncClient, llm) -> None:
    """The point of the endpoint: a six-round turn is mostly spent waiting on
    tools, and without this the wait is indistinguishable from a hung request."""
    llm(calls("list_applications"), says("Two are still open."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    events = await stream(user, "what's open?")

    assert [e["name"] for e in events if e["type"] == "tool"] == ["list_applications"]
    assert final(events)["tools_used"] == ["list_applications"]


async def test_a_proposal_still_has_to_be_confirmed(client: AsyncClient, llm) -> None:
    """Streaming changes the delivery, not the safety boundary."""
    llm(calls("propose_event", query="Amazon", event_type="rejected"), says("Ready."))
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")

    events = await stream(user, "mark Amazon rejected")

    action = final(events)["pending_action"]
    assert action["kind"] == "append_event"
    assert action["payload"]["application_id"] == application["id"]

    async for session in open_user_session(user.user_id):
        # One event: the `applied` that created it. Nothing was written.
        assert (
            await session.execute(text("SELECT count(*) FROM application_events"))
        ).scalar_one() == 1


async def test_the_streamed_turn_is_the_one_that_was_saved(client: AsyncClient, llm) -> None:
    """The transcript and the screen have to agree, or the next turn is answered
    from a history the user never saw."""
    llm(says("Amazon is at screening."))
    user = await Session(client).start()

    events = await stream(user, "How is Amazon going?")

    async for session in open_user_session(user.user_id):
        rows = (
            await session.execute(
                text("SELECT role, content FROM agent_messages ORDER BY created_at")
            )
        ).all()

    assert [r.role for r in rows] == ["user", "assistant"]
    assert rows[1].content == texts(events) == final(events)["message"]


async def test_a_preamble_to_a_tool_call_is_withdrawn(client: AsyncClient, llm) -> None:
    """Narration before a tool call is not part of the reply.

    It is never saved, so leaving it on screen would put a sentence in the
    conversation that vanishes on the next reload.
    """
    llm(
        {**calls("list_applications"), "content": "Let me check."},
        says("Two are still open."),
    )
    user = await Session(client).start()

    events = await stream(user, "what's open?")

    kinds = [e["type"] for e in events]
    assert "superseded" in kinds
    # Withdrawn before the tool is announced, so the client never shows both.
    assert kinds.index("superseded") < kinds.index("tool")
    assert final(events)["message"] == "Two are still open."


async def test_a_model_failure_arrives_as_an_event_not_a_broken_stream(
    client: AsyncClient, llm
) -> None:
    """Once the headers are on the wire there is no status code left to send."""
    llm(error=LLMError("Groq rate limit reached."))
    user = await Session(client).start()

    events = await stream(user, "how is it going?")

    assert final(events) == {"type": "error", "detail": "Groq rate limit reached."}


async def test_a_failed_turn_saves_nothing(client: AsyncClient, llm) -> None:
    """The stream opens its own transaction, so this is the test that it rolls
    one back — a half-written turn would be replayed as history forever."""
    llm(error=LLMError("Groq rate limit reached."))
    user = await Session(client).start()

    await stream(user, "how is it going?")

    async for session in open_user_session(user.user_id):
        assert (
            await session.execute(text("SELECT count(*) FROM agent_messages"))
        ).scalar_one() == 0


async def test_the_stream_requires_authentication(client: AsyncClient) -> None:
    response = await client.post("/api/v1/agent/chat/stream", json={"message": "hi"})
    assert response.status_code == 401
