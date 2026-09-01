"""The assistant, driven by a scripted model.

What is worth testing is not whether the model understands English — that is
the model's problem. It is that a model which misunderstands *cannot cause
damage*: no tool writes, /chat leaves the database unchanged, ambiguity becomes
a question, and a proposal names a row the user sees before anything happens.

The stub is scripted turn by turn so a tool-calling loop can be exercised
without the network. It patches `app.agent.assistant.llm_client`, which is
where the loop resolves it — patching the endpoint's import instead would let
the tests silently reach the real API.
"""

import json
from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.agent.llm_client import LLMError, LLMUsage
from app.db.session import open_user_session
from tests.factories import Session


def says(content: str) -> dict[str, Any]:
    return {"role": "assistant", "content": content}


def calls(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class ScriptedLLM:
    """Returns each scripted turn in order."""

    def __init__(self, *turns: dict[str, Any], error: Exception | None = None) -> None:
        self._turns = list(turns) or [says("Nothing to do.")]
        self._error = error
        self.calls = 0
        self.is_configured = True
        self.messages_seen: list[list[dict[str, Any]]] = []

    async def chat(self, *, messages: list[dict[str, Any]], **_: Any):
        self.messages_seen.append(messages)
        if self._error:
            raise self._error
        turn = self._turns[min(self.calls, len(self._turns) - 1)]
        self.calls += 1
        return turn, LLMUsage(model="stub", total_tokens=500, latency_ms=100)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch):
    def install(*turns: dict[str, Any], error: Exception | None = None) -> ScriptedLLM:
        stub = ScriptedLLM(*turns, error=error)
        monkeypatch.setattr("app.agent.assistant.llm_client", stub)
        monkeypatch.setattr("app.api.v1.agent.llm_client", stub)
        return stub

    return install


async def event_count(user: Session) -> int:
    async for session in open_user_session(user.user_id):
        return (await session.execute(text("SELECT count(*) FROM application_events"))).scalar_one()
    return 0


# --- Answering with tools --------------------------------------------------


async def test_a_question_is_answered_without_proposing_anything(client: AsyncClient, llm) -> None:
    llm(says("Amazon is at screening."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    body = (await user.post("/api/v1/agent/chat", {"message": "How is Amazon going?"})).json()

    assert body["message"] == "Amazon is at screening."
    assert body["pending_action"] is None


async def test_the_model_can_look_up_detail_it_was_not_given(client: AsyncClient, llm) -> None:
    """The gap that made "what skills did it ask for" unanswerable: skills and
    requirements were never in the prompt. A tool fetches them on demand."""
    stub = llm(
        calls("get_application_details", query="Amazon"),
        says("It asks for Python and PostgreSQL."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")

    body = (await user.post("/api/v1/agent/chat", {"message": "what skills?"})).json()

    assert stub.calls == 2, "one call to request the tool, one to answer"
    assert body["message"] == "It asks for Python and PostgreSQL."
    # The tool result must reach the model, or it is answering from nothing.
    tool_turns = [m for m in stub.messages_seen[-1] if m.get("role") == "tool"]
    assert tool_turns and "Python" in tool_turns[0]["content"]


# --- Memory ----------------------------------------------------------------


async def test_earlier_turns_are_replayed_to_the_model(client: AsyncClient, llm) -> None:
    """Without this, "what skills did it ask for" followed by "Amazon" loses
    the question — which is exactly what happened before memory existed."""
    stub = llm(says("Which application?"), says("Python and PostgreSQL."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    await user.post("/api/v1/agent/chat", {"message": "what skills did it ask for"})
    await user.post("/api/v1/agent/chat", {"message": "Amazon"})

    replayed = [
        m["content"] for m in stub.messages_seen[-1] if m.get("role") in {"user", "assistant"}
    ]
    assert "what skills did it ask for" in replayed
    assert "Which application?" in replayed


async def test_history_is_scoped_to_one_user(client: AsyncClient, llm) -> None:
    stub = llm(says("Noted."))
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()

    await alice.post("/api/v1/agent/chat", {"message": "alice's private question"})
    await bob.post("/api/v1/agent/chat", {"message": "bob's question"})

    replayed = " ".join(str(m.get("content")) for m in stub.messages_seen[-1])
    assert "alice's private question" not in replayed


# --- The safety boundary ---------------------------------------------------


async def test_chat_never_writes(client: AsyncClient, llm) -> None:
    """The single most important property. Even an unambiguous instruction
    produces a proposal, not a change."""
    llm(
        calls("propose_event", query="Amazon", event_type="rejected"),
        says("I'll mark Amazon as rejected once you confirm."),
    )
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")
    before = await event_count(user)

    body = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()
    after = (await user.get(f"/api/v1/applications/{application['id']}")).json()

    assert body["pending_action"] is not None
    assert await event_count(user) == before, "/chat must not append anything"
    assert after["current_status"] == "applied"


async def test_a_proposal_names_the_resolved_row_not_the_phrase(client: AsyncClient, llm) -> None:
    llm(
        calls("propose_event", query="Amazon", event_type="rejected"),
        says("Confirm to record it."),
    )
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", title="Backend Engineer", initial_event="applied"
    )

    action = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()[
        "pending_action"
    ]

    assert action["application_id"] == application["id"]
    assert "Backend Engineer" in action["application_label"]


async def test_ambiguity_produces_no_proposal(client: AsyncClient, llm) -> None:
    """Two roles at one company. Picking one would write to a timeline the user
    never chose, and they would not find out for weeks."""
    stub = llm(
        calls("propose_event", query="Amazon", event_type="rejected"),
        says("Which Amazon role do you mean?"),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Amazon", title="Data Engineer")

    body = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()

    assert body["pending_action"] is None, "must not choose between equal matches"
    tool_output = [m for m in stub.messages_seen[-1] if m.get("role") == "tool"][0]["content"]
    assert "Data Engineer" in tool_output, "the model is handed the options to ask about"


async def test_an_unknown_company_proposes_nothing(client: AsyncClient, llm) -> None:
    llm(
        calls("propose_event", query="Spotify", event_type="rejected"),
        says("You are not tracking anything at Spotify."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    body = (await user.post("/api/v1/agent/chat", {"message": "reject Spotify"})).json()

    assert body["pending_action"] is None


async def test_a_hallucinated_tool_does_not_crash_the_request(client: AsyncClient, llm) -> None:
    """A model inventing a tool should be told so and allowed to recover."""
    llm(calls("delete_everything", query="Amazon"), says("I cannot do that."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    response = await user.post("/api/v1/agent/chat", {"message": "delete it all"})

    assert response.status_code == 200
    assert response.json()["pending_action"] is None


async def test_the_loop_terminates_on_a_model_that_never_answers(client: AsyncClient, llm) -> None:
    """A model looping on tool calls must not spend the budget indefinitely."""
    stub = llm(calls("list_applications"))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    body = (await user.post("/api/v1/agent/chat", {"message": "hello"})).json()

    assert stub.calls <= 4
    assert "could not work that out" in body["message"].lower()


# --- Confirming ------------------------------------------------------------


async def test_confirming_appends_an_attributable_event(client: AsyncClient, llm) -> None:
    llm(
        calls("propose_event", query="Amazon", event_type="rejected"),
        says("Confirm to record it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    action = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()[
        "pending_action"
    ]
    confirmed = await user.post(
        "/api/v1/agent/confirm",
        {"application_id": action["application_id"], "event_type": action["event_type"]},
    )

    body = confirmed.json()
    assert confirmed.status_code == 201
    assert body["current_status"] == "rejected"
    agent_events = [e for e in body["events"] if e["source"] == "agent"]
    assert len(agent_events) == 1


async def test_confirming_another_users_application_is_refused(client: AsyncClient, llm) -> None:
    """/confirm takes a raw id, so it must re-check ownership rather than
    trusting the id came from a legitimate proposal."""
    llm(says("ok"))
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application(company_name="Amazon")

    response = await bob.post(
        "/api/v1/agent/confirm",
        {"application_id": application["id"], "event_type": "rejected"},
    )

    assert response.status_code == 404


# --- Degradation -----------------------------------------------------------


async def test_a_model_failure_is_reported_not_swallowed(client: AsyncClient, llm) -> None:
    llm(error=LLMError("Groq rate limit reached."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    response = await user.post("/api/v1/agent/chat", {"message": "how is it going?"})

    assert response.status_code == 422
    assert "rate limit" in response.json()["detail"].lower()


async def test_the_assistant_requires_authentication(client: AsyncClient) -> None:
    assert (await client.post("/api/v1/agent/chat", json={"message": "hi"})).status_code == 401
