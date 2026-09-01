"""The assistant, driven by a stubbed model.

What is worth testing here is not whether the model understands English — that
is the model's problem. It is that a model which misunderstands *cannot cause
damage*: /chat never writes, ambiguity becomes a question, and a proposal is
resolved to a row the user sees before anything happens.
"""

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy import text

from app.agent.llm_client import LLMError, LLMUsage, StructuredResult
from app.db.session import open_user_session
from app.schemas.agent import AgentReply, ProposedAction
from tests.factories import Session


def reply(
    message: str = "Done.",
    kind: str = "none",
    query: str | None = None,
    event_type: str | None = None,
    note: str | None = None,
) -> AgentReply:
    return AgentReply(
        message=message,
        action=ProposedAction(kind=kind, application_query=query, event_type=event_type, note=note),
    )


class StubLLM:
    def __init__(self, result: AgentReply | None = None, error: Exception | None = None) -> None:
        self._result = result or reply()
        self._error = error
        self.calls = 0
        self.is_configured = True

    async def extract(self, **_: Any) -> StructuredResult:
        self.calls += 1
        if self._error:
            raise self._error
        return StructuredResult(
            data=self._result,
            usage=LLMUsage(model="stub", total_tokens=800, latency_ms=900),
        )


def patch_llm(monkeypatch: pytest.MonkeyPatch, stub: StubLLM) -> StubLLM:
    monkeypatch.setattr("app.api.v1.agent.llm_client", stub)
    return stub


async def event_count(user: Session) -> int:
    async for session in open_user_session(user.user_id):
        return (await session.execute(text("SELECT count(*) FROM application_events"))).scalar_one()
    return 0


# --- Answering -------------------------------------------------------------


async def test_a_question_is_answered_without_proposing_anything(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_llm(monkeypatch, StubLLM(reply("Amazon is at screening.")))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    body = (await user.post("/api/v1/agent/chat", {"message": "How is Amazon going?"})).json()

    assert body["message"] == "Amazon is at screening."
    assert body["pending_action"] is None
    assert body["disambiguation"] == []


# --- The safety boundary ---------------------------------------------------


async def test_chat_never_writes(client: AsyncClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The single most important property here. Even a confident, unambiguous
    instruction produces a proposal, not a change."""
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Amazon", event_type="rejected"))
    )
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")
    before = await event_count(user)

    body = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()
    after = (await user.get(f"/api/v1/applications/{application['id']}")).json()

    assert body["pending_action"] is not None
    assert await event_count(user) == before, "/chat must not append anything"
    assert after["current_status"] == "applied", "status must be unchanged until confirmed"


async def test_a_proposal_names_the_resolved_row_not_the_phrase(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirming "mark Amazon as rejected" without seeing *which* Amazon
    defeats the point of confirming."""
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Amazon", event_type="rejected"))
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
    assert "Amazon" in action["application_label"]
    assert action["event_type"] == "rejected"


async def test_ambiguity_becomes_a_question_not_a_guess(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two roles at one company. Picking one would write to a timeline the user
    never chose, and they would not find out for weeks."""
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Amazon", event_type="rejected"))
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Amazon", title="Data Engineer")

    body = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()

    assert body["pending_action"] is None, "must not choose between equal matches"
    assert len(body["disambiguation"]) == 2
    assert "Backend Engineer" in " ".join(body["disambiguation"])


async def test_an_unknown_company_proposes_nothing(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Spotify", event_type="rejected"))
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    body = (await user.post("/api/v1/agent/chat", {"message": "reject Spotify"})).json()

    assert body["pending_action"] is None
    assert "Nothing matches" in body["message"]


async def test_a_proposal_without_a_target_asks_which_one(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_llm(monkeypatch, StubLLM(reply(kind="append_event", query=None, event_type="rejected")))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    body = (await user.post("/api/v1/agent/chat", {"message": "mark it rejected"})).json()

    assert body["pending_action"] is None
    assert "which application" in body["message"].lower()


# --- Confirming ------------------------------------------------------------


async def test_confirming_appends_an_attributable_event(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent writes land on the same append-only timeline as manual ones,
    marked as agent-written so they are visible and reversible."""
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Amazon", event_type="rejected"))
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
    assert agent_events[0]["event_type"] == "rejected"


async def test_an_agent_write_is_reversible_by_a_correction(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is edited or deleted — a mistake is undone by appending, and
    both entries remain visible in the history."""
    patch_llm(
        monkeypatch, StubLLM(reply(kind="append_event", query="Amazon", event_type="rejected"))
    )
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")

    action = (await user.post("/api/v1/agent/chat", {"message": "mark Amazon rejected"})).json()[
        "pending_action"
    ]
    await user.post(
        "/api/v1/agent/confirm",
        {"application_id": action["application_id"], "event_type": "rejected"},
    )
    await user.add_event(application["id"], "recruiter_reply", note="Rejection was a mistake")

    body = (await user.get(f"/api/v1/applications/{application['id']}")).json()

    assert any(e["source"] == "agent" for e in body["events"]), "the agent write is still visible"
    assert len(body["events"]) == 3


async def test_confirming_another_users_application_is_refused(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confirm endpoint takes a raw id, so it must re-check ownership
    rather than trusting that the id came from a legitimate proposal."""
    patch_llm(monkeypatch, StubLLM())
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application(company_name="Amazon")

    response = await bob.post(
        "/api/v1/agent/confirm",
        {"application_id": application["id"], "event_type": "rejected"},
    )

    assert response.status_code == 404


# --- Degradation -----------------------------------------------------------


async def test_a_model_failure_is_reported_not_swallowed(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_llm(monkeypatch, StubLLM(error=LLMError("Groq rate limit reached.")))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    response = await user.post("/api/v1/agent/chat", {"message": "how is it going?"})

    assert response.status_code == 422
    assert "rate limit" in response.json()["detail"].lower()


async def test_the_assistant_requires_authentication(client: AsyncClient) -> None:
    assert (await client.post("/api/v1/agent/chat", json={"message": "hi"})).status_code == 401
