"""What the assistant can now do beyond reading and logging an event.

The drawer could describe someone's job search in detail and change almost none
of it: the only write it could propose was an event on a row that already
existed, so "can you add a new job application?" had no answer. These cover the
three actions that closed that gap, the analytical and drafting tools that came
with them, and — more importantly — the invariants none of them may break.

The pattern throughout is the same one the whole agent design rests on: `/chat`
leaves the database untouched, and the confirm card describes the change
completely enough that agreeing to it is an informed act.
"""

from datetime import UTC, datetime

from httpx import AsyncClient

from tests.conftest import calls, calls_many, days_ago, says
from tests.factories import Session

# --- Creating --------------------------------------------------------------


async def test_it_can_propose_tracking_a_job_it_was_told_about(client: AsyncClient, llm) -> None:
    """The gap the assistant used to report: no tool for creating an entry, so
    the only answer to "add this job" was to go and do it by hand."""
    llm(
        calls(
            "propose_new_application",
            company_name="Zerodha",
            title="Backend Engineer",
            location="Bengaluru",
            status="applied",
            applied_days_ago=3,
        ),
        says("I will start tracking that once you confirm."),
    )
    user = await Session(client).start()

    body = (
        await user.post("/api/v1/agent/chat", {"message": "track the Zerodha backend role"})
    ).json()
    action = body["pending_action"]

    assert action["kind"] == "create_application"
    assert (await user.get("/api/v1/applications")).json()["total"] == 0, "/chat must not write"

    created = await user.post(
        "/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]}
    )

    assert created.status_code == 201
    assert created.json()["job"]["company"]["name"] == "Zerodha"
    assert created.json()["current_status"] == "applied"
    assert created.json()["days_since_activity"] == 3, "backdated, not stamped now"


async def test_a_creation_card_lists_every_field_it_would_write(client: AsyncClient, llm) -> None:
    """The card is the only check between a misread instruction and a real row.
    A field the model filled in but the card omitted would be written without
    anyone having seen it, which makes confirming theatre rather than a check."""
    llm(
        calls(
            "propose_new_application",
            company_name="Zerodha",
            title="Backend Engineer",
            location="Bengaluru",
            work_mode="remote",
            source_platform="LinkedIn",
            notes="They mentioned 40 LPA",
        ),
        says("Confirm to add it."),
    )
    user = await Session(client).start()

    action = (await user.post("/api/v1/agent/chat", {"message": "add it"})).json()["pending_action"]

    shown = " ".join(action["details"])
    for value in ("Zerodha", "Backend Engineer", "Bengaluru", "remote", "LinkedIn", "40 LPA"):
        assert value in shown, f"{value} would be written but is not on the card"


async def test_a_creation_has_no_resolution_confidence_to_show(client: AsyncClient, llm) -> None:
    """Nothing was resolved, so there is no match to be confident about.
    Rendering "100%" there would imply a check that never happened."""
    llm(
        calls("propose_new_application", company_name="Zerodha", title="Backend Engineer"),
        says("Confirm to add it."),
    )
    user = await Session(client).start()

    action = (await user.post("/api/v1/agent/chat", {"message": "add it"})).json()["pending_action"]

    assert action["confidence"] is None
    assert action["application_id"] is None


async def test_it_refuses_to_create_a_second_row_for_a_job_already_tracked(
    client: AsyncClient, llm
) -> None:
    """A duplicate splits the timeline in two, after which neither half tells
    the truth and the follow-up sweep sees two partial stories."""
    stub = llm(
        calls("propose_new_application", company_name="Amazon", title="Backend Engineer"),
        says("You already track that one."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")

    body = (
        await user.post("/api/v1/agent/chat", {"message": "track the Amazon backend role"})
    ).json()

    assert body["pending_action"] is None
    assert "already track" in stub.tool_output()


async def test_creating_without_a_company_asks_rather_than_inventing_one(
    client: AsyncClient, llm
) -> None:
    stub = llm(calls("propose_new_application", title="Backend Engineer"), says("Which company?"))
    user = await Session(client).start()

    body = (await user.post("/api/v1/agent/chat", {"message": "add a backend role"})).json()

    assert body["pending_action"] is None
    assert "company" in stub.tool_output().lower()


# --- Updating --------------------------------------------------------------


async def test_it_cannot_move_status_through_an_update(client: AsyncClient, llm) -> None:
    """Status is derived from the event log. A direct write would let the cache
    diverge from its own source of truth, so the tool has no such field and
    says what to do instead."""
    stub = llm(
        calls("propose_update", query="Amazon", status="rejected"),
        says("I need to log an event for that."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    body = (await user.post("/api/v1/agent/chat", {"message": "set Amazon to rejected"})).json()

    assert body["pending_action"] is None
    assert "propose an event instead" in stub.tool_output()


async def test_updating_priority_takes_effect_only_on_confirmation(
    client: AsyncClient, llm
) -> None:
    llm(
        calls("propose_update", query="Amazon", priority="high", notes="Chase this one"),
        says("Confirm to update it."),
    )
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")

    action = (
        await user.post("/api/v1/agent/chat", {"message": "make Amazon high priority"})
    ).json()["pending_action"]
    unchanged = (await user.get(f"/api/v1/applications/{application['id']}")).json()
    assert unchanged["priority"] == "medium", "/chat must not have changed it"

    updated = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()

    assert updated["priority"] == "high"
    assert updated["notes"] == "Chase this one"


# --- Scheduling ------------------------------------------------------------


async def test_a_scheduled_round_holds_the_future_date_but_the_event_does_not(
    client: AsyncClient, llm
) -> None:
    """The one place a future date belongs is the stage. An event stamped ahead
    of now would push last_activity_at forward and make a stalled application
    look fresh, silently disabling the follow-up detection this project exists
    for."""
    llm(
        calls("propose_interview_round", query="Amazon", stage_type="technical", in_days=5),
        says("Confirm to schedule it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    action = (
        await user.post("/api/v1/agent/chat", {"message": "technical round in 5 days"})
    ).json()["pending_action"]
    body = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()

    now = datetime.now(UTC)
    stage = body["stages"][0]
    assert stage["stage_type"] == "technical"
    assert stage["round_number"] == 1
    assert datetime.fromisoformat(stage["scheduled_at"]) > now, "the stage carries the future date"

    scheduled = [e for e in body["events"] if e["event_type"] == "interview_scheduled"]
    assert len(scheduled) == 1
    assert datetime.fromisoformat(scheduled[0]["occurred_at"]) <= now, "the event happened now"
    assert scheduled[0]["source"] == "agent"
    assert body["current_status"] == "interviewing"


async def test_round_numbers_continue_from_what_is_already_there(client: AsyncClient, llm) -> None:
    """Otherwise the second round collides with the first on the unique
    constraint and the confirmation fails for a reason nobody can act on."""
    llm(
        calls("propose_interview_round", query="Amazon", stage_type="technical"),
        says("Confirm it."),
        calls("propose_interview_round", query="Amazon", stage_type="final"),
        says("Confirm it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    for _ in range(2):
        action = (await user.post("/api/v1/agent/chat", {"message": "schedule a round"})).json()[
            "pending_action"
        ]
        response = await user.post(
            "/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]}
        )
        assert response.status_code == 201, response.text

    assert [s["round_number"] for s in response.json()["stages"]] == [1, 2]


# --- One proposal per turn -------------------------------------------------


async def test_only_the_first_proposal_of_a_turn_survives(client: AsyncClient, llm) -> None:
    """The card renders one action, so two proposals would silently drop one.
    Keeping the first keeps it consistent with what the model went on to say."""
    llm(
        calls_many(
            ("propose_event", {"query": "Amazon", "event_type": "rejected"}),
            ("propose_update", {"query": "Amazon", "priority": "low"}),
        ),
        says("I will mark it rejected."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    action = (
        await user.post("/api/v1/agent/chat", {"message": "reject it and deprioritise"})
    ).json()["pending_action"]

    assert action["kind"] == "append_event"


# --- Analysis --------------------------------------------------------------


async def test_the_small_sample_caveat_reaches_the_model(client: AsyncClient, llm) -> None:
    """The dashboard carries this warning already. Without it in the tool output
    the model quotes "your response rate is 100%" off one application as though
    it were a finding."""
    stub = llm(calls("get_analytics"), says("Too early to say."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    await user.post("/api/v1/agent/chat", {"message": "how am I doing?"})

    assert "CAVEAT" in stub.tool_output()
    assert "not meaningful" in stub.tool_output()


async def test_skill_demand_says_which_skills_are_missing(client: AsyncClient, llm) -> None:
    """ "What should I learn next" answered from the jobs they actually track,
    rather than from an industry trend the model half-remembers."""
    stub = llm(calls("get_skill_demand"), says("Python comes up most."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    await user.post("/api/v1/agent/chat", {"message": "what should I learn?"})

    assert "Python" in stub.tool_output()
    assert "resume" in stub.tool_output(), "each skill must say whether they already have it"


async def test_the_resume_tool_says_plainly_when_there_is_no_resume(
    client: AsyncClient, llm
) -> None:
    """Otherwise the model discusses fit from the job's side and fills the other
    half from assumption, which reads as confident and is invented."""
    stub = llm(calls("get_resume_profile"), says("You have not uploaded one."))
    user = await Session(client).start()

    await user.post("/api/v1/agent/chat", {"message": "am I a good fit anywhere?"})

    assert "No resume" in stub.tool_output()
    assert "Do not guess" in stub.tool_output()


async def test_comparing_needs_at_least_two_roles(client: AsyncClient, llm) -> None:
    stub = llm(calls("compare_applications", queries=["Amazon"]), says("Which two?"))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    await user.post("/api/v1/agent/chat", {"message": "compare them"})

    assert "at least two" in stub.tool_output()


async def test_comparing_puts_both_roles_in_one_result(client: AsyncClient, llm) -> None:
    stub = llm(
        calls("compare_applications", queries=["Amazon", "Razorpay"]),
        says("Amazon pays more."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")
    await user.create_application(company_name="Razorpay", title="Platform Engineer")

    await user.post("/api/v1/agent/chat", {"message": "compare Amazon and Razorpay"})

    output = stub.tool_output()
    assert "Amazon" in output and "Razorpay" in output


async def test_analysis_is_scoped_to_the_caller(client: AsyncClient, llm) -> None:
    """Every analytical tool aggregates, which is exactly where a missing tenant
    filter stops being obvious."""
    stub = llm(calls("get_skill_demand"), says("Nothing yet."))
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    await alice.create_application(company_name="Amazon")

    await bob.post("/api/v1/agent/chat", {"message": "what should I learn?"})

    assert "Python" not in stub.tool_output()


# --- Drafting --------------------------------------------------------------


async def test_drafting_returns_context_and_an_instruction_not_prose(
    client: AsyncClient, llm
) -> None:
    """Generating the message inside the tool would cost a second model call and
    arrive without the conversation — the user may have just said "keep it
    short". The model already in the loop writes it."""
    stub = llm(calls("draft_follow_up", query="Amazon"), says("Subject: Following up"))
    user = await Session(client).start()
    await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=days_ago(12)
    )

    await user.post("/api/v1/agent/chat", {"message": "draft a follow-up for Amazon"})

    output = stub.tool_output()
    assert "silent for 12 days" in output
    assert "Now write the follow-up" in output
    assert "do not invent" in output.lower()


async def test_drafting_a_follow_up_on_a_closed_application_warns_first(
    client: AsyncClient, llm
) -> None:
    """Chasing a role you were rejected from is not what they meant to send."""
    stub = llm(calls("draft_follow_up", query="Amazon"), says("That one is closed."))
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")
    await user.add_event(application["id"], "rejected")

    await user.post("/api/v1/agent/chat", {"message": "follow up with Amazon"})

    assert "closed" in stub.tool_output()


async def test_a_second_follow_up_is_told_about_the_first(client: AsyncClient, llm) -> None:
    """Sending the same message twice reads worse than not sending one."""
    stub = llm(calls("draft_follow_up", query="Amazon"), says("You already chased them."))
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=days_ago(20)
    )
    await user.add_event(application["id"], "follow_up_sent", days_ago(6))

    await user.post("/api/v1/agent/chat", {"message": "follow up with Amazon again"})

    assert "ALREADY sent a follow-up" in stub.tool_output()


async def test_an_interview_brief_is_grounded_in_the_posting(client: AsyncClient, llm) -> None:
    stub = llm(calls("prepare_interview_brief", query="Amazon"), says("Expect Python questions."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    await user.post("/api/v1/agent/chat", {"message": "help me prep for Amazon"})

    output = stub.tool_output()
    assert "3+ years with Python" in output, "the requirements as written must reach the model"
    assert "Ground every point" in output


async def test_a_brief_admits_when_nothing_was_scored(client: AsyncClient, llm) -> None:
    """Gaps come from the match analysis. Without one the model would otherwise
    invent weaknesses, which is worse than useless before an interview."""
    stub = llm(calls("prepare_interview_brief", query="Amazon"), says("Not scored yet."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    await user.post("/api/v1/agent/chat", {"message": "prep me"})

    assert "do not guess" in stub.tool_output().lower()


# --- Wiring ----------------------------------------------------------------


async def test_the_reply_reports_which_tools_ran(client: AsyncClient, llm) -> None:
    """Shown in the drawer so an answer can be traced to its source rather than
    taken on trust."""
    llm(calls("get_analytics"), says("Early days."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    body = (await user.post("/api/v1/agent/chat", {"message": "how am I doing?"})).json()

    assert body["tools_used"] == ["get_analytics"]


def test_optional_tool_arguments_accept_null() -> None:
    """Groq validates tool-call arguments against these schemas and rejects the
    whole request with a 400 when they disagree.

    Models routinely emit `"note": null` for an optional argument they have
    nothing to say about — to them, omitting it and passing null are the same
    intent. A bare {"type": "string"} there turned "mark amazon as rejected"
    into `Tool call validation failed: '/note': expected string, but got null`,
    which reached the user as a red error instead of an answer.
    """
    from app.agent.tools import TOOL_SCHEMAS

    for tool in TOOL_SCHEMAS:
        parameters = tool["function"]["parameters"]
        required = set(parameters["required"])
        for name, schema in parameters["properties"].items():
            if name in required:
                assert schema["type"] != "null", f"{name} is required and cannot be null"
                continue
            assert "null" in schema["type"], (
                f"{tool['function']['name']}.{name} is optional but rejects null"
            )
            if "enum" in schema:
                assert None in schema["enum"], f"{name} is a nullable enum missing null"


async def test_a_null_optional_argument_does_not_break_the_turn(client: AsyncClient, llm) -> None:
    """The same failure from the handler's side: nulls arrive as None and must
    read as "not given" rather than as a value."""
    llm(
        calls("propose_event", query="Amazon", event_type="rejected", note=None),
        says("Confirm to record it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    body = (await user.post("/api/v1/agent/chat", {"message": "mark amazon as rejected"})).json()

    assert body["pending_action"]["payload"]["note"] is None
    assert "Note:" not in " ".join(body["pending_action"]["details"])


def test_every_advertised_tool_has_a_handler() -> None:
    """A tool the model can see but nothing can run produces a baffling "no such
    tool" at runtime, from a list it was explicitly given."""
    from app.agent.tools import _PROPOSERS, _READERS, TOOL_SCHEMAS

    advertised = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert set(advertised) == set(_READERS) | set(_PROPOSERS)
    assert len(set(advertised)) == len(advertised), "a tool name is declared twice"


def test_confirming_requires_saying_which_kind_of_change() -> None:
    """The union is discriminated, so an untagged body is rejected rather than
    guessed at — the four actions write to different tables."""
    import pytest
    from pydantic import TypeAdapter, ValidationError

    from app.schemas.agent import ConfirmRequest

    with pytest.raises(ValidationError):
        TypeAdapter(ConfirmRequest).validate_python(
            {"application_id": "00000000-0000-0000-0000-000000000000", "event_type": "rejected"}
        )
