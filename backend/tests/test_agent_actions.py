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

    assert created.status_code == 200, created.text
    assert created.json()["application"]["job"]["company"]["name"] == "Zerodha"
    assert created.json()["application"]["current_status"] == "applied"
    assert created.json()["application"]["days_since_activity"] == 3, "backdated, not stamped now"


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
    ).json()["application"]

    assert updated["priority"] == "high"
    assert updated["notes"] == "Chase this one"


async def test_any_tracked_detail_can_be_corrected(client: AsyncClient, llm) -> None:
    """The Edit buttons on the page reach every field; the assistant reached
    two. A capability available in one place and not the other is the kind of
    gap you rediscover every time you hit it."""
    llm(
        calls(
            "propose_update",
            query="Amazon",
            title="Senior Backend Engineer",
            location="Bengaluru",
            seniority="senior",
            salary_min=1800000,
            salary_max=2400000,
            salary_currency="INR",
            source_platform="LinkedIn",
        ),
        says("Confirm and I will correct it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", title="Backend Engineer")

    action = (await user.post("/api/v1/agent/chat", {"message": "fix the details"})).json()[
        "pending_action"
    ]
    saved = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()["application"]

    job = saved["job"]
    assert job["title"] == "Senior Backend Engineer"
    assert job["location"] == "Bengaluru"
    assert job["seniority"] == "senior"
    assert float(job["salary_min"]) == 1_800_000
    assert job["source_platform"] == "LinkedIn"


async def test_a_value_can_be_removed(client: AsyncClient, llm) -> None:
    """ "Remove the note" used to be impossible: an absent field means "no
    change", so there was no way to express "make this empty" at all."""
    llm(calls("propose_update", query="Amazon", clear=["notes"]), says("Confirm it."))
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon")
    await user.patch(f"/api/v1/applications/{application['id']}", {"notes": "Old remark"})

    action = (await user.post("/api/v1/agent/chat", {"message": "remove the note"})).json()[
        "pending_action"
    ]
    assert "cleared" in " ".join(action["details"])

    saved = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()["application"]

    assert saved["notes"] is None


async def test_a_null_argument_is_not_read_as_a_request_to_clear(client: AsyncClient, llm) -> None:
    """Optional tool parameters are declared nullable so a model can pass null
    for something it has nothing to say about. Reading those as "empty this
    column" would wipe the salary every time it mentioned a priority."""
    llm(
        calls("propose_update", query="Amazon", priority="high", location=None, salary_min=None),
        says("Confirm it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", location="Pune")

    action = (await user.post("/api/v1/agent/chat", {"message": "make it high"})).json()[
        "pending_action"
    ]
    saved = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()["application"]

    assert saved["priority"] == "high"
    assert saved["job"]["location"] == "Pune", "an unmentioned field must survive"


async def test_editing_the_title_through_chat_refreshes_the_search_vector(
    client: AsyncClient, llm
) -> None:
    from sqlalchemy import text as sql

    from app.db.session import open_user_session

    llm(calls("propose_update", query="Amazon", title="Platform Engineer"), says("Confirm it."))
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", title="Backend Engineer")

    action = (await user.post("/api/v1/agent/chat", {"message": "retitle it"})).json()[
        "pending_action"
    ]
    await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})

    async for session in open_user_session(user.user_id):
        stored = (
            await session.execute(
                sql("SELECT content FROM job_embeddings WHERE job_id = :j"),
                {"j": application["job"]["id"]},
            )
        ).scalar_one()
    assert "Platform Engineer" in stored


# --- Editing a stored description ------------------------------------------


async def test_page_furniture_can_be_cut_out_of_a_description(client: AsyncClient, llm) -> None:
    """A posting pasted from a job board arrives wrapped in the page around it —
    "Application status", "Meet the hiring team", the recruiter's headline. It is
    noise, and it is what the assistant reads when asked about the role."""
    posting = (
        "About the job\nBuild agentic AI systems.\n"
        "Application status\nView resume\nMeet the hiring team\n"
        "Requirements\nPython and LLMs.\n"
    )
    llm(
        calls(
            "propose_description_edit",
            query="Amazon",
            remove_text="Application status\nView resume\nMeet the hiring team",
        ),
        says("Confirm and I will trim it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description=posting)

    action = (await user.post("/api/v1/agent/chat", {"message": "remove this"})).json()[
        "pending_action"
    ]
    assert action["kind"] == "edit_description"

    body = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()["application"]

    assert "Meet the hiring team" not in body["job"]["description"]
    assert "Build agentic AI systems." in body["job"]["description"], "the role survives"
    assert "Python and LLMs." in body["job"]["description"]


async def test_the_card_says_how_much_is_going(client: AsyncClient, llm) -> None:
    llm(
        calls("propose_description_edit", query="Amazon", remove_text="View resume"),
        says("Confirm to trim it."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description="Real content\nView resume\n")

    action = (await user.post("/api/v1/agent/chat", {"message": "trim it"})).json()[
        "pending_action"
    ]

    shown = " ".join(action["details"])
    assert "Removing 1 lines" in shown
    assert "View resume" in shown, "the user must see what is being deleted"


async def test_text_that_is_not_there_removes_nothing(client: AsyncClient, llm) -> None:
    """The model reflows whatever it quotes. Silently deleting the nearest thing
    would be far worse than reporting a miss."""
    stub = llm(
        calls("propose_description_edit", query="Amazon", remove_text="Some invented heading"),
        says("I could not find that."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description="Real content only.")

    body = (await user.post("/api/v1/agent/chat", {"message": "remove that bit"})).json()

    assert body["pending_action"] is None
    assert "None of those lines appear" in stub.tool_output()


async def test_it_refuses_to_empty_a_description(client: AsyncClient, llm) -> None:
    """Quoting the whole thing back is an easy mistake and an expensive one."""
    stub = llm(
        calls("propose_description_edit", query="Amazon", remove_text="Only line here"),
        says("That would delete everything."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description="Only line here")

    body = (await user.post("/api/v1/agent/chat", {"message": "remove it"})).json()

    assert body["pending_action"] is None
    assert "entire description" in stub.tool_output()


async def test_a_description_sized_note_is_refused(client: AsyncClient, llm) -> None:
    """The failure this exists for: with no tool for editing a description, the
    model reached for the update tool and wrote "Removed extraneous details from
    the job description" into notes — then reported the edit as done, while the
    description sat untouched."""
    stub = llm(
        calls("propose_update", query="Amazon", notes="x" * 2000),
        says("Wrong tool."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    body = (await user.post("/api/v1/agent/chat", {"message": "clean up the JD"})).json()

    assert body["pending_action"] is None
    assert "propose_description_edit" in stub.tool_output()


async def test_an_update_card_names_the_field_it_changes(client: AsyncClient, llm) -> None:
    """ "Update Backend Engineer at Amazon" is true of four different actions.
    A card you have to read twice to tell them apart is one you stop reading."""
    llm(calls("propose_update", query="Amazon", priority="high"), says("Confirm it."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon")

    action = (await user.post("/api/v1/agent/chat", {"message": "bump it"})).json()[
        "pending_action"
    ]

    assert "Priority" in action["summary"]


async def test_trimming_updates_the_duplicate_hash(client: AsyncClient, llm) -> None:
    """The hash is derived from the description and is what exact-duplicate
    detection compares. Left stale, re-pasting the original posting stops
    matching the row it created, and the timeline forks."""
    from sqlalchemy import text as sql

    from app.db.session import open_user_session

    llm(
        calls("propose_description_edit", query="Amazon", remove_text="View resume"),
        says("Confirm it."),
    )
    user = await Session(client).start()
    application = await user.create_application(
        company_name="Amazon", description="Real content\nView resume"
    )
    job_id = application["job"]["id"]

    async for session in open_user_session(user.user_id):
        before = (
            await session.execute(sql("SELECT content_hash FROM jobs WHERE id = :j"), {"j": job_id})
        ).scalar_one()

    action = (await user.post("/api/v1/agent/chat", {"message": "trim"})).json()["pending_action"]
    await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})

    async for session in open_user_session(user.user_id):
        after = (
            await session.execute(sql("SELECT content_hash FROM jobs WHERE id = :j"), {"j": job_id})
        ).scalar_one()
    assert after != before


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
    ).json()["application"]

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
        assert response.status_code == 200, response.text

    assert [s["round_number"] for s in response.json()["application"]["stages"]] == [1, 2]


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


# --- Deleting --------------------------------------------------------------


async def test_deleting_is_proposed_with_what_would_be_lost(client: AsyncClient, llm) -> None:
    """The only irreversible action, so the card has to be specific about the
    cost rather than asking for a generic confirmation."""
    stub = llm(calls("propose_delete", query="Amazon"), says("This cannot be undone."))
    user = await Session(client).start()
    application = await user.create_application(company_name="Amazon", initial_event="applied")
    await user.add_event(application["id"], "recruiter_reply")

    action = (await user.post("/api/v1/agent/chat", {"message": "delete the Amazon one"})).json()[
        "pending_action"
    ]

    assert action["kind"] == "delete_application"
    assert action["destructive"] is True, "the UI styles the card from this"
    shown = " ".join(action["details"])
    assert "2 events" in shown, "say how much history goes with it"
    assert "cannot be undone" in shown
    # And the model is told to offer the reversible option first.
    assert "withdrawn" in stub.tool_output()


async def test_a_confirmed_delete_removes_the_application(client: AsyncClient, llm) -> None:
    llm(calls("propose_delete", query="Amazon"), says("Confirm to delete it."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", initial_event="applied")

    action = (await user.post("/api/v1/agent/chat", {"message": "delete Amazon"})).json()[
        "pending_action"
    ]
    assert (await user.get("/api/v1/applications")).json()["total"] == 1, "/chat must not delete"

    result = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()

    assert result["application"] is None, "there is no row left to return"
    assert "permanently deleted" in result["summary"]
    assert (await user.get("/api/v1/applications")).json()["total"] == 0


async def test_deleting_another_users_application_is_refused(client: AsyncClient, llm) -> None:
    """/confirm takes a raw id. For the one irreversible action, trusting that
    the id came from a legitimate proposal would be the worst place to start."""
    llm(says("ok"))
    alice = await Session(client, "alice@example.com").start()
    bob = await Session(client, "bob@example.com").start()
    application = await alice.create_application(company_name="Amazon")

    response = await bob.post(
        "/api/v1/agent/confirm",
        {"kind": "delete_application", "application_id": application["id"]},
    )

    assert response.status_code == 404
    assert (await alice.get("/api/v1/applications")).json()["total"] == 1


# --- Knowing what it did ---------------------------------------------------


async def test_a_confirmed_action_enters_the_conversation(client: AsyncClient, llm) -> None:
    """The bug this exists for: confirmation happens on a different endpoint, so
    the transcript ended at "I'm about to record…" and the assistant answered
    the next question as though nothing had happened — "I haven't created that
    yet, so there's nothing to delete", about a row sitting in the table."""
    stub = llm(
        calls("propose_new_application", company_name="Zerodha", title="SDE"),
        says("Confirm and I will add it."),
        says("It is already tracked."),
    )
    user = await Session(client).start()

    action = (
        await user.post("/api/v1/agent/chat", {"message": "add the Zerodha SDE role"})
    ).json()["pending_action"]
    await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    await user.post("/api/v1/agent/chat", {"message": "actually delete it"})

    replayed = " ".join(
        str(m.get("content")) for m in stub.messages_seen[-1] if m.get("role") == "assistant"
    )
    assert "started tracking SDE at Zerodha" in replayed
    assert "no longer pending" in replayed


async def test_the_thread_stays_in_order(client: AsyncClient, llm) -> None:
    """Postgres now() is the transaction start time, so a question and its
    answer — written in one request — shared a timestamp and the history query
    could return them either way round. A model shown its own reply before the
    message that prompted it answers the wrong question."""
    stub = llm(says("Which application?"), says("Amazon."), says("Anything else?"))
    user = await Session(client).start()

    for message in ("first question", "second question", "third question"):
        await user.post("/api/v1/agent/chat", {"message": message})

    thread = [
        (m["role"], m["content"])
        for m in stub.messages_seen[-1]
        if m.get("role") in {"user", "assistant"}
    ]
    assert thread == [
        ("user", "first question"),
        ("assistant", "Which application?"),
        ("user", "second question"),
        ("assistant", "Amazon."),
        ("user", "third question"),
    ]


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


# --- Documents ---------------------------------------------------------------


async def test_a_job_description_reaches_the_user_without_passing_through_the_model(
    client: AsyncClient, llm
) -> None:
    """The bug: asked to relay a 3,400-character posting the model rewrote it to
    1,900, and sometimes replied "here's the job description" and reproduced
    none of it. A document cannot be re-emitted verbatim on demand, so it does
    not go through the model's output at all."""
    posting = "Key Responsibilities\n" + "\n".join(f"- Build subsystem {i}" for i in range(200))
    stub = llm(
        calls("get_job_description", query="Amazon"),
        says("Here is the posting for the backend role."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description=posting)

    body = (await user.post("/api/v1/agent/chat", {"message": "share the JD"})).json()

    attachment = body["attachments"][0]
    assert attachment["kind"] == "job_description"
    assert attachment["body"] == posting, "the user gets the stored text, whole and unedited"
    assert "Amazon" in attachment["title"]
    # And the model is told not to duplicate what is already on screen.
    assert "ALREADY BEING SHOWN" in stub.tool_output()
    assert "Do NOT reproduce" in stub.tool_output()


async def test_the_model_still_receives_the_posting_to_answer_questions_about(
    client: AsyncClient, llm
) -> None:
    """Attaching it is not enough on its own — "what does it say about
    Kubernetes" needs the text in the context window, so it is sent as well."""
    stub = llm(calls("get_job_description", query="Amazon"), says("It asks for Kubernetes."))
    user = await Session(client).start()
    await user.create_application(
        company_name="Amazon", description="You will operate Kubernetes clusters."
    )

    await user.post("/api/v1/agent/chat", {"message": "does it mention kubernetes?"})

    assert "operate Kubernetes clusters" in stub.tool_output()


async def test_a_posting_is_attached_once_even_if_fetched_twice(client: AsyncClient, llm) -> None:
    """Models retry a lookup with a reworded query. Showing the same document
    twice makes the drawer look broken."""
    llm(
        calls("get_job_description", query="Amazon"),
        calls("get_job_description", query="the Amazon one"),
        says("Here it is."),
    )
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description="Build services.")

    body = (await user.post("/api/v1/agent/chat", {"message": "the JD please"})).json()

    assert len(body["attachments"]) == 1


async def test_nothing_is_attached_when_there_is_no_description(client: AsyncClient, llm) -> None:
    """An empty document block would read as a posting that failed to load."""
    llm(calls("get_job_description", query="Amazon"), says("None was stored."))
    user = await Session(client).start()
    await user.create_application(company_name="Amazon", description=None)

    body = (await user.post("/api/v1/agent/chat", {"message": "the JD please"})).json()

    assert body["attachments"] == []


# --- Pasting a posting into the conversation -------------------------------


async def test_a_pasted_posting_is_tracked_with_everything_in_it(
    client: AsyncClient, llm, monkeypatch
) -> None:
    """Pasting a whole description and being told "salary and skills are not
    captured this way, paste it somewhere else" was the wrong answer — it stored
    two fields out of a document the user had already handed over. The chat path
    now runs the same ingestion graph the paste screen runs."""
    from tests.test_ingest import POSTING, StubLLM

    extractor = StubLLM()
    monkeypatch.setattr("app.agent.graphs.ingestion.llm_client", extractor)
    llm(
        calls("propose_tracked_posting", source_platform="LinkedIn", status="applied"),
        says("Confirm and I will track it with the full posting."),
    )
    user = await Session(client).start()

    action = (
        await user.post("/api/v1/agent/chat", {"message": POSTING + "\n\nAdd this, I applied."})
    ).json()["pending_action"]

    assert action["kind"] == "create_from_posting"
    assert extractor.calls == 1, "the real extraction graph must have run"

    saved = (
        await user.post("/api/v1/agent/confirm", {"kind": action["kind"], **action["payload"]})
    ).json()["application"]
    job = saved["job"]

    assert job["company"]["name"] == "Razorpay"
    assert float(job["salary_min"]) == 4_500_000, "salary survives, because it was verified"
    assert len(job["skills"]) > 0, "skills are normalised and attached"
    assert len(job["requirements"]) > 0
    assert job["description"], "the posting itself is stored, not just a summary"
    assert saved["current_status"] == "applied"


async def test_the_card_says_what_was_captured_and_what_was_dropped(
    client: AsyncClient, llm, monkeypatch
) -> None:
    """A silently absent salary looks like the posting never stated one, rather
    than like the verbatim check fired."""
    from tests.test_ingest import POSTING, StubLLM, extraction

    # A salary the model invented — it appears nowhere in the posting text.
    invented = extraction(
        salary={
            "raw_text": "90 LPA",
            "min_amount": 9_000_000,
            "max_amount": 9_000_000,
            "currency": "INR",
            "period": "year",
        }
    )
    monkeypatch.setattr("app.agent.graphs.ingestion.llm_client", StubLLM(invented))
    llm(calls("propose_tracked_posting"), says("Confirm to track it."))
    user = await Session(client).start()

    action = (await user.post("/api/v1/agent/chat", {"message": POSTING})).json()["pending_action"]

    shown = " ".join(action["details"])
    assert "Skills captured" in shown
    assert "Requirements captured" in shown
    assert "salary" in shown.lower(), "the drop has to be named on the card"


async def test_a_one_line_request_is_sent_back_to_the_simpler_tool(
    client: AsyncClient, llm
) -> None:
    """Running extraction on "add the Amazon SDE role" spends a model call to
    learn what the sentence already says, and returns a confident record built
    from nothing."""
    stub = llm(calls("propose_tracked_posting"), says("Which company and role?"))
    user = await Session(client).start()

    body = (await user.post("/api/v1/agent/chat", {"message": "add the Amazon SDE role"})).json()

    assert body["pending_action"] is None
    assert "propose_new_application" in stub.tool_output()


async def test_a_pasted_posting_already_tracked_is_not_duplicated(
    client: AsyncClient, llm, monkeypatch
) -> None:
    from tests.test_ingest import POSTING, StubLLM

    monkeypatch.setattr("app.agent.graphs.ingestion.llm_client", StubLLM())
    stub = llm(calls("propose_tracked_posting"), says("You already track that."))
    user = await Session(client).start()
    await user.create_application(company_name="Razorpay", title="Backend Engineer")

    body = (await user.post("/api/v1/agent/chat", {"message": POSTING})).json()

    assert body["pending_action"] is None
    assert "already track" in stub.tool_output()


# --- Message size ----------------------------------------------------------


async def test_a_pasted_posting_fits_in_one_message(client: AsyncClient, llm) -> None:
    """2,000 characters cut a real job posting off mid-sentence, and the only
    signal was a red validation error after the paste."""
    llm(says("That looks like a backend role."))
    user = await Session(client).start()

    response = await user.post("/api/v1/agent/chat", {"message": "x" * 9_000})

    assert response.status_code == 200


async def test_an_oversized_message_is_still_refused(client: AsyncClient, llm) -> None:
    """Unbounded would be worse than 2,000 was. The message is replayed as
    history on later turns, so an enormous one is paid for repeatedly."""
    llm(says("ok"))
    user = await Session(client).start()

    response = await user.post("/api/v1/agent/chat", {"message": "x" * 10_001})

    assert response.status_code == 422


async def test_replayed_history_is_bounded_by_size_not_only_by_turns(
    client: AsyncClient, llm
) -> None:
    """Ten turns was a fine budget at 2,000 characters and is not at 10,000:
    replaying them would put ~25,000 tokens in front of every later question,
    so one pasted posting gets paid for on every turn that follows it."""
    from app.agent.assistant import HISTORY_CHARS

    stub = llm(says("noted"))
    user = await Session(client).start()

    for i in range(4):
        await user.post("/api/v1/agent/chat", {"message": f"posting {i} " + "x" * 5_000})
    await user.post("/api/v1/agent/chat", {"message": "what did I just paste?"})

    replayed = sum(
        len(m["content"]) for m in stub.messages_seen[-1] if m.get("role") in {"user", "assistant"}
    )
    assert replayed <= HISTORY_CHARS + 10_000, "the budget is not being applied"

    # The newest turn survives; the oldest is what gets dropped.
    contents = " ".join(
        str(m.get("content")) for m in stub.messages_seen[-1] if m.get("role") == "user"
    )
    assert "posting 3" in contents
    assert "posting 0" not in contents


async def test_a_single_turn_survives_even_if_it_exceeds_the_budget(
    client: AsyncClient, llm
) -> None:
    """Dropping everything would leave the model with no thread at all, which
    is worse than being slightly over budget for one turn."""
    stub = llm(says("noted"))
    user = await Session(client).start()

    await user.post("/api/v1/agent/chat", {"message": "y" * 10_000})
    await user.post("/api/v1/agent/chat", {"message": "and?"})

    replayed = [m for m in stub.messages_seen[-1] if m.get("role") in {"user", "assistant"}]
    assert replayed, "history must not come back empty"


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
    from app.agent.tools import HANDLED, TOOL_SCHEMAS

    advertised = [t["function"]["name"] for t in TOOL_SCHEMAS]
    assert set(advertised) == HANDLED
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
