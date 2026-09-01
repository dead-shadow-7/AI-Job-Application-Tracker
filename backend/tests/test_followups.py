"""The follow-up rule engine.

Detection is SQL, so it is testable — which is the whole reason it is not an
LLM. These pin the boundaries, because "7 days" is exactly the kind of rule
that quietly comes to mean 6 or 8.
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.db.session import open_user_session
from app.domain.enums import ApplicationStatus, FollowUpAction
from app.models.followup import FollowUpRule
from app.services.followups import (
    apply_ghosting,
    ensure_default_rules,
    find_stale_applications,
)
from tests.factories import Session


def iso(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()


async def seeded(client: AsyncClient, email: str = "candidate@example.com") -> Session:
    user = await Session(client, email).start()
    async for session in open_user_session(user.user_id):
        await ensure_default_rules(session, user.user_id)
        await session.commit()
    return user


async def stale_for(user: Session, **kwargs):
    async for session in open_user_session(user.user_id):
        return await find_stale_applications(session, user.user_id, **kwargs)
    return []


async def set_all_rules_enabled(user: Session, enabled: bool) -> None:
    async for session in open_user_session(user.user_id):
        rules = (
            (
                await session.execute(
                    select(FollowUpRule).where(FollowUpRule.user_id == user.user_id)
                )
            )
            .scalars()
            .all()
        )
        for rule in rules:
            rule.enabled = enabled
        await session.commit()


# --- Seeding ---------------------------------------------------------------


async def test_defaults_are_seeded_once(client: AsyncClient) -> None:
    user = await Session(client).start()

    async for session in open_user_session(user.user_id):
        first = await ensure_default_rules(session, user.user_id)
        await session.commit()
    async for session in open_user_session(user.user_id):
        second = await ensure_default_rules(session, user.user_id)
        count = (
            (
                await session.execute(
                    select(FollowUpRule).where(FollowUpRule.user_id == user.user_id)
                )
            )
            .scalars()
            .all()
        )

    assert first == 5
    assert second == 0, "seeding twice would duplicate every rule"
    assert len(count) == 5


# --- Boundaries ------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "should_fire"),
    [(5, False), (6, False), (7, True), (8, True), (30, True)],
)
async def test_the_seven_day_boundary_is_exact(
    client: AsyncClient, days: int, should_fire: bool
) -> None:
    """`applied` has a 7-day threshold. Off by one here means the dashboard
    either nags a day early or stays quiet a day too long, forever."""
    user = await seeded(client)
    await user.create_application(initial_event="applied", occurred_at=iso(days))

    stale = await stale_for(user)

    assert bool(stale) is should_fire


async def test_a_fresh_application_is_never_stale(client: AsyncClient) -> None:
    user = await seeded(client)
    await user.create_application(initial_event="applied")

    assert await stale_for(user) == []


# --- What resets the clock -------------------------------------------------


async def test_a_recruiter_reply_resets_the_clock(client: AsyncClient) -> None:
    """The distinction the schema was built around: contact means the
    application is not stale even though it has not advanced."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(20))
    assert await stale_for(user), "should be stale before the reply"

    await user.add_event(application["id"], "recruiter_reply", iso(1))

    assert await stale_for(user) == [], "a reply yesterday means it is not stale"


async def test_your_own_follow_up_also_resets_it(client: AsyncClient) -> None:
    """You have acted; the ball is in their court again."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(20))

    await user.add_event(application["id"], "follow_up_sent", iso(2))

    assert await stale_for(user) == []


async def test_terminal_applications_are_never_chased(client: AsyncClient) -> None:
    """Nothing is pending from your side on a rejection."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(60))
    await user.add_event(application["id"], "rejected", iso(50))

    assert await stale_for(user) == []


# --- Rule selection --------------------------------------------------------


async def test_the_strongest_matching_rule_wins(client: AsyncClient) -> None:
    """At 25 days both the 7-day suggest and the 21-day ghost rules match.
    Reporting both is noise; the stronger statement is the useful one."""
    user = await seeded(client)
    await user.create_application(initial_event="applied", occurred_at=iso(25))

    stale = await stale_for(user)

    assert len(stale) == 1
    assert stale[0].rule.days_threshold == 21
    assert stale[0].rule.action == FollowUpAction.MARK_GHOSTED.value


async def test_disabled_rules_do_not_fire(client: AsyncClient) -> None:
    user = await seeded(client)
    await user.create_application(initial_event="applied", occurred_at=iso(10))

    await set_all_rules_enabled(user, False)

    assert await stale_for(user) == []


async def test_a_user_with_no_rules_gets_no_suggestions(client: AsyncClient) -> None:
    user = await Session(client).start()
    await user.create_application(initial_event="applied", occurred_at=iso(90))

    assert await stale_for(user) == []


async def test_status_specific_thresholds_apply(client: AsyncClient) -> None:
    """`screening` fires at 5 days where `applied` needs 7 — the point of
    having a rule per status rather than one global number."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(20))
    await user.add_event(application["id"], "screening_scheduled", iso(6))

    stale = await stale_for(user)

    assert len(stale) == 1
    assert stale[0].application.current_status == ApplicationStatus.SCREENING.value
    assert stale[0].rule.days_threshold == 5


# --- The sentence the dashboard shows --------------------------------------


async def test_the_reason_names_the_company_and_the_gap(client: AsyncClient) -> None:
    user = await seeded(client)
    await user.create_application(
        company_name="Amazon", initial_event="applied", occurred_at=iso(9)
    )

    reason = (await stale_for(user))[0].reason

    assert "Amazon" in reason
    assert "9 days" in reason
    assert "applied" in reason


# --- Ghosting writes -------------------------------------------------------


async def test_ghosting_closes_the_application_on_the_timeline(client: AsyncClient) -> None:
    """The only rule action that writes. It goes through append_event, so it is
    visible and reversible rather than an invisible status edit."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(30))

    async for session in open_user_session(user.user_id):
        closed = await apply_ghosting(session, user.user_id)
        await session.commit()

    body = (await user.get(f"/api/v1/applications/{application['id']}")).json()

    assert closed == 1
    assert body["current_status"] == "ghosted"
    ghost = [e for e in body["events"] if e["event_type"] == "marked_ghosted"]
    assert len(ghost) == 1
    assert ghost[0]["source"] == "system", "attributable, not an anonymous edit"
    assert "30 days" in ghost[0]["note"]


async def test_ghosting_leaves_merely_stale_applications_alone(client: AsyncClient) -> None:
    """10 days trips suggest_followup, not mark_ghosted. Closing it would
    discard an application still worth chasing."""
    user = await seeded(client)
    application = await user.create_application(initial_event="applied", occurred_at=iso(10))

    async for session in open_user_session(user.user_id):
        closed = await apply_ghosting(session, user.user_id)
        await session.commit()

    body = (await user.get(f"/api/v1/applications/{application['id']}")).json()

    assert closed == 0
    assert body["current_status"] == "applied"


# --- Isolation -------------------------------------------------------------


async def test_rules_and_suggestions_are_per_user(client: AsyncClient) -> None:
    alice = await seeded(client, "alice@example.com")
    bob = await seeded(client, "bob@example.com")
    await alice.create_application(initial_event="applied", occurred_at=iso(30))

    assert len(await stale_for(alice)) == 1
    assert await stale_for(bob) == []
