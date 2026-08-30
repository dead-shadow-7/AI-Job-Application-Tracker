"""Event-sourcing invariants.

The cached columns on ``applications`` are an optimisation over the event log.
These tests exist to keep them honest — particularly under backdating, which is
the normal way this app gets used ("I actually applied last Tuesday") and the
case an incremental cache update would get wrong.
"""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient

from tests.factories import Session


def iso(days_ago: int = 0, hours_ago: int = 0) -> str:
    stamp = datetime.now(UTC) - timedelta(days=days_ago, hours=hours_ago)
    return stamp.isoformat()


async def test_new_application_starts_saved(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()

    assert application["current_status"] == "saved"
    assert application["applied_at"] is None
    assert [e["event_type"] for e in application["events"]] == ["saved"]


async def test_status_follows_the_latest_event(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()
    app_id = application["id"]

    await user.add_event(app_id, "applied", iso(days_ago=10))
    await user.add_event(app_id, "screening_scheduled", iso(days_ago=8))
    response = await user.add_event(app_id, "interview_scheduled", iso(days_ago=2))

    assert response.status_code == 201
    assert response.json()["current_status"] == "interviewing"


async def test_applied_at_comes_from_the_applied_event(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()

    applied_on = iso(days_ago=12)
    response = await user.add_event(application["id"], "applied", applied_on)

    body = response.json()
    assert body["applied_at"] is not None
    assert body["applied_at"][:10] == applied_on[:10]


async def test_backdated_event_does_not_override_a_later_one(client: AsyncClient) -> None:
    """The bug an incremental cache update would introduce.

    Recording last week's rejection *after* this week's follow-up must not make
    the application look rejected-as-of-now, nor make a later event lose to an
    earlier one merely because it was typed first.
    """
    user = await Session(client).start()
    application = await user.create_application()
    app_id = application["id"]

    await user.add_event(app_id, "applied", iso(days_ago=20))
    await user.add_event(app_id, "interview_scheduled", iso(days_ago=3))
    # Now record something that happened *before* the interview.
    response = await user.add_event(app_id, "screening_done", iso(days_ago=10))

    body = response.json()
    assert body["current_status"] == "interviewing", "an older event overwrote a newer status"
    assert body["current_status_at"][:10] == iso(days_ago=3)[:10]


async def test_contact_events_refresh_activity_without_moving_status(
    client: AsyncClient,
) -> None:
    """A recruiter reply means the application is not stale — but it has not
    advanced either. Phase 4's follow-up rules depend on this distinction."""
    user = await Session(client).start()
    application = await user.create_application(
        initial_event="applied", occurred_at=iso(days_ago=10)
    )
    response = await user.add_event(application["id"], "recruiter_reply", iso(days_ago=1))

    body = response.json()
    assert body["current_status"] == "applied"
    assert body["current_status_at"][:10] == iso(days_ago=10)[:10]
    assert body["last_activity_at"][:10] == iso(days_ago=1)[:10]


@pytest.mark.parametrize(
    ("event_type", "expected"),
    [
        ("applied", "applied"),
        ("assessment_received", "screening"),
        ("screening_done", "screening"),
        ("interview_done", "interviewing"),
        ("offer_received", "offer"),
        ("rejected", "rejected"),
        ("withdrawn", "withdrawn"),
        ("marked_ghosted", "ghosted"),
        ("accepted", "accepted"),
    ],
)
async def test_every_status_event_maps_correctly(
    client: AsyncClient, event_type: str, expected: str
) -> None:
    user = await Session(client).start()
    application = await user.create_application()

    response = await user.add_event(application["id"], event_type)

    assert response.json()["current_status"] == expected


async def test_future_events_are_rejected(client: AsyncClient) -> None:
    """A future-dated event would push last_activity_at forward and make a
    stalled application look fresh, silently disabling follow-up detection."""
    user = await Session(client).start()
    application = await user.create_application()

    future = (datetime.now(UTC) + timedelta(days=3)).isoformat()
    response = await user.add_event(application["id"], "interview_scheduled", future)

    assert response.status_code == 422
    assert "future" in response.json()["detail"].lower()


async def test_timeline_is_ordered_and_append_only(client: AsyncClient) -> None:
    user = await Session(client).start()
    application = await user.create_application()
    app_id = application["id"]

    await user.add_event(app_id, "applied", iso(days_ago=9))
    await user.add_event(app_id, "screening_scheduled", iso(days_ago=2))
    await user.add_event(app_id, "recruiter_reply", iso(days_ago=6))

    events = (await user.get(f"/api/v1/applications/{app_id}/events")).json()

    occurred = [e["occurred_at"] for e in events]
    assert occurred == sorted(occurred), (
        "events should read in the order they happened, not the order they were typed"
    )
    # The backdated trio must precede the `saved` event stamped at creation.
    assert [e["event_type"] for e in events[:3]] == [
        "applied",
        "recruiter_reply",
        "screening_scheduled",
    ]
    assert all(e["source"] == "manual" for e in events)


async def test_cache_equals_a_replay_of_the_log(client: AsyncClient) -> None:
    """The reconciliation guarantee, asserted end to end.

    ``append_event`` refreshes the cache by folding the whole log, so this holds
    by construction today. The test exists to catch a future "optimisation" that
    replaces the fold with an incremental update and quietly breaks it.
    """
    from app.domain.enums import STATUS_BY_EVENT, EventType

    user = await Session(client).start()
    application = await user.create_application(
        initial_event="applied", occurred_at=iso(days_ago=30)
    )
    app_id = application["id"]

    for event_type, days in [
        ("assessment_received", 25),
        ("screening_done", 18),
        ("follow_up_sent", 12),
        ("interview_scheduled", 9),
        ("recruiter_reply", 4),
    ]:
        await user.add_event(app_id, event_type, iso(days_ago=days))

    body = (await user.get(f"/api/v1/applications/{app_id}")).json()
    events = body["events"]

    status_events = [e for e in events if EventType(e["event_type"]) in STATUS_BY_EVENT]
    expected_status = STATUS_BY_EVENT[EventType(status_events[-1]["event_type"])].value

    assert body["current_status"] == expected_status
    assert body["current_status_at"] == status_events[-1]["occurred_at"]
    assert body["last_activity_at"] == events[-1]["occurred_at"]
