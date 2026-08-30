"""The application timeline.

``append_event`` is the **only** writer of ``application_events`` and the only
thing permitted to touch ``applications.current_status``,
``current_status_at`` or ``last_activity_at``. Everything else reads them.

That single-writer rule is what makes the cache trustworthy, and it is what lets
Phase 4 hand an agent write access without risk: an agent action is an ordinary
append with ``source='agent'``, visible on the same timeline as your own entries
and reversible by appending a correction.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import InvalidOperationError, NotFoundError
from app.domain.enums import STATUS_BY_EVENT, ApplicationStatus, EventSource, EventType
from app.models.application import Application, ApplicationEvent

# Tolerance for client/server clock skew when rejecting future-dated events.
FUTURE_TOLERANCE = timedelta(minutes=5)


async def append_event(
    session: AsyncSession,
    *,
    application: Application,
    event_type: EventType,
    occurred_at: datetime | None = None,
    source: EventSource = EventSource.MANUAL,
    note: str | None = None,
    payload: dict[str, Any] | None = None,
) -> ApplicationEvent:
    """Append one event and refresh the application's cached state.

    ``occurred_at`` may be backdated freely — "I actually applied last Tuesday"
    is the common case, and the follow-up rules must see that date rather than
    the date you got round to recording it.

    It may **not** be in the future. An event records something that has already
    happened; a scheduled interview is an ``interview_scheduled`` event that
    happened *now*, carrying the future date in its payload or in
    ``interview_stages``. Allowing future ``occurred_at`` would push
    ``last_activity_at`` forward and make a stalled application look fresh,
    quietly disabling the follow-up detection this whole design exists for.
    """
    now = datetime.now(UTC)
    occurred = occurred_at or now

    if occurred.tzinfo is None:
        raise InvalidOperationError("occurred_at must be timezone-aware")
    if occurred > now + FUTURE_TOLERANCE:
        raise InvalidOperationError(
            "occurred_at cannot be in the future. Record when you learned of the "
            "event; put the scheduled date in the payload or an interview stage."
        )

    event = ApplicationEvent(
        application_id=application.id,
        user_id=application.user_id,
        event_type=event_type.value,
        occurred_at=occurred,
        source=source.value,
        note=note,
        payload=payload or {},
    )
    session.add(event)

    # Flush so the new row participates in the recompute below...
    await session.flush()
    await _refresh_cached_state(session, application)
    # ...and again, so the recomputed cache is actually persisted. Without this
    # second flush the new values live only on the in-memory instance, and any
    # subsequent refresh() or re-query silently reverts them to the stale row.
    await session.flush()
    return event


async def _refresh_cached_state(session: AsyncSession, application: Application) -> None:
    """Recompute the denormalized columns by folding the whole event log.

    A full recompute rather than an incremental update, deliberately. It is O(n)
    in events for *one* application — a few dozen rows — and in exchange the
    cache cannot drift from the log even when events arrive out of order, which
    backdating makes routine. An incremental update would have to reason about
    whether the new event is the latest, and would get it wrong the first time
    someone records last week's rejection after this week's follow-up.
    """
    rows = (
        (
            await session.execute(
                select(ApplicationEvent)
                .where(ApplicationEvent.application_id == application.id)
                # occurred_at is the domain ordering; created_at breaks ties so that
                # two events on the same day resolve to the one recorded later.
                .order_by(ApplicationEvent.occurred_at, ApplicationEvent.created_at)
            )
        )
        .scalars()
        .all()
    )

    if not rows:
        application.current_status = ApplicationStatus.SAVED.value
        application.current_status_at = application.created_at
        application.last_activity_at = application.created_at
        application.applied_at = None
        return

    application.last_activity_at = rows[-1].occurred_at

    # Latest event that actually moves the application; contact-only events
    # (recruiter_reply, follow_up_sent, note_added) refresh activity but leave
    # the status where it was.
    status_events = [r for r in rows if EventType(r.event_type) in STATUS_BY_EVENT]

    # `saved` is a floor, not a destination. Tracking a job stamps a `saved`
    # event at *now*, so backfilling ("I applied nine days ago") puts a real
    # event chronologically *before* it — and a naive latest-wins fold would
    # report the application as merely saved. Since you never return to saved
    # once you have applied, it only counts when nothing else has happened.
    advanced = [r for r in status_events if r.event_type != EventType.SAVED.value]
    considered = advanced or status_events

    if considered:
        latest = considered[-1]
        application.current_status = STATUS_BY_EVENT[EventType(latest.event_type)].value
        application.current_status_at = latest.occurred_at
    else:
        application.current_status = ApplicationStatus.SAVED.value
        application.current_status_at = rows[0].occurred_at

    applied = [r for r in rows if r.event_type == EventType.APPLIED.value]
    application.applied_at = applied[0].occurred_at if applied else None


async def get_application(
    session: AsyncSession, application_id: uuid.UUID, user_id: uuid.UUID
) -> Application:
    """Load an application the caller owns.

    The ``user_id`` filter is redundant with RLS — that is the point. Defence in
    depth: if a policy is ever dropped by a bad migration, this still holds.
    """
    application = (
        await session.execute(
            select(Application).where(
                Application.id == application_id, Application.user_id == user_id
            )
        )
    ).scalar_one_or_none()

    if application is None:
        raise NotFoundError(f"Application {application_id} not found")
    return application


async def reload_application(
    session: AsyncSession, application_id: uuid.UUID, user_id: uuid.UUID
) -> Application:
    """Re-read an application with its relationships freshly loaded.

    Used after appending an event. ``session.refresh()`` is not enough: it
    expires the eager-loaded ``events`` collection without repopulating it, and
    touching it afterwards in async context raises MissingGreenlet. Expiring and
    re-selecting lets the ``selectin`` loaders run again and pick up the row just
    written.
    """
    session.expire_all()
    return await get_application(session, application_id, user_id)


async def rebuild_all_cached_state(session: AsyncSession, user_id: uuid.UUID) -> int:
    """Replay every timeline and rewrite the cached columns.

    A repair tool, not part of the request path. Exists because the cache is an
    optimisation: if it is ever wrong, the event log is still authoritative and
    this restores agreement.
    """
    applications = (
        (await session.execute(select(Application).where(Application.user_id == user_id)))
        .scalars()
        .all()
    )

    for application in applications:
        await _refresh_cached_state(session, application)
    return len(applications)
