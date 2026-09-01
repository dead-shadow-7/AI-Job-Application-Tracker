"""The four things the assistant can ask to change — and it only ever asks.

Every function here returns ``(text for the model, proposal)``. The proposal is
handed back through the API for the user to confirm; nothing in this module
touches a row. That is the property the whole agent design rests on, and it is
only obvious if the write path is visibly absent, so it is kept in one file
where you can check.

Two rules the builders enforce, because the model cannot be relied on for
either:

*Resolution before action.* Anything targeting an existing application goes
through the resolver, which refuses to guess between candidates.

*The card must show everything.* ``details`` is built from the same values that
go into ``payload``. A field the model filled in but the card omitted would be
written without anyone having seen it, which would make the confirmation
theatre rather than a check.
"""

import re
import uuid
from difflib import SequenceMatcher
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.resolving import resolve_one
from app.domain.enums import EventType, InterviewStageType, Priority, WorkMode
from app.services.resolver import Candidate, resolve_application


def _enum_or_none[T](enum_cls: type[T], value: Any) -> T | None:
    try:
        return enum_cls(value)  # type: ignore[call-arg]
    except (ValueError, TypeError):
        return None


def _targeted(
    candidate: Candidate, kind: str, summary: str, details: list[str], payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        "kind": kind,
        "summary": summary,
        "details": details,
        "payload": {"application_id": str(candidate.application.id), **payload},
        "application_id": str(candidate.application.id),
        "application_label": candidate.label,
        "confidence": candidate.score,
        "matched_on": candidate.matched_on,
    }


async def propose_event(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Log something on an existing timeline."""
    query = arguments.get("query", "")
    event_type = _enum_or_none(EventType, arguments.get("event_type"))
    if not query or event_type is None:
        return "I need both an application and a valid event type to propose that.", None

    candidate, problem = await resolve_one(session, user_id, query)
    if candidate is None:
        return problem or "No such application.", None

    days_ago = _non_negative(arguments.get("occurred_days_ago"))
    note = arguments.get("note")

    details = [f"Event: {event_type.value}", f"On: {candidate.label}"]
    if days_ago:
        details.append(f"Dated: {days_ago} days ago")
    if note:
        details.append(f"Note: {note}")

    return (
        f"Prepared, pending confirmation: log '{event_type.value}' on {candidate.label}. "
        "Tell them what you are about to record.",
        _targeted(
            candidate,
            "append_event",
            f"Log {event_type.value} on {candidate.label}",
            details,
            {"event_type": event_type.value, "note": note, "occurred_days_ago": days_ago},
        ),
    )


async def propose_new_application(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Track a job described in conversation.

    Checks for an existing match first. Someone saying "add the Amazon backend
    role" when they already track it means they have lost track, and a second
    row splits the timeline in two — after which neither half tells the truth
    and the follow-up sweep sees two partial stories.
    """
    company = (arguments.get("company_name") or "").strip()
    title = (arguments.get("title") or "").strip()
    if not company or not title:
        return (
            "A company and a role title are both needed. Ask for whichever is missing.",
            None,
        )

    duplicate = await _already_tracked(session, user_id, company, title)
    if duplicate is not None:
        return (
            f"They already track {duplicate}. Do not create a second one — "
            "ask whether they meant to record an event on it instead.",
            None,
        )

    applied = str(arguments.get("status", "saved")).lower() == "applied"
    days_ago = _non_negative(arguments.get("applied_days_ago"))
    work_mode = _enum_or_none(WorkMode, arguments.get("work_mode"))

    payload: dict[str, Any] = {
        "company_name": company,
        "title": title,
        "url": arguments.get("url"),
        "location": arguments.get("location"),
        "work_mode": work_mode.value if work_mode else None,
        "source_platform": arguments.get("source_platform"),
        "notes": arguments.get("notes"),
        "initial_event": EventType.APPLIED.value if applied else EventType.SAVED.value,
        "occurred_days_ago": days_ago,
    }

    details = [f"Company: {company}", f"Role: {title}"]
    for label, key in (
        ("Location", "location"),
        ("Work mode", "work_mode"),
        ("Link", "url"),
        ("Found on", "source_platform"),
        ("Notes", "notes"),
    ):
        if payload[key]:
            details.append(f"{label}: {payload[key]}")
    details.append(
        f"Recorded as: applied {days_ago} days ago"
        if applied
        else "Recorded as: saved, not applied"
    )

    return (
        f"Prepared, pending confirmation: start tracking {title} at {company}. "
        "Say what you are about to add, and mention that salary and skills are not "
        "captured this way — pasting the description is what fills those in.",
        {
            "kind": "create_application",
            "summary": f"Start tracking {title} at {company}",
            "details": details,
            "payload": payload,
        },
    )


async def propose_update(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Change priority or notes.

    Status is not settable and never will be: it is derived from the event log,
    so moving an application means appending an event. A direct write here would
    let the cache diverge from its own source of truth.
    """
    query = arguments.get("query", "")
    priority = _enum_or_none(Priority, arguments.get("priority"))
    notes = arguments.get("notes")

    if priority is None and not notes:
        return (
            "Nothing to change. Priority must be low, medium or high; notes is free text. "
            "To move an application's status, propose an event instead.",
            None,
        )

    candidate, problem = await resolve_one(session, user_id, query)
    if candidate is None:
        return problem or "No such application.", None

    details = []
    if priority:
        details.append(f"Priority: {priority.value}")
    if notes:
        details.append(f"Notes: {notes}")
    details.append(f"On: {candidate.label}")

    return (
        f"Prepared, pending confirmation: update {candidate.label}.",
        _targeted(
            candidate,
            "update_application",
            f"Update {candidate.label}",
            details,
            {"priority": priority.value if priority else None, "notes": notes},
        ),
    )


async def propose_stage(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Schedule an interview round.

    The one place a future date belongs. It goes on the stage, while the event
    that records the scheduling is stamped now — an event dated in the future
    would push last_activity_at forward and make a stalled application look
    fresh, disabling the follow-up detection.
    """
    query = arguments.get("query", "")
    stage_type = _enum_or_none(InterviewStageType, arguments.get("stage_type"))
    if stage_type is None:
        return (
            "I need a round type: hr_screen, recruiter_call, technical, coding, "
            "system_design, managerial, hiring_manager, culture_fit, take_home, final or other.",
            None,
        )

    candidate, problem = await resolve_one(session, user_id, query)
    if candidate is None:
        return problem or "No such application.", None

    in_days = _non_negative(arguments.get("in_days"))
    interviewer = arguments.get("interviewer")
    round_number = arguments.get("round_number")
    if not isinstance(round_number, int) or not 1 <= round_number <= 20:
        round_number = None

    details = [
        f"Round: {stage_type.value}",
        f"When: in {in_days} days" if in_days else "When: today",
        f"On: {candidate.label}",
    ]
    if interviewer:
        details.append(f"With: {interviewer}")
    details.append("Also logs 'interview scheduled' on the timeline.")

    return (
        f"Prepared, pending confirmation: schedule a {stage_type.value} round on "
        f"{candidate.label} in {in_days} days.",
        _targeted(
            candidate,
            "schedule_interview",
            f"Schedule {stage_type.value} on {candidate.label}",
            details,
            {
                "stage_type": stage_type.value,
                "in_days": in_days,
                "round_number": round_number,
                "interviewer": interviewer,
                "notes": arguments.get("notes"),
            },
        ),
    )


# Two titles this similar at the same company are the same opening. Loose enough
# for "SDE" against "SDE I" and for punctuation drift, tight enough that Backend
# and Frontend Engineer stay separate.
SAME_ROLE = 0.8


async def _already_tracked(
    session: AsyncSession, user_id: uuid.UUID, company: str, title: str
) -> str | None:
    """Is this company-and-role pair already on the board?

    Resolved by company alone, then compared on title. Asking the resolver for
    "Backend Engineer Amazon" retrieves nothing at all: retrieval is a substring
    and trigram match over the company and the title *separately*, so a string
    joining the two matches neither — the check silently never fired.
    """
    wanted = _slug(title)
    for candidate in (await resolve_application(session, user_id, company)).candidates:
        found = _slug(candidate.application.job.title)
        if found == wanted or SequenceMatcher(None, found, wanted).ratio() >= SAME_ROLE:
            return candidate.label
    return None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _non_negative(value: Any) -> int:
    """Models pass "3", 3.0 and None interchangeably. Backdating is allowed;
    negative days would be a future date by another name."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
