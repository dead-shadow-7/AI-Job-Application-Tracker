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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.resolving import resolve_one
from app.domain.enums import EventType, InterviewStageType, WorkMode
from app.models.application import ApplicationEvent
from app.schemas.agent import CLEARABLE
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


async def propose_tracked_posting(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any], *, message: str
) -> tuple[str, dict[str, Any] | None]:
    """Track a posting the user pasted into the conversation, in full.

    ``propose_new_application`` records a company and a title because that is
    all a sentence contains. When the posting itself is in the message there is
    no reason to throw the rest away, and doing so was the wrong answer to "add
    this job" — it stored two fields out of a document and told the user to go
    and paste it again somewhere else.

    So this runs the same ingestion graph the paste screen runs: extraction
    against the strict schema, the validation pass that drops a salary not
    present verbatim, company resolution and skill normalisation. Same pipeline,
    same guarantees, one less round trip for the user.

    The posting text is read from the message server-side and never passed
    through the model. Asking it to echo a document back as a tool argument
    costs the tokens twice and arrives rewritten — the same failure that made
    "share the JD" return a summary of itself.
    """
    from app.agent.graphs.ingestion import run_ingestion
    from app.services.job_drafts import MIN_POSTING_CHARS, build_job_draft

    text = message.strip()
    if len(text) < MIN_POSTING_CHARS:
        return (
            "There is no posting in that message — only a sentence. Use "
            "propose_new_application for the company and title, or ask them to paste "
            "the description if they want the skills and salary captured too.",
            None,
        )

    state = await run_ingestion(
        session=session,
        raw_text=text,
        url=arguments.get("url"),
        source_platform=arguments.get("source_platform"),
        user_id=str(user_id),
    )
    if state.get("error") or state.get("extracted") is None:
        return f"Could not read that posting: {state.get('error') or 'extraction failed'}.", None

    draft = build_job_draft(
        state, url=arguments.get("url"), source_platform=arguments.get("source_platform")
    )
    if not draft.company_name or not draft.title:
        return (
            "The posting does not name the company or the role clearly. Ask them which "
            "it is rather than guessing — postings routinely omit the employer.",
            None,
        )

    duplicate = await _already_tracked(session, user_id, draft.company_name, draft.title)
    if duplicate is not None:
        return (
            f"They already track {duplicate}. Do not add a second one — ask whether they "
            "meant to record an event on it instead.",
            None,
        )

    applied = str(arguments.get("status", "applied")).lower() != "saved"
    days_ago = _non_negative(arguments.get("applied_days_ago"))
    report = state["report"]

    details = [f"Company: {draft.company_name}", f"Role: {draft.title}"]
    if draft.location:
        details.append(
            f"Location: {draft.location}{f' ({draft.work_mode})' if draft.work_mode else ''}"
        )
    if draft.salary_min or draft.salary_max:
        details.append(
            f"Salary: {draft.salary_min or '?'}-{draft.salary_max or '?'} "
            f"{draft.salary_currency or ''} per {draft.salary_period or 'year'}"
        )
    details.append(f"Skills captured: {len(draft.skill_slugs)}")
    details.append(f"Requirements captured: {len(draft.requirements)}")
    details.append(f"Full description stored: {len(draft.description or '')} characters")
    if draft.url:
        details.append(f"Link: {draft.url}")
    if report.dropped_fields:
        # Named on the card because a silently absent salary looks like the
        # posting did not state one, rather than like a check that fired.
        details.append(f"Dropped as unverifiable: {', '.join(report.dropped_fields)}")
    details.append(
        f"Recorded as: applied {days_ago} days ago"
        if applied
        else "Recorded as: saved, not applied"
    )

    payload = draft.model_dump(mode="json")
    payload["initial_event"] = EventType.APPLIED.value if applied else EventType.SAVED.value
    payload["occurred_days_ago"] = days_ago

    warned = (
        f" Salary was dropped as unverifiable: {', '.join(report.dropped_fields)}."
        if (report.dropped_fields)
        else ""
    )
    return (
        f"Prepared, pending confirmation: track {draft.title} at {draft.company_name} with the "
        f"full posting, {len(draft.skill_slugs)} skills and {len(draft.requirements)} "
        f"requirements.{warned} Summarise what was captured and ask them to confirm.",
        {
            "kind": "create_from_posting",
            "summary": f"Track {draft.title} at {draft.company_name}, with the full posting",
            "details": details,
            "payload": payload,
        },
    )


async def propose_description_edit(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any], *, message: str
) -> tuple[str, dict[str, Any] | None]:
    """Cut lines out of a stored posting.

    Pasted postings arrive carrying the page around them — "Application status",
    "View resume", "Meet the hiring team", the recruiter's headline. It is
    noise, it is what the assistant reads when asked about the role, and asking
    to be rid of it is reasonable.

    Removal only, never rewriting. Handing the model a 3,000-character
    description and asking for a corrected one gets a *rewrite* — it condenses
    and reflows, the same failure that made "share the JD" return a summary of
    itself, except here the result would be saved over the original. So the
    model says which lines to drop and the server drops exactly those,
    literally. Anything more involved than deleting lines belongs in the
    editor on the page, where the user types the text themselves.
    """
    candidate, problem = await resolve_one(session, user_id, arguments.get("query", ""))
    if candidate is None:
        return problem or "No such application.", None

    job = candidate.application.job
    if not job.description:
        return f"No description is stored for {candidate.label}, so there is nothing to trim.", None

    # Both the model's quote and the user's own message, because neither alone
    # is reliable. Asked to remove a block of six lines the model splits it into
    # six separate tool calls, and only one proposal survives a turn — so acting
    # on its argument alone would delete a sixth of what was asked for. The
    # user's message is where the block actually is; the instruction wrapped
    # around it matches nothing in the description and falls away by itself.
    #
    # Matched line by line on trimmed text: the model reflows whatever it
    # quotes, so a substring match on the block as given fails almost always.
    unwanted = {
        line.strip()
        for source in (arguments.get("remove_text") or "", message)
        for line in source.splitlines()
        if line.strip()
    }
    if not unwanted:
        return "I need the exact text to remove. Quote it from the description.", None
    kept: list[str] = []
    dropped: list[str] = []
    for line in job.description.splitlines():
        (dropped if line.strip() and line.strip() in unwanted else kept).append(line)

    if not dropped:
        return (
            "None of those lines appear in the stored description. Ask them to quote it "
            "exactly as it is shown, or to use the Edit button on the role panel.",
            None,
        )

    cleaned = "\n".join(kept).strip()
    if not cleaned:
        return (
            "That would delete the entire description. Refuse, and say so — if they really "
            "want it empty they can clear it in the editor on the page.",
            None,
        )
    # Trimming page furniture takes a few lines off the top and bottom. Anything
    # taking most of the posting is a mismatch, not a trim — and this deletes
    # text that only exists here once the source tab is closed.
    #
    # Only applied to descriptions long enough for "most of it" to mean
    # something. On a three-line note, removing two lines is an ordinary edit.
    guarded = len(job.description) >= MIN_GUARDED_CHARS
    if guarded and len(cleaned) < len(job.description) * (1 - MAX_TRIM_FRACTION):
        return (
            f"That would cut most of the description away, leaving {len(cleaned)} of "
            f"{len(job.description)} characters. Refuse — that is a rewrite, not a trim. "
            "Point them at the Edit button on the role panel.",
            None,
        )

    missed = len(unwanted) - len({line.strip() for line in dropped})
    preview = [line.strip() for line in dropped[:5]]
    details = [
        f"Removing {len(dropped)} lines, {len(job.description) - len(cleaned)} characters",
        f"Description goes from {len(job.description)} to {len(cleaned)} characters",
        *[f"  − {line[:70]}" for line in preview],
    ]
    if len(dropped) > len(preview):
        details.append(f"  … and {len(dropped) - len(preview)} more lines")
    if missed:
        details.append(f"{missed} of the quoted lines were not found and are left alone")

    return (
        f"Prepared, pending confirmation: remove {len(dropped)} lines from the description of "
        f"{candidate.label}. Say how much is going, and that nothing else in it is touched.",
        _targeted(
            candidate,
            "edit_description",
            f"Trim {len(dropped)} lines from {candidate.label}",
            details,
            {"description": cleaned},
        ),
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
    notes = arguments.get("notes")

    # Every value the model actually supplied. Nulls are dropped rather than
    # read as "empty this column": optional tool parameters are declared
    # nullable, and models emit null for arguments they simply have nothing to
    # say about. Clearing is a separate, explicit list.
    values = {key: arguments[key] for key in EDITABLE if arguments.get(key) not in (None, "")}
    clear = [f for f in (arguments.get("clear") or []) if f in CLEARABLE]

    if not values and not clear:
        return (
            "Nothing to change. Say which field and its new value, or list fields to clear. "
            "Status is not one of them — it comes from the event log, so propose an event instead.",
            None,
        )

    # A note the length of a document is a description that came to the wrong
    # tool. Asked to trim a job description with no tool for it, the model
    # reached for this one and wrote "Removed extraneous details from the job
    # description" into notes — then reported the edit as done, while the
    # description sat untouched. Nothing about that was visible to the user.
    if notes and len(notes) > MAX_NOTE_CHARS:
        return (
            "That is too long for a note. If they are trying to change the job description, "
            "use propose_description_edit; notes are for your own short remarks.",
            None,
        )

    candidate, problem = await resolve_one(session, user_id, query)
    if candidate is None:
        return problem or "No such application.", None

    details = [f"{EDITABLE[key]}: {value}" for key, value in values.items()]
    details += [f"{EDITABLE[key]}: cleared" for key in clear]
    details.append(f"On: {candidate.label}")

    # Names the fields. "Update Backend Engineer at Amazon" is true of half a
    # dozen different actions, and a card you have to read twice to tell them
    # apart is one you stop reading.
    touched = [EDITABLE[k] for k in [*values, *clear]]
    named = ", ".join(touched[:3]) + (f" and {len(touched) - 3} more" if len(touched) > 3 else "")

    return (
        f"Prepared, pending confirmation: set {named} on {candidate.label}. This does NOT "
        "touch the job description — that is propose_description_edit.",
        _targeted(
            candidate,
            "update_application",
            f"Set {named} on {candidate.label}",
            details,
            {**values, "clear": clear},
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


async def propose_delete(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Remove an application and its history.

    The only proposal here that cannot be undone. Everything else the agent
    writes is an event appended to a log, corrected by appending another; this
    destroys the log. It exists because not having it was worse — the assistant
    could add an application but not take one back, so its own mistake became
    the user's manual cleanup.

    The card says how much history goes with it, and the model is told to offer
    `withdrawn` first: someone who has decided against a role usually wants that
    recorded, not erased.
    """
    query = arguments.get("query", "")
    candidate, problem = await resolve_one(session, user_id, query)
    if candidate is None:
        return problem or "No such application.", None

    application = candidate.application
    events = (
        await session.execute(
            select(func.count())
            .select_from(ApplicationEvent)
            .where(ApplicationEvent.application_id == application.id)
        )
    ).scalar_one()

    details = [
        f"Deletes: {candidate.label}",
        f"Also deletes its timeline: {events} event{'' if events == 1 else 's'}",
        "This cannot be undone — there is no correcting event for a deletion.",
    ]

    return (
        f"Prepared, pending confirmation: permanently delete {candidate.label} and its "
        f"{events} timeline events. Warn them it cannot be undone, and ask whether they "
        "would rather log 'withdrawn' — that keeps the history and takes it off the "
        "active list.",
        _targeted(
            candidate,
            "delete_application",
            f"Permanently delete {candidate.label}",
            details,
            {},
        ),
    )


# Two titles this similar at the same company are the same opening. Loose enough
# for "SDE" against "SDE I" and for punctuation drift, tight enough that Backend
# and Frontend Engineer stay separate.
SAME_ROLE = 0.8

# Longer than any remark someone types about their own application, and far
# shorter than a job description. The gap between the two is what makes this a
# usable signal that the model has confused the one for the other.
MAX_NOTE_CHARS = 1000

# Everything the assistant may correct on a tracked application, and the label
# the confirm card shows for it. The same set the Edit buttons on the page
# offer, so "ask it" and "do it yourself" reach the same fields — a capability
# available in one place and not the other is the kind of gap you rediscover
# every time you hit it.
#
# Absent on purpose: status (derived from the event log), company (renaming a
# shared row would rename it for every application against it), and the
# description (a document, and its own tool).
EDITABLE = {
    "title": "Role title",
    "location": "Location",
    "work_mode": "Work mode",
    "seniority": "Seniority",
    "employment_type": "Employment type",
    "salary_min": "Salary from",
    "salary_max": "Salary to",
    "salary_currency": "Currency",
    "salary_period": "Salary period",
    "years_experience_min": "Years from",
    "years_experience_max": "Years to",
    "source_platform": "Found on",
    "url": "Posting link",
    "priority": "Priority",
    "notes": "Your notes",
}

# Removing page furniture takes a few lines. Removing over half the posting is a
# mismatched quote, and the text being deleted usually exists nowhere else.
MAX_TRIM_FRACTION = 0.5
MIN_GUARDED_CHARS = 500


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
