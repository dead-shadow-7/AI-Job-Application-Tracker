"""What the assistant may look up, and what it may propose.

**No tool here writes.** The four ``propose_*`` tools are the only apparent
exceptions and they write nothing either — each records an intention that the
API returns for confirmation. The model therefore cannot change anything at
all, which is the property the whole design rests on: a model that cannot write
cannot write to the wrong row.

Tools rather than a pre-loaded prompt because the useful questions are about
*depth*, not breadth. A job search is tens of applications, so listing them all
fits easily — but each one has requirements, skills, a match breakdown, a
timeline and a description, and putting all of that for all of them into every
prompt would exhaust the token budget on the first message.

A note on the size of this list. Every schema below is sent on every request, so
tools are not free — the block costs roughly two thousand tokens per round.
Each one earns that by answering a question the others cannot, or by collapsing
a chain of three calls into one. ``compare_applications`` and
``prepare_interview_brief`` exist for the second reason: the model could reach
the same facts through repeated detail lookups, but at three round trips and
three full detail blocks apiece.
"""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import analysis, proposals
from app.agent.resolving import resolve_application_only
from app.agent.tracing import hide, traced
from app.domain.enums import (
    EmploymentType,
    EventType,
    InterviewStageType,
    Priority,
    SalaryPeriod,
    Seniority,
    WorkMode,
)
from app.models.job import JobRequirement, JobSkill
from app.models.resume import MatchAnalysis
from app.models.skill import Skill
from app.services.applications import list_applications
from app.services.events import get_application
from app.services.followups import find_stale_applications
from app.services.search import search_applications

# One tool must not be able to swallow the per-minute token budget on its own.
# The cut is announced rather than silent, so the model does not summarise a
# fragment as though it were the whole thing.
MAX_TOOL_OUTPUT_CHARS = 6000


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble one function schema, widening optional parameters to allow null.

    Groq validates tool-call arguments against this schema and rejects the whole
    request with a 400 when they disagree. Models routinely emit ``"note": null``
    for an optional argument they have nothing to say about — omitting it and
    passing null are the same intent to them — so a bare ``{"type": "string"}``
    on an optional field turns a perfectly reasonable call into a failed message.
    Required parameters stay strict: a null there is a real mistake.
    """
    declared = properties or {}
    mandatory = required or []
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    key: schema if key in mandatory else _nullable(schema)
                    for key, schema in declared.items()
                },
                "required": mandatory,
            },
        },
    }


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    widened = dict(schema)
    kind = widened.get("type")
    if isinstance(kind, str):
        widened["type"] = [kind, "null"]
    # An enum has to list null explicitly too, or the union is unsatisfiable.
    if "enum" in widened and None not in widened["enum"]:
        widened["enum"] = [*widened["enum"], None]
    return widened


# Parameter descriptions are omitted where the name already says it. Every
# schema is re-sent on every round, so prose here is rented, not bought: the
# whole block is ~3,300 tokens before the user has typed anything, and a
# multi-step answer pays it several times over. Descriptions are kept only
# where they change what the model does — which value to copy, what a flag
# means, which of two similar tools to reach for.
def _str(description: str | None = None) -> dict[str, Any]:
    return {"type": "string", **({"description": description} if description else {})}


def _enum(enum_cls: type[StrEnum], description: str | None = None) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": [member.value for member in enum_cls],
        **({"description": description} if description else {}),
    }


def _num(description: str | None = None) -> dict[str, Any]:
    return {"type": "number", **({"description": description} if description else {})}


def _int(description: str | None = None) -> dict[str, Any]:
    return {"type": "integer", **({"description": description} if description else {})}


def _days(description: str) -> dict[str, Any]:
    return {"type": "integer", "minimum": 0, "description": description}


# The user's own phrasing, always. The tracker resolves the reference itself and
# refuses when it is ambiguous, which only works if it receives what was said.
_QUERY = _str("How the user referred to it. Copy their words, e.g. 'the Amazon one'.")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    # --- Reading -------------------------------------------------------------
    _tool(
        "list_applications",
        "Every application they track: company, role, status, idle days. Start here for "
        "anything general about their search.",
    ),
    _tool(
        "get_application_details",
        "Full detail for ONE role: required and preferred skills, requirements as written, "
        "salary, match score and breakdown. Use for any question about what a role asks for.",
        {"query": _QUERY},
        ["query"],
    ),
    _tool(
        "get_job_description",
        "The ORIGINAL posting text for one role. Use when asked for the JD, the description, "
        "or 'what does it actually say'. Other tools return a summary; this returns the source.",
        {"query": _QUERY},
        ["query"],
    ),
    _tool(
        "get_timeline",
        "The dated event history of one application — what happened and when.",
        {"query": _QUERY},
        ["query"],
    ),
    _tool(
        "search_applications",
        "Find roles by MEANING rather than exact words — 'the RAG ones', 'anything with "
        "infrastructure'. Use when they describe a kind of job instead of naming one.",
        {"query": _str("The kind of role they described.")},
        ["query"],
    ),
    _tool(
        "find_by_skill",
        "Which tracked roles ask for a named skill, and whether it is required or preferred.",
        {"skill": _str("One skill, e.g. 'Kubernetes'.")},
        ["skill"],
    ),
    _tool(
        "list_needing_attention",
        "Applications that have gone quiet, with how long and which follow-up rule fired. "
        "Use for 'what should I chase'.",
    ),
    _tool(
        "list_follow_up_rules",
        "The configured silence thresholds. Use to explain WHY something was flagged.",
    ),
    _tool(
        "get_upcoming_interviews",
        "Interview rounds still pending, soonest first.",
    ),
    # --- Analysing -----------------------------------------------------------
    _tool(
        "get_analytics",
        "How the search is going overall: response rate, typical wait for a reply, current "
        "standing, per-platform breakdown. Use for 'how am I doing'.",
    ),
    _tool(
        "get_skill_demand",
        "Which skills the jobs they track ask for most, and which are missing from their "
        "resume. Use for 'what should I learn' or 'what am I short on'.",
    ),
    _tool(
        "get_resume_profile",
        "What their resume says: years of experience and recognised skills. Use before "
        "discussing fit — otherwise you are guessing at half of it.",
    ),
    _tool(
        "compare_applications",
        "Two to four roles side by side: status, salary, required skills, match, gaps.",
        {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "How they referred to each role. Two to four entries.",
            }
        },
        ["queries"],
    ),
    # --- Drafting ------------------------------------------------------------
    # These return context and an instruction, not finished prose. You write the
    # text yourself, in the reply, with the conversation still in view.
    _tool(
        "draft_follow_up",
        "Gather what a follow-up message needs — how long it has been silent, what happened "
        "last, whether one was already sent. You then write the draft in your reply.",
        {"query": _QUERY},
        ["query"],
    ),
    _tool(
        "prepare_interview_brief",
        "Gather requirements, resume gaps and history for one role so you can write interview "
        "preparation notes in your reply.",
        {"query": _QUERY},
        ["query"],
    ),
    # --- Proposing (still writes nothing) ------------------------------------
    _tool(
        "propose_event",
        "Propose recording something on a timeline. Does NOT apply it — the user confirms "
        "separately. Use for 'mark X rejected', 'I heard back from Y', 'I applied to Z'.",
        {
            "query": _QUERY,
            "event_type": _enum(EventType),
            "occurred_days_ago": _days("How many days ago it happened. 0 or omit for today."),
            "note": _str(),
        },
        ["query", "event_type"],
    ),
    _tool(
        "propose_tracked_posting",
        "Use this when their message CONTAINS the job posting itself. Reads the posting "
        "straight from their message — do not retype it — and extracts company, role, "
        "salary, skills, requirements and the full text, the same way the paste screen "
        "does. Does NOT apply it. Prefer this over propose_new_application whenever a "
        "description was pasted.",
        {
            "url": _str(),
            "source_platform": _str("e.g. LinkedIn."),
            "status": {
                "type": "string",
                "enum": ["saved", "applied"],
                "description": "Whether they have already applied. Default applied.",
            },
            "applied_days_ago": _days("If already applied, how many days ago. 0 for today."),
        },
    ),
    _tool(
        "propose_new_application",
        "Propose tracking a job they only NAMED, with no description in the message. Does "
        "NOT apply it. Records company and title only — if they pasted the posting, use "
        "propose_tracked_posting instead, which keeps the salary and skills.",
        {
            "company_name": _str(),
            "title": _str(),
            "url": _str(),
            "location": _str(),
            "work_mode": _enum(WorkMode),
            "source_platform": _str("e.g. LinkedIn."),
            "notes": _str("Anything else they mentioned, including any salary they stated."),
            "status": {
                "type": "string",
                "enum": ["saved", "applied"],
                "description": "Whether they have already applied. Default saved.",
            },
            "applied_days_ago": _days("If already applied, how many days ago."),
        },
        ["company_name", "title"],
    ),
    _tool(
        "propose_description_edit",
        "Propose deleting lines from a stored job description — page furniture like "
        "'Application status' or 'Meet the hiring team' that came along with a paste. Quote "
        "the lines to remove — ALL of them in a single call, newline separated. They are "
        "matched and deleted literally. This can only REMOVE "
        "text. To reword anything, tell them to use the Edit button on the role panel.",
        {
            "query": _QUERY,
            "remove_text": _str("The exact lines to delete, copied from the description."),
        },
        ["query", "remove_text"],
    ),
    _tool(
        "propose_update",
        "Correct any tracked detail of an application: role title, location, work mode, "
        "seniority, employment type, salary, years of experience, where they found it, the "
        "link, priority, or your notes. Send only the fields that change. Does NOT apply it, "
        "does NOT touch the job description (use propose_description_edit), and cannot "
        "change status — that comes from the event log, so propose an event instead.",
        {
            "query": _QUERY,
            "title": _str(),
            "location": _str(),
            "work_mode": _enum(WorkMode),
            "seniority": _enum(Seniority),
            "employment_type": _enum(EmploymentType),
            "salary_min": _num(),
            "salary_max": _num(),
            "salary_currency": _str("Three letters, e.g. INR."),
            "salary_period": _enum(SalaryPeriod),
            "years_experience_min": _int(),
            "years_experience_max": _int(),
            "source_platform": _str(),
            "url": _str(),
            "priority": _enum(Priority),
            "notes": _str("Their own short remark, not a document."),
            "clear": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fields to EMPTY, e.g. ['notes']. Omitting a field means no change.",
            },
        },
        ["query"],
    ),
    _tool(
        "propose_delete",
        "Propose permanently deleting an application and its whole history. Does NOT apply "
        "it. Irreversible, unlike everything else you can propose — offer to log 'withdrawn' "
        "instead, which keeps the history and takes it off the active list.",
        {"query": _QUERY},
        ["query"],
    ),
    _tool(
        "propose_interview_round",
        "Propose scheduling an interview round. Does NOT apply it. This is the one place a "
        "future date belongs.",
        {
            "query": _QUERY,
            "stage_type": _enum(InterviewStageType),
            "in_days": _days("How many days from now it is scheduled."),
            "round_number": {"type": "integer", "minimum": 1},
            "interviewer": _str(),
            "notes": _str(),
        },
        ["query", "stage_type"],
    ),
]


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What a tool produced, split by who it is for.

    ``output`` goes to the model. ``attachment`` goes straight to the user,
    bypassing it entirely — which is the only way to hand over a stored document
    intact. Asked to relay a 3,400-character job description, the model rewrote
    it down to 1,900 and sometimes replied with nothing but "here's the job
    description", because re-emitting a long document verbatim is not something
    it reliably does, and paying output tokens to echo text already in hand is
    waste even when it works.
    """

    output: str
    proposal: dict[str, Any] | None = None
    attachment: dict[str, Any] | None = None


Handler = Callable[[AsyncSession, uuid.UUID, dict[str, Any]], Awaitable[str | ToolResult]]


@traced("tool", run_type="tool", process_inputs=hide("session", "message"))
async def run_tool(
    name: str,
    arguments: dict[str, Any],
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    message: str = "",
) -> ToolResult:
    """Execute a tool.

    ``message`` is the user's turn verbatim. One tool needs it: tracking a
    pasted posting reads the text from here rather than from an argument, so
    the model never has to echo a document back — which costs the tokens twice
    and returns it rewritten.

    Unknown names return an error string rather than raising: a model that
    hallucinates a tool should be told so and allowed to correct itself, not
    crash the request.
    """
    needs_message = _MESSAGE_PROPOSERS.get(name)
    if needs_message is not None:
        text, proposal = await needs_message(session, user_id, arguments, message=message)
        return ToolResult(output=_clip(text), proposal=proposal)

    proposer = _PROPOSERS.get(name)
    if proposer is not None:
        text, proposal = await proposer(session, user_id, arguments)
        return ToolResult(output=_clip(text), proposal=proposal)

    reader = _READERS.get(name)
    if reader is None:
        return ToolResult(output=f"No such tool: {name}. Use one of the tools you were given.")

    produced = await reader(session, user_id, arguments)
    if isinstance(produced, ToolResult):
        return ToolResult(output=_clip(produced.output), attachment=produced.attachment)
    return ToolResult(output=_clip(produced))


def _clip(text: str) -> str:
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return text[:MAX_TOOL_OUTPUT_CHARS] + "\n\n[truncated — this is the first part only]"


# --- Read handlers -----------------------------------------------------------


async def _list_applications(session: AsyncSession, user_id: uuid.UUID, _: dict[str, Any]) -> str:
    rows, _total = await list_applications(session, user_id=user_id, limit=100)
    if not rows:
        return "They are not tracking any applications yet."

    now = datetime.now(UTC)
    lines = []
    for a in rows:
        parts = [f"{a.job.title} at {a.job.company.name}", f"status {a.current_status}"]
        if a.match_score is not None:
            parts.append(f"match {a.match_score}/100")
        parts.append(f"idle {(now - a.last_activity_at).days}d")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


async def _details(session: AsyncSession, user_id: uuid.UUID, args: dict[str, Any]) -> str:
    application, problem = await resolve_application_only(session, user_id, args.get("query", ""))
    if application is None:
        return problem or "No such application."

    job = application.job
    lines = [
        f"{job.title} at {job.company.name}",
        f"status: {application.current_status}",
        f"priority: {application.priority}",
    ]
    if job.location:
        lines.append(f"location: {job.location}{f' ({job.work_mode})' if job.work_mode else ''}")
    if job.salary_min or job.salary_max:
        lines.append(
            f"salary: {job.salary_min or '?'}-{job.salary_max or '?'} "
            f"{job.salary_currency or ''} per {job.salary_period or 'year'}"
        )
    if job.years_experience_min is not None:
        lines.append(f"experience wanted: {job.years_experience_min}+ years")
    if application.notes:
        lines.append(f"their notes: {application.notes}")

    skills = (
        await session.execute(
            select(Skill.name, JobSkill.is_required)
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .where(JobSkill.job_id == job.id)
        )
    ).all()
    if skills:
        required = [n for n, req in skills if req]
        preferred = [n for n, req in skills if not req]
        if required:
            lines.append(f"required skills: {', '.join(required)}")
        if preferred:
            lines.append(f"preferred skills: {', '.join(preferred)}")

    requirements = (
        await session.execute(
            select(JobRequirement.text, JobRequirement.kind).where(JobRequirement.job_id == job.id)
        )
    ).all()
    for text, kind in requirements:
        lines.append(f"requirement ({kind}): {text}")

    match = (
        await session.execute(
            select(MatchAnalysis).where(
                MatchAnalysis.job_id == job.id, MatchAnalysis.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if match:
        lines.append(f"match score: {match.overall_score}/100")
        if match.missing_skills:
            lines.append(f"missing from their resume: {', '.join(match.missing_skills)}")
        if match.gaps:
            lines.append("gaps: " + "; ".join(match.gaps))

    return "\n".join(lines)


async def _timeline(session: AsyncSession, user_id: uuid.UUID, args: dict[str, Any]) -> str:
    application, problem = await resolve_application_only(session, user_id, args.get("query", ""))
    if application is None:
        return problem or "No such application."

    full = await get_application(session, application.id, user_id)
    lines = [f"Timeline for {full.job.title} at {full.job.company.name}:"]
    for event in full.events:
        stamp = event.occurred_at.date().isoformat()
        note = f" — {event.note}" if event.note else ""
        lines.append(f"  {stamp}: {event.event_type}{note}")
    return "\n".join(lines)


# A pasted posting can run to tens of thousands of characters, so the model's
# copy is cut harder than the generic clip. The user's copy is not cut at all —
# it does not travel through the context window.
MAX_DESCRIPTION_CHARS = 5000


async def _description(
    session: AsyncSession, user_id: uuid.UUID, args: dict[str, Any]
) -> str | ToolResult:
    """The posting as written.

    Returned twice, deliberately. The attachment is the whole stored text and
    goes to the user untouched; the model gets a copy so it can answer questions
    about the posting, and is told not to repeat it. Making the model relay the
    document was the bug: it rewrote a 3,400-character description down to
    1,900, and sometimes announced it without reproducing anything.

    Kept apart from get_application_details deliberately. That tool returns a
    structured summary, and merging the two would put a full posting into every
    detail lookup — expensive, and usually not what was asked for.
    """
    application, problem = await resolve_application_only(session, user_id, args.get("query", ""))
    if application is None:
        return problem or "No such application."

    job = application.job
    label = f"{job.title} at {job.company.name}"
    if not job.description:
        return (
            f"No description was stored for {label}. "
            "It was probably entered by hand rather than pasted."
        )

    for_model = job.description
    if len(for_model) > MAX_DESCRIPTION_CHARS:
        for_model = for_model[:MAX_DESCRIPTION_CHARS] + "\n\n[truncated for you only]"

    return ToolResult(
        output=(
            f"The full job description for {label} is ALREADY BEING SHOWN to the user, "
            "in full, below your reply. Do NOT reproduce, quote at length, or summarise "
            "it unless they asked a specific question about it — say one line to "
            "introduce it and stop. A copy follows so you can answer questions about "
            f"it:\n\n{for_model}"
        ),
        attachment={"kind": "job_description", "title": label, "body": job.description},
    )


async def _search(session: AsyncSession, user_id: uuid.UUID, args: dict[str, Any]) -> str:
    query = args.get("query", "")
    hits = await search_applications(session, user_id, query)
    if not hits:
        return (
            f"Nothing they track resembles {query!r}. Note that only jobs added by pasting a "
            "description can be searched this way; ones typed by hand have no text to match."
        )
    return "\n".join(
        f"{h.application.job.title} at {h.application.job.company.name} "
        f"(status {h.application.current_status}, similarity {h.similarity})"
        for h in hits
    )


async def _needing_attention(session: AsyncSession, user_id: uuid.UUID, _: dict[str, Any]) -> str:
    stale = await find_stale_applications(session, user_id)
    if not stale:
        return "Nothing has gone quiet — no follow-up rules have fired."
    return "\n".join(f"{item.reason} (rule: {item.rule.days_threshold} days)" for item in stale)


_READERS: dict[str, Handler] = {
    "list_applications": _list_applications,
    "get_application_details": _details,
    "get_job_description": _description,
    "get_timeline": _timeline,
    "search_applications": _search,
    "list_needing_attention": _needing_attention,
    "find_by_skill": lambda s, u, a: analysis.by_skill(s, u, a.get("skill", "")),
    "list_follow_up_rules": lambda s, u, _a: analysis.follow_up_rules(s, u),
    "get_upcoming_interviews": lambda s, u, _a: analysis.upcoming_interviews(s, u),
    "get_analytics": lambda s, u, _a: analysis.analytics_summary(s, u),
    "get_skill_demand": lambda s, u, _a: analysis.skill_demand(s, u),
    "get_resume_profile": lambda s, u, _a: analysis.resume_profile(s, u),
    "compare_applications": lambda s, u, a: analysis.compare(s, u, list(a.get("queries") or [])),
    "draft_follow_up": lambda s, u, a: analysis.follow_up_context(s, u, a.get("query", "")),
    "prepare_interview_brief": lambda s, u, a: analysis.interview_brief(s, u, a.get("query", "")),
}

Proposer = Callable[
    [AsyncSession, uuid.UUID, dict[str, Any]], Awaitable[tuple[str, dict[str, Any] | None]]
]

_PROPOSERS: dict[str, Proposer] = {
    "propose_event": proposals.propose_event,
    "propose_new_application": proposals.propose_new_application,
    "propose_update": proposals.propose_update,
    "propose_interview_round": proposals.propose_stage,
    "propose_delete": proposals.propose_delete,
}


# Cheap insurance against a schema and its handler drifting apart — a tool the
# model can see but nothing can run produces a baffling "No such tool" at
# runtime, from a list the model was explicitly given.
class MessageProposer(Protocol):
    """A proposer that also needs the user's turn verbatim.

    Its own registry rather than a branch in the dispatch, so the coverage
    assertion below still sees every tool. A tool reachable only through a
    special case is exactly the one that gets dropped when the dispatch is next
    rearranged.
    """

    async def __call__(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        arguments: dict[str, Any],
        *,
        message: str,
    ) -> tuple[str, dict[str, Any] | None]: ...


_MESSAGE_PROPOSERS: dict[str, MessageProposer] = {
    "propose_tracked_posting": proposals.propose_tracked_posting,
    "propose_description_edit": proposals.propose_description_edit,
}

HANDLED = set(_READERS) | set(_PROPOSERS) | set(_MESSAGE_PROPOSERS)

assert {t["function"]["name"] for t in TOOL_SCHEMAS} == HANDLED
