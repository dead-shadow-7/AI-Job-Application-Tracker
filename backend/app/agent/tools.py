"""What the assistant may look up, and what it may propose.

**Every tool here is read-only.** ``propose_event`` is the sole apparent
exception and it writes nothing either — it records an intention that the API
returns for confirmation. The model therefore cannot change anything at all,
which is the property the whole design rests on: a model that cannot write
cannot write to the wrong row.

Tools rather than a pre-loaded prompt because the useful questions are about
*depth*, not breadth. A job search is only tens of applications, so listing
them all fits easily — but each one has requirements, skills, a match
breakdown and a timeline, and putting all of that for all of them into every
prompt would exhaust the token budget on the first message. Fetching detail for
the one application actually being discussed is what makes "what skills did it
ask for" answerable.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.application import Application
from app.models.job import JobRequirement, JobSkill
from app.models.resume import MatchAnalysis
from app.models.skill import Skill
from app.services.applications import list_applications
from app.services.events import get_application
from app.services.followups import find_stale_applications
from app.services.resolver import resolve_application

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_applications",
            "description": (
                "List the user's tracked applications with company, role, status and "
                "how many days each has been idle. Start here when asked anything "
                "general about their search."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_application_details",
            "description": (
                "Full detail for ONE application: required and preferred skills, the "
                "requirements as written, salary, match score and its breakdown. Use "
                "this for any question about what a role asks for or how well it fits."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "How the user referred to it, e.g. 'Amazon'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline",
            "description": "The dated event history of one application — what happened and when.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "How the user referred to it."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_description",
            "description": (
                "The original job posting text, as it was pasted or written. Use this "
                "when asked for the JD, the description, the responsibilities, or "
                "anything phrased 'what does the posting actually say'. The other "
                "tools return a structured summary; this returns the source."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "How the user referred to it."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_needing_attention",
            "description": (
                "Applications that have gone quiet, with how long and which follow-up "
                "rule fired. Use for 'what should I chase' or 'what is stale'."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_event",
            "description": (
                "Propose recording something on an application's timeline. This does "
                "NOT happen until the user confirms it, so say what you are proposing. "
                "Use for 'mark X as rejected', 'I heard back from Y', 'I applied to Z'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "How the user referred to it. Copy their words.",
                    },
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "applied",
                            "assessment_received",
                            "screening_scheduled",
                            "screening_done",
                            "interview_scheduled",
                            "interview_done",
                            "offer_received",
                            "accepted",
                            "rejected",
                            "withdrawn",
                            "recruiter_reply",
                            "follow_up_sent",
                            "note_added",
                        ],
                    },
                    "note": {"type": "string", "description": "Optional note to attach."},
                },
                "required": ["query", "event_type"],
            },
        },
    },
]


async def run_tool(
    name: str, arguments: dict[str, Any], session: AsyncSession, user_id: uuid.UUID
) -> tuple[str, dict[str, Any] | None]:
    """Execute a tool. Returns (text for the model, proposal if any).

    Unknown names return an error string rather than raising: a model that
    hallucinates a tool should be told so and allowed to correct itself, not
    crash the request.
    """
    if name == "list_applications":
        return await _list_applications(session, user_id), None
    if name == "get_application_details":
        return await _details(session, user_id, arguments.get("query", "")), None
    if name == "get_timeline":
        return await _timeline(session, user_id, arguments.get("query", "")), None
    if name == "get_job_description":
        return await _description(session, user_id, arguments.get("query", "")), None
    if name == "list_needing_attention":
        return await _needing_attention(session, user_id), None
    if name == "propose_event":
        return await _propose(session, user_id, arguments)
    return f"No such tool: {name}.", None


async def _list_applications(session: AsyncSession, user_id: uuid.UUID) -> str:
    rows, _ = await list_applications(session, user_id=user_id, limit=100)
    if not rows:
        return "They are not tracking any applications yet."

    lines = []
    for a in rows:
        parts = [f"{a.job.title} at {a.job.company.name}", f"status {a.current_status}"]
        if a.match_score is not None:
            parts.append(f"match {a.match_score}/100")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


async def _resolved(
    session: AsyncSession, user_id: uuid.UUID, query: str
) -> tuple[Application | None, str | None]:
    """Shared resolve step. Returns the application, or a string to report back."""
    resolution = await resolve_application(session, user_id, query)
    if resolution.best is None:
        return None, resolution.describe()
    return resolution.best.application, None


async def _details(session: AsyncSession, user_id: uuid.UUID, query: str) -> str:
    application, problem = await _resolved(session, user_id, query)
    if application is None:
        return problem or "No such application."

    job = application.job
    lines = [
        f"{job.title} at {job.company.name}",
        f"status: {application.current_status}",
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

    analysis = (
        await session.execute(
            select(MatchAnalysis).where(
                MatchAnalysis.job_id == job.id, MatchAnalysis.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if analysis:
        lines.append(f"match score: {analysis.overall_score}/100")
        if analysis.missing_skills:
            lines.append(f"missing from their resume: {', '.join(analysis.missing_skills)}")
        if analysis.gaps:
            lines.append("gaps: " + "; ".join(analysis.gaps))

    return "\n".join(lines)


async def _timeline(session: AsyncSession, user_id: uuid.UUID, query: str) -> str:
    application, problem = await _resolved(session, user_id, query)
    if application is None:
        return problem or "No such application."

    full = await get_application(session, application.id, user_id)
    lines = [f"Timeline for {full.job.title} at {full.job.company.name}:"]
    for event in full.events:
        stamp = event.occurred_at.date().isoformat()
        note = f" — {event.note}" if event.note else ""
        lines.append(f"  {stamp}: {event.event_type}{note}")
    return "\n".join(lines)


# A pasted posting can run to tens of thousands of characters. Truncated so one
# request cannot swallow the 8000 token-per-minute budget on its own; the cut is
# announced rather than silent, so the model does not summarise a fragment as
# though it were the whole thing.
MAX_DESCRIPTION_CHARS = 6000


async def _description(session: AsyncSession, user_id: uuid.UUID, query: str) -> str:
    """The posting as written.

    Kept apart from get_application_details deliberately. That tool returns a
    structured summary, and merging the two would put a full posting into every
    detail lookup — expensive, and usually not what was asked for.
    """
    application, problem = await _resolved(session, user_id, query)
    if application is None:
        return problem or "No such application."

    job = application.job
    if not job.description:
        return (
            f"No description was stored for {job.title} at {job.company.name}. "
            "It was probably entered by hand rather than pasted."
        )

    text = job.description
    if len(text) > MAX_DESCRIPTION_CHARS:
        text = text[:MAX_DESCRIPTION_CHARS] + "\n\n[truncated — this is the first part only]"

    return f"Job description for {job.title} at {job.company.name}:\n\n{text}"


async def _needing_attention(session: AsyncSession, user_id: uuid.UUID) -> str:
    stale = await find_stale_applications(session, user_id)
    if not stale:
        return "Nothing has gone quiet — no follow-up rules have fired."
    return "\n".join(f"{item.reason} (rule: {item.rule.days_threshold} days)" for item in stale)


async def _propose(
    session: AsyncSession, user_id: uuid.UUID, arguments: dict[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Record an intention. Writes nothing.

    The model supplies the user's own phrasing; the tracker resolves it. If that
    is ambiguous the proposal is refused and the candidates are handed back, so
    the model asks rather than picking one.
    """
    query = arguments.get("query", "")
    event_type = arguments.get("event_type")
    if not query or not event_type:
        return "Both an application and an event type are needed to propose a change.", None

    resolution = await resolve_application(session, user_id, query)
    if resolution.best is None:
        return resolution.describe(), None

    candidate = resolution.best
    proposal = {
        "kind": "append_event",
        "event_type": event_type,
        "note": arguments.get("note"),
        "application_id": str(candidate.application.id),
        "application_label": candidate.label,
        "confidence": candidate.score,
        "matched_on": candidate.matched_on,
    }
    return (
        f"Prepared, pending the user's confirmation: log '{event_type}' on "
        f"{candidate.label}. Tell them what you are about to do.",
        proposal,
    )
