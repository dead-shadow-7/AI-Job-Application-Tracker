"""The analytical and drafting tools.

Two kinds of thing live here, and they share a principle worth naming.

The *analytical* tools aggregate across the whole tracker — how the search is
going, which skills keep coming up, how two roles compare. They are computed in
SQL and handed to the model as facts. The model is good at explaining a number
and bad at deriving one, so it is never asked to derive one.

The *drafting* tools — follow-ups, interview briefs — return **context, not
prose**. They gather everything a good draft needs and hand it back with an
instruction. Generating the text inside the tool would mean a second model call
nested in the first: slower, twice the tokens, and the result would arrive with
none of the conversation's context. The model already in the loop writes it.

Nothing here writes.
"""

import uuid
from datetime import UTC, datetime

from sqlalchemy import Numeric, case, cast, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.resolving import resolve_application_only
from app.domain.enums import TERMINAL_STATUSES
from app.models.application import Application, ApplicationEvent, InterviewStage
from app.models.followup import FollowUpRule
from app.models.job import Job, JobRequirement, JobSkill
from app.models.resume import MatchAnalysis
from app.models.skill import Skill
from app.services.analytics import compute_analytics
from app.services.resumes import get_default_resume
from app.services.skills import extract_skills_from_text

# Enough to reason over, short enough that one call does not swallow the
# per-minute token budget. Lists here are ranked, so a cut takes the tail.
TOP_SKILLS = 18
MAX_COMPARE = 4


async def analytics_summary(session: AsyncSession, user_id: uuid.UUID) -> str:
    """How the search is going, as numbers rather than impressions."""
    data = await compute_analytics(session, user_id)
    if data.total == 0:
        return "They are not tracking anything yet, so there is nothing to analyse."

    lines = [f"Tracked: {data.total}. Actually sent: {data.submitted}."]

    if data.submitted:
        rate = "unknown" if data.response_rate is None else f"{round(data.response_rate * 100)}%"
        lines.append(f"Replies: {data.responses} of {data.submitted} sent ({rate}).")
        if data.median_days_to_response is not None:
            lines.append(f"Median wait for a reply: {data.median_days_to_response} days.")
        else:
            lines.append("Nobody has replied yet, so there is no wait time to report.")

    reached = [f"{s.status} {s.count}" for s in data.funnel if s.count]
    lines.append("Current standing: " + (", ".join(reached) or "none"))

    if data.by_platform:
        lines.append(
            "By platform: "
            + "; ".join(
                f"{p.platform} {p.responses}/{p.applications}"
                + (f" ({round(p.response_rate * 100)}%)" if p.response_rate is not None else "")
                for p in data.by_platform
            )
        )

    if data.sample_is_small:
        # Said to the model, not just to the UI. Otherwise it reports "your
        # response rate is 33%" off three applications as though it meant
        # something, which is exactly the failure the analytics module is
        # written to avoid.
        lines.append(
            "CAVEAT: fewer than 10 sent, so these proportions are not meaningful yet. "
            "Say so if you quote them — one outcome moves them a long way."
        )
    return "\n".join(lines)


async def compare(session: AsyncSession, user_id: uuid.UUID, queries: list[str]) -> str:
    """Two to four roles side by side.

    A tool rather than repeated detail lookups: each lookup is a round trip and
    a full detail block, and the comparison is usually about four fields.
    """
    wanted = [q for q in queries if q and q.strip()][:MAX_COMPARE]
    if len(wanted) < 2:
        return "Comparing needs at least two applications. Which ones?"

    blocks: list[str] = []
    for query in wanted:
        application, problem = await resolve_application_only(session, user_id, query)
        if application is None:
            blocks.append(f"{query}: {problem}")
            continue

        job = application.job
        parts = [
            f"{job.title} at {job.company.name}",
            f"  status: {application.current_status}",
        ]
        if job.location:
            parts.append(f"  location: {job.location} {job.work_mode or ''}".rstrip())
        if job.salary_min or job.salary_max:
            parts.append(
                f"  salary: {job.salary_min or '?'}-{job.salary_max or '?'} "
                f"{job.salary_currency or ''} per {job.salary_period or 'year'}"
            )
        if application.match_score is not None:
            parts.append(f"  match: {application.match_score}/100")

        required = (
            (
                await session.execute(
                    select(Skill.name)
                    .join(JobSkill, JobSkill.skill_id == Skill.id)
                    .where(JobSkill.job_id == job.id, JobSkill.is_required.is_(True))
                )
            )
            .scalars()
            .all()
        )
        if required:
            parts.append(f"  needs: {', '.join(required)}")

        analysis = (
            await session.execute(
                select(MatchAnalysis).where(
                    MatchAnalysis.job_id == job.id, MatchAnalysis.user_id == user_id
                )
            )
        ).scalar_one_or_none()
        if analysis and analysis.missing_skills:
            parts.append(f"  missing from resume: {', '.join(analysis.missing_skills)}")

        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


async def resume_profile(session: AsyncSession, user_id: uuid.UUID) -> str:
    """What their resume actually says.

    Without this the model discusses fit from the job's side only, and fills the
    other half from assumption — which reads as confident and is invented.
    """
    resume = await get_default_resume(session, user_id)
    if resume is None:
        return (
            "No resume has been uploaded, so nothing is known about their background. "
            "Do not guess at it — say it is missing and that uploading one enables match scoring."
        )

    lines = [f"Resume: {resume.label}"]
    if resume.years_experience is not None:
        lines.append(f"Experience: {resume.years_experience} years")

    skills = await extract_skills_from_text(session, resume.parsed_text)
    if skills:
        lines.append(f"Skills found on it: {', '.join(s.name for s in skills)}")
    else:
        lines.append("No known skills were recognised on it.")

    excerpt = " ".join(resume.parsed_text.split())[:600]
    lines.append(f"Opening text: {excerpt}")
    return "\n".join(lines)


async def by_skill(session: AsyncSession, user_id: uuid.UUID, skill_query: str) -> str:
    """Which tracked roles want a given skill."""
    wanted = skill_query.strip().lower()
    if not wanted:
        return "Which skill?"

    rows = (
        await session.execute(
            select(Job.title, Skill.name, JobSkill.is_required, Application.current_status)
            .join(JobSkill, JobSkill.job_id == Job.id)
            .join(Skill, Skill.id == JobSkill.skill_id)
            .join(Application, Application.job_id == Job.id)
            .where(Application.user_id == user_id, func.lower(Skill.name).contains(wanted))
            .order_by(desc(JobSkill.is_required))
            .limit(30)
        )
    ).all()

    if not rows:
        return f"None of their tracked roles list a skill matching {skill_query!r}."

    return "\n".join(
        f"{title} — {name} ({'required' if required else 'preferred'}), status {status}"
        for title, name, required, status in rows
    )


async def skill_demand(session: AsyncSession, user_id: uuid.UUID) -> str:
    """Which skills keep coming up, and which of them they do not have.

    The honest version of "what should I learn next": ranked by how often the
    jobs *they chose to track* ask for it, not by an industry trend the model
    half-remembers.
    """
    rows = (
        await session.execute(
            select(
                Skill.name,
                func.count().label("demand"),
                cast(func.sum(case((JobSkill.is_required, 1), else_=0)), Numeric).label("required"),
            )
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .join(Job, Job.id == JobSkill.job_id)
            .join(Application, Application.job_id == Job.id)
            .where(Application.user_id == user_id)
            .group_by(Skill.name)
            .order_by(desc("demand"), Skill.name)
            .limit(TOP_SKILLS)
        )
    ).all()

    if not rows:
        return (
            "None of their tracked jobs have skills attached yet. Skills are extracted when a "
            "job description is pasted; ones added by hand have none."
        )

    resume = await get_default_resume(session, user_id)
    have: set[str] = set()
    if resume is not None:
        have = {s.name.lower() for s in await extract_skills_from_text(session, resume.parsed_text)}

    lines = ["Skill demand across their tracked jobs (most asked first):"]
    for name, demand, required in rows:
        held = "on their resume" if name.lower() in have else "NOT on their resume"
        lines.append(f"  {name}: wanted by {demand} ({int(required or 0)} as required) — {held}")

    if resume is None:
        lines.append("No resume uploaded, so nothing is marked as held or missing.")
    return "\n".join(lines)


async def follow_up_rules(session: AsyncSession, user_id: uuid.UUID) -> str:
    """The thresholds, so "why is this flagged" has an answer."""
    rules = (
        (
            await session.execute(
                select(FollowUpRule)
                .where(FollowUpRule.user_id == user_id)
                .order_by(FollowUpRule.days_threshold)
            )
        )
        .scalars()
        .all()
    )
    if not rules:
        return "No follow-up rules are configured."

    return "\n".join(
        f"{r.applies_to_status}: after {r.days_threshold} days of silence → {r.action}"
        + ("" if r.enabled else " (disabled)")
        for r in rules
    )


async def upcoming_interviews(session: AsyncSession, user_id: uuid.UUID) -> str:
    """Rounds still pending, soonest first."""
    rows = (
        await session.execute(
            select(InterviewStage, Application)
            .join(Application, Application.id == InterviewStage.application_id)
            .where(InterviewStage.user_id == user_id, InterviewStage.outcome == "pending")
            .order_by(InterviewStage.scheduled_at.nulls_last(), InterviewStage.round_number)
            .limit(25)
        )
    ).all()

    if not rows:
        return "No interview rounds are pending."

    now = datetime.now(UTC)
    lines = []
    for stage, application in rows:
        job = application.job
        when = "no date set"
        if stage.scheduled_at is not None:
            days = (stage.scheduled_at - now).days
            when = f"in {days} days" if days >= 0 else f"{abs(days)} days ago, still pending"
        who = f", with {stage.interviewer}" if stage.interviewer else ""
        lines.append(
            f"{job.title} at {job.company.name} — round {stage.round_number} "
            f"{stage.stage_type}, {when}{who}"
        )
    return "\n".join(lines)


async def follow_up_context(session: AsyncSession, user_id: uuid.UUID, query: str) -> str:
    """Everything a follow-up needs, plus the instruction to write it.

    Deliberately not generated here. A nested model call would cost a second
    round trip and would arrive without the conversation — the user may have
    just said "keep it short" or "they mentioned a Q3 start", and only the model
    already in the loop knows that.
    """
    application, problem = await resolve_application_only(session, user_id, query)
    if application is None:
        return problem or "No such application."

    job = application.job
    idle = (datetime.now(UTC) - application.last_activity_at).days

    recent = (
        (
            await session.execute(
                select(ApplicationEvent)
                .where(ApplicationEvent.application_id == application.id)
                .order_by(desc(ApplicationEvent.occurred_at))
                .limit(5)
            )
        )
        .scalars()
        .all()
    )

    lines = [
        f"Context for a follow-up on {job.title} at {job.company.name}:",
        f"  status: {application.current_status}, silent for {idle} days",
    ]
    if application.current_status in TERMINAL_STATUSES:
        lines.append(
            "  WARNING: this application is closed. Say so — a follow-up on a "
            "rejected role is not what they meant to send."
        )

    for event in recent:
        note = f" ({event.note})" if event.note else ""
        lines.append(f"  {event.occurred_at.date().isoformat()}: {event.event_type}{note}")

    already = [e for e in recent if e.event_type == "follow_up_sent"]
    if already:
        lines.append(
            f"  They ALREADY sent a follow-up on {already[0].occurred_at.date().isoformat()}. "
            "A second one should acknowledge the first rather than repeat it."
        )

    interviewer = (
        await session.execute(
            select(InterviewStage.interviewer)
            .where(
                InterviewStage.application_id == application.id,
                InterviewStage.interviewer.is_not(None),
            )
            .order_by(desc(InterviewStage.round_number))
            .limit(1)
        )
    ).scalar_one_or_none()
    if interviewer:
        lines.append(f"  last known contact: {interviewer}")

    lines.append(
        "\nNow write the follow-up: a subject line and a short body, three or four "
        "sentences. Reference the actual last event above and nothing else — do not "
        "invent a conversation, a name, a date or an enthusiasm they did not express. "
        "Say it is a draft they should read before sending."
    )
    return "\n".join(lines)


async def interview_brief(session: AsyncSession, user_id: uuid.UUID, query: str) -> str:
    """Requirements, gaps and history in one call, plus what to do with them."""
    application, problem = await resolve_application_only(session, user_id, query)
    if application is None:
        return problem or "No such application."

    job = application.job
    lines = [f"Interview prep for {job.title} at {job.company.name}:", ""]
    if job.seniority:
        lines.append(f"seniority: {job.seniority}")
    if job.years_experience_min is not None:
        lines.append(f"experience wanted: {job.years_experience_min}+ years")

    skills = (
        await session.execute(
            select(Skill.name, JobSkill.is_required)
            .join(JobSkill, JobSkill.skill_id == Skill.id)
            .where(JobSkill.job_id == job.id)
        )
    ).all()
    required = [n for n, req in skills if req]
    preferred = [n for n, req in skills if not req]
    if required:
        lines.append(f"required: {', '.join(required)}")
    if preferred:
        lines.append(f"preferred: {', '.join(preferred)}")

    requirements = (
        (
            await session.execute(
                select(JobRequirement.text).where(JobRequirement.job_id == job.id).limit(15)
            )
        )
        .scalars()
        .all()
    )
    for text in requirements:
        lines.append(f"  requirement: {text}")

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
            lines.append(f"NOT on their resume: {', '.join(analysis.missing_skills)}")
        if analysis.strengths:
            lines.append("strengths: " + "; ".join(analysis.strengths))
        if analysis.gaps:
            lines.append("gaps: " + "; ".join(analysis.gaps))
    else:
        lines.append("Not scored against their resume yet, so gaps are unknown — do not guess.")

    stage = (
        await session.execute(
            select(InterviewStage)
            .where(
                InterviewStage.application_id == application.id, InterviewStage.outcome == "pending"
            )
            .order_by(InterviewStage.round_number)
            .limit(1)
        )
    ).scalar_one_or_none()
    if stage is not None:
        lines.append(f"next round: {stage.stage_type} (round {stage.round_number})")

    lines.append(
        "\nNow write the brief: the likely question areas drawn from the requirements above, "
        "and where they are weakest. Ground every point in something listed here. Be blunt "
        "about the gaps — reassurance before an interview is worth nothing."
    )
    return "\n".join(lines)
