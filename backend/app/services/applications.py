"""Application creation and querying."""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import Select, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, InvalidOperationError, NotFoundError
from app.domain.enums import TERMINAL_STATUSES, ApplicationStatus, EventSource, EventType
from app.models.application import Application
from app.models.company import Company
from app.models.job import Job, JobRequirement, JobSkill
from app.models.skill import Skill
from app.schemas.job import JobCreate
from app.services.companies import resolve_company
from app.services.events import append_event, reload_application


def content_hash(text: str | None) -> str | None:
    """Stable hash of a description for exact-duplicate detection.

    Whitespace-normalised first: the same posting copied from two sites differs
    only in line wrapping far more often than it differs in substance.
    """
    if not text:
        return None
    normalized = " ".join(text.split()).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


async def create_job(session: AsyncSession, payload: JobCreate, user_id: uuid.UUID) -> Job:
    """Create a job posting, resolving its company and skills."""
    company: Company = await resolve_company(
        session, payload.company_name, domain=payload.company_domain
    )

    job = Job(
        company_id=company.id,
        created_by_user_id=user_id,
        title=payload.title.strip(),
        seniority=payload.seniority.value if payload.seniority else None,
        employment_type=payload.employment_type.value if payload.employment_type else None,
        work_mode=payload.work_mode.value if payload.work_mode else None,
        location=payload.location,
        url=str(payload.url) if payload.url else None,
        source_platform=payload.source_platform,
        posted_at=payload.posted_at,
        description=payload.description,
        responsibilities=payload.responsibilities,
        salary_min=payload.salary_min,
        salary_max=payload.salary_max,
        salary_currency=payload.salary_currency.upper() if payload.salary_currency else None,
        salary_period=payload.salary_period.value if payload.salary_period else None,
        years_experience_min=payload.years_experience_min,
        years_experience_max=payload.years_experience_max,
        content_hash=content_hash(payload.description),
    )
    session.add(job)
    await session.flush()

    for requirement in payload.requirements:
        session.add(
            JobRequirement(
                job_id=job.id,
                text=requirement.text,
                kind=requirement.kind.value,
                category=requirement.category,
            )
        )

    if payload.skill_slugs:
        slugs = list(dict.fromkeys(payload.skill_slugs))  # de-dupe, keep order
        found = (await session.execute(select(Skill).where(Skill.slug.in_(slugs)))).scalars().all()

        missing = sorted(set(slugs) - {s.slug for s in found})
        if missing:
            # Fail loudly rather than silently dropping. A typo'd slug that
            # vanishes without complaint produces a job with quietly incomplete
            # skills, which then scores wrongly in Phase 3.
            raise InvalidOperationError(f"Unknown skill slugs: {', '.join(missing)}")

        for skill in found:
            session.add(JobSkill(job_id=job.id, skill_id=skill.id, is_required=True))

    await session.flush()
    await session.refresh(job)
    return job


async def create_application(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    job_id: uuid.UUID | None,
    job_payload: JobCreate | None,
    priority: str,
    notes: str | None,
    initial_event: EventType,
    occurred_at: datetime | None,
) -> Application:
    if (job_id is None) == (job_payload is None):
        raise InvalidOperationError("Provide exactly one of job_id or job")

    if job_payload is not None:
        job = await create_job(session, job_payload, user_id)
    else:
        existing = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
        if existing is None:
            raise NotFoundError(f"Job {job_id} not found")
        job = existing

    now = datetime.now(UTC)
    application = Application(
        user_id=user_id,
        job_id=job.id,
        priority=priority,
        notes=notes,
        current_status=ApplicationStatus.SAVED.value,
        current_status_at=now,
        last_activity_at=now,
    )
    session.add(application)

    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        # The (user_id, job_id) unique constraint. Re-applying is a new event on
        # the existing timeline, not a second application.
        raise ConflictError("You are already tracking this job") from exc

    await append_event(
        session,
        application=application,
        event_type=initial_event,
        occurred_at=occurred_at,
        source=EventSource.MANUAL,
    )
    return await reload_application(session, application.id, user_id)


def _apply_filters(
    stmt: Select,
    *,
    status: list[str] | None,
    search: str | None,
    priority: str | None,
    work_mode: str | None,
    active_only: bool,
) -> Select:
    if status:
        stmt = stmt.where(Application.current_status.in_(status))
    if active_only:
        stmt = stmt.where(Application.current_status.notin_([s.value for s in TERMINAL_STATUSES]))
    if priority:
        stmt = stmt.where(Application.priority == priority)
    if work_mode:
        stmt = stmt.where(Job.work_mode == work_mode)
    if search:
        # ILIKE over title and company name. pg_trgm indexes make this fast
        # enough well past the volume one person's job search will ever reach;
        # full-text search would be premature.
        pattern = f"%{search.strip()}%"
        stmt = stmt.where(or_(Job.title.ilike(pattern), Company.name.ilike(pattern)))
    return stmt


SORTABLE = {
    "created_at": Application.created_at,
    "last_activity_at": Application.last_activity_at,
    "applied_at": Application.applied_at,
    "current_status_at": Application.current_status_at,
    "match_score": Application.match_score,
    "job_score": Application.job_score,
    "title": Job.title,
    "company": Company.name,
}


async def list_applications(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    status: list[str] | None = None,
    search: str | None = None,
    priority: str | None = None,
    work_mode: str | None = None,
    active_only: bool = False,
    sort: str = "last_activity_at",
    order: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Application], int]:
    if sort not in SORTABLE:
        raise InvalidOperationError(f"Cannot sort by '{sort}'. Try one of: {', '.join(SORTABLE)}")

    base = (
        select(Application)
        .join(Job, Job.id == Application.job_id)
        .join(Company, Company.id == Job.company_id)
        .where(Application.user_id == user_id)
    )
    base = _apply_filters(
        base,
        status=status,
        search=search,
        priority=priority,
        work_mode=work_mode,
        active_only=active_only,
    )

    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await session.execute(count_stmt)).scalar_one()

    column = SORTABLE[sort]
    # NULLS LAST both ways: an unscored application sorting above a scored one
    # is never what the user meant.
    ordering = column.desc().nullslast() if order == "desc" else column.asc().nullslast()

    rows = (
        (await session.execute(base.order_by(ordering, Application.id).limit(limit).offset(offset)))
        .unique()
        .scalars()
        .all()
    )

    return list(rows), total


async def get_stats(session: AsyncSession, user_id: uuid.UUID, stale_days: int = 7) -> dict:
    """Dashboard counters.

    ``needs_attention`` is a preview of the Phase 4 rule engine using a single
    flat threshold. The real version reads per-status rules from
    ``follow_up_rules``; this is the same query shape so swapping it in later is
    a change of predicate, not of plumbing.
    """
    rows = (
        await session.execute(
            select(Application.current_status, func.count())
            .where(Application.user_id == user_id)
            .group_by(Application.current_status)
        )
    ).all()

    by_status = {status: count for status, count in rows}
    total = sum(by_status.values())
    terminal = {s.value for s in TERMINAL_STATUSES}
    active = sum(count for status, count in by_status.items() if status not in terminal)

    cutoff = datetime.now(UTC) - timedelta(days=stale_days)
    needs_attention = (
        await session.execute(
            select(func.count())
            .select_from(Application)
            .where(
                Application.user_id == user_id,
                Application.current_status.notin_(terminal),
                Application.last_activity_at < cutoff,
            )
        )
    ).scalar_one()

    return {
        "total": total,
        "by_status": [{"status": s, "count": c} for s, c in sorted(by_status.items())],
        "active": active,
        "needs_attention": needs_attention,
    }
