"""Reference data and job records.

Note there is no "list all jobs" endpoint. Jobs are shared rows (see migration
0002's tenancy note), so listing them would expose the union of every user's
tracked postings. Jobs are reachable only through an application you own, or by
id once you have one.
"""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import NotFoundError
from app.models.application import Application
from app.models.company import Company
from app.models.job import Job
from app.models.skill import Skill
from app.schemas.job import CompanyRead, JobRead, JobUpdate, RequirementIn, SkillRead
from app.services.applications import content_hash, embed_job, set_requirements, set_skills

router = APIRouter(tags=["catalog"])


@router.get("/skills", response_model=list[SkillRead], summary="The canonical skill taxonomy")
async def list_skills(
    session: DbSession,
    _: CurrentUser,
    search: Annotated[str | None, Query(max_length=80)] = None,
    category: Annotated[str | None, Query(max_length=60)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[SkillRead]:
    stmt = select(Skill).order_by(Skill.name).limit(limit)
    if category:
        stmt = stmt.where(Skill.category == category)
    if search:
        pattern = f"%{search.strip().lower()}%"
        # Match the canonical name or any alias, so typing "reactjs" finds
        # React. Flattening the array to a string is the cheapest way to ILIKE
        # across it; the taxonomy is small enough that the scan cost is moot.
        stmt = stmt.where(
            or_(
                Skill.name.ilike(pattern),
                func.array_to_string(Skill.aliases, " ").ilike(pattern),
            )
        )
    rows = (await session.execute(stmt)).scalars().all()
    return [SkillRead.model_validate(s) for s in rows]


@router.get("/companies", response_model=list[CompanyRead], summary="Company lookup")
async def list_companies(
    session: DbSession,
    _: CurrentUser,
    search: Annotated[str | None, Query(max_length=200)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[CompanyRead]:
    stmt = select(Company).order_by(Company.name).limit(limit)
    if search:
        stmt = stmt.where(Company.name.ilike(f"%{search.strip()}%"))
    rows = (await session.execute(stmt)).scalars().all()
    return [CompanyRead.model_validate(c) for c in rows]


@router.get("/jobs/{job_id}", response_model=JobRead, summary="One job you track")
async def read_job(job_id: UUID, user: CurrentUser, session: DbSession) -> JobRead:
    job = await _job_you_track(session, job_id, user.id)
    return JobRead.model_validate(job)


@router.patch("/jobs/{job_id}", response_model=JobRead, summary="Correct a job's details")
async def update_job(
    job_id: UUID, payload: JobUpdate, user: CurrentUser, session: DbSession
) -> JobRead:
    """Fix what extraction got wrong, or fill in what it never had.

    Every field is optional and only the ones sent are applied, so a null is an
    instruction to clear rather than an accident of serialisation.
    """
    job = await _job_you_track(session, job_id, user.id)

    changes = payload.model_dump(exclude_unset=True)
    # Relationships, not columns — setattr would assign a list of dicts to a
    # mapped collection and fail somewhere much less obvious than here.
    requirements = changes.pop("requirements", None)
    skill_slugs = changes.pop("skill_slugs", None)

    for field, value in changes.items():
        setattr(job, field, value.value if hasattr(value, "value") else value)

    # The hash is derived from the description and is what exact-duplicate
    # detection compares against. Editing one without the other means a
    # re-paste of the original posting stops matching the row it created.
    if "description" in changes:
        job.content_hash = content_hash(job.description)

    if requirements is not None:
        await set_requirements(session, job, [RequirementIn(**r) for r in requirements])
    if skill_slugs is not None:
        await set_skills(session, job, skill_slugs)
    await session.flush()

    # The stored vector is built from title, seniority, location and the
    # requirements. Editing any of them without re-embedding leaves semantic
    # search and duplicate detection answering from the old text — quietly, and
    # for as long as nobody re-saves the job.
    if changes.keys() & _EMBEDDED_FIELDS or requirements is not None:
        await embed_job(session, job)

    await session.refresh(job)
    return JobRead.model_validate(job)


_EMBEDDED_FIELDS = {"title", "seniority", "location", "responsibilities"}


async def _job_you_track(session: AsyncSession, job_id: UUID, user_id: UUID) -> Job:
    """Reachability check standing in for RLS.

    ``jobs`` has no row policy — it is shared reference data — so access is
    gated here instead: you may see a job if you have an application against it.
    Without this, any job id would be readable by any user.
    """
    job = (
        (
            await session.execute(
                select(Job)
                .join(Application, Application.job_id == Job.id)
                .where(Job.id == job_id, Application.user_id == user_id)
            )
        )
        .unique()
        .scalar_one_or_none()
    )

    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    return job
