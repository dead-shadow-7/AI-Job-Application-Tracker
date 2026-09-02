"""Resume-to-job scoring endpoints."""

import logging
from uuid import UUID

from fastapi import APIRouter, status
from sqlalchemy import select

from app.agent.llm_client import LLMError, llm_client
from app.agent.prompts.rubric import (
    RUBRIC_PROMPT_VERSION,
    RUBRIC_SYSTEM_PROMPT,
    build_rubric_prompt,
)
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError, NotFoundError
from app.models.job import JobRequirement
from app.models.resume import MatchAnalysis
from app.schemas.matching import MatchRead, RubricJudgment
from app.services.events import get_application
from app.services.matching import (
    combine_with_rubric,
    compute_deterministic_match,
    retrieve_evidence,
)
from app.services.resumes import get_default_resume, get_resume

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/applications", tags=["matching"])

# Requirements sent to the rubric. Capped because each one costs a vector
# search, and past a dozen the marginal requirement is boilerplate ("excellent
# communication skills") that the rubric cannot meaningfully judge anyway.
MAX_REQUIREMENTS_JUDGED = 12


@router.get(
    "/{application_id}/match",
    response_model=MatchRead | None,
    summary="The cached match score, if one has been computed",
)
async def read_match(
    application_id: UUID, user: CurrentUser, session: DbSession
) -> MatchRead | None:
    application = await get_application(session, application_id, user.id)
    analysis = (
        await session.execute(
            select(MatchAnalysis).where(
                MatchAnalysis.job_id == application.job_id,
                MatchAnalysis.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    return MatchRead.model_validate(analysis) if analysis else None


@router.post(
    "/{application_id}/match",
    response_model=MatchRead,
    status_code=status.HTTP_201_CREATED,
    summary="Score this job against your resume",
)
async def compute_match(
    application_id: UUID,
    user: CurrentUser,
    session: DbSession,
    resume_id: UUID | None = None,
) -> MatchRead:
    """Score, cache, and return.

    The deterministic components are computed first and stand on their own. The
    LLM rubric is folded in afterwards at 15%, and if the model is unavailable
    or rate-limited the score is still produced from the other 85% rather than
    the request failing — a partial score you can read beats an error you
    cannot act on.
    """
    application = await get_application(session, application_id, user.id)

    resume = (
        await get_resume(session, resume_id, user.id)
        if resume_id
        else await get_default_resume(session, user.id)
    )
    if resume is None:
        raise InvalidOperationError(
            "Upload a resume first — there is nothing to score this job against."
        )

    job = application.job
    deterministic = await compute_deterministic_match(session, resume, job)

    requirements = list(
        (
            await session.execute(
                select(JobRequirement)
                .where(JobRequirement.job_id == job.id)
                .order_by(JobRequirement.kind)
                .limit(MAX_REQUIREMENTS_JUDGED)
            )
        )
        .scalars()
        .all()
    )

    # Retrieval is the point of pgvector here: for each requirement, pull the
    # resume passages that actually bear on it, so the model judges against the
    # candidate's real words rather than a summary.
    evidence: list[tuple[str, list[str]]] = []
    for requirement in requirements:
        chunks = await retrieve_evidence(session, resume.id, requirement.text)
        evidence.append((requirement.text, [c.content for c in chunks]))

    judgment: RubricJudgment | None = None
    if llm_client.is_configured and evidence:
        try:
            result = await llm_client.extract(
                schema=RubricJudgment,
                system=RUBRIC_SYSTEM_PROMPT,
                user=build_rubric_prompt(
                    title=job.title,
                    company=job.company.name,
                    requirements_with_evidence=evidence,
                    matched_skills=deterministic.matched_skills,
                    missing_skills=deterministic.missing_skills,
                ),
            )
            judgment = result.data
        except LLMError as exc:
            # Degrade rather than fail: 85% of the score needs no model.
            logger.warning("Rubric unavailable, scoring deterministically only: %s", exc)

    if judgment is not None:
        # Clamped once and used for both, so the breakdown adds up to the total.
        # The schema says 0.0-1.0 but nothing enforces it, and storing the raw
        # value beside a total computed from the clamped one meant a model
        # answering 1.4 produced a card whose parts did not make its whole.
        rubric_score = max(0.0, min(1.0, judgment.score))
        overall = combine_with_rubric(deterministic, rubric_score)
        subscores = {**deterministic.subscores, "rubric": rubric_score}
    else:
        overall = deterministic.overall_score
        subscores = deterministic.subscores

    existing = (
        await session.execute(
            select(MatchAnalysis).where(
                MatchAnalysis.resume_id == resume.id, MatchAnalysis.job_id == job.id
            )
        )
    ).scalar_one_or_none()

    analysis = existing or MatchAnalysis(user_id=user.id, resume_id=resume.id, job_id=job.id)
    analysis.overall_score = overall
    analysis.subscores = subscores
    analysis.matched_skills = deterministic.matched_skills
    analysis.missing_skills = deterministic.missing_skills
    analysis.strengths = judgment.strengths if judgment else []
    analysis.gaps = judgment.gaps if judgment else []
    analysis.narrative = judgment.narrative if judgment else None
    analysis.model = settings.extraction_model if judgment else None
    analysis.prompt_version = RUBRIC_PROMPT_VERSION if judgment else None

    if existing is None:
        session.add(analysis)

    # Denormalized onto the application so the dashboard can sort by score
    # without joining every row to its analysis.
    application.match_score = overall

    await session.flush()
    await session.refresh(analysis)
    return MatchRead.model_validate(analysis)


@router.delete(
    "/{application_id}/match",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Discard the cached score so it is recomputed",
)
async def clear_match(application_id: UUID, user: CurrentUser, session: DbSession) -> None:
    application = await get_application(session, application_id, user.id)
    analysis = (
        await session.execute(
            select(MatchAnalysis).where(
                MatchAnalysis.job_id == application.job_id, MatchAnalysis.user_id == user.id
            )
        )
    ).scalar_one_or_none()
    if analysis is None:
        raise NotFoundError("No cached score for this application")
    await session.delete(analysis)
    application.match_score = None
