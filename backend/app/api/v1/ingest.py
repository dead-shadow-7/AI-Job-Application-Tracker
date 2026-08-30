"""Paste a job description, get a structured record back.

This endpoint **writes nothing**. It returns a preview the user reviews and
corrects, which is then posted to the ordinary create endpoint. Extraction is
good but not perfect, and a wrong row silently saved is far more expensive to
discover later than an edit made now — especially for salary, which is the
field most likely to be wrong and least likely to be re-checked.
"""

import logging
from decimal import Decimal
from uuid import UUID

from fastapi import APIRouter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graphs.ingestion import run_ingestion
from app.agent.llm_client import llm_client
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError
from app.models.application import Application
from app.models.job import Job
from app.schemas.extraction import ExtractedJob
from app.schemas.ingest import IngestPreview, IngestRequest, JobDraft
from app.schemas.job import RequirementIn
from app.services.applications import content_hash
from app.services.skills import SkillResolution

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["ingest"])

# Below this, the review screen opens with fields flagged rather than merely
# available. The model reports its own confidence; validation warnings override
# it, since a confident extraction with a discarded salary still needs eyes.
REVIEW_CONFIDENCE_THRESHOLD = 0.75


@router.post(
    "/ingest",
    response_model=IngestPreview,
    summary="Extract a job posting from pasted text (does not save)",
)
async def ingest(payload: IngestRequest, user: CurrentUser, session: DbSession) -> IngestPreview:
    if not llm_client.is_configured:
        raise InvalidOperationError(
            "GROQ_API_KEY is not set, so extraction is unavailable. "
            "Add a job by hand, or set the key and restart the API."
        )

    state = await run_ingestion(
        session=session,
        raw_text=payload.raw_text,
        url=payload.url,
        source_platform=payload.source_platform,
        user_id=str(user.id),
    )

    extracted: ExtractedJob | None = state.get("extracted")
    if state.get("error") or extracted is None:
        raise InvalidOperationError(state.get("error") or "Extraction failed.")

    report = state["report"]
    skills: SkillResolution = state.get("skills") or SkillResolution()
    usage = state.get("usage") or []

    job = JobDraft(
        company_name=extracted.company_name,
        title=extracted.title,
        seniority=extracted.seniority,
        employment_type=extracted.employment_type,
        work_mode=extracted.work_mode,
        location=extracted.location,
        url=payload.url,
        source_platform=payload.source_platform,
        description=state["cleaned_text"],
        responsibilities=extracted.responsibilities,
        salary_min=_decimal(extracted.salary.min_amount),
        salary_max=_decimal(extracted.salary.max_amount),
        salary_currency=extracted.salary.currency,
        salary_period=extracted.salary.period,
        years_experience_min=extracted.years_experience_min,
        years_experience_max=extracted.years_experience_max,
        requirements=[RequirementIn(text=r.text, kind=r.kind) for r in extracted.requirements],
        skill_slugs=skills.slugs,
    )

    needs_review = (
        extracted.confidence < REVIEW_CONFIDENCE_THRESHOLD
        or bool(report.warnings)
        or bool(report.dropped_fields)
    )

    return IngestPreview(
        job=job,
        confidence=Decimal(str(round(extracted.confidence, 2))),
        needs_review=needs_review,
        warnings=report.warnings,
        dropped_fields=report.dropped_fields,
        unmatched_skills=skills.unmatched,
        model=settings.extraction_model,
        prompt_version=state["prompt_version"],
        tokens_used=sum(u.total_tokens for u in usage),
        latency_ms=sum(u.latency_ms for u in usage),
        duplicate_of=await _find_duplicate(session, user.id, state["cleaned_text"]),
    )


def _decimal(value: float | None) -> Decimal | None:
    return None if value is None else Decimal(str(value))


async def _find_duplicate(session: AsyncSession, user_id: UUID, cleaned_text: str) -> UUID | None:
    """Exact-duplicate check on the normalised description.

    Catches the common case of pasting the same posting twice, or the same job
    found on two boards. Near-duplicate detection over embeddings arrives in
    Phase 3; this costs one indexed lookup and no tokens.
    """
    digest = content_hash(cleaned_text)
    if digest is None:
        return None

    return (
        await session.execute(
            select(Application.id)
            .join(Job, Job.id == Application.job_id)
            .where(Application.user_id == user_id, Job.content_hash == digest)
            .limit(1)
        )
    ).scalar_one_or_none()
