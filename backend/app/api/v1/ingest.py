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
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graphs.ingestion import run_ingestion
from app.agent.llm_client import llm_client
from app.api.deps import CurrentUser, DbSession
from app.core.config import settings
from app.core.exceptions import InvalidOperationError
from app.schemas.extraction import ExtractedJob
from app.schemas.ingest import DuplicateHint, IngestPreview, IngestRequest
from app.services.applications import content_hash, job_embedding_text
from app.services.job_drafts import build_job_draft
from app.services.search import find_near_duplicate
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

    job = build_job_draft(state, url=payload.url, source_platform=payload.source_platform)

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
        duplicate_of=await _find_duplicate(session, user.id, state["cleaned_text"], extracted),
    )


async def _find_duplicate(
    session: AsyncSession, user_id: UUID, cleaned_text: str, extracted: ExtractedJob
) -> DuplicateHint | None:
    """Exact hash first, then embedding similarity.

    The hash catches pasting the same posting twice. The embedding catches what
    it cannot: the same role reposted with edited wording, or found on two
    boards with different boilerplate wrapped around it — which is the case
    that actually costs you a duplicate timeline.

    The probe is built by ``job_embedding_text`` from the *extraction*, not from
    the pasted text. Stored job vectors deliberately exclude the culture-and-
    benefits padding, so comparing a raw posting against one measures the
    distance between a full document and a distilled one and finds nothing —
    which would leave the exact hash quietly doing all the work.

    Which of the two fired is reported, not hidden: the near match is a
    judgement and the review screen should say so.
    """
    probe = job_embedding_text(
        title=extracted.title,
        company_name=extracted.company_name,
        seniority=extracted.seniority,
        location=extracted.location,
        requirements=[r.text for r in extracted.requirements],
        responsibilities=extracted.responsibilities,
    )

    match = await find_near_duplicate(
        session, user_id, probe, content_digest=content_hash(cleaned_text)
    )
    if match is None:
        return None

    job = match.application.job
    return DuplicateHint(
        application_id=match.application.id,
        label=f"{job.title} at {job.company.name}",
        is_exact=match.is_exact,
    )
