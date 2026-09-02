"""Resume storage, chunking, and embedding."""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import NotFoundError
from app.models.application import Application
from app.models.resume import MatchAnalysis, Resume, ResumeChunk
from app.services import embeddings
from app.services.resume_parser import (
    Chunk,
    chunk_resume,
    estimate_years_experience,
    parse_positions,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ParsedResume:
    chunks: list[Chunk]
    years_experience: float | None
    years_source: str | None
    positions: list[dict[str, Any]]


def _parse(text: str) -> ParsedResume:
    """Every synchronous pass over a resume, in one threadpool hop."""
    positions = parse_positions(text)
    estimate = estimate_years_experience(text, positions)
    return ParsedResume(
        chunks=chunk_resume(text),
        years_experience=estimate.years,
        years_source=estimate.source,
        positions=[p.as_dict() for p in positions],
    )


async def create_resume(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    label: str,
    filename: str | None,
    text: str,
    make_default: bool = True,
) -> Resume:
    """Store a resume and embed its chunks.

    Embedding happens inline rather than in a background job: it takes a couple
    of hundred milliseconds locally, and a resume that exists but is not yet
    searchable is a confusing state to have to explain in the UI.
    """
    # Every regex pass in one threadpool hop. Small — 19ms on a 200-page
    # document, a fraction of a millisecond on a real resume — but it is the
    # last CPU left inline on this path now that parsing and embedding are
    # both offloaded, and doing it here keeps that true.
    parsed = await run_in_threadpool(_parse, text)

    resume = Resume(
        user_id=user_id,
        label=label.strip(),
        filename=filename,
        parsed_text=text,
        years_experience=parsed.years_experience,
        years_experience_source=parsed.years_source,
        positions=parsed.positions,
    )
    session.add(resume)
    await session.flush()

    await _embed_chunks(session, resume, parsed.chunks)

    if make_default:
        await set_default(session, user_id=user_id, resume_id=resume.id)

    await session.flush()
    return resume


async def reparse_resume(session: AsyncSession, resume: Resume) -> Resume:
    """Re-run the parser over the stored text and replace what it produced.

    The whole reason ``parsed_text`` is kept is that chunking, date reading, and
    title extraction all improve over time, and asking someone to find and
    re-upload a file to benefit from that is a poor trade. Chunks are deleted
    and rebuilt rather than diffed: they are cheap to produce, and matching
    old rows to new ones across a changed chunking strategy is not a problem
    worth having.
    """
    parsed = await run_in_threadpool(_parse, resume.parsed_text)

    await session.execute(delete(ResumeChunk).where(ResumeChunk.resume_id == resume.id))

    resume.years_experience = parsed.years_experience
    resume.years_experience_source = parsed.years_source
    resume.positions = parsed.positions

    await _embed_chunks(session, resume, parsed.chunks)
    await discard_cached_scores(session, resume)
    await session.flush()
    return resume


async def discard_cached_scores(session: AsyncSession, resume: Resume) -> int:
    """Drop every cached score computed against this resume.

    A re-parse changes both the passages the rubric was shown and the years the
    arithmetic used, so a score cached before it no longer corresponds to
    anything. Leaving it would be worse than having none, because a stale score
    is indistinguishable from a current one.
    """
    job_ids = list(
        (
            await session.execute(
                select(MatchAnalysis.job_id).where(MatchAnalysis.resume_id == resume.id)
            )
        )
        .scalars()
        .all()
    )
    if not job_ids:
        return 0

    await session.execute(delete(MatchAnalysis).where(MatchAnalysis.resume_id == resume.id))
    # match_score is denormalized onto the application so the dashboard can sort
    # by it without a join, which means it has to be cleared alongside.
    await session.execute(
        update(Application)
        .where(Application.user_id == resume.user_id, Application.job_id.in_(job_ids))
        .values(match_score=None)
    )
    return len(job_ids)


async def _embed_chunks(session: AsyncSession, resume: Resume, chunks: list[Chunk]) -> int:
    if not chunks:
        logger.warning("Resume %s produced no chunks", resume.id)
        return 0

    vectors = await embeddings.embedding_provider.embed_documents([c.content for c in chunks])

    for chunk, vector in zip(chunks, vectors, strict=True):
        session.add(
            ResumeChunk(
                resume_id=resume.id,
                user_id=resume.user_id,
                ordinal=chunk.ordinal,
                section=chunk.section,
                content=chunk.content,
                embedding=vector,
            )
        )
    await session.flush()
    return len(chunks)


async def set_default(session: AsyncSession, *, user_id: uuid.UUID, resume_id: uuid.UUID) -> None:
    """Exactly one default per user.

    Cleared then set in the same transaction, so there is never a moment with
    two defaults — which would make "score against my resume" ambiguous.
    """
    await session.execute(update(Resume).where(Resume.user_id == user_id).values(is_default=False))
    await session.execute(
        update(Resume)
        .where(Resume.id == resume_id, Resume.user_id == user_id)
        .values(is_default=True)
    )


async def get_resume(session: AsyncSession, resume_id: uuid.UUID, user_id: uuid.UUID) -> Resume:
    resume = (
        await session.execute(
            select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id)
        )
    ).scalar_one_or_none()
    if resume is None:
        raise NotFoundError(f"Resume {resume_id} not found")
    return resume


async def get_default_resume(session: AsyncSession, user_id: uuid.UUID) -> Resume | None:
    """The default, or the most recent if none is marked."""
    return (
        await session.execute(
            select(Resume)
            .where(Resume.user_id == user_id)
            .order_by(Resume.is_default.desc(), Resume.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def count_chunks(session: AsyncSession, resume_ids: list[uuid.UUID]) -> dict[uuid.UUID, int]:
    """Chunk counts by resume, in one aggregate query.

    An aggregate rather than ``len(resume.chunks)``: each chunk carries a
    384-dimensional vector, so materialising them merely to count them would
    make listing resumes progressively slower for no benefit.
    """
    if not resume_ids:
        return {}
    rows = (
        await session.execute(
            select(ResumeChunk.resume_id, func.count())
            .where(ResumeChunk.resume_id.in_(resume_ids))
            .group_by(ResumeChunk.resume_id)
        )
    ).all()
    return {resume_id: count for resume_id, count in rows}


async def list_resumes(session: AsyncSession, user_id: uuid.UUID) -> list[Resume]:
    return list(
        (
            await session.execute(
                select(Resume)
                .where(Resume.user_id == user_id)
                .order_by(Resume.is_default.desc(), Resume.created_at.desc())
            )
        )
        .scalars()
        .all()
    )
