"""Resume storage, chunking, and embedding."""

import logging
import uuid

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.core.exceptions import NotFoundError
from app.models.resume import Resume, ResumeChunk
from app.services import embeddings
from app.services.resume_parser import Chunk, chunk_resume, guess_years_experience

logger = logging.getLogger(__name__)


def _parse(text: str) -> tuple[list[Chunk], float | None]:
    """The two synchronous regex passes over a resume, for one threadpool hop."""
    return chunk_resume(text), guess_years_experience(text)


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
    # Both regex passes in one threadpool hop. Small — 19ms on a 200-page
    # document, a fraction of a millisecond on a real resume — but it is the
    # last CPU left inline on this path now that parsing and embedding are
    # both offloaded, and doing it here keeps that true.
    chunks, years = await run_in_threadpool(_parse, text)

    resume = Resume(
        user_id=user_id,
        label=label.strip(),
        filename=filename,
        parsed_text=text,
        years_experience=years,
    )
    session.add(resume)
    await session.flush()

    await _embed_chunks(session, resume, chunks)

    if make_default:
        await set_default(session, user_id=user_id, resume_id=resume.id)

    await session.flush()
    return resume


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
