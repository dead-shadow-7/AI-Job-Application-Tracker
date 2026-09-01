"""Semantic search and near-duplicate detection over tracked jobs.

The first real use of the embeddings Phase 3 stored. Until now they were
written on every job creation and never read — the HNSW index existed but
nothing queried it.

Both features here are the same operation with different thresholds: "find jobs
like this text" and "is this posting one I already have". Keeping them in one
module keeps the distance semantics in one place, because cosine distance
thresholds are the kind of magic number that quietly drifts apart when copied.
"""

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.application import Application
from app.models.job import Job
from app.models.resume import JobEmbedding
from app.services import embeddings

logger = logging.getLogger(__name__)

# pgvector cosine_distance is 1 - cosine_similarity, so lower is closer.
#
# 0.15 was chosen against real postings: two ML-engineer roles at different
# companies land around 0.10-0.20, while an ML role against a mobile role is
# past 0.5. Set tighter and genuine reposts are missed; looser and every
# backend job looks like every other backend job, which is worse — a false
# duplicate warning teaches you to ignore duplicate warnings.
NEAR_DUPLICATE_DISTANCE = 0.15

# Search is deliberately more permissive. A search returning nothing is a dead
# end, whereas a weak result is visibly weak and costs one glance.
SEARCH_MAX_DISTANCE = 0.65


@dataclass(frozen=True, slots=True)
class SearchHit:
    application: Application
    distance: float

    @property
    def similarity(self) -> float:
        """0-1, for display. Distance is the useful number; similarity reads better."""
        return round(max(0.0, 1.0 - self.distance), 3)


async def search_applications(
    session: AsyncSession,
    user_id: uuid.UUID,
    query: str,
    *,
    limit: int = 10,
) -> list[SearchHit]:
    """Find tracked applications by meaning rather than keyword.

    Answers "show me the RAG and agent roles I applied to" without either job
    containing the word you searched for — which is the whole point, and what
    ILIKE cannot do.

    Scoped by joining through ``applications``: ``job_embeddings`` is shared
    reference data with no row policy, so the tenant filter has to come from
    the tenant-scoped side of the join.
    """
    cleaned = query.strip()
    if not cleaned:
        return []

    vector = await embeddings.embedding_provider.embed_query(cleaned)
    distance = JobEmbedding.embedding.cosine_distance(vector)

    rows = (
        (
            await session.execute(
                select(Application, distance.label("distance"))
                .join(Job, Job.id == Application.job_id)
                .join(JobEmbedding, JobEmbedding.job_id == Job.id)
                .where(Application.user_id == user_id, distance < SEARCH_MAX_DISTANCE)
                .order_by(distance)
                .limit(limit)
            )
        )
        .unique()
        .all()
    )

    return [SearchHit(application=application, distance=float(d)) for application, d in rows]


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    application: Application
    distance: float
    is_exact: bool


async def find_near_duplicate(
    session: AsyncSession,
    user_id: uuid.UUID,
    text: str,
    *,
    content_digest: str | None = None,
) -> DuplicateMatch | None:
    """Is this posting one the user already tracks?

    Exact hash first — it is free and unambiguous. The embedding check then
    catches what the hash cannot: the same role reposted with edited wording,
    or found on two boards with different boilerplate wrapped around it.
    """
    if content_digest:
        exact = (
            (
                await session.execute(
                    select(Application)
                    .join(Job, Job.id == Application.job_id)
                    .where(Application.user_id == user_id, Job.content_hash == content_digest)
                    .limit(1)
                )
            )
            .unique()
            .scalar_one_or_none()
        )
        if exact is not None:
            return DuplicateMatch(application=exact, distance=0.0, is_exact=True)

    if not text.strip():
        return None

    vector = await embeddings.embedding_provider.embed_query(text)
    distance = JobEmbedding.embedding.cosine_distance(vector)

    row = (
        (
            await session.execute(
                select(Application, distance.label("distance"))
                .join(Job, Job.id == Application.job_id)
                .join(JobEmbedding, JobEmbedding.job_id == Job.id)
                .where(Application.user_id == user_id, distance < NEAR_DUPLICATE_DISTANCE)
                .order_by(distance)
                .limit(1)
            )
        )
        .unique()
        .first()
    )
    if row is None:
        return None

    application, dist = row
    return DuplicateMatch(application=application, distance=float(dist), is_exact=False)
