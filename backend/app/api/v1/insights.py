from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser, DbSession
from app.schemas.job import JobSummary
from app.services.analytics import MEANINGFUL_SAMPLE, compute_analytics
from app.services.search import search_applications

router = APIRouter(tags=["insights"])


class SearchHitRead(BaseModel):
    application_id: UUID
    job: JobSummary
    current_status: str
    similarity: float


class FunnelStageRead(BaseModel):
    status: str
    count: int


class PlatformStatsRead(BaseModel):
    platform: str
    applications: int
    responses: int
    response_rate: float | None = None


class AnalyticsRead(BaseModel):
    total: int
    funnel: list[FunnelStageRead] = Field(default_factory=list)
    by_platform: list[PlatformStatsRead] = Field(default_factory=list)

    submitted: int = 0
    responses: int = 0
    response_rate: float | None = None
    median_days_to_response: float | None = None

    sample_is_small: bool = True
    caveat: str | None = Field(
        default=None,
        description="Set when the history is too short for the proportions to mean anything.",
    )


@router.get(
    "/search",
    response_model=list[SearchHitRead],
    summary="Find tracked applications by meaning, not keyword",
)
async def semantic_search(
    user: CurrentUser,
    session: DbSession,
    q: Annotated[str, Query(min_length=2, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[SearchHitRead]:
    """Answers "the RAG and agent roles I applied to" without either posting
    containing the word searched for — which is what the keyword filter on the
    dashboard cannot do."""
    hits = await search_applications(session, user.id, q, limit=limit)
    return [
        SearchHitRead(
            application_id=hit.application.id,
            job=JobSummary.model_validate(hit.application.job),
            current_status=hit.application.current_status,
            similarity=hit.similarity,
        )
        for hit in hits
    ]


@router.get("/analytics", response_model=AnalyticsRead, summary="What the history says")
async def analytics(user: CurrentUser, session: DbSession) -> AnalyticsRead:
    """Every proportion carries the count behind it.

    A response rate over two applications is noise, and presenting it without
    saying so invites acting on it.
    """
    data = await compute_analytics(session, user.id)

    return AnalyticsRead(
        total=data.total,
        funnel=[FunnelStageRead(status=s.status, count=s.count) for s in data.funnel],
        by_platform=[
            PlatformStatsRead(
                platform=p.platform,
                applications=p.applications,
                responses=p.responses,
                response_rate=p.response_rate,
            )
            for p in data.by_platform
        ],
        submitted=data.submitted,
        responses=data.responses,
        response_rate=data.response_rate,
        median_days_to_response=data.median_days_to_response,
        sample_is_small=data.sample_is_small,
        caveat=(
            f"Based on fewer than {MEANINGFUL_SAMPLE} submitted applications, so these "
            "proportions are not yet meaningful — one outcome moves them a long way."
            if data.sample_is_small
            else None
        ),
    )
