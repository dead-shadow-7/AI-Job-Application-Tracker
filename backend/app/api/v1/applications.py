from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.exceptions import NotFoundError
from app.domain.enums import ApplicationStatus, EventSource, Priority, WorkMode
from app.models.application import InterviewStage
from app.schemas.application import (
    ApplicationCreate,
    ApplicationPage,
    ApplicationRead,
    ApplicationStats,
    ApplicationUpdate,
    EventCreate,
    EventRead,
    InterviewStageCreate,
    InterviewStageRead,
    InterviewStageUpdate,
    detail,
    summarize,
)
from app.services.applications import create_application, get_stats, list_applications
from app.services.events import append_event, get_application, reload_application

router = APIRouter(prefix="/applications", tags=["applications"])


@router.get("", response_model=ApplicationPage, summary="List and filter applications")
async def list_all(
    user: CurrentUser,
    session: DbSession,
    status_filter: Annotated[list[ApplicationStatus] | None, Query(alias="status")] = None,
    search: Annotated[str | None, Query(max_length=200)] = None,
    priority: Priority | None = None,
    work_mode: WorkMode | None = None,
    active_only: bool = False,
    sort: str = "last_activity_at",
    order: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ApplicationPage:
    rows, total = await list_applications(
        session,
        user_id=user.id,
        status=[s.value for s in status_filter] if status_filter else None,
        search=search,
        priority=priority.value if priority else None,
        work_mode=work_mode.value if work_mode else None,
        active_only=active_only,
        sort=sort,
        order=order,
        limit=limit,
        offset=offset,
    )
    return ApplicationPage(
        items=[summarize(r) for r in rows], total=total, limit=limit, offset=offset
    )


@router.get("/stats", response_model=ApplicationStats, summary="Dashboard counters")
async def stats(user: CurrentUser, session: DbSession) -> ApplicationStats:
    return ApplicationStats.model_validate(await get_stats(session, user.id))


@router.post(
    "",
    response_model=ApplicationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Track a job — by id, or with a new job inline",
)
async def create(
    payload: ApplicationCreate, user: CurrentUser, session: DbSession
) -> ApplicationRead:
    application = await create_application(
        session,
        user_id=user.id,
        job_id=payload.job_id,
        job_payload=payload.job,
        priority=payload.priority.value,
        notes=payload.notes,
        initial_event=payload.initial_event,
        occurred_at=payload.occurred_at,
    )
    return detail(application)


@router.get("/{application_id}", response_model=ApplicationRead, summary="One application")
async def read_one(application_id: UUID, user: CurrentUser, session: DbSession) -> ApplicationRead:
    return detail(await get_application(session, application_id, user.id))


@router.patch("/{application_id}", response_model=ApplicationRead, summary="Edit priority or notes")
async def update(
    application_id: UUID, payload: ApplicationUpdate, user: CurrentUser, session: DbSession
) -> ApplicationRead:
    application = await get_application(session, application_id, user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(application, field, value.value if hasattr(value, "value") else value)
    await session.flush()
    await session.refresh(application)
    return detail(application)


@router.delete("/{application_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Stop tracking")
async def delete(application_id: UUID, user: CurrentUser, session: DbSession) -> None:
    application = await get_application(session, application_id, user.id)
    await session.delete(application)


# --- Timeline ---------------------------------------------------------------


@router.get(
    "/{application_id}/events",
    response_model=list[EventRead],
    summary="The application's timeline, oldest first",
)
async def read_events(
    application_id: UUID, user: CurrentUser, session: DbSession
) -> list[EventRead]:
    application = await get_application(session, application_id, user.id)
    return [EventRead.model_validate(e) for e in application.events]


@router.post(
    "/{application_id}/events",
    response_model=ApplicationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Append an event — this is how status changes",
)
async def add_event(
    application_id: UUID, payload: EventCreate, user: CurrentUser, session: DbSession
) -> ApplicationRead:
    """Returns the whole application, not just the event.

    Appending is the only way status moves, so the caller almost always needs
    the recomputed status and activity dates immediately — returning them here
    saves the client a follow-up round trip and a window of stale UI.
    """
    application = await get_application(session, application_id, user.id)
    await append_event(
        session,
        application=application,
        event_type=payload.event_type,
        occurred_at=payload.occurred_at,
        source=EventSource.MANUAL,
        note=payload.note,
        payload=payload.payload,
    )
    return detail(await reload_application(session, application_id, user.id))


# --- Interview stages -------------------------------------------------------


@router.post(
    "/{application_id}/stages",
    response_model=InterviewStageRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add an interview round",
)
async def add_stage(
    application_id: UUID, payload: InterviewStageCreate, user: CurrentUser, session: DbSession
) -> InterviewStageRead:
    application = await get_application(session, application_id, user.id)
    stage = InterviewStage(
        application_id=application.id,
        user_id=user.id,
        round_number=payload.round_number,
        stage_type=payload.stage_type.value,
        scheduled_at=payload.scheduled_at,
        outcome=payload.outcome.value,
        interviewer=payload.interviewer,
        notes=payload.notes,
    )
    session.add(stage)
    await session.flush()
    return InterviewStageRead.model_validate(stage)


@router.patch(
    "/{application_id}/stages/{stage_id}",
    response_model=InterviewStageRead,
    summary="Update a round's schedule or outcome",
)
async def update_stage(
    application_id: UUID,
    stage_id: UUID,
    payload: InterviewStageUpdate,
    user: CurrentUser,
    session: DbSession,
) -> InterviewStageRead:
    await get_application(session, application_id, user.id)
    stage = (
        await session.execute(
            select(InterviewStage).where(
                InterviewStage.id == stage_id,
                InterviewStage.application_id == application_id,
                InterviewStage.user_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if stage is None:
        raise NotFoundError(f"Interview stage {stage_id} not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(stage, field, value.value if hasattr(value, "value") else value)
    await session.flush()
    return InterviewStageRead.model_validate(stage)
