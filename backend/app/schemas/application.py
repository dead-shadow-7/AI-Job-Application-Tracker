from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import (
    ApplicationStatus,
    EventSource,
    EventType,
    InterviewStageType,
    Priority,
    StageOutcome,
)
from app.schemas.job import JobCreate, JobRead, JobSummary


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    event_type: EventType
    occurred_at: datetime
    created_at: datetime
    source: EventSource
    note: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class EventCreate(BaseModel):
    event_type: EventType
    # Omit for "just now". Backdating is expected and supported; future dates
    # are rejected — see services.events.append_event.
    occurred_at: datetime | None = None
    note: str | None = Field(default=None, max_length=4000)
    payload: dict[str, Any] = Field(default_factory=dict)


class InterviewStageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    round_number: int
    stage_type: InterviewStageType
    scheduled_at: datetime | None = None
    outcome: StageOutcome
    interviewer: str | None = None
    notes: str | None = None


class InterviewStageCreate(BaseModel):
    round_number: int = Field(ge=1, le=20)
    stage_type: InterviewStageType
    scheduled_at: datetime | None = None
    outcome: StageOutcome = StageOutcome.PENDING
    interviewer: str | None = Field(default=None, max_length=200)
    notes: str | None = None


class InterviewStageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage_type: InterviewStageType | None = None
    scheduled_at: datetime | None = None
    outcome: StageOutcome | None = None
    interviewer: str | None = Field(default=None, max_length=200)
    notes: str | None = None


class ApplicationCreate(BaseModel):
    """Create an application against an existing job, or a new one inline.

    Both paths exist because the two real workflows differ: adding a job you
    just found (inline) versus applying to one you saved earlier (by id).
    """

    job_id: UUID | None = None
    job: JobCreate | None = None
    priority: Priority = Priority.MEDIUM
    notes: str | None = None
    # Defaults to `saved`. Pass `applied` with applied_at to record one you have
    # already sent, which is the common case when backfilling.
    initial_event: EventType = EventType.SAVED
    occurred_at: datetime | None = None


class ApplicationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: Priority | None = None
    notes: str | None = None
    # Status is intentionally absent. It is derived from the event log; to move
    # an application you append an event. Allowing a direct write here would let
    # the cache diverge from its own source of truth.


class ApplicationSummary(BaseModel):
    """One row in the dashboard table."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job: JobSummary
    current_status: ApplicationStatus
    current_status_at: datetime
    last_activity_at: datetime
    applied_at: datetime | None = None
    priority: Priority
    job_score: int | None = None
    match_score: int | None = None
    created_at: datetime

    days_since_activity: int = 0


class ApplicationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job: JobRead
    current_status: ApplicationStatus
    current_status_at: datetime
    last_activity_at: datetime
    applied_at: datetime | None = None
    priority: Priority
    notes: str | None = None
    job_score: int | None = None
    match_score: int | None = None
    created_at: datetime
    events: list[EventRead] = Field(default_factory=list)
    stages: list[InterviewStageRead] = Field(default_factory=list)

    # Computed server-side, like the list rows. Keeping "how idle is this?" in
    # one place stops the dashboard and the detail page from disagreeing, and
    # keeps the client from doing clock arithmetic during render.
    days_since_activity: int = 0


class ApplicationPage(BaseModel):
    items: list[ApplicationSummary]
    total: int
    limit: int
    offset: int


class StatusCount(BaseModel):
    status: ApplicationStatus
    count: int


class ApplicationStats(BaseModel):
    """Header counters for the dashboard."""

    total: int
    by_status: list[StatusCount]
    active: int
    needs_attention: int
