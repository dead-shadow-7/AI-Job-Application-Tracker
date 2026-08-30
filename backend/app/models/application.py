import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.domain.enums import ApplicationStatus

if TYPE_CHECKING:  # imported for typing only; avoids a circular import at runtime
    from app.models.job import Job


class Application(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Your tracking of one job. The tenant-scoped heart of the app."""

    __tablename__ = "applications"
    __table_args__ = (
        # One application per job per user. Re-applying to the same posting is a
        # new event on the existing timeline, not a second row — otherwise the
        # history fragments and the follow-up sweep sees two half-stories.
        UniqueConstraint("user_id", "job_id", name="uq_applications_user_id_job_id"),
        # The Phase 4 sweep's core query: "open applications whose last activity
        # is older than N days". Composite so it can be served by an index scan
        # rather than a filter over every row the user owns.
        Index(
            "ix_applications_user_status_activity", "user_id", "current_status", "last_activity_at"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # FK added in Phase 3

    # --- Denormalized cache of the event log --------------------------------
    # The event log is the source of truth. These three columns exist so that
    # "open applications idle for 7+ days" is an index scan rather than a fold
    # over every event. They are written *only* by services.events.append_event,
    # inside the same transaction as the event insert; a reconciliation test
    # replays the log and asserts they match.
    current_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=ApplicationStatus.SAVED, index=True
    )
    current_status_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Distinct from current_status_at: a recruiter reply or a follow-up you sent
    # resets the staleness clock without advancing the status. Follow-up rules
    # measure against *this*.
    last_activity_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    priority: Mapped[str] = mapped_column(String(10), nullable=False, default="medium")
    notes: Mapped[str | None] = mapped_column(Text)

    # Phase 3 fills these; nullable so Phase 1 works without any scoring.
    job_score: Mapped[int | None] = mapped_column(Integer)
    match_score: Mapped[int | None] = mapped_column(Integer)

    job: Mapped["Job"] = relationship(lazy="joined")
    events: Mapped[list["ApplicationEvent"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="ApplicationEvent.occurred_at",
        lazy="selectin",
    )
    stages: Mapped[list["InterviewStage"]] = relationship(
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="InterviewStage.round_number",
        lazy="selectin",
    )


class ApplicationEvent(UUIDPrimaryKeyMixin, Base):
    """One entry on the timeline. Append-only.

    Nothing updates or deletes these rows. Correcting a mistake means appending
    a corrective event, which is what makes an agent safe to give write access:
    every action it takes is visible and reversible rather than destructive.
    """

    __tablename__ = "application_events"
    __table_args__ = (
        Index("ix_application_events_application_occurred", "application_id", "occurred_at"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("applications.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalized from the parent so RLS can be enforced on this table directly.
    # Joining to applications for every policy check would make the timeline
    # query notably slower, and a policy that cannot be evaluated locally is a
    # policy that gets dropped under deadline pressure.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    event_type: Mapped[str] = mapped_column(String(40), nullable=False, index=True)

    # When it happened in the real world — backdate freely. Distinct from
    # created_at, which is when it was recorded. The follow-up rules read
    # occurred_at; "I applied last Tuesday" must not read as activity today.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    source: Mapped[str] = mapped_column(String(10), nullable=False, default="manual", index=True)
    note: Mapped[str | None] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")

    application: Mapped["Application"] = relationship(back_populates="events")


class InterviewStage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A structured interview round.

    A dedicated table rather than only events, because a round has state that
    changes — scheduled, then an outcome — while events are immutable. The two
    stay in step: scheduling or completing a stage also appends an event.
    """

    __tablename__ = "interview_stages"
    __table_args__ = (
        UniqueConstraint("application_id", "round_number", name="uq_interview_stages_app_round"),
    )

    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    stage_type: Mapped[str] = mapped_column(String(30), nullable=False)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    outcome: Mapped[str] = mapped_column(String(15), nullable=False, default="pending")
    interviewer: Mapped[str | None] = mapped_column(String(200))
    notes: Mapped[str | None] = mapped_column(Text)

    application: Mapped["Application"] = relationship(back_populates="stages")
