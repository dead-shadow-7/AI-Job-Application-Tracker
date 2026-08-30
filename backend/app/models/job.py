import uuid
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:  # imported for typing only; avoids a circular import at runtime
    from app.models.company import Company
    from app.models.skill import Skill


class Job(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A job posting.

    Deliberately separate from ``Application``: this is what the *employer*
    published, while an application is what *you* did about it. Splitting them
    lets you save a posting before applying, keeps the extracted JD stable while
    your status moves, and gives Phase 2 an immutable record to re-extract
    against when a prompt improves.
    """

    __tablename__ = "jobs"

    company_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # Attribution, not authorisation. Jobs are shared reference data (see the
    # tenancy note in migration 0002); this records who first entered the row so
    # writes can be narrowed to the creator later without a data migration.
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True
    )

    title: Mapped[str] = mapped_column(String(300), nullable=False)
    seniority: Mapped[str | None] = mapped_column(String(30))
    employment_type: Mapped[str | None] = mapped_column(String(30))
    work_mode: Mapped[str | None] = mapped_column(String(20), index=True)
    location: Mapped[str | None] = mapped_column(String(255))

    url: Mapped[str | None] = mapped_column(Text)
    source_platform: Mapped[str | None] = mapped_column(String(60), index=True)
    posted_at: Mapped[date | None] = mapped_column()

    description: Mapped[str | None] = mapped_column(Text)
    responsibilities: Mapped[str | None] = mapped_column(Text)

    # Numeric, not Float: money compared with == in tests and summed in Phase 5
    # analytics should not carry binary rounding error.
    salary_min: Mapped[float | None] = mapped_column(Numeric(12, 2))
    salary_max: Mapped[float | None] = mapped_column(Numeric(12, 2))
    salary_currency: Mapped[str | None] = mapped_column(String(3))
    salary_period: Mapped[str | None] = mapped_column(String(10))

    years_experience_min: Mapped[int | None] = mapped_column(Integer)
    years_experience_max: Mapped[int | None] = mapped_column(Integer)

    # Set by Phase 2. Null for hand-entered jobs, which is how the UI knows not
    # to show a "review the extraction" prompt for something you typed yourself.
    extraction_confidence: Mapped[float | None] = mapped_column(Numeric(3, 2))
    extraction_model: Mapped[str | None] = mapped_column(String(80))

    # sha256 of the normalized description. Cheap exact-duplicate check on
    # ingest; Phase 3 adds semantic near-duplicate detection over pgvector.
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    company: Mapped["Company"] = relationship(lazy="joined")
    requirements: Mapped[list["JobRequirement"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )
    skills: Mapped[list["JobSkill"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", lazy="selectin"
    )


class JobRequirement(UUIDPrimaryKeyMixin, Base):
    """A single requirement line, split must-have vs nice-to-have.

    Kept as rows rather than two text blobs because Phase 3 scores must-have
    coverage at 45% and nice-to-have at 15% — they need to be counted
    separately, and each one is retrieved against your resume individually.
    """

    __tablename__ = "job_requirements"

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False, default="must")
    category: Mapped[str | None] = mapped_column(String(60))

    job: Mapped["Job"] = relationship(back_populates="requirements")


class JobSkill(Base):
    """Join between a job and the canonical skill taxonomy."""

    __tablename__ = "job_skills"
    __table_args__ = (UniqueConstraint("job_id", "skill_id", name="uq_job_skills_job_id_skill_id"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), primary_key=True
    )
    skill_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("skills.id", ondelete="CASCADE"), primary_key=True
    )
    is_required: Mapped[bool] = mapped_column(nullable=False, default=True)
    years_required: Mapped[int | None] = mapped_column(Integer)

    job: Mapped["Job"] = relationship(back_populates="skills")
    skill: Mapped["Skill"] = relationship(lazy="joined")
