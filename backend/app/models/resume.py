import uuid
from typing import TYPE_CHECKING, Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.job import Job

# 384 for BAAI/bge-small-en-v1.5. Fixed at migration time: changing it means
# re-embedding every row, so it is configuration for the model choice, not
# something to vary per environment.
EMBEDDING_DIM = settings.embedding_dim


class Resume(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An uploaded resume. Tenant-scoped — this is the most personal data here."""

    __tablename__ = "resumes"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(255))

    # The extracted text, kept so a resume can be re-chunked when the strategy
    # improves without asking the user to upload the file again. The original
    # binary is deliberately not stored: it is never needed after parsing, and
    # not keeping it is the simplest way not to leak it.
    parsed_text: Mapped[str] = mapped_column(Text, nullable=False)

    is_default: Mapped[bool] = mapped_column(nullable=False, default=False)
    years_experience: Mapped[float | None] = mapped_column(Numeric(4, 1))

    # lazy="raise": chunks carry a 384-dimensional vector each, so loading them
    # to answer "how many?" would pull the entire embedding set for every
    # resume in a list. Anything needing a count uses an aggregate query;
    # anything needing the chunks themselves asks for them explicitly. Raising
    # makes an accidental lazy load a loud error rather than a slow endpoint.
    chunks: Mapped[list["ResumeChunk"]] = relationship(
        back_populates="resume", cascade="all, delete-orphan", lazy="raise"
    )


class ResumeChunk(UUIDPrimaryKeyMixin, Base):
    """One embedded passage of a resume.

    Chunks rather than one vector per resume, because the retrieval that matters
    is "which of my bullet points evidence *this* requirement" — a whole-resume
    embedding averages everything into a blur that answers nothing.
    """

    __tablename__ = "resume_chunks"
    __table_args__ = (Index("ix_resume_chunks_resume_ordinal", "resume_id", "ordinal"),)

    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    # Denormalized so RLS can be enforced without a join, exactly as on
    # application_events.
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str | None] = mapped_column(String(60))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    resume: Mapped["Resume"] = relationship(back_populates="chunks")


class JobEmbedding(UUIDPrimaryKeyMixin, Base):
    """Embedded job text, for semantic search and near-duplicate detection.

    Not tenant-scoped, matching ``jobs`` itself — see the tenancy note in
    migration 0002.
    """

    __tablename__ = "job_embeddings"
    __table_args__ = (UniqueConstraint("job_id", "ordinal", name="uq_job_embeddings_job_ordinal"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    job: Mapped["Job"] = relationship()


class MatchAnalysis(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A cached resume-to-job score.

    Cached per (resume, job) because scoring costs an LLM call, and the inputs
    only change when the resume is replaced or the job is re-extracted.
    ``model`` and ``prompt_version`` are stored so a prompt change does not
    silently make old scores incomparable to new ones.
    """

    __tablename__ = "match_analyses"
    __table_args__ = (UniqueConstraint("resume_id", "job_id", name="uq_match_analyses_resume_job"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("resumes.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )

    overall_score: Mapped[int] = mapped_column(Integer, nullable=False)

    # Every component that produced overall_score, kept so the UI can show *why*
    # a job scored 72 rather than asking the user to trust a bare number.
    subscores: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    matched_skills: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    missing_skills: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    strengths: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    gaps: Mapped[list[str]] = mapped_column(JSONB, nullable=False, server_default="[]")
    narrative: Mapped[str | None] = mapped_column(Text)

    model: Mapped[str | None] = mapped_column(String(80))
    prompt_version: Mapped[str | None] = mapped_column(String(40))
