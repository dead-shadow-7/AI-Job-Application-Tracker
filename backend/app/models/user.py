import uuid
from typing import Any

from sqlalchemy import String, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class User(TimestampMixin, Base):
    """Profile row mirroring Supabase ``auth.users``.

    ``id`` is *not* generated here — it is the Supabase user id taken from the
    verified JWT subject, so the two systems share one identity.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(200))

    # Job-search preferences: target locations, work mode, salary floor,
    # years of experience. Read by the Phase 3 job_score. Schema-on-read while
    # the shape is still settling; promoted to columns once it stops moving.
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
