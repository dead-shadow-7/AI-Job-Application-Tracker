import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, UUIDPrimaryKeyMixin


class AgentMessage(UUIDPrimaryKeyMixin, Base):
    """One turn of conversation with the assistant.

    Stored server-side rather than held in the browser so a thread survives a
    reload and closing the drawer — a conversation that forgets itself every
    refresh is barely a conversation.

    Deliberately flat: no thread or session id. The drawer is one continuous
    conversation per person, and inventing threads would add a concept the UI
    does not expose and nobody asked for. Adding one later is a column, not a
    redesign.
    """

    __tablename__ = "agent_messages"
    __table_args__ = (Index("ix_agent_messages_user_created", "user_id", "created_at"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    role: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Stamped in Python, not by the server. Postgres `now()` returns the
    # *transaction* start time, so a question and its answer — written in one
    # request — landed on the identical timestamp, and the history query's
    # ORDER BY could then return them either way round. A model shown its own
    # reply before the message that prompted it answers the wrong question.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
