import uuid
from datetime import datetime

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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
