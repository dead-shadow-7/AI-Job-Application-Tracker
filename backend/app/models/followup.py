import uuid

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class FollowUpRule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """When silence on an application becomes worth acting on.

    Rules are rows rather than constants because the right interval is a
    personal judgement that differs by stage: a week of quiet after applying is
    normal, a week after a final interview is not. They are also what makes the
    detection explicable — the dashboard can say *which* rule fired rather than
    presenting a number the user cannot trace.

    Evaluated by a SQL query, never by a model. "Seven days since the screening"
    is a date subtraction; routing it through an LLM would make a deterministic
    rule non-deterministic, cost tokens on every sweep, and remove the ability
    to unit-test the one part of this feature that must not be wrong.
    """

    __tablename__ = "follow_up_rules"
    __table_args__ = (
        # One threshold per (status, action). A status legitimately needs more
        # than one rule — `applied` suggests a follow-up at 7 days and gives up
        # at 21 — but two *suggest* rules for the same status would both fire
        # and the UI could not say which one meant anything.
        UniqueConstraint(
            "user_id", "applies_to_status", "action", name="uq_follow_up_rules_user_status_action"
        ),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    applies_to_status: Mapped[str] = mapped_column(String(20), nullable=False)
    days_threshold: Mapped[int] = mapped_column(Integer, nullable=False)

    # `suggest_followup` surfaces it for you to act on. `mark_ghosted` is the
    # long-stop that closes an application nobody was ever going to answer, so
    # the active list stays honest.
    action: Mapped[str] = mapped_column(String(20), nullable=False, default="suggest_followup")
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True)
