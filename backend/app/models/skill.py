from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Skill(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Canonical skill taxonomy — shared, not tenant-scoped.

    Phase 2 maps free-text JD skills onto these rows via ``aliases`` so that
    "React.js", "ReactJS" and "React" score as one skill rather than three, and
    Phase 5's trend analysis has something stable to count.
    """

    __tablename__ = "skills"

    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(120), nullable=False, unique=True, index=True)
    category: Mapped[str | None] = mapped_column(String(60), index=True)

    # Lowercased spellings that resolve to this skill. This deterministic lookup
    # runs before any LLM call — cheaper, and it cannot invent a skill.
    aliases: Mapped[list[str]] = mapped_column(
        ARRAY(String(120)), nullable=False, server_default="{}"
    )
