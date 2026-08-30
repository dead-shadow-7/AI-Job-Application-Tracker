from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class Company(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An employer.

    Shared across users rather than tenant-scoped: "Google" is the same company
    whoever is tracking it, and duplicating it per user would fragment the
    Phase 5 analytics ("which companies reply fastest?") for no benefit. Nothing
    user-identifying lives here — that all hangs off ``applications``.
    """

    __tablename__ = "companies"

    name: Mapped[str] = mapped_column(String(300), nullable=False)

    # Deduplication key: lowercased, legal suffixes stripped, punctuation
    # collapsed. See app.services.companies.normalize_company_name.
    normalized_name: Mapped[str] = mapped_column(
        String(300), nullable=False, unique=True, index=True
    )

    domain: Mapped[str | None] = mapped_column(String(255), index=True)
    industry: Mapped[str | None] = mapped_column(String(120))
    size: Mapped[str | None] = mapped_column(String(60))
    location: Mapped[str | None] = mapped_column(String(255))
