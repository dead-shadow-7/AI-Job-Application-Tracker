"""Employment history read out of a resume

Most resumes never write "4 years of experience" anywhere — they write the
dates. Until now that left ``years_experience`` null on almost every upload,
and both scoring components that depend on it fall back to 0.5 when it is
unknown, so a quarter of every match score was a constant.

``positions`` caches the roles the parser found, and ``years_experience_source``
records whether the number beside them was claimed by the resume or summed from
those dates. Both are derived from ``parsed_text`` and are rebuilt by a
re-parse; existing rows are backfilled the next time they are re-parsed rather
than in this migration, which has no parser available to it.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0007_resume_positions"
down_revision: str | None = "0006_agent_messages"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "resumes",
        sa.Column(
            "positions",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "resumes", sa.Column("years_experience_source", sa.String(length=20), nullable=True)
    )
    # Every existing row's number came from an explicit statement, because that
    # was the only way one could be produced before this migration.
    op.execute(
        "UPDATE resumes SET years_experience_source = 'stated' "
        "WHERE years_experience IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("resumes", "years_experience_source")
    op.drop_column("resumes", "positions")
