"""Initial: extensions, users profile table, RLS baseline

Revision ID: 0001_initial_users
Revises:
Create Date: 2026-08-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_user_isolation, enable_user_isolation

revision: str = "0001_initial_users"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Declared here rather than relying on the local init script, so the
    # migration bootstraps any empty database (CI, a colleague's machine, a
    # restored backup) without out-of-band setup.
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
    op.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm"')

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=True),
        sa.Column(
            "preferences",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # Grants live in the migration, not in the one-shot init script: default
    # privileges are per-database, so a database created later (CI, the test
    # database, a restore) would otherwise come up with no runtime access.
    op.execute("GRANT USAGE ON SCHEMA public TO app_user")

    # users is keyed on `id`, not `user_id` — it *is* the tenant row.
    for statement in enable_user_isolation("users", user_column="id"):
        op.execute(statement)


def downgrade() -> None:
    for statement in disable_user_isolation("users"):
        op.execute(statement)
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
