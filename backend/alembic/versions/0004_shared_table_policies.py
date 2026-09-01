"""Explicit permissive policies on the shared, non-tenant tables

Found while migrating to Supabase, and it would have broken the application
completely.

Supabase enables row level security by default on every table created in the
`public` schema. **RLS enabled with no policy denies everything** to any role
without BYPASSRLS — so the runtime role could read zero skills, zero companies
and zero jobs, while the identical migration against a local Postgres worked
perfectly because RLS was simply off there.

Nothing in the application would have reported this as a permissions problem.
The skill picker would just be empty, extraction would match no skills, and
every match score would show total coverage failure.

The fix makes both environments state the same thing: RLS on, with an explicit
`USING (true)` policy recording that the sharing is deliberate. See
app/db/rls.py — the two environments now converge rather than differing by
whatever the host happens to default to.

Revision ID: 0004_shared_table_policies
Revises: 0003_vector_schema
Create Date: 2026-09-01
"""

from collections.abc import Sequence

from alembic import op
from app.db.rls import share_table

revision: str = "0004_shared_table_policies"
down_revision: str | None = "0003_vector_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

WRITABLE = ["companies", "jobs", "job_requirements", "job_skills", "job_embeddings"]
# The taxonomy is curated: every user reads it, only migrations write it.
READ_ONLY = ["skills"]


def upgrade() -> None:
    for table in WRITABLE:
        for statement in share_table(table, writable=True):
            op.execute(statement)
    for table in READ_ONLY:
        for statement in share_table(table, writable=False):
            op.execute(statement)


def downgrade() -> None:
    for table in WRITABLE + READ_ONLY:
        op.execute(f"DROP POLICY IF EXISTS {table}_shared_access ON {table}")
        op.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
