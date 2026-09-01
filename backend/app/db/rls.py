"""Row Level Security policy SQL, generated in one place.

Phase 1 adds this to roughly fifteen tables. Hand-writing the policy each time
invites exactly the two mistakes that are easy to make and hard to notice:
forgetting ``FORCE`` (so the owner silently bypasses it) and comparing against a
raw ``current_setting`` (see ``USER_ID_EXPR`` below).

Returns plain SQL strings rather than calling ``op.execute`` so the module stays
free of an Alembic import and can be unit-tested directly.
"""

# NULLIF is load-bearing, not defensive noise.
#
# `set_config(..., is_local => true)` scopes the value to a transaction. When
# that transaction ends the custom GUC does not become undefined again — it
# reverts to the empty string. So `current_setting('app.user_id', true)` returns
# NULL only on a connection that has *never* been scoped, and '' on any pooled
# connection that has been. Casting '' to uuid raises 22P02 rather than
# filtering, which turns an unscoped query into a 500 instead of an empty
# result. NULLIF collapses both cases to NULL, and `col = NULL` is NULL, so the
# row is filtered. Fail closed, quietly.
USER_ID_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"

RUNTIME_ROLE = "app_user"


def enable_user_isolation(
    table: str,
    *,
    user_column: str = "user_id",
    role: str = RUNTIME_ROLE,
) -> list[str]:
    """Enable + force RLS, add the tenant policy, and grant runtime access."""
    policy = f"{table}_user_isolation"
    predicate = f"{user_column} = {USER_ID_EXPR}"
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        # FORCE, not just ENABLE: the schema owner bypasses plain ENABLE, and
        # migrations run as the owner.
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        f"CREATE POLICY {policy} ON {table} USING ({predicate}) WITH CHECK ({predicate})",
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {role}",
    ]


def disable_user_isolation(table: str, *, role: str = RUNTIME_ROLE) -> list[str]:
    return [
        f"REVOKE ALL ON {table} FROM {role}",
        f"DROP POLICY IF EXISTS {table}_user_isolation ON {table}",
    ]


def share_table(table: str, *, writable: bool = True, role: str = RUNTIME_ROLE) -> list[str]:
    """For non-tenant tables: company records, job postings, the skill taxonomy.

    These carry nothing private. A job description is information the employer
    published, and a per-user copy of every posting would break deduplication
    and cross-company analytics. What *is* private — that you applied, and what
    happened next — lives on the tenant-scoped tables.

    RLS is enabled with an explicit permissive policy rather than left off.
    That is not ceremony: Supabase enables RLS by default on every table in the
    `public` schema, and **RLS enabled with no policy denies everything**. A
    migration that merely granted SELECT would leave the runtime role able to
    read exactly zero rows there while working perfectly against a local
    Postgres, which had RLS off. Stating the policy makes both environments
    agree, and records that the sharing is deliberate rather than an oversight.
    """
    verbs = "SELECT, INSERT, UPDATE, DELETE" if writable else "SELECT"
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"DROP POLICY IF EXISTS {table}_shared_access ON {table}",
        f"CREATE POLICY {table}_shared_access ON {table} USING (true) WITH CHECK (true)",
        f"GRANT {verbs} ON {table} TO {role}",
    ]


# Retained under the old names so existing migrations keep working unchanged.
def grant_readonly_reference(table: str, *, role: str = RUNTIME_ROLE) -> list[str]:
    return share_table(table, writable=False, role=role)


def grant_shared_table(table: str, *, role: str = RUNTIME_ROLE) -> list[str]:
    return share_table(table, writable=True, role=role)
