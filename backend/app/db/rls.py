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


def grant_readonly_reference(table: str, *, role: str = RUNTIME_ROLE) -> list[str]:
    """For curated reference tables (the skill taxonomy) that every user reads
    but only migrations write."""
    return [f"GRANT SELECT ON {table} TO {role}"]


def grant_shared_table(table: str, *, role: str = RUNTIME_ROLE) -> list[str]:
    """For shared, non-tenant tables that users create rows in — companies and
    job postings.

    Deliberately no RLS: a job description is public information the employer
    published, and a per-user copy of every posting would break deduplication
    and cross-company analytics. What is private — that *you* applied, and what
    happened — lives on the tenant-scoped tables instead.
    """
    return [f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {role}"]
