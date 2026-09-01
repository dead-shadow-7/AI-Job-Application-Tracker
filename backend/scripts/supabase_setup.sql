-- Run once against a fresh Supabase project, in the SQL Editor.
--
-- Supabase's `postgres` user is not a superuser, but it does carry BYPASSRLS,
-- which bypasses row level security just as completely. FORCE ROW LEVEL
-- SECURITY does not cover it either. Connecting the API as `postgres` would
-- leave every tenant isolation policy
-- decorative while the whole test suite still passed. So the runtime role is
-- created here, exactly as scripts/init-db.sql does locally.
--
-- Change the password before running, and put the same value in DATABASE_URL.
--
-- Note the pooler username format: DATABASE_URL must use `app_user.<project-ref>`,
-- not a bare `app_user`. Supavisor routes tenants by that suffix and refuses a
-- bare role name with "no tenant identifier provided", which reads like an
-- authentication failure and is not.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user
            LOGIN PASSWORD 'CHANGE_ME_BEFORE_RUNNING'
            NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;

-- Table-level grants are issued by the Alembic migrations themselves, so run
-- `alembic upgrade head` after this and the runtime role gets exactly the
-- privileges each table needs — no more.

-- Verify afterwards. Both columns must read `f` for app_user:
--   SELECT rolname, rolsuper, rolbypassrls FROM pg_roles
--   WHERE rolname IN ('postgres', 'app_user');
