-- Run once against a fresh Supabase project, in the SQL Editor.
--
-- Supabase's `postgres` user is a SUPERUSER, and superusers bypass row level
-- security unconditionally — FORCE ROW LEVEL SECURITY does not cover them.
-- Connecting the API as `postgres` would leave every tenant isolation policy
-- decorative while the whole test suite still passed. So the runtime role is
-- created here, exactly as scripts/init-db.sql does locally.
--
-- Change the password before running, and put the same value in DATABASE_URL.

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
