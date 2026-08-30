-- Runs once, on first creation of the local Postgres volume.
-- Supabase ships these extensions pre-installed; this keeps local parity.

CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS "pg_trgm";    -- fuzzy company/application matching

-- ---------------------------------------------------------------------------
-- Runtime role.
--
-- POSTGRES_USER (jobtracker) is a SUPERUSER, and superusers bypass row level
-- security unconditionally — FORCE ROW LEVEL SECURITY does not apply to them.
-- Connecting the application as that role would make every policy decorative.
--
-- So: `jobtracker` owns the schema and runs migrations; `app_user` is
-- NOSUPERUSER NOBYPASSRLS and is what the API connects as. Table grants are
-- issued per-table in the migrations, which keeps any freshly-created database
-- (CI, a restored backup) correct without out-of-band setup.
--
-- Roles are cluster-wide, so this covers the test database too.
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user
            LOGIN PASSWORD 'app_password'
            NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS NOINHERIT;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE jobtracker TO app_user;
