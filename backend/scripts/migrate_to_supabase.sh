#!/usr/bin/env bash
# Move the local tracker data to Supabase.
#
#   ./scripts/migrate_to_supabase.sh 'postgresql://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres'
#
# Run supabase_setup.sql and `alembic upgrade head` against the target first —
# this copies rows only, it does not create the schema.
set -euo pipefail

TARGET="${1:?Pass the Supabase session-pooler URI as the first argument}"
SOURCE="${SOURCE_DSN:-postgresql://jobtracker:jobtracker@localhost:5432/jobtracker}"
DUMP="/tmp/jobtracker-data-$(date +%s).sql"

# Excluded deliberately:
#   skills          seeded by migration 0002; copying it would collide on the
#                   unique slug and abort the whole restore
#   alembic_version already at head on the target after `alembic upgrade head`
echo "==> Dumping data from local"
pg_dump "$SOURCE" \
    --data-only \
    --no-owner \
    --no-privileges \
    --exclude-table=skills \
    --exclude-table=alembic_version \
    --file="$DUMP"

echo "==> Rows to copy:"
grep -c '^INSERT\|^COPY' "$DUMP" || true

echo "==> Restoring into Supabase"
# As the owner (postgres), which is a superuser and therefore bypasses RLS.
# Restoring through app_user would be filtered by the very policies being
# populated, and most rows would silently vanish.
psql "$TARGET" --set ON_ERROR_STOP=on -f "$DUMP"

echo "==> Verifying"
psql "$TARGET" -c "
SELECT 'users' AS table, count(*) FROM users
UNION ALL SELECT 'companies', count(*) FROM companies
UNION ALL SELECT 'jobs', count(*) FROM jobs
UNION ALL SELECT 'applications', count(*) FROM applications
UNION ALL SELECT 'application_events', count(*) FROM application_events
UNION ALL SELECT 'resumes', count(*) FROM resumes
UNION ALL SELECT 'resume_chunks', count(*) FROM resume_chunks
UNION ALL SELECT 'skills (seeded)', count(*) FROM skills
ORDER BY 1;"

rm -f "$DUMP"
echo
echo "Done. Point DATABASE_URL at app_user on the pooler, then:"
echo "  docker compose up -d --force-recreate backend"
