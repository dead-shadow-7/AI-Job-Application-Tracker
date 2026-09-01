#!/usr/bin/env bash
# Dump the tracker to a timestamped file.
#
# Usage:
#   ./scripts/backup.sh                      # local Docker database
#   ./scripts/backup.sh "$SUPABASE_URI"      # anywhere else
#
# Runs as the owner, because a dump taken through the RLS-constrained runtime
# role would silently contain only the rows visible to whichever tenant was
# scoped — that is, usually none. A backup that restores empty is worse than no
# backup, because you believe you have one.
set -euo pipefail

DSN="${1:-postgresql://jobtracker:jobtracker@localhost:5432/jobtracker}"
DEST="${BACKUP_DIR:-./backups}"
STAMP=$(date +%Y%m%d-%H%M%S)

mkdir -p "$DEST"
OUT="$DEST/jobtracker-$STAMP.sql"

pg_dump "$DSN" \
    --no-owner \
    --no-privileges \
    --file="$OUT"

echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo
echo "Restore with:"
echo "  psql <target-dsn> --set ON_ERROR_STOP=on -f $OUT"
