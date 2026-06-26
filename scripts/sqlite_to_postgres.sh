#!/usr/bin/env bash
# sqlite_to_postgres.sh — dump SQLite golden DB to PostgreSQL-compatible SQL
#
# Usage:
#   ./scripts/sqlite_to_postgres.sh                    # → data/collaborative_pg.sql
#   ./scripts/sqlite_to_postgres.sh /path/to/db.sqlite  # custom input
#
# Output is a plain SQL file ready for:
#   psql -U postgres -d echo_dsrn -f data/collaborative_pg.sql
set -euo pipefail

INPUT="${1:-data/collaborative.db}"
OUTPUT="data/collaborative_pg.sql"

if [ ! -f "$INPUT" ]; then
    echo "Error: input file not found: $INPUT" >&2
    exit 1
fi

echo "Dumping $INPUT → $OUTPUT ..."

sqlite3 "$INPUT" .dump | python3 -c "
import sys, re

for line in sys.stdin:
    # Only keep INSERT statements — schema is created by alembic
    if not line.startswith('INSERT INTO'):
        continue
    # Remove double-quotes around identifiers
    line = re.sub(r'\"(sqlite_|alembic_|\w+)\"', r'\1', line)
    # Replace datetime('now') with now()
    line = line.replace(\"datetime('now')\", 'now()')
    # SQLite stores booleans as 0/1 — cast to TRUE/FALSE for PostgreSQL
    line = re.sub(r\"(,)\s*1\s*(,|\))\", r'\\1 TRUE\\2', line)
    line = re.sub(r\"(,)\s*0\s*(,|\))\", r'\\1 FALSE\\2', line)
    sys.stdout.write(line)
" > "$OUTPUT"

echo "Done. $(wc -l < "$OUTPUT") lines."
echo "Import with:  psql -U postgres -d echo_dsrn -f $OUTPUT"
