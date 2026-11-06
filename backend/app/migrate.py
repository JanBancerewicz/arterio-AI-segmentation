"""Versioned SQL migrations, run once by the Compose init service."""
from pathlib import Path

from .db import connect


def main():
    with connect() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(73190421)")
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY)")
        for path in sorted(Path(__file__).with_name("migrations").glob("*.sql")):
            if conn.execute("SELECT 1 FROM schema_migrations WHERE name = %s", (path.name,)).fetchone():
                continue
            conn.execute(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            print(f"Applied {path.name}", flush=True)


if __name__ == "__main__":
    main()
