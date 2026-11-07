"""Durable cleanup after deletion; a filesystem failure can be retried."""
from contextlib import contextmanager
import logging
from pathlib import Path
import shutil
from threading import Event, Thread
from uuid import UUID

from .db import connect
from .limits import prune_login_limits

log = logging.getLogger(__name__)


def cleanup_pending(storage_dir):
    root = Path(storage_dir).resolve()
    with connect() as conn:
        conn.execute("SET LOCAL lock_timeout = '1s'")
        rows = conn.execute("SELECT analysis_id FROM file_deletions ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 20").fetchall()
        for row in rows:
            analysis_id = UUID(str(row["analysis_id"]))
            folder = root / str(analysis_id)
            assert folder.parent == root
            try:
                if folder.is_symlink():
                    folder.unlink()  # Never follow a link outside the storage volume.
                elif folder.exists():
                    shutil.rmtree(folder)
            except OSError:
                log.exception("File deletion for analysis %s will be retried", analysis_id)
                continue
            conn.execute("DELETE FROM file_deletions WHERE analysis_id = %s", (analysis_id,))


@contextmanager
def cleanup_lifetime(storage_dir):
    stopped = Event()

    def watch():
        while not stopped.is_set():
            try:
                cleanup_pending(storage_dir)
                prune_login_limits()
            except Exception:
                log.exception("File cleanup failed; retrying")
            stopped.wait(2)

    thread = Thread(target=watch, name="file-cleanup", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=6)
