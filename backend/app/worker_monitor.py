"""Database liveness checks shared by the API and worker supervisor."""
from contextlib import contextmanager
import logging
from threading import Event, Thread

from .config import WORKER_HEARTBEAT_SECONDS, WORKER_STALE_SECONDS
from .db import connect

log = logging.getLogger(__name__)


@contextmanager
def monitoring_connection():
    """Keep a database lock from blocking supervision indefinitely."""
    with connect() as conn:
        conn.execute("SET LOCAL statement_timeout = '2s'")
        conn.execute("SET LOCAL lock_timeout = '1s'")
        yield conn


def heartbeat(worker_id):
    with monitoring_connection() as conn:
        conn.execute(
            """INSERT INTO workers (id) VALUES (%s) ON CONFLICT (id)
               DO UPDATE SET heartbeat_at = clock_timestamp()""", (worker_id,),
        )


def recover_jobs():
    """A missing/stale supervisor cannot keep a job in processing indefinitely."""
    with monitoring_connection() as conn:
        rows = conn.execute(
            """UPDATE analyses AS a SET status = 'failed', error_code = 'WORKER_LOST',
               error_message = 'Utracono połączenie z workerem przetwarzającym analizę.',
               finished_at = clock_timestamp(), processing_stage = NULL
               WHERE a.status = 'processing' AND a.user_id IS NOT NULL
                 AND a.started_at < clock_timestamp() - (%s * interval '1 second')
                 AND NOT EXISTS (SELECT 1 FROM workers AS w WHERE w.id = a.worker_id
                     AND w.heartbeat_at > clock_timestamp() - (%s * interval '1 second'))
               RETURNING a.id""", (WORKER_STALE_SECONDS, WORKER_STALE_SECONDS),
        ).fetchall()
    for row in rows:
        log.warning("Lost worker for analysis %s", row["id"])
    return len(rows)


def fail_job(analysis_id, worker_id, code, message):
    with monitoring_connection() as conn:
        conn.execute(
            """UPDATE analyses SET status = 'failed', processing_stage = NULL,
               finished_at = clock_timestamp(), error_code = %s, error_message = %s
               WHERE id = %s AND worker_id = %s AND status = 'processing'""",
            (code, message, analysis_id, worker_id),
        )


def unregister(worker_id):
    with monitoring_connection() as conn:
        conn.execute(
            """UPDATE analyses SET status = 'failed', error_code = 'WORKER_LOST',
               error_message = 'Worker został zatrzymany podczas przetwarzania.',
               processing_stage = NULL, finished_at = clock_timestamp()
               WHERE worker_id = %s AND status = 'processing' AND user_id IS NOT NULL""",
            (worker_id,),
        )
        conn.execute("DELETE FROM workers WHERE id = %s", (worker_id,))


@contextmanager
def monitor_lifetime():
    stopped = Event()

    def watch():
        while not stopped.is_set():
            try:
                recover_jobs()
            except Exception:
                log.exception("Worker monitoring failed; retrying")
            stopped.wait(WORKER_HEARTBEAT_SECONDS)

    thread = Thread(target=watch, name="worker-monitor", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=6)
