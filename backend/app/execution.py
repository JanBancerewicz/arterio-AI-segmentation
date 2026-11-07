"""Reusable, isolated computation process. Only the supervisor publishes results."""
import logging
import multiprocessing
from pathlib import Path

from . import db

log = logging.getLogger(__name__)


def _execute_loop(connection, database_url, storage_dir, task):
    from . import worker

    db.DATABASE_URL = database_url
    worker.STORAGE_DIR = Path(storage_dir)
    run = task or worker.run_smoke_inference
    try:
        while True:
            row, worker_id = connection.recv()
            try:
                run(row, worker_id)
                connection.send("ok")
            except Exception:
                log.exception("Computation failed for analysis %s", row["id"])
                connection.send("error")
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class JobExecutor:
    def __init__(self, storage_dir, task=None):
        self.storage_dir = str(storage_dir)
        self.task = task
        self.process = None
        self.connection = None

    def submit(self, row, worker_id):
        if self.process is not None and not self.process.is_alive():
            self.close()
        if self.process is None:
            context = multiprocessing.get_context("spawn")
            self.connection, child_connection = context.Pipe()
            self.process = context.Process(
                target=_execute_loop,
                args=(child_connection, db.DATABASE_URL, self.storage_dir, self.task),
                name="arterio-computation", daemon=True,
            )
            try:
                self.process.start()
            except Exception:
                self.connection.close()
                self.process = self.connection = None
                raise
            finally:
                child_connection.close()
        self.connection.send((row, worker_id))

    def result(self, timeout):
        if self.connection.poll(timeout):
            try:
                return self.connection.recv()
            except EOFError as exc:
                raise ChildProcessError("Computation process exited") from exc
        if not self.process.is_alive():
            raise ChildProcessError("Computation process exited")
        return None

    def close(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            if self.process.is_alive():
                raise RuntimeError("Could not stop computation process")
            self.process.join()
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None
