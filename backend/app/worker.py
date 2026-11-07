"""One job at a time; replace run_smoke_inference when adding a real model."""
import logging
import signal
from threading import Event
from time import monotonic
from uuid import uuid4

from PIL import Image, ImageOps

from .config import JOB_TIMEOUT_SECONDS, STORAGE_DIR
from .db import connect

log = logging.getLogger(__name__)
stop = Event()


def claim_job(worker_id):
    with connect() as conn:
        conn.execute(
            """INSERT INTO workers (id) VALUES (%s) ON CONFLICT (id)
               DO UPDATE SET heartbeat_at = clock_timestamp()""", (worker_id,),
        )
        conn.execute(
            """UPDATE analyses SET status = 'failed', error_code = 'TIMEOUT',
               error_message = 'Przetwarzanie zostało przerwane lub przekroczyło limit czasu.',
               finished_at = clock_timestamp(), processing_stage = NULL
               WHERE status = 'processing' AND user_id IS NOT NULL
               AND started_at < clock_timestamp() - (%s * interval '1 second')""",
            (JOB_TIMEOUT_SECONDS,),
        )
        return conn.execute(
            """UPDATE analyses SET status = 'processing', started_at = clock_timestamp(),
               processing_stage = 'preprocessing', worker_id = %s
               WHERE id = (SELECT id FROM analyses WHERE status = 'queued' AND user_id IS NOT NULL
                           ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING *""", (worker_id,),
        ).fetchone()


def set_stage(analysis_id, worker_id, stage):
    with connect() as conn:
        row = conn.execute(
            """UPDATE analyses SET processing_stage = %s
               WHERE id = %s AND worker_id = %s AND status = 'processing' RETURNING id""",
            (stage, analysis_id, worker_id),
        ).fetchone()
        if not row:
            raise RuntimeError("Job is no longer owned by this worker")


def run_smoke_inference(row, worker_id):
    """Synthetic brightness threshold, NOT a vessel segmentation algorithm."""
    folder = STORAGE_DIR / str(row["id"])
    with Image.open(folder / "original.png") as source:
        image = source.convert("RGB")
        grayscale = ImageOps.grayscale(image)
    set_stage(row["id"], worker_id, "inference")
    mask = grayscale.point(lambda value: 255 if value >= 128 else 0)
    set_stage(row["id"], worker_id, "postprocessing")
    tint = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.composite(Image.blend(image, tint, 0.5), image, mask)
    for kind, result in (("mask", mask), ("overlay", overlay)):
        temporary = folder / f"{kind}.tmp"
        result.save(temporary, format="PNG")
        temporary.replace(folder / f"{kind}.png")


def process_job(row, worker_id):
    started = monotonic()
    try:
        run_smoke_inference(row, worker_id)
        if monotonic() - started > JOB_TIMEOUT_SECONDS:
            raise TimeoutError("Job timeout")
        with connect() as conn:
            completed = conn.execute(
                """UPDATE analyses SET status = 'completed', processing_stage = NULL,
                   finished_at = clock_timestamp()
                   WHERE id = %s AND worker_id = %s AND status = 'processing' RETURNING id""",
                (row["id"], worker_id),
            ).fetchone()
        if completed:
            log.info("Completed %s", row["id"])
    except Exception as exc:
        log.exception("Job %s failed", row["id"])
        code = "TIMEOUT" if isinstance(exc, TimeoutError) else "INFERENCE_FAILED"
        with connect() as conn:
            conn.execute(
                """UPDATE analyses SET status = 'failed', processing_stage = NULL,
                   finished_at = clock_timestamp(), error_code = %s,
                   error_message = 'Próbny worker nie ukończył przetwarzania.'
                   WHERE id = %s AND worker_id = %s AND status = 'processing'""",
                (code, row["id"], worker_id),
            )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker_id = uuid4()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("Starting SMOKE-TEST worker %s (no AI model)", worker_id)
    try:
        while not stop.is_set():
            try:
                row = claim_job(worker_id)
                if row:
                    process_job(row, worker_id)
                else:
                    stop.wait(0.5)
            except Exception:
                log.exception("Worker iteration failed; retrying")
                stop.wait(2)
    finally:
        with connect() as conn:
            conn.execute("DELETE FROM workers WHERE id = %s", (worker_id,))


if __name__ == "__main__":
    main()
