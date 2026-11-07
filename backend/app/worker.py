"""One supervised job at a time, with a reusable computation subprocess."""
import logging
import signal
from datetime import datetime, timezone
from threading import Event
from time import monotonic
from uuid import uuid4

from PIL import Image, ImageOps

from . import config
from .config import JOB_TIMEOUT_SECONDS, STORAGE_DIR, WORKER_HEARTBEAT_SECONDS
from .execution import ComputationFailed, JobExecutor
from .inference import ModelLoadError
from .models import UNET_ID
from .worker_monitor import fail_job, heartbeat, monitoring_connection, recover_jobs, unregister

log = logging.getLogger(__name__)
stop = Event()


def claim_job(worker_id):
    heartbeat(worker_id)
    recover_jobs()
    with monitoring_connection() as conn:
        return conn.execute(
            """UPDATE analyses SET status = 'processing', started_at = clock_timestamp(),
               processing_stage = 'preprocessing', worker_id = %s
               WHERE id = (SELECT id FROM analyses WHERE status = 'queued' AND user_id IS NOT NULL
                           ORDER BY created_at, id FOR UPDATE SKIP LOCKED LIMIT 1)
               RETURNING *""", (worker_id,),
        ).fetchone()


def set_stage(analysis_id, worker_id, stage):
    with monitoring_connection() as conn:
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
    save_result(folder, image, mask)


def save_result(folder, image, mask):
    tint = Image.new("RGB", image.size, (255, 0, 0))
    overlay = Image.composite(Image.blend(image, tint, 0.5), image, mask)
    for kind, result in (("mask", mask), ("overlay", overlay)):
        temporary = folder / f"{kind}.tmp"
        result.save(temporary, format="PNG")
        temporary.replace(folder / f"{kind}.png")


def run_inference(row, worker_id):
    if row["model_id"] == config.MODEL_ID:
        return run_smoke_inference(row, worker_id)
    if row["model_id"] != UNET_ID:
        raise ModelLoadError("Unknown model assigned to worker")
    if row["model_version"] != config.UNET_MODEL_VERSION:
        raise ModelLoadError("Queued model version differs from the worker configuration")
    try:
        from .unet import get_segmenter
    except ImportError as exc:
        raise ModelLoadError("U-Net dependencies are missing from the worker image") from exc
    folder = STORAGE_DIR / str(row["id"])
    segmenter = get_segmenter(str(config.UNET_CHECKPOINT_PATH), row["model_version"],
                              config.UNET_DEVICE, config.UNET_THRESHOLD)
    tensor, size = segmenter.preprocess(folder / "original.png")
    set_stage(row["id"], worker_id, "inference")
    mask = segmenter.predict(tensor)
    set_stage(row["id"], worker_id, "postprocessing")
    mask = segmenter.restore_mask(mask, size)
    with Image.open(folder / "original.png") as source:
        image = source.convert("RGB")
    save_result(folder, image, mask)


def owned_job(analysis_id, worker_id):
    with monitoring_connection() as conn:
        return conn.execute(
            "SELECT started_at FROM analyses WHERE id = %s AND worker_id = %s AND status = 'processing'",
            (analysis_id, worker_id),
        ).fetchone()


class WorkerStopped(Exception):
    pass


def process_job(row, worker_id, executor=None, stop_event=None):
    owns_executor = executor is None
    executor = executor or JobExecutor(STORAGE_DIR)
    stop_event = stop_event if stop_event is not None else stop
    try:
        current = owned_job(row["id"], worker_id)
        if not current:
            return
        elapsed = max(0, (datetime.now(timezone.utc) - current["started_at"]).total_seconds())
        deadline = monotonic() + max(0, JOB_TIMEOUT_SECONDS - elapsed)
        heartbeat(worker_id)
        next_heartbeat = monotonic() + WORKER_HEARTBEAT_SECONDS
        if monotonic() >= deadline:
            raise TimeoutError("Job timeout")
        if stop_event.is_set():
            raise WorkerStopped()
        executor.submit(row, worker_id)
        while True:
            if stop_event.is_set():
                raise WorkerStopped()
            if monotonic() >= deadline:
                raise TimeoutError("Job timeout")
            if monotonic() >= next_heartbeat:
                heartbeat(worker_id)
                if not owned_job(row["id"], worker_id):
                    executor.close()
                    return
                next_heartbeat = monotonic() + WORKER_HEARTBEAT_SECONDS
            result = executor.result(min(0.2, max(0, deadline - monotonic())))
            if isinstance(result, tuple) and result[0] == "error":
                raise ComputationFailed(result[1])
            if result == "ok":
                break
        if stop_event.is_set():
            raise WorkerStopped()
        if monotonic() >= deadline:
            raise TimeoutError("Job timeout")
        with monitoring_connection() as conn:
            completed = conn.execute(
                """UPDATE analyses SET status = 'completed', processing_stage = NULL,
                   finished_at = clock_timestamp()
                   WHERE id = %s AND worker_id = %s AND status = 'processing'
                     AND started_at > clock_timestamp() - (%s * interval '1 second')
                   RETURNING id""",
                (row["id"], worker_id, JOB_TIMEOUT_SECONDS),
            ).fetchone()
        if completed:
            log.info("Completed %s", row["id"])
        else:
            fail_job(row["id"], worker_id, "TIMEOUT", "Przekroczono limit czasu przetwarzania.")
    except Exception as exc:
        executor.close()  # Kill computation before marking a terminal state.
        if isinstance(exc, TimeoutError):
            code, message = "TIMEOUT", "Przekroczono limit czasu przetwarzania."
        elif isinstance(exc, WorkerStopped):
            code, message = "WORKER_LOST", "Worker został zatrzymany podczas przetwarzania."
        elif isinstance(exc, ComputationFailed):
            code = exc.code
            message = {"MODEL_LOAD_FAILED": "Nie udało się załadować skonfigurowanego modelu.",
                       "OUT_OF_MEMORY": "Zabrakło pamięci GPU podczas przetwarzania.",
                       "INFERENCE_FAILED": "Proces obliczeniowy nie ukończył przetwarzania."}[code]
        elif isinstance(exc, ChildProcessError):
            code, message = "INFERENCE_FAILED", "Proces obliczeniowy nie ukończył przetwarzania."
        else:
            code, message = "WORKER_LOST", "Przerwano nadzór nad przetwarzaniem analizy."
        log.warning("Job %s failed: %s", row["id"], code, exc_info=not isinstance(exc, (TimeoutError, WorkerStopped)))
        fail_job(row["id"], worker_id, code, message)
    finally:
        if owns_executor:
            executor.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker_id = uuid4()
    stop.clear()
    executor = JobExecutor(STORAGE_DIR)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    log.info("Starting worker %s (smoke-test and optional U-Net)", worker_id)
    try:
        while not stop.is_set():
            try:
                row = claim_job(worker_id)
                if row:
                    process_job(row, worker_id, executor=executor)
                else:
                    stop.wait(0.5)
            except Exception:
                log.exception("Worker iteration failed; retrying")
                stop.wait(2)
    finally:
        executor.close()
        unregister(worker_id)


if __name__ == "__main__":
    main()
