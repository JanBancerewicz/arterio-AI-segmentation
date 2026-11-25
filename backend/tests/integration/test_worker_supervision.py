from concurrent.futures import ThreadPoolExecutor
import os
import signal
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

from app import db, main, worker, worker_monitor
from app.execution import JobExecutor
from test_pipeline import create


def hanging_task(row, worker_id):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (worker.STORAGE_DIR / str(row["id"]) / "started").write_text("started")
    sleep(30)
    (worker.STORAGE_DIR / str(row["id"]) / "late-result").write_text("unexpected")


def slow_task(row, worker_id):
    sleep(12)
    worker.run_smoke_inference(row, worker_id)


def crashing_task(row, worker_id):
    os._exit(17)


def test_real_hang_is_killed_and_next_job_completes(client, monkeypatch):
    monkeypatch.setattr(worker, "JOB_TIMEOUT_SECONDS", 2)
    analysis = create(client)
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    executor = JobExecutor(main.STORAGE_DIR, task=hanging_task)
    started = monotonic()
    worker.process_job(row, worker_id, executor=executor)
    assert monotonic() - started < 10
    assert executor.process is None
    folder = main.STORAGE_DIR / analysis["id"]
    assert (folder / "started").is_file()  # The task ran, not just interpreter startup.
    assert not (folder / "late-result").exists()
    result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
    assert result["status"] == "failed" and result["errorCode"] == "TIMEOUT"
    assert result["files"]["mask"] is None
    # A replacement computation process can run the following queued task.
    executor.task = None
    next_analysis = create(client)
    try:
        worker.process_job(worker.claim_job(worker_id), worker_id, executor=executor)
        assert client.get(f"/api/v1/analyses/{next_analysis['id']}").json()["status"] == "completed"
    finally:
        executor.close()


def test_heartbeat_continues_during_computation(client):
    # Exercise the real 2s/10s settings instead of a scheduler-sensitive 1s cutoff.
    analysis = create(client)
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    executor = JobExecutor(main.STORAGE_DIR, task=slow_task)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(worker.process_job, row, worker_id, executor)
            while not future.done():
                sleep(0.3)
                assert worker_monitor.recover_jobs() == 0
                assert client.get("/api/v1/health").json()["workersOnline"] >= 1
            future.result()
        assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["status"] == "completed"
    finally:
        executor.close()


def test_api_detects_lost_worker_without_another_worker(client):
    analysis = create(client)
    queued = create(client)
    worker_id = uuid4()
    worker.claim_job(worker_id)
    with db.connect() as conn:
        conn.execute("UPDATE workers SET heartbeat_at = clock_timestamp() - interval '11 seconds'")
        conn.execute("UPDATE analyses SET started_at = clock_timestamp() - interval '11 seconds' WHERE id = %s", (analysis["id"],))
    deadline = monotonic() + 5
    while monotonic() < deadline:
        result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
        if result["status"] == "failed":
            break
        sleep(0.1)
    assert result["status"] == "failed" and result["errorCode"] == "WORKER_LOST"
    assert client.get(f"/api/v1/analyses/{queued['id']}").json()["status"] == "queued"
    # Late supervisor activity must not resurrect the failed analysis.
    worker.process_job({"id": row_id(analysis)}, worker_id)
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["errorCode"] == "WORKER_LOST"


def row_id(analysis):
    from uuid import UUID
    return UUID(analysis["id"])


def test_computation_crash_and_supervisor_stop(client):
    for task, stopped, expected in ((crashing_task, Event(), "INFERENCE_FAILED"), (slow_task, Event(), "WORKER_LOST")):
        if expected == "WORKER_LOST":
            stopped.set()
        analysis = create(client)
        worker_id = uuid4()
        row = worker.claim_job(worker_id)
        executor = JobExecutor(main.STORAGE_DIR, task=task)
        worker.process_job(row, worker_id, executor=executor, stop_event=stopped)
        assert executor.process is None
        result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
        assert result["status"] == "failed" and result["errorCode"] == expected


def test_computation_process_reused_and_unregister_is_immediate(client):
    worker_id = uuid4()
    executor = JobExecutor(main.STORAGE_DIR)
    try:
        pids = []
        for _ in range(2):
            analysis = create(client)
            worker.process_job(worker.claim_job(worker_id), worker_id, executor=executor)
            pids.append(executor.process.pid)
            assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["status"] == "completed"
        assert pids[0] == pids[1]
    finally:
        executor.close()
    analysis = create(client)
    worker.claim_job(worker_id)
    worker_monitor.unregister(worker_id)
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["errorCode"] == "WORKER_LOST"
    assert client.get("/api/v1/health").json()["workersOnline"] == 0
