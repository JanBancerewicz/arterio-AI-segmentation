from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from uuid import uuid4

from PIL import Image, ImageDraw

from app import db, main, migrate, worker


def png():
    image = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(image).line((20, 250, 490, 250), fill=255, width=12)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create(client):
    response = client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                           files={"file": ("frame.png", png(), "image/png")})
    assert response.status_code == 201, response.text
    return response.json()


def test_pipeline_and_binary_mask(client):
    analysis = create(client)
    assert analysis["status"] == "queued"
    assert analysis["queuePosition"] == 1
    assert analysis["files"]["mask"] is None
    assert client.get(f"/api/v1/analyses/{analysis['id']}/files/mask").status_code == 409
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    assert str(row["id"]) == analysis["id"]
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["processingStage"] == "preprocessing"
    worker.process_job(row, worker_id)
    result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
    assert result["status"] == "completed"
    assert result["processingMs"] >= 0
    assert result["processingStage"] is None
    for kind in ("original", "mask", "overlay"):
        response = client.get(result["files"][kind])
        assert response.status_code == 200
        with Image.open(BytesIO(response.content)) as image:
            assert image.size == (512, 512)
            if kind == "mask":
                assert set(image.tobytes()) == {0, 255}
    assert client.post(f"/api/v1/analyses/{analysis['id']}/cancel").status_code == 409


def test_cancellation_and_queue_position(client):
    first, second = create(client), create(client)
    assert second["queuePosition"] == 2
    url = f"/api/v1/analyses/{first['id']}/cancel"
    assert client.post(url).json()["status"] == "cancelled"
    assert client.post(url).status_code == 200
    assert client.get(f"/api/v1/analyses/{second['id']}").json()["queuePosition"] == 1
    claimed = worker.claim_job(uuid4())
    assert str(claimed["id"]) == second["id"]
    assert client.post(f"/api/v1/analyses/{second['id']}/cancel").status_code == 409


def test_invalid_uploads_do_not_create_jobs(client, monkeypatch):
    for content, expected in ((b"", 422), (b"jpeg", 415), (b"\x89PNG\r\n\x1a\ninvalid", 422)):
        response = client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                               files={"file": ("image.png", content, "image/png")})
        assert response.status_code == expected
        assert response.json()["error"]["requestId"] == response.headers["X-Request-ID"]
    monkeypatch.setattr(main, "MAX_FILE_BYTES", 10)
    assert client.post("/api/v1/analyses", data={"modelId": "smoke-test"},
                       files={"file": ("big.png", png(), "image/png")}).status_code == 413
    assert client.get("/api/v1/analyses").json()["total"] == 0
    assert not list(main.STORAGE_DIR.iterdir())


def test_pagination_filters_and_repeat_migration(client):
    first, second = create(client), create(client)
    client.post(f"/api/v1/analyses/{first['id']}/cancel")
    migrate.main()  # Re-running initialization must retain data.
    page = client.get("/api/v1/analyses?pageSize=1").json()
    assert page["total"] == 2
    assert page["items"][0]["id"] == second["id"]
    assert client.get("/api/v1/analyses?status=cancelled").json()["total"] == 1
    assert client.get("/api/v1/analyses?status=invalid").status_code == 422
    assert client.get("/api/v1/analyses?page=10").json()["items"] == []


def test_two_workers_do_not_claim_same_job(client):
    jobs = {create(client)["id"], create(client)["id"]}
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed = list(executor.map(worker.claim_job, (uuid4(), uuid4())))
    assert {str(row["id"]) for row in claimed} == jobs
    assert worker.claim_job(uuid4()) is None


def test_cancel_and_claim_race(client):
    analysis = create(client)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claim = executor.submit(worker.claim_job, uuid4())
        cancel = executor.submit(client.post, f"/api/v1/analyses/{analysis['id']}/cancel")
        row, response = claim.result(), cancel.result()
    final = client.get(f"/api/v1/analyses/{analysis['id']}").json()
    if row:
        assert response.status_code == 409
        assert final["status"] == "processing"
    else:
        assert response.status_code == 200
        assert final["status"] == "cancelled"


def test_lost_job_times_out_and_cannot_publish_late_result(client):
    analysis = create(client)
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    with db.connect() as conn:
        conn.execute("UPDATE analyses SET started_at = clock_timestamp() - interval '121 seconds' WHERE id = %s", (row["id"],))
    assert worker.claim_job(uuid4()) is None
    worker.process_job(row, worker_id)
    result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
    assert result["status"] == "failed"
    assert result["errorCode"] == "TIMEOUT"
    assert result["files"]["mask"] is None


def test_worker_failure_is_analysis_state(client):
    analysis = create(client)
    worker_id = uuid4()
    row = worker.claim_job(worker_id)
    (main.STORAGE_DIR / analysis["id"] / "original.png").unlink()
    worker.process_job(row, worker_id)
    response = client.get(f"/api/v1/analyses/{analysis['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["errorCode"] == "INFERENCE_FAILED"


def test_missing_analysis_and_validation(client):
    assert client.get(f"/api/v1/analyses/{uuid4()}").status_code == 404
    assert client.get("/api/v1/analyses/not-a-uuid").status_code == 422
    assert client.get("/api/v1/health").json()["status"] == "degraded"
    assert client.get("/api/v1/models").json()[0]["id"] == "smoke-test"
    schema = client.get("/openapi.json").json()
    assert "modelId" in str(schema)
