from io import BytesIO

from PIL import Image

from app import config, worker
from app.execution import JobExecutor
from app.inference import ModelLoadError, OutOfMemoryError


def upload_unet(client):
    image = BytesIO()
    Image.new("L", (256, 256), 128).save(image, format="PNG")
    return client.post("/api/v1/analyses", data={"modelId": "unet-resnet34"},
                       files={"file": ("unet.png", image.getvalue(), "image/png")})


def test_unet_without_checkpoint_unavailable(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UNET_CHECKPOINT_PATH", tmp_path / "missing.pth")
    models = client.get("/api/v1/models").json()
    unet = next(model for model in models if model["id"] == "unet-resnet34")
    assert unet["available"] is False and unet["isDefault"] is False
    assert sum(model["isDefault"] for model in models) == 1
    response = upload_unet(client)
    assert response.status_code == 503 and response.json()["error"]["code"] == "MODEL_UNAVAILABLE"
    assert client.get("/api/v1/analyses").json()["total"] == 0


def test_model_snapshot_does_not_follow_configuration(client, tmp_path, monkeypatch):
    path = tmp_path / "configured.pth"
    path.write_bytes(b"test preflight only; not trained weights")
    monkeypatch.setattr(config, "UNET_CHECKPOINT_PATH", path)
    monkeypatch.setattr(config, "UNET_MODEL_VERSION", "snapshot-1")
    models = client.get("/api/v1/models").json()
    assert next(model for model in models if model["id"] == "unet-resnet34")["isDefault"] is True
    response = upload_unet(client)
    assert response.status_code == 201
    analysis = response.json()
    assert analysis["modelName"] == "U-Net (ResNet-34)" and analysis["modelVersion"] == "snapshot-1"
    monkeypatch.setattr(config, "UNET_MODEL_VERSION", "snapshot-2")
    assert client.get(f"/api/v1/analyses/{analysis['id']}").json()["modelVersion"] == "snapshot-1"


def load_failure(row, worker_id):
    raise ModelLoadError("Synthetic load failure")


def memory_failure(row, worker_id):
    raise OutOfMemoryError("Synthetic out of memory")


def test_child_preserves_model_error_codes(client):
    from uuid import uuid4
    from test_pipeline import create
    for task, expected in ((load_failure, "MODEL_LOAD_FAILED"), (memory_failure, "OUT_OF_MEMORY")):
        analysis = create(client)
        worker_id = uuid4()
        row = worker.claim_job(worker_id)
        executor = JobExecutor(worker.STORAGE_DIR, task=task)
        try:
            worker.process_job(row, worker_id, executor=executor)
            result = client.get(f"/api/v1/analyses/{analysis['id']}").json()
            assert result["status"] == "failed" and result["errorCode"] == expected
        finally:
            executor.close()
