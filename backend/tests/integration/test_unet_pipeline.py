from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
smp = pytest.importorskip("segmentation_models_pytorch")

from app import config, db, worker
from app.inference import ModelLoadError


@pytest.fixture
def unet_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "unet_random.pth"
    torch.manual_seed(0)
    network = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    torch.save({"state_dict": network.state_dict()}, path)
    monkeypatch.setattr(config, "UNET_CHECKPOINT_PATH", path)
    monkeypatch.setattr(config, "UNET_MODEL_VERSION", "random-weights")
    return path


def frame(size):
    image = Image.new("L", size, 40)
    ImageDraw.Draw(image).line((0, 0, size[0], size[1]), fill=220, width=15)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def queue_unet(client, size=(700, 400)):
    response = client.post("/api/v1/analyses", data={"modelId": "unet-resnet34"},
                           files={"file": ("angio.png", frame(size), "image/png")})
    assert response.status_code == 201, response.text
    worker_id = uuid4()
    return response.json(), worker.claim_job(worker_id), worker_id


def test_unet_pipeline_writes_mask_in_original_resolution(client, unet_checkpoint):
    analysis, row, worker_id = queue_unet(client)
    assert analysis["modelVersion"] == "random-weights"
    worker.run_inference(row, worker_id)
    folder = worker.STORAGE_DIR / analysis["id"]
    with Image.open(folder / "mask.png") as mask, Image.open(folder / "overlay.png") as overlay:
        assert mask.size == overlay.size == (700, 400)
        assert set(mask.tobytes()) <= {0, 255}
        assert overlay.mode == "RGB"
    with db.connect() as conn:
        stage = conn.execute("SELECT processing_stage FROM analyses WHERE id = %s", (analysis["id"],)).fetchone()
    assert stage["processing_stage"] == "postprocessing"


def test_worker_refuses_job_for_other_model_version(client, unet_checkpoint, monkeypatch):
    analysis, row, worker_id = queue_unet(client)
    monkeypatch.setattr(config, "UNET_MODEL_VERSION", "newer-weights")
    with pytest.raises(ModelLoadError, match="version"):
        worker.run_inference(row, worker_id)
    assert not (worker.STORAGE_DIR / analysis["id"] / "mask.png").exists()


def test_worker_refuses_unknown_model(client, unet_checkpoint):
    _, row, worker_id = queue_unet(client)
    with pytest.raises(ModelLoadError, match="Unknown"):
        worker.run_inference({**row, "model_id": "not-a-model"}, worker_id)
