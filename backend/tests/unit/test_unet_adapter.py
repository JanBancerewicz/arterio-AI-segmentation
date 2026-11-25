import pytest

torch = pytest.importorskip("torch")
smp = pytest.importorskip("segmentation_models_pytorch")

import numpy as np
from PIL import Image

from app.inference import ModelLoadError
from app.unet import IMAGE_SIZE, MEAN, STD, UnetSegmenter


def network():
    return smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("checkpoint") / "unet.pth"
    torch.manual_seed(0)
    torch.save({"state_dict": network().state_dict(), "cfg": {"encoder": "resnet34", "image_size": 512}}, path)
    return path


@pytest.fixture(scope="module")
def segmenter(checkpoint):
    return UnetSegmenter(checkpoint)


def test_missing_checkpoint(tmp_path):
    with pytest.raises(ModelLoadError):
        UnetSegmenter(tmp_path / "missing.pth")


def test_checkpoint_without_state_dict(tmp_path):
    path = tmp_path / "raw.pth"
    torch.save({"weights": {}}, path)
    with pytest.raises(ModelLoadError, match="state_dict"):
        UnetSegmenter(path)


@pytest.mark.parametrize("cfg", [{"encoder": "resnet50"}, {"image_size": 256}])
def test_checkpoint_with_other_architecture(tmp_path, cfg):
    path = tmp_path / "other.pth"
    torch.save({"state_dict": {}, "cfg": cfg}, path)
    with pytest.raises(ModelLoadError, match="ResNet-34"):
        UnetSegmenter(path)


def test_checkpoint_with_mismatched_weights(tmp_path):
    path = tmp_path / "broken.pth"
    state = network().state_dict()
    state.pop(next(iter(state)))
    torch.save({"state_dict": state}, path)
    with pytest.raises(ModelLoadError):
        UnetSegmenter(path)


def test_data_parallel_checkpoint_is_accepted(tmp_path, checkpoint):
    path = tmp_path / "parallel.pth"
    state = torch.load(checkpoint, weights_only=True)["state_dict"]
    torch.save({"state_dict": {"module." + key: value for key, value in state.items()}}, path)
    UnetSegmenter(path)


def test_unavailable_cuda_device(checkpoint, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelLoadError, match="CUDA"):
        UnetSegmenter(checkpoint, device="cuda")


def test_preprocess_matches_training_input(segmenter, tmp_path):
    path = tmp_path / "frame.png"
    Image.new("L", (800, 600), 255).save(path)
    tensor, size = segmenter.preprocess(path)
    assert size == (800, 600)
    assert tensor.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE) and tensor.dtype == torch.float32
    expected = (1.0 - MEAN) / STD
    assert np.allclose(tensor[0, :, 0, 0].numpy(), expected, atol=1e-5)


def test_preprocess_rejects_undecodable_file(segmenter, tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        segmenter.preprocess(path)


def test_prediction_is_binary_mask(segmenter):
    mask = segmenter.predict(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert mask.shape == (IMAGE_SIZE, IMAGE_SIZE) and mask.dtype == np.uint8
    assert set(np.unique(mask)) <= {0, 255}


def test_prediction_is_deterministic(segmenter):
    tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert np.array_equal(segmenter.predict(tensor), segmenter.predict(tensor))


def test_threshold_controls_mask(checkpoint):
    tensor = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    low = UnetSegmenter(checkpoint, threshold=0.01).predict(tensor)
    high = UnetSegmenter(checkpoint, threshold=0.99).predict(tensor)
    assert (low > 0).sum() >= (high > 0).sum()


def test_restored_mask_keeps_original_size_and_values():
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    mask[100:200, 50:450] = 255
    restored = UnetSegmenter.restore_mask(mask, (1000, 700))
    assert restored.size == (1000, 700)
    assert set(np.unique(np.array(restored))) == {0, 255}
