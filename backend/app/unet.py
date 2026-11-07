"""U-Net adapter matching train_unet.py and augmentations.py validation input."""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch

from .inference import ModelLoadError, OutOfMemoryError

MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
IMAGE_SIZE = 512


class UnetSegmenter:
    def __init__(self, checkpoint_path, device="cpu", threshold=0.5):
        self.threshold = threshold
        self.device = torch.device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(2)
        try:
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise ModelLoadError("Configured CUDA device is unavailable")
            checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise ModelLoadError("Expected a checkpoint produced by train_unet.py with state_dict")
            cfg = checkpoint.get("cfg", {})
            if cfg.get("encoder", "resnet34") != "resnet34" or cfg.get("image_size", IMAGE_SIZE) != IMAGE_SIZE:
                raise ModelLoadError("Checkpoint must describe ResNet-34 U-Net with 512x512 input")
            self.model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                                  in_channels=3, classes=1, activation=None)
            state = checkpoint["state_dict"]
            # Also accept checkpoints saved directly from DataParallel.
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)
            self.model.to(self.device).eval()
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError("Insufficient GPU memory while loading U-Net") from exc
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError("Cannot load the configured U-Net checkpoint") from exc

    def preprocess(self, image_path):
        # The training dataset reads 8-bit grayscale with OpenCV, then replicates RGB.
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError("Cannot decode inference input")
        size = (gray.shape[1], gray.shape[0])
        resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)
        normalized = (rgb / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
        return tensor, size

    def predict(self, tensor):
        try:
            with torch.inference_mode():
                probabilities = self.model(tensor.to(self.device)).sigmoid()
                mask = (probabilities[0, 0] > self.threshold).to(torch.uint8).cpu().numpy() * 255
            return mask
        except torch.cuda.OutOfMemoryError as exc:
            raise OutOfMemoryError("Insufficient GPU memory during U-Net inference") from exc

    @staticmethod
    def restore_mask(mask, size):
        return Image.fromarray(mask).resize(size, Image.Resampling.NEAREST)


@lru_cache(maxsize=1)
def get_segmenter(checkpoint_path, version, device, threshold):
    # A new checkpoint/version requires restarting the computation process.
    return UnetSegmenter(checkpoint_path, device=device, threshold=threshold)
