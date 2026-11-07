"""Lightweight catalogue; importing the API must not import PyTorch."""
from dataclasses import dataclass

from . import config

UNET_ID = "unet-resnet34"


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    version: str
    description: str
    available: bool


def catalogue():
    return [
        ModelSpec(config.MODEL_ID, "Test infrastruktury — bez modelu AI", "0.1.0",
                  "Próbny worker progowania obrazu. Wynik nie jest segmentacją naczyń.", True),
        ModelSpec(UNET_ID, "U-Net (ResNet-34)", config.UNET_MODEL_VERSION,
                  "Segmentacja naczyń: U-Net, wejście 512×512, normalizacja ImageNet.",
                  config.UNET_CHECKPOINT_PATH.is_file()),
    ]


def find_model(model_id):
    return next((model for model in catalogue() if model.id == model_id), None)
