"""
dataset.py
===========
PyTorch Dataset for ARCADE syntax coronary vessel segmentation.

Used by train_unet.py, train_qwen_seg_new.py, evaluate.py.

Expected layout (after convert_mask.py):
    data/syntax/{train,val,test}/images/*.png
    data/masks/{train,val,test}/*.png

Usage:
    from dataset import ArcadeDataset, get_loaders
    from augmentations import get_transforms

    train_loader, val_loader, test_loader = get_loaders(
        data_root="data", batch_size=8, num_workers=4,
    )
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from augmentations import get_transforms


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class ArcadeDataset(Dataset):
    """
    Loads (XCA image, binary vessel mask) pairs from ARCADE syntax dataset.

    Images are grayscale coronary angiography frames replicated to 3 channels
    so they are compatible with ImageNet-pretrained encoders (U-Net/Qwen3-VL).

    Masks are single-channel binary PNGs: 255 = vessel, 0 = background.
    Returned masks are float32 tensors with values 0.0 / 1.0.

    Args:
        split      : "train", "val", or "test"
        data_root  : root folder containing syntax/ and masks/ subfolders
        transform  : albumentations Compose pipeline (from augmentations.py)
                     If None, only resizes and normalizes.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        split: str,
        data_root: str = "data",
        transform=None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")

        self.split     = split
        self.transform = transform or get_transforms("val")  # safe default
        root           = Path(data_root)

        img_dir  = root / "syntax" / split / "images"
        mask_dir = root / "masks"  / split

        self._validate_dirs(img_dir, mask_dir, split)

        # Build list of (image_path, mask_path) pairs
        # Only include pairs where BOTH files exist
        self.pairs: list[tuple[Path, Path]] = []
        skipped = 0

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))
            else:
                skipped += 1

        if skipped:
            print(f"[{split}] WARNING: {skipped} images have no matching mask — skipped.")
            print(f"         Run convert_masks.py to generate missing masks.")

        if not self.pairs:
            raise RuntimeError(
                f"No valid image-mask pairs found for split='{split}'.\n"
                f"  images dir : {img_dir}\n"
                f"  masks dir  : {mask_dir}\n"
                "Run convert_masks.py first."
            )

        print(f"[{split}] {len(self.pairs)} pairs ready.")

    # ── Core interface ────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        img  = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        aug  = self.transform(image=img, mask=mask)

        return {
            "image": aug["image"],               # torch.Tensor (3, H, W) float32
            "mask":  aug["mask"].unsqueeze(0),    # torch.Tensor (1, H, W) float32  {0, 1}
            "stem":  img_path.stem,               # filename without extension (for saving results)
        }

    # ── Private helpers ───────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """
        Loads a grayscale XCA frame and converts to 3-channel RGB.
        ImageNet-pretrained encoders expect 3 channels.
        """
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # (H, W, 3) uint8

    @staticmethod
    def _load_mask(path: Path) -> np.ndarray:
        """
        Loads a binary mask PNG (0/255) and normalizes to float32 (0.0/1.0).
        Thresholds at 127 to handle any JPEG-style compression artifacts.
        """
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        return (mask > 127).astype(np.float32)  # (H, W) float32

    @staticmethod
    def _validate_dirs(img_dir: Path, mask_dir: Path, split: str) -> None:
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                f"Check --data-root and ARCADE dataset structure."
            )
        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Mask directory not found: {mask_dir}\n"
                f"Run: python convert_masks.py"
            )

    # ── Utility ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Computes foreground (vessel) pixel ratio across the dataset.
        Useful for verifying class imbalance before training.
        Reads masks from disk — call once, not during training.
        """
        ratios = []
        for _, mask_path in self.pairs:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                ratios.append(np.mean(mask > 127))

        return {
            "n_images":      len(self.pairs),
            "fg_mean_pct":   float(np.mean(ratios) * 100),
            "fg_min_pct":    float(np.min(ratios) * 100),
            "fg_max_pct":    float(np.max(ratios) * 100),
            "fg_std_pct":    float(np.std(ratios) * 100),
        }


# ──────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────

def get_loaders(
    data_root:   str = "data",
    batch_size:  int = 8,
    num_workers: int = 4,
    pin_memory:  bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Creates train / val / test DataLoaders in one call.

    Returns:
        (train_loader, val_loader, test_loader)
        test_loader is None if test split is not available.

    Example:
        train_loader, val_loader, test_loader = get_loaders(
            data_root="data", batch_size=8
        )
    """
    train_ds = ArcadeDataset("train", data_root, transform=get_transforms("train"))
    val_ds   = ArcadeDataset("val",   data_root, transform=get_transforms("val"))

    try:
        test_ds = ArcadeDataset("test", data_root, transform=get_transforms("val"))
    except (FileNotFoundError, RuntimeError):
        print("[test] Split not available — test_loader will be None.")
        test_ds = None

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs) if test_ds else None

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────
# Quick sanity check — run directly to verify everything works
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this before training to verify:
      1. Dataset loads correctly
      2. Tensor shapes are right
      3. Mask foreground % is reasonable (expect 5-20%)
      4. Augmentations work without errors

    Usage:
        python dataset.py
        python dataset.py --data-root /other/path
    """
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="dataset_check.png",
                        help="Output path for sample visualization")
    args = parser.parse_args()

    print("=" * 50)
    print("Dataset sanity check")
    print("=" * 50)

    # Load all splits
    train_loader, val_loader, test_loader = get_loaders(
        data_root=args.data_root, batch_size=4, num_workers=0
    )

    # Print stats for each split
    for split in ("train", "val", "test"):
        try:
            ds = ArcadeDataset(split, args.data_root)
            stats = ds.get_stats()
            print(f"\n[{split}]")
            print(f"  images       : {stats['n_images']}")
            print(f"  vessel px %  : {stats['fg_mean_pct']:.2f}% avg  "
                  f"(min {stats['fg_min_pct']:.2f}%  max {stats['fg_max_pct']:.2f}%)")
        except Exception as e:
            print(f"[{split}] skipped — {e}")

    # Check one batch shape
    print("\nChecking batch shapes from train_loader...")
    batch = next(iter(train_loader))
    img  = batch["image"]
    mask = batch["mask"]
    print(f"  image shape  : {tuple(img.shape)}   dtype={img.dtype}")
    print(f"  mask shape   : {tuple(mask.shape)}   dtype={mask.dtype}")
    print(f"  mask values  : min={mask.min():.1f}  max={mask.max():.1f}")
    print(f"  stems        : {batch['stem']}")

    assert img.shape[1]  == 3, "Image should have 3 channels"
    assert mask.shape[1] == 1, "Mask should have 1 channel"
    assert img.dtype  == torch.float32, "Image should be float32"
    assert mask.dtype == torch.float32, "Mask should be float32"
    print("\nAll assertions passed.")

    # Save visualization of first 4 samples
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(4):
        # Denormalize image
        img_np = img[i].numpy().transpose(1, 2, 0)
        img_np = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        mask_np = mask[i, 0].numpy()

        axes[0, i].imshow(img_gray, cmap="gray")
        axes[0, i].set_title(f"XCA — {batch['stem'][i]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"mask  fg={mask_np.mean()*100:.1f}%")
        axes[1, i].axis("off")

    plt.suptitle("Dataset check — top: XCA image, bottom: binary vessel mask", y=1.01)
    plt.tight_layout()
    plt.savefig(args.out, bbox_inches="tight", dpi=100)
    print(f"\nVisualization saved → {args.out}")
    print("Open it to visually confirm images and masks align correctly.")