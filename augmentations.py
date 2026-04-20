"""
augmentations.py
=================
Augmentation pipelines for X-ray coronary angiography (XCA) images.

Design choices for medical XCA images:
  - Geometric transforms: safe — vessels can appear at any angle/position
  - Elastic deformation: moderate — simulates slight patient movement
  - Brightness/contrast: yes — contrast agent concentration varies between patients
  - Gaussian noise: yes — X-ray noise is realistic
  - Color jitter / saturation: NO — XCA images are grayscale
  - Heavy crops: NO — thin vessels could be completely cropped out
  - Coarse dropout: small only — don't remove large vessel regions

Usage:
    from augmentations import get_transforms

    train_transform = get_transforms("train")
    val_transform   = get_transforms("val")

    # With albumentations:
    augmented = train_transform(image=img_np, mask=mask_np)
    image = augmented["image"]   # torch.Tensor (3, H, W)
    mask  = augmented["mask"]    # torch.Tensor (H, W)
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization — used because encoders (ResNet, ViT) are pretrained on ImageNet
# XCA images are grayscale but replicated to 3 channels, so same stats apply
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE = 512  # ARCADE native resolution


def get_transforms(split: str) -> A.Compose:
    """
    Returns albumentations Compose pipeline for the given split.

    Args:
        split: "train", "val", or "test"

    Returns:
        A.Compose object — call as transform(image=np_array, mask=np_array)
        Output image is a normalized float32 tensor (3, H, W).
        Output mask is a float32 tensor (H, W) with values 0.0 or 1.0.
    """
    if split == "train":
        return _train_transforms()
    else:
        return _val_test_transforms()


# ──────────────────────────────────────────────────────────────
# Train pipeline
# ──────────────────────────────────────────────────────────────

def _train_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # ── Geometric ─────────────────────────────────────────
        # Horizontal flip: safe — no anatomical constraint in 2D projection
        A.HorizontalFlip(p=0.5),

        # Vertical flip: less common but valid for XCA projections
        A.VerticalFlip(p=0.2),

        # Rotation up to 20 degrees — XCA images are acquired at various angles
        A.Rotate(limit=20, border_mode=0, p=0.5),

        # Slight zoom/shift — simulates different patient positioning
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=0,     # rotation already handled above
            border_mode=0,
            p=0.4,
        ),

        # Elastic deformation — simulates slight cardiac motion between frames
        # Keep alpha low — too much distorts thin vessel topology
        A.ElasticTransform(
            alpha=30,
            sigma=5,
            p=0.3,
        ),

        # ── Photometric ───────────────────────────────────────
        # Brightness/contrast: contrast agent concentration varies per patient
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.25,
            p=0.6,
        ),

        # Gamma correction: simulates different X-ray exposure settings
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),

        # CLAHE: commonly used in XCA preprocessing — augmenting with it
        # teaches the model to handle both enhanced and raw images
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # ── Noise ─────────────────────────────────────────────
        # Gaussian noise: realistic X-ray noise simulation
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),

        # Slight blur: simulates slight motion blur from heartbeat
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

        # ── Regularization ────────────────────────────────────
        # Small random rectangle dropout — forces model not to rely on
        # a single region; keep max_holes small to avoid erasing vessels
        A.CoarseDropout(
            max_holes=4,
            max_height=20,
            max_width=20,
            min_holes=1,
            min_height=5,
            min_width=5,
            fill_value=0,
            mask_fill_value=0,  # dropped region → background in mask too
            p=0.2,
        ),

        # ── Normalize + Tensorize ─────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
# Val / Test pipeline — no random transforms
# ──────────────────────────────────────────────────────────────

def _val_test_transforms() -> A.Compose:
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
# Quick visual test — run this file directly to preview augmentations
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Preview augmentations on a single image + mask pair.
    Usage:
        python augmentations.py --image data/syntax/train/images/676.png \
                                --mask  data/masks/train/676.png \
                                --out   aug_preview.png \
                                --n     8
    """
    import argparse
    import cv2
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to source XCA image")
    parser.add_argument("--mask",  required=True, help="Path to binary mask PNG")
    parser.add_argument("--out",   default="aug_preview.png")
    parser.add_argument("--n",     type=int, default=8,
                        help="Number of augmented versions to generate")
    args = parser.parse_args()

    img  = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(args.mask,  cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print("Could not read image or mask — check paths.")
        exit(1)

    # Convert grayscale to 3-channel RGB for encoder compatibility
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    transform = _train_transforms()

    rows = []
    for i in range(args.n):
        aug = transform(image=img_rgb, mask=(mask > 127).astype(np.float32))

        # Denormalize image for display
        img_t = aug["image"].numpy().transpose(1, 2, 0)  # (H,W,3)
        mean  = np.array(IMAGENET_MEAN)
        std   = np.array(IMAGENET_STD)
        img_t = ((img_t * std + mean) * 255).clip(0, 255).astype(np.uint8)
        img_t = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)

        mask_t = (aug["mask"].numpy() * 255).astype(np.uint8)

        # Side-by-side: image | mask
        pair = np.hstack([img_t, mask_t])
        rows.append(pair)

    # Stack all versions vertically
    grid = np.vstack(rows)
    cv2.imwrite(args.out, grid)
    print(f"Saved {args.n} augmented pairs → {args.out}")
    print("Left column = augmented image, Right column = augmented mask")