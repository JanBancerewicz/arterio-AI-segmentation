"""
convert_mask.py
================
Convert ARCADE SYNTAX COCO polygons to binary vessel mask PNGs.

Merges all polygon categories into one foreground (255) / background (0) mask
per image. Optional overlay previews with --vis.

Usage:
    python convert_mask.py
    python convert_mask.py --vis 8 --data-root data
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Core function
# ──────────────────────────────────────────────────────────────

def convert_split(
    json_path: Path,
    images_dir: Path,
    masks_out_dir: Path,
    vis_dir: Path | None = None,
    n_vis: int = 0,
    seed: int = 42,
) -> dict:
    """
    Reads one COCO JSON (train / val / test), renders merged binary masks,
    optionally saves overlay visualizations.

    Returns dict with basic statistics for sanity-checking.
    """
    masks_out_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, encoding="utf-8") as f:
        coco = json.load(f)

    # Build image lookup
    images_meta = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    anns_by_image: dict[int, list] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Decide which images to visualize
    vis_ids: set[int] = set()
    if n_vis > 0 and vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(seed)
        vis_ids = set(rng.sample(list(images_meta.keys()), min(n_vis, len(images_meta))))

    stats = {
        "total": len(images_meta),
        "missing_image": 0,
        "empty_mask": 0,
        "fg_ratios": [],
    }

    for image_id, img_info in tqdm(images_meta.items(), leave=False):
        filename = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        img_path  = images_dir / filename
        mask_path = masks_out_dir / filename  # same filename as source image

        # ── Build binary mask ─────────────────────────────────
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns_by_image.get(image_id, []):
            # Each annotation can have multiple polygon segments
            # (e.g. a vessel hidden behind bone → two separate polygons)
            # We draw ALL of them onto the same mask → merge
            for seg in ann.get("segmentation", []):
                if len(seg) < 6:
                    # Degenerate polygon (< 3 points) — skip
                    continue
                pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(mask, [pts], color=255)

        # ── Save mask ─────────────────────────────────────────
        cv2.imwrite(str(mask_path), mask)

        # ── Stats ─────────────────────────────────────────────
        if not img_path.exists():
            stats["missing_image"] += 1
        fg = float(np.sum(mask > 0)) / (w * h)
        stats["fg_ratios"].append(fg)
        if fg == 0.0:
            stats["empty_mask"] += 1

        # ── Overlay visualization ──────────────────────────────
        if image_id in vis_ids and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                # Green overlay on vessel pixels
                overlay = img.copy()
                overlay[mask > 0] = (
                    overlay[mask > 0] * 0.5 + np.array([0, 180, 0]) * 0.5
                ).astype(np.uint8)

                # Side-by-side: original | overlay
                panel = np.hstack([img, overlay])

                fg_pct = fg * 100
                label = f"{filename}  vessel: {fg_pct:.1f}%  anns: {len(anns_by_image.get(image_id, []))}"
                cv2.putText(panel, label, (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 80), 1, cv2.LINE_AA)

                cv2.imwrite(str(vis_dir / f"overlay_{filename}"), panel)

    return stats


def print_stats(split_name: str, stats: dict) -> None:
    fg = stats["fg_ratios"]
    if not fg:
        return
    print(f"\n  {split_name}")
    print(f"    images        : {stats['total']}")
    print(f"    missing imgs  : {stats['missing_image']}")
    print(f"    empty masks   : {stats['empty_mask']}")
    print(f"    foreground %  : mean={np.mean(fg)*100:.2f}%  "
          f"min={np.min(fg)*100:.2f}%  max={np.max(fg)*100:.2f}%")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    root     = Path(args.data_root)
    syn_root = root / "syntax"
    mask_root = root / "masks"
    vis_root  = root / "mask_vis" if args.vis > 0 else None

    splits = [
        ("train", syn_root / "train" / "annotations" / "train.json",
                  syn_root / "train" / "images",
                  mask_root / "train"),
        ("val",   syn_root / "val"   / "annotations" / "val.json",
                  syn_root / "val"   / "images",
                  mask_root / "val"),
        ("test",  syn_root / "test"  / "annotations" / "test.json",
                  syn_root / "test"  / "images",
                  mask_root / "test"),
    ]

    print("ARCADE syntax → binary masks")
    print(f"Data root : {root.resolve()}")
    print(f"Output    : {mask_root.resolve()}")
    print()

    all_stats = {}
    for split_name, json_path, images_dir, masks_dir in splits:
        if not json_path.exists():
            print(f"  [{split_name}] JSON not found, skipping: {json_path}")
            continue

        print(f"  [{split_name}] processing...")
        vis_dir = (vis_root / split_name) if vis_root else None

        stats = convert_split(
            json_path    = json_path,
            images_dir   = images_dir,
            masks_out_dir= masks_dir,
            vis_dir      = vis_dir,
            n_vis        = args.vis,
            seed         = args.seed,
        )
        all_stats[split_name] = stats

    print("\n── Statistics ──────────────────────────────────")
    for split_name, stats in all_stats.items():
        print_stats(split_name, stats)

    if vis_root:
        print(f"\nVisualizations saved to: {vis_root}/")
        print("Open overlay_*.png to verify masks look correct before training.")

    print("\nDone. Next step: python train_unet.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data",
                        help="Root folder containing syntax/ subfolder")
    parser.add_argument("--vis", type=int, default=5,
                        help="Number of overlay images per split (0 = none)")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())