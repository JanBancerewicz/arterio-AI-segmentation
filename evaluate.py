"""
evaluate.py
============
Evaluation script for coronary vessel segmentation.

Computes all metrics needed for the thesis results table:
    - Dice Score      — primary metric for binary segmentation
    - IoU             — Jaccard index (stricter than Dice)
    - Pixel Accuracy  — fraction of correct pixels
    - Precision       — TP / (TP + FP)
    - Recall          — TP / (TP + FN)  [sensitivity]
    - clDice          — centerline Dice, topology-aware metric for tubular
                        structures. Penalises predictions that "cut" a vessel
                        even when volumetric overlap is high.
                        Reference: Shit et al., CVPR 2021, arXiv:2003.07311

Supports three evaluation modes:
    1. U-Net checkpoint (from train_unet.py)
    2. Qwen3-VL checkpoint (from train_qwen_seg.py)
    3. Folder of pre-saved prediction PNGs (model-agnostic)

Usage:
    # Evaluate U-Net checkpoint on test split
    python evaluate.py --model unet \
                       --checkpoint checkpoints/unet_best.pth \
                       --split test

    # Evaluate Qwen frozen encoder
    python evaluate.py --model qwen \
                       --checkpoint checkpoints/qwen_seg_best.pth \
                       --split test

    # Evaluate from pre-saved prediction PNGs
    python evaluate.py --pred-dir results/my_predictions \
                       --mask-dir data/masks/test

    # Compare all models and print thesis table
    python evaluate.py --compare
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import skeletonize
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def dice_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Dice = 2|P∩G| / (|P|+|G|)
    Both inputs: uint8 arrays (0/255).
    """
    p = (pred > 127).astype(np.float32).ravel()
    g = (gt   > 127).astype(np.float32).ravel()
    return float((2.0 * (p * g).sum() + smooth) / (p.sum() + g.sum() + smooth))


def iou_score(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """IoU = |P∩G| / |P∪G|"""
    p = (pred > 127).astype(np.float32).ravel()
    g = (gt   > 127).astype(np.float32).ravel()
    inter = (p * g).sum()
    return float((inter + smooth) / (p.sum() + g.sum() - inter + smooth))


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of correctly labelled pixels (TP+TN / total)."""
    return float(np.mean((pred > 127) == (gt > 127)))


def precision_recall(
    pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6
) -> tuple[float, float]:
    p  = (pred > 127).astype(np.float32).ravel()
    g  = (gt   > 127).astype(np.float32).ravel()
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    prec = float((tp + smooth) / (tp + fp + smooth))
    rec  = float((tp + smooth) / (tp + fn + smooth))
    return prec, rec


def cl_dice(pred: np.ndarray, gt: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Centerline Dice — topology-aware metric for tubular structures.

    Standard Dice rewards volumetric overlap but ignores connectivity.
    A model can score Dice=0.80 while severing vessels in two places,
    which is clinically unacceptable. clDice penalises this by checking
    whether predicted and ground-truth centrelines are mutually covered:

        clPrecision = |skel(pred) ∩ GT|   / |skel(pred)|
        clRecall    = |skel(GT)   ∩ pred| / |skel(GT)|
        clDice      = 2 · clPrec · clRec  / (clPrec + clRec)

    Reference:
        Shit S. et al. "clDice - a Novel Topology-Preserving Loss Function
        for Tubular Structure Segmentation." CVPR 2021. arXiv:2003.07311
    """
    pred_bin = (pred > 127)
    gt_bin   = (gt   > 127)

    # Edge cases — empty masks
    if not pred_bin.any() and not gt_bin.any():
        return 1.0
    if not pred_bin.any() or not gt_bin.any():
        return 0.0

    skel_pred = skeletonize(pred_bin)
    skel_gt   = skeletonize(gt_bin)

    cl_prec = float((skel_pred & gt_bin).sum() + smooth) / float(skel_pred.sum() + smooth)
    cl_rec  = float((skel_gt & pred_bin).sum() + smooth) / float(skel_gt.sum()   + smooth)

    return float(2.0 * cl_prec * cl_rec / (cl_prec + cl_rec + smooth))


def compute_all(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """All metrics for one image pair. Both inputs: uint8 (0/255)."""
    prec, rec = precision_recall(pred, gt)
    return {
        "dice":      dice_score(pred, gt),
        "iou":       iou_score(pred, gt),
        "pixel_acc": pixel_accuracy(pred, gt),
        "precision": prec,
        "recall":    rec,
        "cl_dice":   cl_dice(pred, gt),
    }


def aggregate(per_image: dict[str, list[float]]) -> dict[str, dict]:
    """Returns mean ± std for each metric."""
    out = {}
    for k, vals in per_image.items():
        arr = np.array(vals)
        out[k] = {
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "min":  float(arr.min()),
            "max":  float(arr.max()),
        }
    return out


# ──────────────────────────────────────────────────────────────
# Evaluation from prediction PNG folder (model-agnostic)
# ──────────────────────────────────────────────────────────────

def eval_from_folders(
    pred_dir:  Path,
    mask_dir:  Path,
    vis_dir:   Path | None = None,
) -> dict:
    """
    Evaluates PNG predictions against ground-truth masks.
    Filenames must match between the two directories.
    """
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No PNGs found in {pred_dir}")

    per_image: dict[str, list] = {
        k: [] for k in ["dice", "iou", "pixel_acc", "precision", "recall", "cl_dice"]
    }

    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for pred_path in tqdm(pred_files, desc="Evaluating"):
        mask_path = mask_dir / pred_path.name
        if not mask_path.exists():
            print(f"  [SKIP] no matching mask for {pred_path.name}")
            continue

        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue

        m = compute_all(pred, gt)
        for k, v in m.items():
            per_image[k].append(v)

        if vis_dir:
            _save_comparison(pred_path.stem, pred, gt, m, vis_dir)

    return aggregate(per_image)


# ──────────────────────────────────────────────────────────────
# Evaluation from a model checkpoint
# ──────────────────────────────────────────────────────────────

def eval_unet_checkpoint(
    checkpoint_path: Path,
    split: str,
    data_root: str = "data",
    batch_size: int = 8,
    save_preds: bool = True,
    results_dir: Path = Path("results"),
) -> dict:
    """Loads a U-Net checkpoint and evaluates on the given split."""
    import segmentation_models_pytorch as smp
    from augmentations import get_transforms
    from dataset import ArcadeDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt    = torch.load(checkpoint_path, map_location=device)
    cfg     = ckpt.get("cfg", {})
    encoder = cfg.get("encoder", "resnet34")

    model = smp.Unet(
        encoder_name    = encoder,
        encoder_weights = None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded U-Net  epoch={ckpt.get('epoch','?')}  "
          f"val_dice={ckpt.get('val_dice', 0):.4f}")

    ds     = ArcadeDataset(split, data_root, transform=get_transforms("val"))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2
    )

    pred_dir = results_dir / "unet_predictions" / split
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    per_image: dict[str, list] = {
        k: [] for k in ["dice", "iou", "pixel_acc", "precision", "recall", "cl_dice"]
    }
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np  = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"U-Net [{split}]"):
            images  = batch["image"].to(device)
            targets = batch["mask"]
            stems   = batch["stem"]

            preds_bin = (torch.sigmoid(model(images)) > 0.5).cpu().numpy()
            tgt_np    = targets.numpy()

            for i, stem in enumerate(stems):
                pred_np = (preds_bin[i, 0] * 255).astype(np.uint8)
                gt_np   = (tgt_np[i, 0]   * 255).astype(np.uint8)

                m = compute_all(pred_np, gt_np)
                for k, v in m.items():
                    per_image[k].append(v)

                if save_preds:
                    # Denormalize image for panel
                    img_t = images[i].cpu().numpy().transpose(1, 2, 0)
                    img_t = ((img_t * std_np + mean_np) * 255).clip(0, 255).astype(np.uint8)
                    img_gray = cv2.cvtColor(img_t, cv2.COLOR_RGB2GRAY)
                    _save_comparison(stem, pred_np, gt_np, m,
                                     pred_dir, img_gray)

    return aggregate(per_image)


def eval_qwen_checkpoint(
    checkpoint_path: Path,
    split: str,
    data_root: str = "data",
    batch_size: int = 4,
    save_preds: bool = True,
    results_dir: Path = Path("results"),
) -> dict:
    """Loads a Qwen3-VL segmentation checkpoint and evaluates on the given split."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from train_qwen_seg import QwenArcadeDataset, QwenSegmenter, SegDecoder, qwen_collate_fn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(checkpoint_path, map_location=device)
    model_id = ckpt.get("model_id", "Qwen/Qwen3-VL-8B-Instruct")
    use_lora = ckpt.get("lora", False)
    mode     = "LoRA" if use_lora else "frozen"

    print(f"Loaded Qwen checkpoint  epoch={ckpt.get('epoch','?')}  "
          f"val_dice={ckpt.get('val_dice', 0):.4f}  mode={mode}")

    encoder   = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    decoder   = SegDecoder(in_channels=1152, out_size=512).to(device)
    decoder.load_state_dict(ckpt["decoder"])

    model = QwenSegmenter(encoder, decoder)
    model.eval()

    ds     = QwenArcadeDataset(split, data_root, processor)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=2, collate_fn=qwen_collate_fn
    )

    suffix   = "lora" if use_lora else "frozen"
    pred_dir = results_dir / f"qwen_{suffix}_predictions" / split
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    per_image: dict[str, list] = {
        k: [] for k in ["dice", "iou", "pixel_acc", "precision", "recall", "cl_dice"]
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Qwen {mode} [{split}]"):
            pv  = batch["pixel_values"].to(device)
            thw = batch["image_grid_thw"].to(device)
            tgt = batch["mask"].numpy()
            stems = batch["stem"]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(pv, thw)
            preds_bin = (torch.sigmoid(logits) > 0.5).cpu().numpy()

            for i, stem in enumerate(stems):
                pred_np = (preds_bin[i, 0] * 255).astype(np.uint8)
                gt_np   = (tgt[i, 0]       * 255).astype(np.uint8)
                m = compute_all(pred_np, gt_np)
                for k, v in m.items():
                    per_image[k].append(v)
                if save_preds:
                    _save_comparison(stem, pred_np, gt_np, m, pred_dir)

    return aggregate(per_image)


# ──────────────────────────────────────────────────────────────
# Visualization helper
# ──────────────────────────────────────────────────────────────

def _save_comparison(
    stem:     str,
    pred:     np.ndarray,
    gt:       np.ndarray,
    metrics:  dict,
    out_dir:  Path,
    img_gray: np.ndarray | None = None,
) -> None:
    """
    Saves a side-by-side comparison panel:
        [XCA image (if available)] | GT mask | predicted mask | diff map
    Filename encodes Dice score for quick sorting.
    """
    diff = np.zeros((*gt.shape, 3), dtype=np.uint8)
    p_bin = pred > 127
    g_bin = gt   > 127
    diff[p_bin &  g_bin]  = [255, 255, 255]   # TP — white
    diff[p_bin & ~g_bin]  = [255,   0,   0]   # FP — red
    diff[~p_bin & g_bin]  = [  0,   0, 255]   # FN — blue

    panels = []
    if img_gray is not None:
        panels.append(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR))
    panels += [
        cv2.cvtColor(gt,   cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR),
        diff,
    ]
    panel = np.hstack(panels)

    dice_val = metrics.get("dice", 0.0)
    label = (f"dice={dice_val:.3f}  iou={metrics.get('iou',0):.3f}  "
             f"cl_dice={metrics.get('cl_dice',0):.3f}")
    cv2.putText(panel, label, (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    fname = f"{stem}_dice{dice_val:.3f}.png"
    cv2.imwrite(str(out_dir / fname), panel)


# ──────────────────────────────────────────────────────────────
# Results table printer
# ──────────────────────────────────────────────────────────────

def print_table(results: dict[str, dict]) -> None:
    """
    Prints a formatted results table ready to transcribe into the thesis.

    results: {"Model Name": aggregated_metrics_dict, ...}
    """
    metrics = ["dice", "iou", "pixel_acc", "precision", "recall", "cl_dice"]
    header  = f"{'Model':<35}" + "".join(f"  {m:>10}" for m in metrics)

    print("\n" + "=" * len(header))
    print("RESULTS TABLE")
    print("=" * len(header))
    print(header)
    print("─" * len(header))

    for model_name, agg in results.items():
        row = f"{model_name:<35}"
        for m in metrics:
            val = agg.get(m, {}).get("mean", float("nan"))
            row += f"  {val:>10.4f}"
        print(row)

    print("─" * len(header))
    print("All values: mean over test split")
    print("clDice reference: Shit et al., CVPR 2021, arXiv:2003.07311\n")


def compare_all(results_dir: Path = Path("results"), split: str = "test") -> None:
    """
    Reads all *_metrics*.json files from results_dir and prints comparison table.
    Run after evaluating all models.
    """
    json_files = sorted(results_dir.glob("*metrics*.json"))
    if not json_files:
        print(f"No *metrics*.json files found in {results_dir}/")
        print("Run evaluation for each model first.")
        return

    all_results = {}
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        model_name = data.get("model", jf.stem)
        metrics    = data.get("test_metrics") or data.get("metrics", {})
        # Wrap in mean/std format if flat dict
        if metrics and not isinstance(next(iter(metrics.values())), dict):
            metrics = {k: {"mean": v, "std": 0.0} for k, v in metrics.items()}
        if metrics:
            all_results[model_name] = metrics

    print_table(all_results)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Mode 1: compare all saved JSON results
    if args.compare:
        compare_all(results_dir, split=args.split)
        return

    # Mode 2: evaluate from prediction PNG folder
    if args.pred_dir:
        print(f"Evaluating predictions from: {args.pred_dir}")
        agg = eval_from_folders(
            pred_dir = Path(args.pred_dir),
            mask_dir = Path(args.mask_dir),
            vis_dir  = results_dir / "comparison_vis" if args.save_vis else None,
        )
        print_table({"Custom predictions": agg})
        out = results_dir / "custom_metrics.json"
        with open(out, "w") as f:
            json.dump({"model": "custom", "metrics": agg}, f, indent=2)
        print(f"Saved → {out}")
        return

    # Mode 3: evaluate a checkpoint
    if not args.checkpoint:
        print("Provide --checkpoint, --pred-dir, or --compare")
        return

    ckpt_path = Path(args.checkpoint)
    if args.model == "unet":
        agg = eval_unet_checkpoint(
            checkpoint_path = ckpt_path,
            split           = args.split,
            data_root       = args.data_root,
            batch_size      = args.batch_size,
            save_preds      = args.save_preds,
            results_dir     = results_dir,
        )
        model_label = "U-Net (ResNet-34)"
        out_name    = f"unet_metrics_{args.split}.json"

    elif args.model == "qwen":
        agg = eval_qwen_checkpoint(
            checkpoint_path = ckpt_path,
            split           = args.split,
            data_root       = args.data_root,
            batch_size      = args.batch_size,
            save_preds      = args.save_preds,
            results_dir     = results_dir,
        )
        model_label = "Qwen3-VL + SegDecoder"
        out_name    = f"qwen_metrics_{args.split}.json"

    else:
        raise ValueError(f"Unknown model type: {args.model}")

    print_table({model_label: agg})

    out = results_dir / out_name
    with open(out, "w") as f:
        json.dump({
            "model":    model_label,
            "split":    args.split,
            "metrics":  agg,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate coronary vessel segmentation models"
    )

    # What to evaluate
    parser.add_argument("--model",      choices=["unet", "qwen"],
                        help="Model type to evaluate from checkpoint")
    parser.add_argument("--checkpoint", help="Path to .pth checkpoint file")
    parser.add_argument("--pred-dir",   help="Folder of prediction PNGs")
    parser.add_argument("--mask-dir",   default="data/masks/test",
                        help="Folder of ground-truth masks (for --pred-dir mode)")
    parser.add_argument("--compare",    action="store_true",
                        help="Print comparison table from all saved JSON results")

    # Settings
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--data-root",  default="data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--results-dir",default="results")
    parser.add_argument("--save-preds", action="store_true", default=True,
                        help="Save prediction panel PNGs")
    parser.add_argument("--save-vis",   action="store_true",
                        help="Save GT/pred/diff comparison images")

    main(parser.parse_args())
