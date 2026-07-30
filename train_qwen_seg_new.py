"""
train_qwen_seg_new.py
=====================
Fine-tuning Qwen3-VL-8B-Instruct for binary coronary vessel segmentation on ARCADE.

Architecture:
    Qwen3-VL vision tower (model.visual)  — frozen or LoRA
      → last_hidden_state (+ optional deepstack features)
      → reshape to (B, 4096, 16, 16)
      → optional DeepStackFusion (learnable weighted sum)
      → SegDecoder (upsample to 512×512)
      → logits (B, 1, 512, 512)

Included functionality:
    - Qwen3VLForConditionalGeneration (transformers >= 4.57)
    - vision features from vision_config.out_hidden_size (4096), patch 16, merge 2
    - LoRA on vision tower: target_modules ["qkv", "linear_fc1", "linear_fc2"]
    - optional --deepstack fusion of last + intermediate ViT features
    - bf16 training without GradScaler
    - custom collate for flattened Qwen image patches
    - decoder on cuda:0 when the backbone uses device_map="auto"
    - test-split evaluation and JSON metrics dump at the end

Resolved issues from earlier drafts:
    - wrong model class left vision MLP / merger randomly initialised
    - mismatched feature dim / patch size broke the spatial reshape

Usage:
    python train_qwen_seg_new.py --no-lora --deepstack --epochs 30 --batch-size 1
    python train_qwen_seg_new.py --lora --deepstack --epochs 30 --batch-size 1
"""

import argparse
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
IMG_SIZE = 512
DEC_DEVICE = "cuda:0"   # decoder + loss stay on cuda:0; encoder is split by device_map="auto"


# ══════════════════════════════════════════════════════════════
# SegDecoder  —  small CNN that upsamples (B, 4096, 16, 16) → (B, 1, 512, 512)
# ══════════════════════════════════════════════════════════════

class SegDecoder(nn.Module):
    """
    CNN decoder: Qwen3-VL spatial features → binary logits.

    - 1×1 projection 4096 → 512 before 3×3 convs
    - five bilinear 2× upsample stages to 512×512
    - head bias init ≈ logit(0.1) for sparse vessel pixels
    """

    def __init__(self, in_channels: int = 4096, base_channels: int = 512):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        channels = [base_channels, 256, 128, 64, 32, 16]
        self.stages = nn.ModuleList(
            [self._block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)
        self._init_weights()

    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# DeepStack fusion  —  learnable weighted sum of 4 feature banks
# ══════════════════════════════════════════════════════════════

class DeepStackFusion(nn.Module):
    """
    Fuses 4 feature tensors of identical shape into a single tensor:
        fused = Σ softmax(w)_i · features_i

    Qwen3-VL exposes 4 feature banks at the same spatial resolution and channel
    count: last_hidden_state plus deepstack_features from ViT layers 8, 16, 24.
    Inside the LLM, DeepStack adds these as residuals to layers 1-3. Here, for
    segmentation, we combine all of them into one tensor before decoding.

    Starting weights are uniform (after softmax of zeros → all 0.25).
    """

    def __init__(self, n_sources: int = 4):
        super().__init__()
        self.n_sources = n_sources
        self.logits    = nn.Parameter(torch.zeros(n_sources))

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == self.n_sources
        w = torch.softmax(self.logits, dim=0)
        out = w[0] * feats[0]
        for i in range(1, self.n_sources):
            out = out + w[i] * feats[i]
        return out


# ══════════════════════════════════════════════════════════════
# Full segmentation model
# ══════════════════════════════════════════════════════════════

class QwenSegmenter(nn.Module):
    """
    Qwen3-VL vision tower (`model.visual`) + CNN decoder.

    forward(pixel_values, image_grid_thw) returns (B, 1, 512, 512) logits.

    - pixel_values is concatenated across batch: shape (B*N_patches, patch_dim)
    - image_grid_thw: shape (B, 3) each row = [T, H_patches, W_patches]
    - All images must share the same grid (we assert this) → constant 256 tokens/image
    """

    def __init__(
        self,
        vl_model,            # Qwen3VLForConditionalGeneration (or PEFT-wrapped)
        decoder: SegDecoder,
        use_deepstack: bool = False,
        spatial_merge: int = 2,
        patch_size:    int = 16,
        img_size:      int = 512,
    ):
        super().__init__()
        self.vl_model      = vl_model
        self.decoder       = decoder
        self.use_deepstack = use_deepstack
        self.grid          = img_size // patch_size // spatial_merge   # = 16 for 512×512
        assert self.grid * patch_size * spatial_merge == img_size

        self.fusion = DeepStackFusion(n_sources=4) if use_deepstack else None

    @property
    def visual(self):
        m = self.vl_model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model        # unwrap PEFT
        return m.visual

    def forward(self, pixel_values: torch.Tensor,
                image_grid_thw: torch.Tensor) -> torch.Tensor:
        B = image_grid_thw.shape[0]
        grid = self.grid
        tokens_per_img = grid * grid

        # Vision tower returns (last_hidden_state, deepstack_features_list)
        last_hidden, deepstack = self.visual(
            hidden_states = pixel_values,
            grid_thw      = image_grid_thw,
        )

        C = last_hidden.shape[-1]  # should be 4096 (out_hidden_size)

        def to_spatial(x: torch.Tensor) -> torch.Tensor:
            # (B*T, C) → (B, grid, grid, C) → (B, C, grid, grid)
            return (x.view(B, grid, grid, C)
                     .permute(0, 3, 1, 2)
                     .contiguous()
                     .to(DEC_DEVICE))

        last_spatial = to_spatial(last_hidden)

        if self.fusion is not None:
            ds_spatial = [to_spatial(ds) for ds in deepstack]
            fused = self.fusion([last_spatial, *ds_spatial])
        else:
            fused = last_spatial

        # Decoder expects fp32 for stability of BN; autocast will re-cast as needed
        return self.decoder(fused.float())


# ══════════════════════════════════════════════════════════════
# Loss + metrics  (same as UNet baseline for fair comparison)
# ══════════════════════════════════════════════════════════════

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = self.bce(logits, targets)
        p    = torch.sigmoid(logits).view(-1)
        t    = targets.view(-1)
        dice = 1.0 - (2.0 * (p * t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return 0.5 * bce + 0.5 * dice


def compute_metrics(preds_bin, targets, smooth: float = 1e-6) -> dict:
    p  = preds_bin.float().view(-1)
    t  = targets.float().view(-1)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()
    return {
        "dice":      ((2 * tp + smooth) / (2 * tp + fp + fn + smooth)).item(),
        "iou":       ((tp + smooth) / (tp + fp + fn + smooth)).item(),
        "px_acc":    ((tp + tn) / (tp + tn + fp + fn + smooth)).item(),
        "precision": ((tp + smooth) / (tp + fp + smooth)).item(),
        "recall":    ((tp + smooth) / (tp + fn + smooth)).item(),
    }


# ══════════════════════════════════════════════════════════════
# Dataset  (Qwen-style: processor flattens patches, no batch dim)
# ══════════════════════════════════════════════════════════════

class QwenArcadeDataset(Dataset):
    def __init__(self, split: str, data_root: str, processor, img_size: int = 512):
        self.processor = processor
        self.img_size  = img_size
        self.split     = split

        # Geometric-only augmentation — Qwen processor does its own normalisation
        if split == "train":
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, border_mode=0, p=0.5),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.25, p=0.5,
                ),
            ])
        else:
            self.transform = A.Compose([A.Resize(img_size, img_size)])

        root    = Path(data_root)
        img_dir = root / "syntax" / split / "images"
        msk_dir = root / "masks"  / split
        if not img_dir.exists(): raise FileNotFoundError(img_dir)
        if not msk_dir.exists(): raise FileNotFoundError(msk_dir)

        self.pairs = [(p, msk_dir / p.name) for p in sorted(img_dir.glob("*.png"))
                      if (msk_dir / p.name).exists()]
        if not self.pairs:
            raise RuntimeError(f"No image-mask pairs for split={split}")
        print(f"[{split}] {len(self.pairs)} pairs (Qwen mode)")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask_bin = (mask > 127).astype(np.float32)

        aug = self.transform(image=rgb, mask=mask_bin)
        img_aug  = aug["image"]
        mask_aug = aug["mask"]
        if img_aug.dtype != np.uint8:
            img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        pil  = PILImage.fromarray(img_aug)
        proc = self.processor.image_processor(images=[pil], return_tensors="pt")

        return {
            "pixel_values":   proc["pixel_values"],              # (N_patches, patch_dim)
            "image_grid_thw": proc["image_grid_thw"][0],          # (3,)
            "mask":           torch.from_numpy(mask_aug).unsqueeze(0),
            "stem":           img_path.stem,
        }


def qwen_collate_fn(batch: list[dict]) -> dict:
    """Qwen flattens patches; batching = concat along tokens, stack grid_thw and mask."""
    return {
        "pixel_values":   torch.cat(  [b["pixel_values"]   for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch], dim=0),
        "mask":           torch.stack([b["mask"]            for b in batch], dim=0),
        "stem":           [b["stem"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════
# Train / eval loops
# ══════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion) -> dict:
    model.train()
    total_loss = 0.0
    totals = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  train", leave=False):
        pv  = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw = batch["image_grid_thw"].to(DEC_DEVICE)
        tgt = batch["mask"].to(DEC_DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv, thw)
        loss = criterion(logits.float(), tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).long()
            for k, v in compute_metrics(preds_bin, tgt).items():
                totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


@torch.no_grad()
def eval_epoch(model, loader, criterion) -> dict:
    model.eval()
    total_loss = 0.0
    totals = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  eval ", leave=False):
        pv  = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw = batch["image_grid_thw"].to(DEC_DEVICE)
        tgt = batch["mask"].to(DEC_DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv, thw)
        total_loss += criterion(logits.float(), tgt).item()

        preds_bin = (torch.sigmoid(logits) > 0.5).long()
        for k, v in compute_metrics(preds_bin, tgt).items():
            totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main(args: argparse.Namespace) -> None:
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            n = torch.cuda.get_device_name(i)
            v = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {n}  ({v:.1f} GB)")

    mode = "LoRA"       if args.lora      else "frozen"
    fuse = "deepstack"  if args.deepstack else "last_only"
    tag  = f"{mode}_{fuse}"
    print(f"\nModel       : {MODEL_ID}")
    print(f"Encoder     : {mode}")
    print(f"Fusion      : {fuse}")
    print(f"Epochs      : {args.epochs}  batch: {args.batch_size}  lr: {args.lr}")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Load Qwen3-VL (CORRECT class + correct `dtype=` arg) ────────────────
    print(f"\nLoading {MODEL_ID} ...")
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype             = torch.bfloat16,     # 'torch_dtype=' is deprecated in 4.57
        device_map        = "auto",
        low_cpu_mem_usage = True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    vc  = vl_model.config.vision_config
    print(f"\n  vision_config.hidden_size      : {vc.hidden_size}")
    print(f"  vision_config.out_hidden_size  : {vc.out_hidden_size}")
    print(f"  vision_config.patch_size       : {vc.patch_size}")
    print(f"  spatial_merge_size             : {vc.spatial_merge_size}")
    print(f"  deepstack_visual_indexes       : {vc.deepstack_visual_indexes}")

    # ── LoRA / freeze ───────────────────────────────────────────────────────
    if args.lora:
        # These leaf names are unique to Qwen3-VL vision tower:
        #   - qkv          : FUSED Q/K/V linear in each ViT attn block
        #   - linear_fc1/2 : MLP in each ViT block and in the mergers
        # They do NOT collide with LLM leaves (which use q_proj/k_proj/v_proj/...).
        # Skip vision attn `proj` — name collides with LLM leaves.
        lora_cfg = LoraConfig(
            r               = args.lora_r,
            lora_alpha      = args.lora_r * 2,
            target_modules  = ["qkv", "linear_fc1", "linear_fc2"],
            lora_dropout    = 0.05,
            bias            = "none",
            modules_to_save = [],
        )
        vl_model = get_peft_model(vl_model, lora_cfg)
        vl_model.print_trainable_parameters()
    else:
        for p in vl_model.parameters():
            p.requires_grad = False
        print("Encoder FROZEN — only decoder trains.")

    # ── Decoder + fusion ───────────────────────────────────────────────────
    decoder = SegDecoder(in_channels=vc.out_hidden_size, base_channels=512).to(DEC_DEVICE)
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

    segmenter = QwenSegmenter(
        vl_model      = vl_model,
        decoder       = decoder,
        use_deepstack = args.deepstack,
        spatial_merge = vc.spatial_merge_size,
        patch_size    = vc.patch_size,
        img_size      = IMG_SIZE,
    )
    if segmenter.fusion is not None:
        segmenter.fusion.to(DEC_DEVICE)

    # ── Data ────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = QwenArcadeDataset("train", args.data_root, processor, IMG_SIZE)
    val_ds   = QwenArcadeDataset("val",   args.data_root, processor, IMG_SIZE)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True, collate_fn=qwen_collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)

    # ── Optimizer / scheduler ──────────────────────────────────────────────
    if args.lora:
        enc_params = [p for p in vl_model.parameters() if p.requires_grad]
        dec_params = list(decoder.parameters()) + (
            list(segmenter.fusion.parameters()) if segmenter.fusion else []
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": dec_params, "lr": args.lr},
                {"params": enc_params, "lr": args.lr * 0.1},   # LoRA adapters: 10× smaller
            ],
            weight_decay = 1e-4,
        )
    else:
        params = list(decoder.parameters()) + (
            list(segmenter.fusion.parameters()) if segmenter.fusion else []
        )
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = BCEDiceLoss()
    # bf16 — no GradScaler

    n_trainable = sum(p.numel() for p in optimizer.param_groups[0]["params"]) / 1e6
    print(f"Trainable params (primary group): {n_trainable:.2f}M")

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_dice = 0.0
    ckpt_path     = Path(args.out_dir) / f"qwen_seg_best_{tag}.pth"
    history       = []

    header = f"{'Epoch':>5} | {'Loss':>7} | {'Tr.Dice':>7} | {'Val.Dice':>8} | {'Val.IoU':>7}"
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(segmenter, train_loader, optimizer, criterion)
        vl = eval_epoch(segmenter,  val_loader,   criterion)
        scheduler.step()
        sec = time.time() - t0

        print(f"{epoch:>5} | {tr['loss']:>7.4f} | {tr['dice']:>7.4f} | "
              f"{vl['dice']:>8.4f} | {vl['iou']:>7.4f}  ({sec:.0f}s)")
        history.append({"epoch": epoch, "train": tr, "val": vl})

        if vl["dice"] > best_val_dice:
            best_val_dice = vl["dice"]
            ckpt = {
                "epoch":    epoch,
                "decoder":  decoder.state_dict(),
                "fusion":   segmenter.fusion.state_dict() if segmenter.fusion else None,
                "val_dice": best_val_dice,
                "config": {
                    "model_id":      MODEL_ID,
                    "use_deepstack": args.deepstack,
                    "use_lora":      args.lora,
                    "lora_r":        args.lora_r if args.lora else None,
                },
            }
            if args.lora:
                # LoRA adapters only (full state_dict is huge)
                ckpt["lora_state"] = get_peft_model_state_dict(vl_model)
            torch.save(ckpt, ckpt_path)
            print(f"       ↑ best val Dice {best_val_dice:.4f} — saved")

    # ── Test evaluation on best checkpoint ─────────────────────────────────
    print("\n" + "=" * 55)
    print("Loading best checkpoint for TEST evaluation...")
    ckpt = torch.load(ckpt_path, map_location=DEC_DEVICE)
    decoder.load_state_dict(ckpt["decoder"])
    if segmenter.fusion is not None and ckpt.get("fusion") is not None:
        segmenter.fusion.load_state_dict(ckpt["fusion"])

    try:
        test_ds = QwenArcadeDataset("test", args.data_root, processor, IMG_SIZE)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
        test_m = eval_epoch(segmenter, test_loader, criterion)

        print("\nTEST RESULTS")
        for k, v in test_m.items():
            print(f"  {k:<10} {v:.4f}")

        results = {
            "model":         f"Qwen3-VL-8B + SegDecoder ({mode}, {fuse})",
            "dataset":       "ARCADE syntax — binary vessel segmentation",
            "split":         "test",
            "best_val_dice": round(best_val_dice, 4),
            "test_metrics":  {k: round(v, 4) for k, v in test_m.items()},
            "history":       history,
            "config": {
                "model_id":     MODEL_ID,
                "encoder_mode": mode,
                "fusion_mode":  fuse,
                "epochs":       args.epochs,
                "batch_size":   args.batch_size,
                "lr":           args.lr,
                "lora_r":       args.lora_r if args.lora else None,
                "vision_hidden_size":     vc.hidden_size,
                "vision_out_hidden_size": vc.out_hidden_size,
                "patch_size":             vc.patch_size,
                "spatial_merge_size":     vc.spatial_merge_size,
                "deepstack_indexes":      list(vc.deepstack_visual_indexes),
            },
        }
        out_json = Path(args.results_dir) / f"qwen_metrics_{tag}.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {out_json}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[SKIP] Test evaluation skipped: {e}")

    print(f"Checkpoint → {ckpt_path}")
    print("Done.")


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-VL segmentation fine-tuning on ARCADE")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lora",    action="store_true", help="LoRA fine-tune vision tower")
    mode.add_argument("--no-lora", action="store_true", help="Freeze encoder, train decoder only")

    parser.add_argument("--deepstack",   action="store_true",
                        help="Fuse last_hidden + 3 deepstack feature banks via learnable weighted sum")

    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--out-dir",     default="checkpoints")
    parser.add_argument("--results-dir", default="results")

    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=1)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int,   default=2)
    parser.add_argument("--lora-r",      type=int,   default=16)

    main(parser.parse_args())
