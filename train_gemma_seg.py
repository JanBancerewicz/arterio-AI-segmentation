# ten plik jest obecnie nieaktywny, ani nigdzie niewykorzystywany, jest w .gitignore i wymaga zaaktualizowania do gemma4b quantized oraz sam był wzorowany na train_qwen_seg.py, który jest deprecated
"""
train_gemma_seg.py
===================
Fine-tuning Gemma 3-4B (multimodal) for binary coronary vessel segmentation
on ARCADE. Mirrors train_qwen_seg.py for fair thesis comparison.

Architecture:
    Gemma 3 vision tower (SigLIP-So400m)  →  spatial token grid
        ↓  adaptive avg-pool to 16x16  (matches Qwen pipeline grid)
    CNN decoder (same SegDecoder as Qwen)  →  (B, 1, 512, 512)
        ↓
    Sigmoid → binary vessel mask

Two training modes (--lora / --no-lora):
    --no-lora  : vision tower frozen, only decoder trains  (~9 GB VRAM)
    --lora     : vision tower fine-tuned with LoRA r=16    (~12 GB VRAM)

This adds two more rows to the thesis results table:
    1. U-Net baseline                       (train_unet.py)
    2. Qwen3-VL frozen   + SegDecoder       (train_qwen_seg.py --no-lora)
    3. Qwen3-VL LoRA     + SegDecoder       (train_qwen_seg.py --lora)
    4. Gemma 3-4B frozen + SegDecoder       (this script, --no-lora)
    5. Gemma 3-4B LoRA   + SegDecoder       (this script, --lora)

Usage:
    # Frozen encoder (faster, less VRAM):
    python train_gemma_seg.py --no-lora

    # LoRA fine-tuning (main experiment):
    python train_gemma_seg.py --lora

    # Custom settings:
    python train_gemma_seg.py --lora --epochs 30 --batch-size 2 --lr 5e-5

Requirements:
    pip install "transformers>=4.51.0" peft accelerate
    augmentations.py and dataset.py must be in the same folder.

Notes on VRAM (2x RTX 4080 16 GB, bf16):
    - Gemma 3-4B in bf16 ~ 8 GB total weights — fits with device_map="auto".
    - Default image size 896x896 → 64x64 patch grid (vs Qwen's effective 16x16).
      We avg-pool to 16x16 so the SAME SegDecoder works and comparison is fair.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

# Gemma 3 multimodal class. Requires transformers >= 4.50.
# If you hit ImportError, bump: pip install -U "transformers>=4.51.0"
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from augmentations import get_transforms  # noqa: F401  (kept for parity)
from dataset import ArcadeDataset

MODEL_ID = "google/gemma-3-4b-it"


# ──────────────────────────────────────────────────────────────
# CNN Decoder (identical to SegDecoder in train_qwen_seg.py)
# ──────────────────────────────────────────────────────────────

class SegDecoder(nn.Module):
    """
    Lightweight CNN decoder.

    Input  : (B, in_channels, 16, 16)  — spatial features after pool
    Output : (B, 1, 512, 512)          — logits (apply sigmoid for mask)

    Why bilinear upsample + Conv (not ConvTranspose):
        Avoids checkerboard artifacts that hurt thin vessel structures.

    Why init the final 1x1 conv bias to logit(0.1) ≈ -2.2:
        Vessels are ~10% of pixels. Starting from this prior lets the model
        focus on WHERE vessels are, not on re-learning the class imbalance.
        See Lin et al., Focal Loss, ICCV 2017, §4.1.
    """

    def __init__(self, in_channels: int = 1152, out_size: int = 512):
        super().__init__()
        self.out_size = out_size

        self.stages = nn.ModuleList([
            self._block(in_channels, 256),
            self._block(256, 128),
            self._block(128, 64),
            self._block(64, 32),
        ])
        self.head = nn.Conv2d(32, 1, kernel_size=1)
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
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        x = F.interpolate(x, size=(self.out_size, self.out_size),
                          mode="bilinear", align_corners=False)
        return x


# ──────────────────────────────────────────────────────────────
# Full segmentation model
# ──────────────────────────────────────────────────────────────

class GemmaSegmenter(nn.Module):
    """
    Gemma 3 vision tower (SigLIP) + CNN decoder for binary segmentation.

    Forward:
        pixel_values  → vision_tower → (B, N_patches, D)
        reshape       → (B, D, H, W)         where H = W = sqrt(N_patches)
        avg-pool      → (B, D, 16, 16)       (matches Qwen pipeline grid)
        SegDecoder    → (B, 1, 512, 512)     logits
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        decoder:      SegDecoder,
        pool_size:    int = 16,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.decoder      = decoder
        self.pool_size    = pool_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # SigLIP forward — returns BaseModelOutputWithPooling
        out = self.vision_tower(pixel_values=pixel_values)
        feats = out.last_hidden_state                          # (B, N, D)

        B, N, D = feats.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise RuntimeError(
                f"Vision features not square: N={N}. "
                "Check processor image_size vs patch_size."
            )

        # (B, N, D) → (B, D, H, W)
        feats = feats.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Adaptive avg-pool to 16x16 — keeps decoder identical to Qwen pipeline
        if H != self.pool_size:
            feats = F.adaptive_avg_pool2d(feats, output_size=self.pool_size)

        return self.decoder(feats)


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """BCE + Dice loss (equal weights). See train_unet.py for rationale."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        p   = torch.sigmoid(logits).view(-1)
        t   = targets.view(-1)
        dice = 1.0 - (2.0 * (p * t).sum() + self.smooth) / (
            p.sum() + t.sum() + self.smooth
        )
        return 0.5 * bce + 0.5 * dice


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def compute_metrics(
    preds_bin: torch.Tensor,
    targets:   torch.Tensor,
    smooth:    float = 1e-6,
) -> dict[str, float]:
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


# ──────────────────────────────────────────────────────────────
# Dataset wrapper for Gemma 3 processor
# ──────────────────────────────────────────────────────────────

class GemmaArcadeDataset(ArcadeDataset):
    """
    Extends ArcadeDataset to produce pixel_values expected by Gemma 3.

    Gemma 3 processor handles:
        - Resizing to model's native input size (896x896 by default)
        - Normalization (own stats, NOT ImageNet)

    We bypass the language modeling part entirely — no text input needed.
    """

    def __init__(self, split: str, data_root: str, processor):
        import albumentations as A
        if split == "train":
            transform = A.Compose([
                A.Resize(512, 512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, border_mode=0, p=0.5),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.25, p=0.5
                ),
                A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
            ])
        else:
            transform = A.Compose([A.Resize(512, 512)])

        self.transform_geo = transform
        self.processor     = processor
        self.split         = split

        root     = Path(data_root)
        img_dir  = root / "syntax" / split / "images"
        mask_dir = root / "masks"  / split

        self._validate_dirs(img_dir, mask_dir, split)
        self.pairs = []
        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))

        print(f"[{split}] {len(self.pairs)} pairs (Gemma mode)")

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask     = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        mask_bin = (mask > 127).astype(np.float32)

        aug      = self.transform_geo(image=img_rgb, mask=mask_bin)
        img_aug  = aug["image"]
        mask_aug = aug["mask"]

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_aug)

        # Use image_processor directly — we don't need text tokenization
        proc_out = self.processor.image_processor(
            images         = [pil_img],
            return_tensors = "pt",
        )

        return {
            "pixel_values": proc_out["pixel_values"].squeeze(0),
            "mask":         torch.from_numpy(mask_aug).unsqueeze(0),
            "stem":         img_path.stem,
        }


def gemma_collate_fn(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "mask":         torch.stack([b["mask"]          for b in batch]),
        "stem":         [b["stem"] for b in batch],
    }


# ──────────────────────────────────────────────────────────────
# Train / eval loops
# ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler) -> dict:
    model.train()
    total_loss = 0.0
    totals = {k: 0.0 for k in ["dice", "iou", "px_acc", "precision", "recall"]}

    for batch in tqdm(loader, desc="  train", leave=False):
        pv  = batch["pixel_values"].to(device)
        tgt = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv)
            loss   = criterion(logits, tgt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).long()
            for k, v in compute_metrics(preds_bin, tgt).items():
                totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    totals = {k: 0.0 for k in ["dice", "iou", "px_acc", "precision", "recall"]}

    for batch in tqdm(loader, desc="  val  ", leave=False):
        pv  = batch["pixel_values"].to(device)
        tgt = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv)
            total_loss += criterion(logits, tgt).item()

        preds_bin = (torch.sigmoid(logits) > 0.5).long()
        for k, v in compute_metrics(preds_bin, tgt).items():
            totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name}  ({vram:.1f} GB)")

    mode = "LoRA" if args.lora else "frozen encoder"
    print(f"\nMode      : {mode}")
    print(f"Model     : {MODEL_ID}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch     : {args.batch_size}")
    print(f"LR        : {args.lr}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load Gemma 3 ──────────────────────────────────────────
    print(f"\nLoading {MODEL_ID}...")
    print("(first run downloads ~8 GB)")

    encoder = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        low_cpu_mem_usage = True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── Free the language model — we only use the vision tower ──
    # Saves ~7 GB VRAM. Safe because we never call encoder.forward();
    # we only forward through vision_tower directly.
    if hasattr(encoder, "language_model"):
        del encoder.language_model
        torch.cuda.empty_cache()
        print("Language model removed — vision tower only.")

    # ── Freeze / LoRA ─────────────────────────────────────────
    if args.lora:
        # Restrict LoRA to SigLIP attention projections via regex.
        # SigLIP uses 'out_proj' (not 'o_proj'), letting us match
        # only vision layers cleanly.
        lora_cfg = LoraConfig(
            r              = 16,
            lora_alpha     = 32,
            target_modules = r".*vision_tower.*\.(q_proj|k_proj|v_proj|out_proj)$",
            lora_dropout   = 0.05,
            bias           = "none",
            modules_to_save = [],
        )
        encoder = get_peft_model(encoder, lora_cfg)
        encoder.print_trainable_parameters()
        # After PEFT wrap, vision_tower lives at encoder.base_model.model.vision_tower
        vision_tower = encoder.base_model.model.vision_tower
    else:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — only decoder will train.")
        vision_tower = encoder.vision_tower

    # ── Decoder ───────────────────────────────────────────────
    # SigLIP-So400m hidden size = 1152 (queried from config to be safe)
    vision_hidden = encoder.config.vision_config.hidden_size
    print(f"Vision hidden dim : {vision_hidden}")

    decoder = SegDecoder(in_channels=vision_hidden, out_size=512).to(device)
    n_dec   = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"Decoder parameters: {n_dec:.2f}M")

    model = GemmaSegmenter(vision_tower, decoder, pool_size=16)

    # ── Data ──────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = GemmaArcadeDataset("train", args.data_root, processor)
    val_ds   = GemmaArcadeDataset("val",   args.data_root, processor)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=gemma_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=gemma_collate_fn,
    )

    # ── Optimizer ─────────────────────────────────────────────
    if args.lora:
        param_groups = [
            {"params": decoder.parameters(),                           "lr": args.lr},
            {"params": [p for p in encoder.parameters() if p.requires_grad],
             "lr": args.lr * 0.1},   # LoRA adapters: 10x smaller LR
        ]
    else:
        param_groups = [{"params": decoder.parameters(), "lr": args.lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = BCEDiceLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # ── Training loop ─────────────────────────────────────────
    best_val_dice = 0.0
    ckpt_path     = out_dir / "gemma_seg_best.pth"

    header = (f"{'Epoch':>6} | {'Loss':>8} | {'Tr.Dice':>8} | "
              f"{'Val.Dice':>9} | {'Val.IoU':>8}")
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        vl = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        sec = time.time() - t0
        print(f"{epoch:>6} | {tr['loss']:>8.4f} | {tr['dice']:>8.4f} | "
              f"{vl['dice']:>9.4f} | {vl['iou']:>8.4f}  ({sec:.0f}s)")

        if vl["dice"] > best_val_dice:
            best_val_dice = vl["dice"]
            torch.save({
                "epoch":      epoch,
                "decoder":    decoder.state_dict(),
                "val_dice":   best_val_dice,
                "lora":       args.lora,
                "model_id":   MODEL_ID,
            }, ckpt_path)
            print(f"         ↑ best val Dice {best_val_dice:.4f} — saved")

    # ── Final results ─────────────────────────────────────────
    results = {
        "model":         f"Gemma 3-4B + SegDecoder ({mode})",
        "dataset":       "ARCADE syntax — binary vessel segmentation",
        "best_val_dice": round(best_val_dice, 4),
        "config": {
            "model_id":   MODEL_ID,
            "mode":       mode,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
        },
    }
    suffix = "lora" if args.lora else "frozen"
    results_path = out_dir / f"gemma_metrics_{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults → {results_path}")
    print(f"Checkpoint → {ckpt_path}")
    print("\nDone.")


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gemma 3-4B segmentation fine-tuning on ARCADE"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lora",    action="store_true",
                       help="Fine-tune vision tower with LoRA r=16 (~12 GB VRAM)")
    group.add_argument("--no-lora", action="store_true",
                       help="Freeze vision tower, train only decoder (~9 GB VRAM)")

    parser.add_argument("--data-root", default="data")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=2,
                        help="Per-GPU batch. 2 for LoRA, 4 for frozen on 16 GB.")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--out-dir", default="checkpoints")

    main(parser.parse_args())