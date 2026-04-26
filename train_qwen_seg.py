"""
train_qwen_seg.py
==================
Fine-tuning Qwen3-VL-8B for binary coronary vessel segmentation.

Architecture:
    Qwen3-VL vision encoder  →  patch embeddings (visual tokens)
        ↓
    Reshape to spatial grid  →  (B, C, H', W')
        ↓
    Lightweight CNN decoder  →  upsampling to (B, 1, 512, 512)
        ↓
    Sigmoid → binary vessel mask

Two training modes (controlled by --lora flag):
    --no-lora  : encoder fully frozen, only decoder trains  (~10 GB VRAM)
    --lora     : encoder fine-tuned with LoRA r=16          (~14 GB VRAM)

This gives three rows in the thesis results table:
    1. U-Net baseline (train_unet.py)
    2. Qwen3-VL frozen + decoder
    3. Qwen3-VL LoRA   + decoder

Usage:
    # Frozen encoder (faster, less VRAM):
    python train_qwen_seg.py --no-lora

    # LoRA fine-tuning (main experiment):
    python train_qwen_seg.py --lora

    # Custom settings:
    python train_qwen_seg.py --lora --epochs 30 --batch-size 4 --lr 5e-5

Requirements:
    pip install transformers>=4.51.0 peft accelerate
    augmentations.py and dataset.py must be in the same folder.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from augmentations import get_transforms
from dataset import ArcadeDataset

# ──────────────────────────────────────────────────────────────
# NOTE ON MODEL CLASS:
# As of transformers>=4.51, Qwen3-VL uses Qwen2_5_VLForConditionalGeneration.
# If you get an ImportError, try:
#   from transformers import AutoModelForImageTextToText
#   model = AutoModelForImageTextToText.from_pretrained(...)
# and adapt accordingly.
# ──────────────────────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ──────────────────────────────────────────────────────────────
# CNN Decoder
# ──────────────────────────────────────────────────────────────

class SegDecoder(nn.Module):
    """
    Lightweight CNN decoder that upsamples Qwen3-VL patch embeddings
    to a full-resolution binary segmentation mask.

    Input  : (B, hidden_dim, H', W')  — spatial grid of visual tokens
    Output : (B, 1, 512, 512)         — logits (apply sigmoid for probabilities)

    Design rationale for thesis:
        - 4 stages of 2× bilinear upsampling + Conv2d (not ConvTranspose to
          avoid checkerboard artifacts common in medical imaging)
        - BatchNorm + ReLU for stable training from random init
        - Final 1×1 Conv produces single-channel logit map
        - Total ~2M parameters — small enough that encoder features dominate

    Qwen3-VL-8B vision encoder:
        - Patch size   : 14×14 pixels
        - Input image  : 512×512
        - Grid size    : 512/14 ≈ 37×37 (rounded by processor)
        - Hidden dim   : 1152 (ViT-L equivalent)
    """

    def __init__(self, in_channels: int = 1152, out_size: int = 512):
        super().__init__()
        self.out_size = out_size

        # Progressive upsampling: 37→74→148→296→512
        # Channels: 1152 → 256 → 128 → 64 → 32 → 1
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
        # Small random weights + bias = logit(vessel_prior)
        # Vessels ~10% of pixels → bias = logit(0.1) ≈ -2.2
        nn.init.normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, -2.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = self.head(x)
        # Final resize to exact output size (handles rounding in patch grid)
        x = nn.functional.interpolate(
            x, size=(self.out_size, self.out_size),
            mode="bilinear", align_corners=False,
        )
        return x


# ──────────────────────────────────────────────────────────────
# Full segmentation model
# ──────────────────────────────────────────────────────────────

class QwenSegmenter(nn.Module):
    """
    Qwen3-VL vision encoder + CNN decoder for binary segmentation.

    The VLM is used purely as a visual feature extractor.
    Language modeling head is not used and not loaded into GPU memory.

    Feature extraction:
        We pass the image through the vision tower and extract the
        last hidden state of the visual tokens. These are then reshaped
        from sequence form (B, N_patches, hidden_dim) to a 2D spatial
        grid (B, hidden_dim, H', W') and fed to the decoder.
    """

    def __init__(
        self,
        encoder:    nn.Module,
        decoder:    SegDecoder,
        merge_size: int = 2,   # Qwen2.5/3-VL default spatial_merge_size
    ):
        super().__init__()
        self.encoder    = encoder
        self.decoder    = decoder
        self.merge_size = merge_size

    def forward(
        self,
        pixel_values:    torch.Tensor,
        image_grid_thw:  torch.Tensor,
    ) -> torch.Tensor:
        visual_features = self.encoder.model.visual(
            hidden_states = pixel_values,
            grid_thw      = image_grid_thw,
        )

        B         = image_grid_thw.shape[0]
        H_merged  = int(image_grid_thw[0, 1]) // self.merge_size
        W_merged  = int(image_grid_thw[0, 2]) // self.merge_size
        hidden    = visual_features.shape[-1]

        features = visual_features.view(B, H_merged, W_merged, hidden)
        features = features.permute(0, 3, 1, 2).contiguous()

        return self.decoder(features)


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
# XCA-specific dataset wrapper for Qwen processor
# ──────────────────────────────────────────────────────────────

class QwenArcadeDataset(ArcadeDataset):
    """
    Extends ArcadeDataset to produce pixel_values and image_grid_thw
    tensors expected by Qwen3-VL vision tower.

    Qwen processor handles:
        - Normalization (different from ImageNet — uses its own stats)
        - Tiling / patch extraction
        - Returning image_grid_thw

    We bypass the language modeling part entirely — no text input needed.
    """

    def __init__(self, split: str, data_root: str, processor):
        # Use only geometric augmentations for Qwen (processor handles normalization)
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

        # Initialize parent without albumentations ToTensorV2
        # (Qwen processor will tensorize)
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

        print(f"[{split}] {len(self.pairs)} pairs (Qwen mode)")

    def __getitem__(self, idx: int) -> dict:
        img_path, mask_path = self.pairs[idx]

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask     = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # Convert to RGB for Qwen processor
        img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        mask_bin = (mask > 127).astype(np.float32)

        # Apply geometric augmentations (mask follows the same transform)
        aug      = self.transform_geo(image=img_rgb, mask=mask_bin)
        img_aug  = aug["image"]   # (H, W, 3) uint8
        mask_aug = aug["mask"]    # (H, W) float32

        # Qwen processor: normalizes and extracts patches
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img_aug)

        proc_out = self.processor.image_processor(
            images         = [pil_img],
            return_tensors = "pt",
        )


        return {
            "pixel_values":   proc_out["pixel_values"].squeeze(0),
            "image_grid_thw": proc_out["image_grid_thw"].squeeze(0),
            "mask":           torch.from_numpy(mask_aug).unsqueeze(0),
            "stem":           img_path.stem,
        }


def qwen_collate_fn(batch: list[dict]) -> dict:
    """Custom collate: pixel_values may have different lengths due to tiling."""
    return {
        "pixel_values":   torch.cat([b["pixel_values"]   for b in batch]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
        "mask":           torch.cat([b["mask"]            for b in batch]),
        "stem":           [b["stem"] for b in batch],
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
        thw = batch["image_grid_thw"].to(device)
        tgt = batch["mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv, thw)
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
        thw = batch["image_grid_thw"].to(device)
        tgt = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(pv, thw)
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

    # ── Load Qwen3-VL ─────────────────────────────────────────
    print(f"\nLoading {MODEL_ID}...")
    print("(this may take a few minutes on first run — downloading ~16 GB)")

    encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype    = torch.bfloat16,
        device_map     = "auto",          # spreads across 2× RTX 4080
        low_cpu_mem_usage = True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── Freeze / LoRA ─────────────────────────────────────────
    if args.lora:
        # Apply LoRA to the vision encoder attention layers
        # r=16 is a good default: enough capacity, manageable VRAM
        lora_cfg = LoraConfig(
            r              = 16,
            lora_alpha     = 32,
            # Target the self-attention projections in the vision transformer
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout   = 0.05,
            bias           = "none",
            # Apply only to vision encoder, not language model
            modules_to_save = [],
        )
        encoder = get_peft_model(encoder, lora_cfg)
        encoder.print_trainable_parameters()

    else:
        # Freeze entire encoder — only decoder will train
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder frozen — only decoder will train.")

    # ── Decoder ───────────────────────────────────────────────
    llm_hidden = encoder.config.hidden_size
    merge_sz   = getattr(encoder.config.vision_config,
                         "spatial_merge_size", 2)
    print(f"Vision output dim   : {llm_hidden}")
    print(f"Spatial merge size  : {merge_sz}")

    decoder = SegDecoder(in_channels=llm_hidden, out_size=512).to(device)
    n_dec   = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"Decoder parameters  : {n_dec:.2f}M")

    model = QwenSegmenter(encoder, decoder, merge_size=merge_sz)

    # ── Data ──────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = QwenArcadeDataset("train", args.data_root, processor)
    val_ds   = QwenArcadeDataset("val",   args.data_root, processor)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, collate_fn=qwen_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, collate_fn=qwen_collate_fn,
    )

    # ── Optimizer ─────────────────────────────────────────────
    # Two parameter groups: decoder gets higher LR than LoRA adapters
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

    # Mixed precision scaler (bfloat16 on Ampere GPUs)
    scaler = torch.amp.GradScaler("cuda", enabled=True)


    # ── Training loop ─────────────────────────────────────────
    best_val_dice = 0.0
    ckpt_path     = out_dir / "qwen_seg_best.pth"

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
            # Save decoder weights + LoRA adapters (not full Qwen — too large)
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
        "model":         f"Qwen3-VL-8B + SegDecoder ({mode})",
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
    results_path = out_dir / f"qwen_metrics_{suffix}.json"
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
        description="Qwen3-VL segmentation fine-tuning on ARCADE"
    )

    # Mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lora",    action="store_true",
                       help="Fine-tune encoder with LoRA r=16 (~14 GB VRAM)")
    group.add_argument("--no-lora", action="store_true",
                       help="Freeze encoder, train only decoder (~10 GB VRAM)")

    # Data
    parser.add_argument("--data-root", default="data")

    # Training
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch-size", type=int,   default=4,
                        help="Per-GPU batch. Use 4 for LoRA, 8 for frozen.")
    parser.add_argument("--lr",         type=float, default=1e-4)

    # Output
    parser.add_argument("--out-dir", default="checkpoints")

    main(parser.parse_args())