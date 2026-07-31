"""
qwen_unet_pipeline.py
=====================
Train a U-Net refiner on top of a frozen Qwen3-VL vessel mask predictor (ARCADE).

Pipeline:
    XCA image → Qwen (frozen ckpt) → vessel mask
             → concat with RGB → U-Net → refined mask

Included functionality:
    - curriculum target: dilated GT early, taper to thin GT
    - soft Qwen mask as U-Net input during training (clamped sigmoid)
    - connectivity loss (penalise vessel breaks)
    - Qwen-guided loss (discourage deleting high-confidence Qwen vessels)
    - EMA weights, early stopping, threshold search, optional TTA at test

Usage:
    python qwen_unet_pipeline.py \\
        --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \\
        --epochs 50 --batch-size 2

    python qwen_unet_pipeline.py \\
        --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \\
        --encoder efficientnet-b4 --epochs 50 --batch-size 2
"""

import argparse
import json
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ── Stałe ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-VL-8B-Instruct"
IMG_SIZE      = 512
DEC_DEVICE    = "cuda:0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Qwen mask predictor
# ══════════════════════════════════════════════════════════════════════════════

class QwenMaskPredictor(nn.Module):
    def __init__(self, vl_model, decoder, fusion=None,
                 spatial_merge: int = 2, patch_size: int = 16,
                 img_size: int = 512):
        super().__init__()
        self.vl_model = vl_model
        self.decoder  = decoder
        self.fusion   = fusion
        self.grid     = img_size // patch_size // spatial_merge

    @property
    def visual(self):
        m = self.vl_model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        return m.visual

    def forward(self, pixel_values, image_grid_thw):
        B    = image_grid_thw.shape[0]
        grid = self.grid
        last_hidden, deepstack = self.visual(
            hidden_states=pixel_values,
            grid_thw=image_grid_thw,
        )
        C = last_hidden.shape[-1]

        def to_spatial(x):
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

        logits = self.decoder(fused.float())
        return torch.sigmoid(logits)


# ══════════════════════════════════════════════════════════════════════════════
# SegDecoder + DeepStackFusion
# ══════════════════════════════════════════════════════════════════════════════

class SegDecoder(nn.Module):
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

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.project(x)
        for stage in self.stages:
            x = stage(x)
        return self.head(x)


class DeepStackFusion(nn.Module):
    def __init__(self, n_sources: int = 4):
        super().__init__()
        self.n_sources = n_sources
        self.logits    = nn.Parameter(torch.zeros(n_sources))

    def forward(self, feats):
        w   = torch.softmax(self.logits, dim=0)
        out = w[0] * feats[0]
        for i in range(1, self.n_sources):
            out = out + w[i] * feats[i]
        return out


# ══════════════════════════════════════════════════════════════════════════════
# Curriculum dilation of GT targets
# ══════════════════════════════════════════════════════════════════════════════

def get_dilation_kernel_size(epoch: int, total_epochs: int,
                              max_dil: int = 9, min_dil: int = 1) -> int:
    """
    Curriculum GT dilation: large kernel early (≈ Qwen thickness),
    shrink toward 1 by the last epoch. Kernel size is always odd.
    """
    progress = (epoch - 1) / max(total_epochs - 1, 1)   # 0.0 → 1.0
    size     = max_dil - progress * (max_dil - min_dil)
    size     = int(round(size))
    if size % 2 == 0:
        size += 1
    return max(size, 1)


def dilate_mask(mask_bin: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Dylatuje binarną maskę (float32, 0/1) z eliptycznym kernelem.
    kernel_size=1 → brak dylatacji (GT jak jest).
    """
    if kernel_size <= 1:
        return mask_bin
    k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.dilate(mask_bin.astype(np.uint8), k, iterations=1)
    return out.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Load Qwen checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_predictor(ckpt_path: str, finetune: bool = False) -> QwenMaskPredictor:
    print(f"\nŁadowanie Qwen3-VL z: {MODEL_ID}")
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    ckpt = torch.load(ckpt_path, map_location=DEC_DEVICE)
    cfg  = ckpt.get("config", {})
    print(f"  Checkpoint config: {cfg}")

    use_deepstack = cfg.get("use_deepstack", True)
    use_lora      = cfg.get("use_lora", False) or ("lora_state" in ckpt)

    if use_lora and "lora_state" in ckpt:
        print("  Aplikuję LoRA adaptery...")
        lora_r = cfg.get("lora_r") or 16
        lora_cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=["qkv", "linear_fc1", "linear_fc2"],
            lora_dropout=0.05, bias="none",
        )
        vl_model = get_peft_model(vl_model, lora_cfg)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(vl_model, ckpt["lora_state"])
        print("  LoRA wagi załadowane.")

    base = vl_model
    if hasattr(vl_model, "base_model") and hasattr(vl_model.base_model, "model"):
        base = vl_model.base_model.model
    vc = base.config.vision_config

    decoder = SegDecoder(in_channels=vc.out_hidden_size, base_channels=512).to(DEC_DEVICE)
    decoder.load_state_dict(ckpt["decoder"])
    print(f"  Decoder załadowany. Params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

    fusion = None
    if use_deepstack and ckpt.get("fusion") is not None:
        fusion = DeepStackFusion(n_sources=4).to(DEC_DEVICE)
        fusion.load_state_dict(ckpt["fusion"])
        print("  DeepStack fusion załadowany.")

    predictor = QwenMaskPredictor(
        vl_model=vl_model, decoder=decoder, fusion=fusion,
        spatial_merge=vc.spatial_merge_size, patch_size=vc.patch_size,
        img_size=IMG_SIZE,
    )

    if not finetune:
        for p in predictor.parameters():
            p.requires_grad = False
        predictor.eval()
        print("  Qwen ZAMROŻONY — trenujemy tylko U-Net.")
    else:
        for name, p in predictor.named_parameters():
            if "lora_" in name or "decoder" in name or "fusion" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        print("  Qwen częściowo odblokowany (LoRA + decoder).")

    return predictor


# ══════════════════════════════════════════════════════════════════════════════
# Dataset — z curriculum dilation
# ══════════════════════════════════════════════════════════════════════════════

class HybridArcadeDataset(Dataset):
    """
    Dataset with a per-epoch `dilation_kernel` for curriculum GT targets.
    """

    def __init__(self, split: str, data_root: str, qwen_processor,
                 img_size: int = 512, dilation_kernel: int = 9):
        self.processor       = qwen_processor
        self.img_size        = img_size
        self.split           = split
        self.dilation_kernel = dilation_kernel   # ← aktualizowane co epokę

        if split == "train":
            self.geo_aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, border_mode=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=0, border_mode=0, p=0.4),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            ])
            self.photo_aug = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.25, p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.4),
            ])
        else:
            self.geo_aug   = A.Compose([A.Resize(img_size, img_size)])
            self.photo_aug = None

        self.unet_norm = A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

        root    = Path(data_root)
        img_dir = root / "syntax" / split / "images"
        msk_dir = root / "masks"  / split

        if not img_dir.exists():
            raise FileNotFoundError(img_dir)
        if not msk_dir.exists():
            raise FileNotFoundError(msk_dir)

        self.pairs = [
            (p, msk_dir / p.name)
            for p in sorted(img_dir.glob("*.png"))
            if (msk_dir / p.name).exists()
        ]
        if not self.pairs:
            raise RuntimeError(f"Brak par obraz-maska dla split={split}")
        print(f"[{split}] {len(self.pairs)} par")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        gray     = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask     = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb      = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask_raw = (mask > 127).astype(np.float32)

        aug      = self.geo_aug(image=rgb, mask=mask_raw)
        img_aug  = aug["image"]
        mask_aug = aug["mask"]

        if img_aug.dtype != np.uint8:
            img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        if self.photo_aug is not None:
            img_aug = self.photo_aug(image=img_aug)["image"]
            if img_aug.dtype != np.uint8:
                img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        # ── CURRICULUM: dilate target ──────────────────────────────────────
        # mask_aug     = GT (cienkie linie) — do ewaluacji Dice (nie zmieniamy)
        # mask_target  = dilatowana GT — jako target treningu U-Neta
        mask_target = dilate_mask(mask_aug, self.dilation_kernel)

        pil  = PILImage.fromarray(img_aug)
        proc = self.processor.image_processor(images=[pil], return_tensors="pt")

        unet_out   = self.unet_norm(image=img_aug, mask=mask_aug)
        image_unet = unet_out["image"]

        return {
            "pixel_values":   proc["pixel_values"],
            "image_grid_thw": proc["image_grid_thw"][0],
            "image_unet":     image_unet,
            "mask":           torch.from_numpy(mask_aug).unsqueeze(0),     # GT do Dice
            "mask_target":    torch.from_numpy(mask_target).unsqueeze(0),  # do loss
            "stem":           img_path.stem,
        }


def hybrid_collate_fn(batch):
    return {
        "pixel_values":   torch.cat(  [b["pixel_values"]   for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch], dim=0),
        "image_unet":     torch.stack([b["image_unet"]     for b in batch], dim=0),
        "mask":           torch.stack([b["mask"]            for b in batch], dim=0),
        "mask_target":    torch.stack([b["mask_target"]     for b in batch], dim=0),
        "stem":           [b["stem"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# U-Net refinement model
# ══════════════════════════════════════════════════════════════════════════════

def build_unet_refiner(encoder_name: str = "resnet34") -> nn.Module:
    model = smp.Unet(
        encoder_name           = encoder_name,
        encoder_weights        = "imagenet",
        in_channels            = 3,
        classes                = 1,
        activation             = None,
        decoder_attention_type = "scse",
    )

    first_conv = first_conv_name = None
    for name, m in model.encoder.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv      = m
            first_conv_name = name
            break

    if first_conv is None:
        raise RuntimeError("Nie znaleziono Conv2d w encoderze")

    print(f"  Patchuję Conv2d: '{first_conv_name}'  shape={tuple(first_conv.weight.shape)}")

    old_weight              = first_conv.weight.data.clone()
    out_ch, in_ch, kH, kW  = old_weight.shape
    new_weight              = torch.zeros(out_ch, in_ch + 1, kH, kW)
    new_weight[:, :in_ch]  = old_weight
    nn.init.kaiming_normal_(new_weight[:, in_ch:], mode="fan_out", nonlinearity="relu")

    new_conv = nn.Conv2d(
        in_ch + 1, out_ch,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight

    parts  = first_conv_name.split(".")
    parent = model.encoder
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)
    model.encoder._in_channels = 4

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net ({encoder_name} + scSE): {n_params:.1f}M params  (4ch wejście)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Loss: FocalTversky + Connectivity + QwenGuided
# ══════════════════════════════════════════════════════════════════════════════

class ConnectivityLoss(nn.Module):
    """
    Karze za przerwy w ciągłości naczyń.

    Idea: jeśli predykcja ma "dziurę" (lokalnie niski sigmoid) w miejscu
    gdzie Qwen był pewny (prob > qwen_thresh) → duża kara.

    Implementacja: różnica między predykcją a jej dylatowaną wersją w
    miejscach pewnej predykcji Qwena.
    """

    def __init__(self, kernel_size: int = 7, qwen_thresh: float = 0.6):
        super().__init__()
        self.qwen_thresh = qwen_thresh
        # Morfologiczna dylatacja przez max-pooling (różniczkowa)
        self.pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

    def forward(self, logits: torch.Tensor,
                qwen_prob: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(logits)

        # Maska pewnych naczyń Qwena
        qwen_certain = (qwen_prob > self.qwen_thresh).float()

        # Dylatowana predykcja (lokalne maksimum)
        pred_dilated = self.pool(pred)

        # Przerwa = miejsce gdzie lokalnie jest wysoka wartość (sąsiedztwo)
        # ale w danym pikselu nie ma (pred mała), a Qwen był pewny
        gap = (pred_dilated - pred).clamp(0) * qwen_certain

        return gap.mean()


class QwenGuidedLoss(nn.Module):
    """
    Karze U-Net za usuwanie pikseli które Qwen wykrył pewnie.

    Jeśli Qwen_prob > high_thresh → U-Net powinien też to wykryć.
    Jeśli Qwen_prob < low_thresh  → U-Net może robić co chce (Qwen się mylił).

    To jest "leash" który trzyma U-Net blisko maski Qwena dla pewnych naczyń.
    """

    def __init__(self, high_thresh: float = 0.75, low_thresh: float = 0.3):
        super().__init__()
        self.high_thresh = high_thresh
        self.low_thresh  = low_thresh

    def forward(self, logits: torch.Tensor,
                qwen_prob: torch.Tensor) -> torch.Tensor:
        # Gdzie Qwen był pewny że jest naczynie
        qwen_high = (qwen_prob > self.high_thresh).float()

        # Kara gdy U-Net nie wykrywa tego co Qwen widział z dużą pewnością.
        # Używamy binary_cross_entropy_with_logits (bezpieczne z autocast)
        # z target=ones → maksymalizuj predykcję tam gdzie Qwen był pewny
        loss = F.binary_cross_entropy_with_logits(
            logits,
            torch.ones_like(logits),
            reduction="none",
        ) * qwen_high

        return loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 2.0,
                 smooth: float = 1.0, focal_weight: float = 0.4,
                 pos_weight: float = 20.0):
        super().__init__()
        self.alpha        = alpha
        self.gamma        = gamma
        self.smooth       = smooth
        self.focal_weight = focal_weight
        self.register_buffer("pw", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)

        bce_raw = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pw.to(logits.device),
            reduction="none",
        )
        p_t   = p * targets + (1 - p) * (1 - targets)
        focal = ((1 - p_t) ** self.gamma * bce_raw).mean()

        pf = p.view(-1);  tf = targets.view(-1)
        tp = (pf * tf).sum()
        fp = (pf * (1 - tf)).sum()
        fn = ((1 - pf) * tf).sum()
        tversky = 1.0 - (tp + self.smooth) / (
            tp + (1 - self.alpha) * fp + self.alpha * fn + self.smooth
        )

        return self.focal_weight * focal + (1 - self.focal_weight) * tversky


class CombinedLossV4(nn.Module):
    """
    Combined loss:

      L = w_main * FocalTversky(pred, dilated_GT)
        + w_conn * ConnectivityLoss(pred, qwen_prob)
        + w_qwen * QwenGuidedLoss(pred, qwen_prob)

    Wagi:
      w_conn: rośnie od 0 do conn_max w pierwszych warmup_conn epokach.
              Na początku U-Net uczy się podstawowego kształtu,
              potem connectivity zaczyna wymuszać ciągłość.
      w_qwen: stały — zawsze trzyma U-Net blisko pewnych naczyń Qwena.
    """

    def __init__(self, pos_weight: float = 20.0,
                 conn_max: float = 0.3, qwen_w: float = 0.2,
                 warmup_conn: int = 5):
        super().__init__()
        self.focal_tversky  = FocalTverskyLoss(pos_weight=pos_weight)
        self.connectivity   = ConnectivityLoss(kernel_size=7, qwen_thresh=0.6)
        self.qwen_guided    = QwenGuidedLoss(high_thresh=0.75, low_thresh=0.3)
        self.conn_max       = conn_max
        self.qwen_w         = qwen_w
        self.warmup_conn    = warmup_conn
        self.current_epoch  = 1

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_conn_weight(self) -> float:
        """Connectivity loss rośnie stopniowo przez pierwsze warmup_conn epok."""
        progress = min(self.current_epoch / self.warmup_conn, 1.0)
        return self.conn_max * progress

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                qwen_prob: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits:    (B, 1, H, W) — surowe logity U-Neta
            targets:   (B, 1, H, W) — DILATOWANA GT (curriculum target)
            qwen_prob: (B, 1, H, W) — sigmoid Qwena (ciągły, nie binarny)
        """
        main_loss = self.focal_tversky(logits, targets)
        conn_loss = self.connectivity(logits, qwen_prob)
        qwen_loss = self.qwen_guided(logits, qwen_prob)

        w_conn = self.get_conn_weight()
        w_main = 1.0 - w_conn * 0.3    # lekko zmniejszamy main gdy conn rośnie

        total = w_main * main_loss + w_conn * conn_loss + self.qwen_w * qwen_loss

        return total, {
            "loss_main": main_loss.item(),
            "loss_conn": conn_loss.item(),
            "loss_qwen": qwen_loss.item(),
            "w_conn":    w_conn,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(preds_bin, targets, smooth=1e-6):
    p  = preds_bin.float().view(-1)
    t  = targets.float().view(-1)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()
    return {
        "dice":      ((2*tp + smooth) / (2*tp + fp + fn + smooth)).item(),
        "iou":       ((tp + smooth) / (tp + fp + fn + smooth)).item(),
        "px_acc":    ((tp + tn) / (tp + tn + fp + fn + smooth)).item(),
        "precision": ((tp + smooth) / (tp + fp + smooth)).item(),
        "recall":    ((tp + smooth) / (tp + fn + smooth)).item(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Early stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = 0.0

    def __call__(self, val_dice: float) -> bool:
        if val_dice > self.best + self.min_delta:
            self.best    = val_dice
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ══════════════════════════════════════════════════════════════════════════════
# Test-time augmentation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(qwen, unet, pv, thw, img_un, device, use_tta: bool = True,
                     soft_qwen: bool = False):
    """
    soft_qwen=True  → pass clamped Qwen probabilities into the U-Net (training).
    soft_qwen=False → binary Qwen mask (inference).

    Soft maska: sigmoid Qwena clampowany do [0.15, 1.0].
    Minimalny clamp 0.15 zapewnia że słabe naczynia nie giną całkowicie.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        qwen_prob = qwen(pv, thw).to(device).float()

    if soft_qwen:
        # Soft: zachowaj gradient pewności, minimalny sygnał = 0.15
        qwen_mask = qwen_prob.clamp(0.15, 1.0)
    else:
        # Binary mask for inference
        qwen_mask = (qwen_prob > 0.5).float()

    if not use_tta:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit = unet(torch.cat([img_un, qwen_mask], dim=1))
        return torch.sigmoid(logit), qwen_prob

    preds = []
    for flip_h, flip_v in [(False, False), (True, False), (False, True), (True, True)]:
        img = img_un.clone()
        qm  = qwen_mask.clone()
        if flip_h:
            img = torch.flip(img, dims=[3])
            qm  = torch.flip(qm,  dims=[3])
        if flip_v:
            img = torch.flip(img, dims=[2])
            qm  = torch.flip(qm,  dims=[2])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit = unet(torch.cat([img, qm], dim=1))
        pred = torch.sigmoid(logit)
        if flip_h: pred = torch.flip(pred, dims=[3])
        if flip_v: pred = torch.flip(pred, dims=[2])
        preds.append(pred)

    return torch.stack(preds).mean(0), qwen_prob


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval loops
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(qwen, unet, loader, optimizer, criterion, device, scaler):
    unet.train()
    qwen.eval()

    total_loss = 0.0
    loss_parts = {"loss_main": 0.0, "loss_conn": 0.0, "loss_qwen": 0.0}
    totals     = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  train", leave=False):
        pv      = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw     = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un  = batch["image_unet"].to(device)
        tgt_gt  = batch["mask"].to(device)         # GT — do Dice
        tgt_dil = batch["mask_target"].to(device)  # dilatowana GT — do loss

        # Qwen → soft maska (clamp 0.15) zamiast binarnej
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_prob = qwen(pv, thw).to(device).float()

        # Soft Qwen mask as extra input channel
        qwen_soft = qwen_prob.clamp(0.15, 1.0).detach()
        unet_input = torch.cat([img_un, qwen_soft], dim=1)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = unet(unet_input)
            # Dilated GT target + Qwen probs for guided loss
            loss, parts = criterion(logits, tgt_dil, qwen_prob.detach())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in loss_parts:
            loss_parts[k] += parts.get(k, 0.0)

        # Dice liczymy względem oryginalnego GT (nie dilatowanego)
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).long()
            for k, v in compute_metrics(preds_bin, tgt_gt).items():
                totals[k] += v

    n = len(loader)
    result = {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}
    result.update({k: v / n for k, v in loss_parts.items()})
    return result


@torch.no_grad()
def eval_epoch(qwen, unet, loader, criterion, device,
               threshold: float = 0.5, use_tta: bool = False):
    unet.eval(); qwen.eval()

    total_loss = 0.0
    totals = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  eval ", leave=False):
        pv      = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw     = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un  = batch["image_unet"].to(device)
        tgt     = batch["mask"].to(device)
        tgt_dil = batch["mask_target"].to(device)

        if use_tta:
            prob, qwen_prob = predict_with_tta(
                qwen, unet, pv, thw, img_un, device, use_tta=True, soft_qwen=False)
            logits    = torch.log(prob.clamp(1e-6, 1-1e-6) / (1 - prob.clamp(1e-6, 1-1e-6)))
            preds_bin = (prob > threshold).long()
        else:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_prob = qwen(pv, thw).to(device).float()
            qwen_soft  = qwen_prob.clamp(0.15, 1.0)
            unet_input = torch.cat([img_un, qwen_soft], dim=1)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = unet(unet_input)
            preds_bin  = (torch.sigmoid(logits) > threshold).long()

        loss, _ = criterion(logits, tgt_dil, qwen_prob)
        total_loss += loss.item()
        for k, v in compute_metrics(preds_bin, tgt).items():
            totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Threshold search on validation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def find_best_threshold(qwen, unet, val_loader, device, thresholds=None) -> float:
    if thresholds is None:
        thresholds = np.arange(0.25, 0.75, 0.025)

    unet.eval(); qwen.eval()
    all_probs, all_targets = [], []

    for batch in tqdm(val_loader, desc="  threshold search", leave=False):
        pv     = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw    = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un = batch["image_unet"].to(device)
        tgt    = batch["mask"]

        prob, _ = predict_with_tta(
            qwen, unet, pv, thw, img_un, device, use_tta=False, soft_qwen=False)
        all_probs.append(prob.cpu())
        all_targets.append(tgt)

    probs   = torch.cat(all_probs)
    targets = torch.cat(all_targets)

    best_t, best_dice = 0.5, 0.0
    for t in thresholds:
        d = compute_metrics((probs > t).long(), targets.long())["dice"]
        if d > best_dice:
            best_dice, best_t = d, float(t)

    print(f"  Optymalny próg: {best_t:.3f}  (val Dice: {best_dice:.4f})")
    return best_t


# ══════════════════════════════════════════════════════════════════════════════
# Save prediction panels
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_predictions(qwen, unet, loader, device, out_dir: Path,
                     threshold: float = 0.5, n: int = 12):
    out_dir.mkdir(parents=True, exist_ok=True)
    unet.eval(); qwen.eval()

    mean  = np.array(IMAGENET_MEAN)
    std   = np.array(IMAGENET_STD)
    saved = 0

    for batch in loader:
        if saved >= n:
            break

        pv     = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw    = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un = batch["image_unet"].to(device)
        tgt    = batch["mask"]
        stems  = batch["stem"]

        prob, qwen_prob = predict_with_tta(
            qwen, unet, pv, thw, img_un, device, use_tta=True, soft_qwen=False)
        preds_bin = (prob > threshold).cpu().numpy()
        qwen_np   = (qwen_prob > 0.5).cpu().numpy()

        for i in range(len(stems)):
            if saved >= n:
                break

            img_np   = img_un[i].cpu().numpy().transpose(1, 2, 0)
            img_np   = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            gt_np = tgt[i, 0].numpy().astype(bool)
            pr_np = preds_bin[i, 0].astype(bool)
            qw_np = qwen_np[i, 0].astype(bool)

            gt_vis = (gt_np * 255).astype(np.uint8)
            pr_vis = (pr_np * 255).astype(np.uint8)
            qw_vis = (qw_np * 255).astype(np.uint8)

            overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            overlay[pr_np &  gt_np] = (0,   200,   0)
            overlay[pr_np & ~gt_np] = (0,   0,   200)
            overlay[~pr_np & gt_np] = (200, 0,     0)
            overlay = cv2.addWeighted(
                cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR), 0.45,
                overlay, 0.55, 0)

            header_h = 22
            cols   = [img_gray, qw_vis, pr_vis, gt_vis,
                      cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)]
            labels = ["XCA", "Qwen", "Unet+Qwen", "GT", "Overlay"]

            panels = []
            for col, lbl in zip(cols, labels):
                hdr = np.zeros((header_h, IMG_SIZE), dtype=np.uint8)
                cv2.putText(hdr, lbl, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)
                panels.append(np.vstack([hdr, col]))

            panel   = np.hstack(panels)
            metrics = compute_metrics(
                torch.from_numpy(preds_bin[i:i+1]),
                tgt[i:i+1].long(),
            )
            fname = f"{stems[i]}_dice{metrics['dice']:.3f}_rec{metrics['recall']:.3f}.png"
            cv2.imwrite(str(out_dir / fname), panel)
            saved += 1

    print(f"  Zapisano {saved} paneli → {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name}  ({vram:.1f} GB)")

    device = torch.device(DEC_DEVICE)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    qwen      = load_qwen_predictor(args.qwen_ckpt, finetune=False)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    unet      = build_unet_refiner(encoder_name=args.encoder).to(device)

    ema_unet = AveragedModel(unet, multi_avg_fn=get_ema_multi_avg_fn(decay=0.9999))
    print(f"EMA skonfigurowane (decay=0.9999, aktywne od epoki {args.ema_start})")

    # ── Dane z curriculum dilation ─────────────────────────────────────────
    print("\nŁaduję datasety...")
    init_dil = get_dilation_kernel_size(1, args.epochs,
                                         max_dil=args.max_dilation,
                                         min_dil=args.min_dilation)
    print(f"  Curriculum: dilation {args.max_dilation}→{args.min_dilation}px "
          f"przez {args.epochs} epok. Epoka 1: kernel={init_dil}")

    train_ds = HybridArcadeDataset("train", args.data_root, processor, IMG_SIZE,
                                    dilation_kernel=init_dil)
    val_ds   = HybridArcadeDataset("val",   args.data_root, processor, IMG_SIZE,
                                    dilation_kernel=init_dil)

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True,
                        collate_fn=hybrid_collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    criterion = CombinedLossV4(
        pos_weight=args.pos_weight,
        conn_max=args.conn_weight,
        qwen_w=args.qwen_weight,
        warmup_conn=args.warmup_conn,
    )
    scaler     = torch.cuda.amp.GradScaler()
    early_stop = EarlyStopping(patience=args.patience, min_delta=1e-4)

    print(f"\nEncoder: {args.encoder} | Epochs: {args.epochs} | "
          f"Batch: {args.batch_size} | LR: {args.lr}")
    print(f"Loss: FocalTversky + Connectivity(w={args.conn_weight}) "
          f"+ QwenGuided(w={args.qwen_weight})")
    print(f"Curriculum: dilation {args.max_dilation}→{args.min_dilation}px")

    best_val_dice = 0.0
    ckpt_path     = Path(args.out_dir) / "qwen_unet_best.pth"
    history       = []

    header = (f"{'Epoch':>5} | {'Loss':>7} | {'Tr.Dice':>7} | "
              f"{'Val.Dice':>8} | {'Lconn':>7} | {'Lqwen':>6} | {'Dil':>4} | {'LR':>8}")
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ── Aktualizuj curriculum dilation ────────────────────────────────
        dil_k = get_dilation_kernel_size(epoch, args.epochs,
                                          max_dil=args.max_dilation,
                                          min_dil=args.min_dilation)
        train_ds.dilation_kernel = dil_k
        val_ds.dilation_kernel   = dil_k
        criterion.set_epoch(epoch)

        tr = train_epoch(qwen, unet, train_loader, optimizer, criterion, device, scaler)

        if epoch >= args.ema_start:
            ema_unet.update_parameters(unet)

        vl = eval_epoch(qwen, unet, val_loader, criterion, device,
                        threshold=0.5, use_tta=False)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        sec        = time.time() - t0

        print(f"{epoch:>5} | {tr['loss']:>7.4f} | {tr['dice']:>7.4f} | "
              f"{vl['dice']:>8.4f} | {tr.get('loss_conn', 0):>7.4f} | "
              f"{tr.get('loss_qwen', 0):>6.4f} | {dil_k:>4} | {current_lr:>8.2e}  "
              f"({sec:.0f}s)")

        history.append({
            "epoch": epoch, "lr": current_lr, "dilation": dil_k,
            "train": tr, "val": vl,
        })

        if vl["dice"] > best_val_dice:
            best_val_dice = vl["dice"]
            save_weights  = (ema_unet.module.state_dict()
                             if epoch >= args.ema_start else unet.state_dict())
            torch.save({
                "epoch":    epoch,
                "unet":     save_weights,
                "val_dice": best_val_dice,
                "config": {
                    "qwen_ckpt":     args.qwen_ckpt,
                    "encoder":       args.encoder,
                    "batch_size":    args.batch_size,
                    "lr":            args.lr,
                    "pos_weight":    args.pos_weight,
                    "max_dilation":  args.max_dilation,
                    "min_dilation":  args.min_dilation,
                    "conn_weight":   args.conn_weight,
                    "qwen_weight":   args.qwen_weight,
                },
            }, ckpt_path)
            ema_str = "(EMA)" if epoch >= args.ema_start else "(raw)"
            print(f"       ↑ best val Dice {best_val_dice:.4f} {ema_str}  dil={dil_k}")

        if early_stop(vl["dice"]):
            print(f"\nEarly stopping w epoce {epoch}")
            break

    # ── Optymalizacja progu ─────────────────────────────────────────────────
    print("\nSzukam optymalnego progu decyzyjnego...")
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    best_threshold = find_best_threshold(qwen, unet, val_loader, device)

    # ── Test z TTA ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Ewaluacja testowa z TTA i optymalnym progiem...")

    try:
        test_ds = HybridArcadeDataset("test", args.data_root, processor, IMG_SIZE,
                                       dilation_kernel=1)   # test: bez dylatacji
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, **loader_kw)

        test_m = eval_epoch(qwen, unet, test_loader, criterion, device,
                            threshold=best_threshold, use_tta=True)

        print("\nWYNIKI TESTOWE (EMA + TTA + optymalny próg)")
        print("┌──────────────────────────────────────────┐")
        for k, v in test_m.items():
            if k != "loss":
                print(f"│  {k:<14} {v:.4f}                   │")
        print(f"│  {'threshold':<14} {best_threshold:.3f}                   │")
        print("└──────────────────────────────────────────┘")

        pred_dir = Path(args.results_dir) / "qwen_unet_v4_predictions"
        save_predictions(qwen, unet, test_loader, device,
                         pred_dir, threshold=best_threshold, n=12)

        results = {
            "model":         f"Qwen3-VL → U-Net ({args.encoder}+scSE+curriculum)",
            "dataset":       "ARCADE syntax — binary vessel segmentation",
            "split":         "test",
            "best_val_dice": round(best_val_dice, 4),
            "threshold":     round(best_threshold, 3),
            "test_metrics":  {k: round(v, 4) for k, v in test_m.items()},
            "history":       history,
            "config": vars(args),
        }
        out_json = Path(args.results_dir) / "qwen_unet_v4_metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nWyniki → {out_json}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"[SKIP] Ewaluacja testowa pominięta: {e}")

    print(f"Checkpoint → {ckpt_path}")
    print("Gotowe!")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL + U-Net refinement (curriculum + connectivity loss)"
    )
    parser.add_argument("--qwen-ckpt",     required=True)
    parser.add_argument("--encoder",       default="resnet34",
                        choices=["resnet34", "resnet50", "efficientnet-b4",
                                 "resnext50_32x4d", "timm-efficientnet-b5"])
    parser.add_argument("--data-root",     default="data")
    parser.add_argument("--out-dir",       default="checkpoints")
    parser.add_argument("--results-dir",   default="results")
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch-size",    type=int,   default=2)
    parser.add_argument("--lr",            type=float, default=1e-5)
    parser.add_argument("--num-workers",   type=int,   default=2)
    parser.add_argument("--patience",      type=int,   default=10)
    parser.add_argument("--pos-weight",    type=float, default=20.0)
    parser.add_argument("--ema-start",     type=int,   default=5)
    # ── Curriculum ──────────────────────────────────────────────────────────
    parser.add_argument("--max-dilation",  type=int,   default=9,
                        help="Rozmiar kernela dylatacji w epoce 1 (styl Qwena)")
    parser.add_argument("--min-dilation",  type=int,   default=1,
                        help="Rozmiar kernela dylatacji w ostatniej epoce (GT)")
    # ── Loss weights ────────────────────────────────────────────────────────
    parser.add_argument("--conn-weight",   type=float, default=0.3,
                        help="Max waga connectivity loss")
    parser.add_argument("--qwen-weight",   type=float, default=0.2,
                        help="Waga QwenGuided loss (trzyma U-Net blisko Qwena)")
    parser.add_argument("--warmup-conn",   type=int,   default=5,
                        help="Epoki warmup dla connectivity loss")

    main(parser.parse_args())