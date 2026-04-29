"""
train_qwen_unet.py  —  Qwen3-VL (frozen) → U-Net refinement pipeline
======================================================================
Coronary vessel segmentation on ARCADE syntax dataset.

IDEA:
    Qwen3-VL (twój wytrenowany checkpoint: qwen_seg_best_LoRA_deepstack.pth)
    produkuje wstępną maskę. U-Net dostaje CONCAT oryginalnego obrazu i tej
    maski (2 kanały wejścia) i uczy się ją poprawiać.

    XCA obraz (512×512)
          │
          ├──────────────────────────────────────────┐
          │                                          │
    Qwen3-VL (zamrożony, twój ckpt)           oryginalny obraz
          │                                    (normalizowany)
    maska wstępna (sigmoid → float)                  │
          │                                          │
          └──────────── concat ──────────────────────┘
                            │
                     (B, 2, 512, 512)
                            │
                   U-Net ResNet-34 encoder
                   (wagi ImageNet na kanale 1,
                    kanal 2 = maska Qwena)
                            │
                   finalna maska (B, 1, 512, 512)
                            │
                   loss vs ground truth

UŻYCIE:
    # Podstawowe — tylko U-Net refinement, Qwen zamrożony
    python train_qwen_unet.py \
        --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
        --epochs 50 --batch-size 4

    # Z LoRA w Qwenie (fine-tune Qwena razem z U-Netem, bardzo mało lr)
    python train_qwen_unet.py \
        --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
        --finetune-qwen --epochs 30 --batch-size 2

WYNIKI DO TABELI W PRACY:
    Row 1: U-Net baseline                 test Dice ~0.7955
    Row 2: Qwen3-VL frozen + DeepStack    (twój wynik)
    Row 3: Qwen3-VL LoRA  + DeepStack     (twój wynik)
    Row 4: Qwen + U-Net refinement        ← TEN SKRYPT
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
from albumentations.pytorch import ToTensorV2
from PIL import Image as PILImage
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ── Stałe ─────────────────────────────────────────────────────────────────────
MODEL_ID   = "Qwen/Qwen3-VL-8B-Instruct"
IMG_SIZE   = 512
DEC_DEVICE = "cuda:0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Qwen wrapper  —  produkuje maskę wstępną, zamrożony podczas treningu U-Neta
# ══════════════════════════════════════════════════════════════════════════════

class QwenMaskPredictor(nn.Module):
    """
    Ładuje wytrenowanego Qwena (SegDecoder + opcjonalnie LoRA + DeepStackFusion)
    i produkuje binarną maskę prawdopodobieństwa (sigmoid, nie thresholded).

    Zawsze zamrożony podczas treningu U-Neta (chyba że --finetune-qwen).
    """

    def __init__(self, vl_model, decoder, fusion=None,
                 spatial_merge: int = 2, patch_size: int = 16,
                 img_size: int = 512):
        super().__init__()
        self.vl_model = vl_model
        self.decoder  = decoder
        self.fusion   = fusion
        self.grid     = img_size // patch_size // spatial_merge  # = 16

    @property
    def visual(self):
        m = self.vl_model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            m = m.base_model.model
        return m.visual

    def forward(self, pixel_values: torch.Tensor,
                image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Zwraca prawdopodobieństwo (0-1), shape (B, 1, 512, 512)."""
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
        return torch.sigmoid(logits)   # (B, 1, 512, 512)  wartości 0-1


# ══════════════════════════════════════════════════════════════════════════════
# SegDecoder + DeepStackFusion  —  skopiowane z train_qwen_seg_1.py
# (musimy zrekonstruować architekturę żeby załadować wagi)
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
# Ładowanie checkpointu Qwena
# ══════════════════════════════════════════════════════════════════════════════

def load_qwen_predictor(ckpt_path: str, finetune: bool = False) -> QwenMaskPredictor:
    """
    Ładuje Qwen3-VL + SegDecoder + DeepStackFusion z checkpointu.
    Jeśli checkpoint zawiera 'lora_state', aplikuje LoRA adaptery.
    """
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

    vc = vl_model.config.vision_config

    # Aplikuj LoRA jeśli checkpoint go zawiera
    if use_lora and "lora_state" in ckpt:
        print("  Aplikuję LoRA adaptery...")
        lora_r = cfg.get("lora_r") or 16
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["qkv", "linear_fc1", "linear_fc2"],
            lora_dropout=0.05,
            bias="none",
        )
        vl_model = get_peft_model(vl_model, lora_cfg)
        # Ładuj wagi adapterów
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(vl_model, ckpt["lora_state"])
        print("  LoRA wagi załadowane.")

    # Decoder
    decoder = SegDecoder(
        in_channels=vc.out_hidden_size,
        base_channels=512,
    ).to(DEC_DEVICE)
    decoder.load_state_dict(ckpt["decoder"])
    print(f"  Decoder załadowany. Params: {sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

    # DeepStack fusion
    fusion = None
    if use_deepstack and ckpt.get("fusion") is not None:
        fusion = DeepStackFusion(n_sources=4).to(DEC_DEVICE)
        fusion.load_state_dict(ckpt["fusion"])
        print("  DeepStack fusion załadowany.")

    predictor = QwenMaskPredictor(
        vl_model=vl_model,
        decoder=decoder,
        fusion=fusion,
        spatial_merge=vc.spatial_merge_size,
        patch_size=vc.patch_size,
        img_size=IMG_SIZE,
    )

    # Zamroź Qwena (chyba że finetune)
    if not finetune:
        for p in predictor.parameters():
            p.requires_grad = False
        predictor.eval()
        print("  Qwen ZAMROŻONY — trenujemy tylko U-Net.")
    else:
        # Zamroź backbone, odblokuj tylko LoRA + decoder
        for name, p in predictor.named_parameters():
            if "lora_" in name or "decoder" in name or "fusion" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        print("  Qwen częściowo odblokowany (LoRA + decoder).")

    return predictor


# ══════════════════════════════════════════════════════════════════════════════
# Dataset  —  zwraca obraz w DWÓCH formatach:
#   1. dla Qwena  (pixel_values + image_grid_thw)
#   2. dla U-Neta (znormalizowany tensor 3ch ImageNet)
# ══════════════════════════════════════════════════════════════════════════════

class HybridArcadeDataset(Dataset):
    """
    Każdy __getitem__ zwraca:
        pixel_values   — dla Qwena (jego własny preprocessor)
        image_grid_thw — dla Qwena
        image_unet     — (3, 512, 512) ImageNet-normalized tensor dla U-Neta
        mask           — (1, 512, 512) binary ground truth
        stem           — nazwa pliku
    """

    def __init__(self, split: str, data_root: str, qwen_processor,
                 img_size: int = 512):
        self.processor = qwen_processor
        self.img_size  = img_size
        self.split     = split

        # Augmentacje geometryczne — aplikowane do obu wejść jednocześnie
        if split == "train":
            self.geo_aug = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=20, border_mode=0, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=0, border_mode=0, p=0.4),
                A.ElasticTransform(alpha=30, sigma=5, p=0.3),
            ])
            self.photo_aug = A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.25, p=0.6),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ])
        else:
            self.geo_aug   = A.Compose([A.Resize(img_size, img_size)])
            self.photo_aug = None

        # Normalizacja ImageNet dla U-Neta
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
        print(f"[{split}] {len(self.pairs)} par (tryb hybrydowy)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        mask_bin = (mask > 127).astype(np.float32)

        # Geometryczna augmentacja (wspólna dla obrazu i maski)
        aug      = self.geo_aug(image=rgb, mask=mask_bin)
        img_aug  = aug["image"]
        mask_aug = aug["mask"]

        if img_aug.dtype != np.uint8:
            img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        # Fotograficzna augmentacja (tylko obraz, nie maska)
        if self.photo_aug is not None:
            img_aug = self.photo_aug(image=img_aug)["image"]
            if img_aug.dtype != np.uint8:
                img_aug = np.clip(img_aug, 0, 255).astype(np.uint8)

        # Format dla Qwena (jego własny preprocessor)
        pil  = PILImage.fromarray(img_aug)
        proc = self.processor.image_processor(images=[pil], return_tensors="pt")

        # Format dla U-Neta (ImageNet normalizacja)
        unet_out   = self.unet_norm(image=img_aug, mask=mask_aug)
        image_unet = unet_out["image"]  # (3, 512, 512)

        return {
            "pixel_values":   proc["pixel_values"],       # (N_patches, patch_dim)
            "image_grid_thw": proc["image_grid_thw"][0],  # (3,)
            "image_unet":     image_unet,                  # (3, 512, 512)
            "mask":           torch.from_numpy(mask_aug).unsqueeze(0),  # (1, 512, 512)
            "stem":           img_path.stem,
        }


def hybrid_collate_fn(batch):
    """Qwen wymaga concat patchy, reszta to standardowy stack."""
    return {
        "pixel_values":   torch.cat(  [b["pixel_values"]   for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch], dim=0),
        "image_unet":     torch.stack([b["image_unet"]     for b in batch], dim=0),
        "mask":           torch.stack([b["mask"]            for b in batch], dim=0),
        "stem":           [b["stem"] for b in batch],
    }


# ══════════════════════════════════════════════════════════════════════════════
# U-Net refinement model
# ══════════════════════════════════════════════════════════════════════════════

def build_unet_refiner() -> nn.Module:
    """
    U-Net ResNet-34 z 4 kanałami wejścia:
        kanał 0-2: oryginalny obraz XCA (ImageNet normalized)
        kanał 3:   maska wstępna Qwena (0-1 float)

    Strategia: budujemy model z in_channels=3 (żeby załadować wagi ImageNet),
    potem ręcznie patchujemy pierwszą Conv2d na 4 kanały zachowując wagi RGB.
    """
    # Buduj z 3 kanałami żeby załadować wagi ImageNet
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )

    # Znajdź pierwszą Conv2d w encoderze (niezależnie od nazwy warstwy w SMP)
    first_conv = None
    first_conv_name = None
    for name, m in model.encoder.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv      = m
            first_conv_name = name
            break

    if first_conv is None:
        raise RuntimeError("Nie znaleziono Conv2d w encoderze ResNet-34")

    print(f"  Patchuję pierwszą Conv2d: '{first_conv_name}'  "
          f"shape={tuple(first_conv.weight.shape)}")

    # Zachowaj wagi ImageNet dla kanałów RGB, dodaj kanał dla maski
    old_weight = first_conv.weight.data.clone()          # (64, 3, 7, 7)
    out_ch, in_ch, kH, kW = old_weight.shape
    new_weight = torch.zeros(out_ch, in_ch + 1, kH, kW)
    new_weight[:, :in_ch, :, :] = old_weight            # ImageNet RGB
    nn.init.kaiming_normal_(                             # kanał maski Qwena
        new_weight[:, in_ch:, :, :], mode="fan_out", nonlinearity="relu"
    )

    # Zamień Conv2d na nową z 4 kanałami wejścia
    new_conv = nn.Conv2d(
        in_ch + 1, out_ch,
        kernel_size = first_conv.kernel_size,
        stride      = first_conv.stride,
        padding     = first_conv.padding,
        bias        = first_conv.bias is not None,
    )
    new_conv.weight.data = new_weight

    # Wstaw nową Conv2d w odpowiednie miejsce (obsługa zagnieżdżonych modułów)
    parts = first_conv_name.split(".")
    parent = model.encoder
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    # Poinformuj SMP o nowej liczbie kanałów
    model.encoder._in_channels = 4

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"U-Net refiner params: {n_params:.1f}M  (wejście: 4 kanały)")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Loss + metryki
# ══════════════════════════════════════════════════════════════════════════════

class BCEDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce  = self.bce(logits, targets)
        p    = torch.sigmoid(logits).view(-1)
        t    = targets.view(-1)
        dice = 1.0 - (2.0 * (p * t).sum() + self.smooth) / (
            p.sum() + t.sum() + self.smooth)
        return 0.5 * bce + 0.5 * dice


def compute_metrics(preds_bin, targets, smooth=1e-6):
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


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval loops
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(qwen, unet, loader, optimizer, criterion, device):
    unet.train()
    # Qwen: eval jeśli zamrożony, train jeśli finetune
    if any(p.requires_grad for p in qwen.parameters()):
        qwen.train()
    else:
        qwen.eval()

    total_loss = 0.0
    totals = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  train", leave=False):
        pv      = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw     = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un  = batch["image_unet"].to(device)
        tgt     = batch["mask"].to(device)

        # Krok 1: Qwen → maska wstępna
        with torch.no_grad() if not any(p.requires_grad for p in qwen.parameters()) \
                else torch.enable_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                qwen_mask = qwen(pv, thw)  # (B, 1, 512, 512)  wartości 0-1
        qwen_mask = qwen_mask.to(device).float().detach() \
            if not any(p.requires_grad for p in qwen.parameters()) \
            else qwen_mask.to(device).float()

        # Krok 2: U-Net → finalna maska
        unet_input = torch.cat([img_un, qwen_mask], dim=1)  # (B, 4, 512, 512)

        optimizer.zero_grad()
        logits = unet(unet_input)
        loss   = criterion(logits, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds_bin = (torch.sigmoid(logits) > 0.5).long()
            for k, v in compute_metrics(preds_bin, tgt).items():
                totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


@torch.no_grad()
def eval_epoch(qwen, unet, loader, criterion, device):
    unet.eval()
    qwen.eval()

    total_loss = 0.0
    totals = {k: 0.0 for k in ("dice", "iou", "px_acc", "precision", "recall")}

    for batch in tqdm(loader, desc="  eval ", leave=False):
        pv      = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw     = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un  = batch["image_unet"].to(device)
        tgt     = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qwen_mask = qwen(pv, thw)
        qwen_mask = qwen_mask.to(device).float()

        unet_input = torch.cat([img_un, qwen_mask], dim=1)
        logits     = unet(unet_input)
        total_loss += criterion(logits, tgt).item()

        preds_bin = (torch.sigmoid(logits) > 0.5).long()
        for k, v in compute_metrics(preds_bin, tgt).items():
            totals[k] += v

    n = len(loader)
    return {"loss": total_loss / n, **{k: v / n for k, v in totals.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Zapis przykładowych predykcji (do pracy)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def save_predictions(qwen, unet, loader, device, out_dir: Path, n: int = 12):
    """
    Zapisuje panele: XCA | maska Qwena | maska U-Net | ground truth
    Bezpośrednio do rozdziału wynikowego pracy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    unet.eval(); qwen.eval()

    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    saved = 0

    for batch in loader:
        if saved >= n:
            break

        pv     = batch["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
        thw    = batch["image_grid_thw"].to(DEC_DEVICE)
        img_un = batch["image_unet"].to(device)
        tgt    = batch["mask"]
        stems  = batch["stem"]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            qwen_mask = qwen(pv, thw)
        qwen_mask = qwen_mask.to(device).float()

        unet_input = torch.cat([img_un, qwen_mask], dim=1)
        logits     = unet(unet_input)
        preds_bin  = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        qwen_np    = (qwen_mask > 0.5).cpu().numpy()

        for i in range(len(stems)):
            if saved >= n:
                break

            img_np = img_un[i].cpu().numpy().transpose(1, 2, 0)
            img_np = ((img_np * std + mean) * 255).clip(0, 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            gt_mask    = (tgt[i, 0].numpy() * 255).astype(np.uint8)
            qwen_vis   = (qwen_np[i, 0] * 255).astype(np.uint8)
            unet_vis   = (preds_bin[i, 0] * 255).astype(np.uint8)

            panel = np.hstack([img_gray, qwen_vis, unet_vis, gt_mask])

            # Nagłówki
            h = panel.shape[0]
            labeled = np.zeros((h + 20, panel.shape[1]), dtype=np.uint8)
            labeled[20:, :] = panel
            for j, label in enumerate(["XCA", "Qwen", "Unet+Qwen", "GT"]):
                cv2.putText(labeled, label,
                            (j * IMG_SIZE + 5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 200, 1)

            dice = compute_metrics(
                torch.from_numpy(preds_bin[i:i+1]),
                tgt[i:i+1].long(),
            )["dice"]

            cv2.imwrite(str(out_dir / f"{stems[i]}_dice{dice:.3f}.png"), labeled)
            saved += 1

    print(f"  Zapisano {saved} paneli → {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print(f"\nCUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            n = torch.cuda.get_device_name(i)
            v = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {n}  ({v:.1f} GB)")

    device = torch.device(DEC_DEVICE)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Załaduj Qwena ──────────────────────────────────────────────────────
    qwen = load_qwen_predictor(args.qwen_ckpt, finetune=args.finetune_qwen)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # ── Zbuduj U-Net ───────────────────────────────────────────────────────
    unet = build_unet_refiner().to(device)

    # ── Dane ───────────────────────────────────────────────────────────────
    print("\nŁaduję datasety...")
    train_ds = HybridArcadeDataset("train", args.data_root, processor, IMG_SIZE)
    val_ds   = HybridArcadeDataset("val",   args.data_root, processor, IMG_SIZE)

    loader_kw = dict(
        num_workers  = args.num_workers,
        pin_memory   = True,
        collate_fn   = hybrid_collate_fn,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, **loader_kw)

    # ── Optimizer ──────────────────────────────────────────────────────────
    unet_params = list(unet.parameters())
    if args.finetune_qwen:
        qwen_params = [p for p in qwen.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": unet_params, "lr": args.lr},
            {"params": qwen_params, "lr": args.lr * 0.01},  # Qwen: 100x mniejszy lr
        ], weight_decay=1e-4)
        print(f"Finetune Qwen: {sum(p.numel() for p in qwen_params)/1e6:.2f}M params")
    else:
        optimizer = torch.optim.AdamW(unet_params, lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    criterion = BCEDiceLoss()

    print(f"U-Net trainable: {sum(p.numel() for p in unet_params)/1e6:.2f}M params")
    print(f"Epochs: {args.epochs}  batch: {args.batch_size}  lr: {args.lr}")

    # ── Trening ────────────────────────────────────────────────────────────
    best_val_dice = 0.0
    ckpt_path = Path(args.out_dir) / "qwen_unet_best.pth"
    history   = []

    header = f"{'Epoch':>5} | {'Loss':>7} | {'Tr.Dice':>7} | {'Val.Dice':>8} | {'Val.IoU':>7}"
    print(f"\n{header}")
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(qwen, unet, train_loader, optimizer, criterion, device)
        vl = eval_epoch(qwen,  unet, val_loader,   criterion, device)
        scheduler.step()
        sec = time.time() - t0

        print(f"{epoch:>5} | {tr['loss']:>7.4f} | {tr['dice']:>7.4f} | "
              f"{vl['dice']:>8.4f} | {vl['iou']:>7.4f}  ({sec:.0f}s)")
        history.append({"epoch": epoch, "train": tr, "val": vl})

        if vl["dice"] > best_val_dice:
            best_val_dice = vl["dice"]
            torch.save({
                "epoch":    epoch,
                "unet":     unet.state_dict(),
                "val_dice": best_val_dice,
                "config": {
                    "qwen_ckpt":     args.qwen_ckpt,
                    "finetune_qwen": args.finetune_qwen,
                    "batch_size":    args.batch_size,
                    "lr":            args.lr,
                },
            }, ckpt_path)
            print(f"       ↑ best val Dice {best_val_dice:.4f} — zapisany")

    # ── Test ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Ładuję najlepszy checkpoint do ewaluacji testowej...")
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])

    try:
        test_ds = HybridArcadeDataset("test", args.data_root, processor, IMG_SIZE)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                                 shuffle=False, **loader_kw)
        test_m = eval_epoch(qwen, unet, test_loader, criterion, device)

        print("\nWYNIKI TESTOWE")
        print("┌──────────────────────────────────────┐")
        for k, v in test_m.items():
            if k != "loss":
                print(f"│  {k:<14} {v:.4f}                 │")
        print("└──────────────────────────────────────┘")

        # Panele predykcji do pracy
        pred_dir = Path(args.results_dir) / "qwen_unet_predictions"
        save_predictions(qwen, unet, test_loader, device, pred_dir, n=12)

        results = {
            "model":         "Qwen3-VL (LoRA+DeepStack) → U-Net refinement",
            "dataset":       "ARCADE syntax — binary vessel segmentation",
            "split":         "test",
            "best_val_dice": round(best_val_dice, 4),
            "test_metrics":  {k: round(v, 4) for k, v in test_m.items()},
            "history":       history,
            "config": {
                "qwen_ckpt":     args.qwen_ckpt,
                "unet_encoder":  "resnet34",
                "in_channels":   4,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "finetune_qwen": args.finetune_qwen,
            },
        }
        out_json = Path(args.results_dir) / "qwen_unet_metrics.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWyniki → {out_json}")

    except (FileNotFoundError, RuntimeError) as e:
        print(f"[SKIP] Ewaluacja testowa pominięta: {e}")

    print(f"Checkpoint → {ckpt_path}")
    print("Gotowe.")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-VL + U-Net refinement — segmentacja naczyń wieńcowych ARCADE"
    )
    parser.add_argument(
        "--qwen-ckpt", required=True,
        help="Ścieżka do checkpointu Qwena (np. checkpoints/qwen_seg_best_LoRA_deepstack.pth)"
    )
    parser.add_argument(
        "--finetune-qwen", action="store_true",
        help="Jeśli ustawione: fine-tune LoRA adaptery Qwena razem z U-Netem (wolniejsze)"
    )
    parser.add_argument("--data-root",   default="data")
    parser.add_argument("--out-dir",     default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=2,
                        help="2 jest bezpieczne przy 2× RTX 4080 SUPER; zwiększ do 4 jeśli VRAM pozwala")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int,   default=2)

    main(parser.parse_args())