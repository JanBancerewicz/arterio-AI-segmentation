"""
inference_qwen_unet.py
======================
Inference: frozen Qwen3-VL mask → U-Net refinement → vessel masks.

Usage:
  # Single image
  python inference_qwen_unet.py \\
      --qwen-ckpt  checkpoints/qwen_seg_best_LoRA_deepstack.pth \\
      --unet-ckpt  checkpoints/qwen_unet_best.pth \\
      --input      data/syntax/test/images/1.png \\
      --out-dir    results/infer

  # Full folder + GT metrics + TTA
  python inference_qwen_unet.py \\
      --qwen-ckpt  checkpoints/qwen_seg_best_LoRA_deepstack.pth \\
      --unet-ckpt  checkpoints/qwen_unet_best.pth \\
      --input      data/syntax/test/images/ \\
      --masks-dir  data/masks/test/ \\
      --out-dir    results/infer \\
      --threshold  0.375 \\
      --tta

Outputs (under --out-dir):
  masks/       — binary U-Net masks
  qwen_masks/  — binary Qwen-only masks
  panels/      — XCA | Qwen | Qwen+UNet | GT | overlay
  unet_probs/  — probability maps (if --save-probs)
  metrics.json — metrics when --masks-dir is set

Notes:
  - Qwen mask is binarised (>0.5) before the U-Net, matching training
  - TTA uses the same binary Qwen mask
  - U-Net encoder name / scSE come from the checkpoint config
"""

import argparse
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from peft import LoraConfig, get_peft_model
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ── Stałe ─────────────────────────────────────────────────────────────────────
MODEL_ID      = "Qwen/Qwen3-VL-8B-Instruct"
IMG_SIZE      = 512
DEC_DEVICE    = "cuda:0" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Model components (shared with qwen_unet_pipeline.py)
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

    @torch.no_grad()
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


def build_unet_refiner(encoder_name: str = "resnet34") -> nn.Module:
    """
    U-Net with scSE, 4 input channels (RGB + Qwen mask).
    First conv is patched from 3→4 channels; weights come from the checkpoint.
    """
    model = smp.Unet(
        encoder_name           = encoder_name,
        encoder_weights        = None,          # wagi załadujemy z checkpointu
        in_channels            = 3,
        classes                = 1,
        activation             = None,
        decoder_attention_type = "scse",        # ← musi pasować do ckpt
    )

    first_conv      = None
    first_conv_name = None
    for name, m in model.encoder.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv      = m
            first_conv_name = name
            break

    if first_conv is None:
        raise RuntimeError("Nie znaleziono Conv2d w encoderze")

    out_ch, in_ch, kH, kW = first_conv.weight.shape
    new_conv = nn.Conv2d(
        in_ch + 1, out_ch,
        kernel_size = first_conv.kernel_size,
        stride      = first_conv.stride,
        padding     = first_conv.padding,
        bias        = first_conv.bias is not None,
    )

    parts  = first_conv_name.split(".")
    parent = model.encoder
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)
    model.encoder._in_channels = 4
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Ładowanie modeli
# ══════════════════════════════════════════════════════════════════════════════

def load_models(qwen_ckpt: str, unet_ckpt: str):
    # ── Qwen ──────────────────────────────────────────────────────────────
    print(f"\nŁadowanie Qwen3-VL z: {MODEL_ID}")
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype             = torch.bfloat16,
        device_map        = "auto",
        low_cpu_mem_usage = True,
    )

    ckpt_q        = torch.load(qwen_ckpt, map_location=DEC_DEVICE)
    cfg_q         = ckpt_q.get("config", {})
    use_lora      = cfg_q.get("use_lora", False) or ("lora_state" in ckpt_q)
    use_deepstack = cfg_q.get("use_deepstack", True)

    if use_lora and "lora_state" in ckpt_q:
        print("  Aplikuję LoRA adaptery...")
        lora_r = cfg_q.get("lora_r") or 16
        lora_cfg = LoraConfig(
            r              = lora_r,
            lora_alpha     = lora_r * 2,
            target_modules = ["qkv", "linear_fc1", "linear_fc2"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        vl_model = get_peft_model(vl_model, lora_cfg)
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(vl_model, ckpt_q["lora_state"])
        print("  LoRA wagi załadowane.")

    # Spójny dostęp do vision_config niezależnie od LoRA
    base = vl_model
    if hasattr(vl_model, "base_model") and hasattr(vl_model.base_model, "model"):
        base = vl_model.base_model.model
    vc = base.config.vision_config

    decoder = SegDecoder(in_channels=vc.out_hidden_size).to(DEC_DEVICE)
    decoder.load_state_dict(ckpt_q["decoder"])
    print("  Decoder załadowany.")

    fusion = None
    if use_deepstack and ckpt_q.get("fusion") is not None:
        fusion = DeepStackFusion(n_sources=4).to(DEC_DEVICE)
        fusion.load_state_dict(ckpt_q["fusion"])
        print("  DeepStack fusion załadowany.")

    qwen = QwenMaskPredictor(
        vl_model      = vl_model,
        decoder       = decoder,
        fusion        = fusion,
        spatial_merge = vc.spatial_merge_size,
        patch_size    = vc.patch_size,
        img_size      = IMG_SIZE,
    ).eval()
    for p in qwen.parameters():
        p.requires_grad = False
    print("  Qwen gotowy (zamrożony).")

    # ── U-Net ─────────────────────────────────────────────────────────────
    print(f"\nŁadowanie U-Net z: {unet_ckpt}")
    ckpt_u       = torch.load(unet_ckpt, map_location=DEC_DEVICE)
    cfg_u        = ckpt_u.get("config", {})
    encoder_name = cfg_u.get("encoder", "resnet34")
    print(f"  Encoder: {encoder_name}")

    unet = build_unet_refiner(encoder_name=encoder_name).to(DEC_DEVICE)

    # Strict load — kształty muszą się zgadzać z checkpointem
    try:
        unet.load_state_dict(ckpt_u["unet"], strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Błąd ładowania wag U-Neta — niezgodność kształtów tensora.\n"
            f"Sprawdź czy encoder w checkpoincie ('{encoder_name}') i "
            f"decoder_attention_type='scse' zgadzają się z zapisanym modelem.\n"
            f"Szczegóły: {e}"
        ) from e

    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    val_dice = ckpt_u.get("val_dice", "?")
    epoch    = ckpt_u.get("epoch", "?")
    print(f"  U-Net załadowany. Epoka: {epoch}  Best val Dice: {val_dice}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return qwen, unet, processor, encoder_name


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

unet_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])


def preprocess_image(img_path: Path, processor):
    """Zwraca (pixel_values, image_grid_thw, image_unet_tensor)."""
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Nie można wczytać: {img_path}")

    rgb = cv2.cvtColor(
        cv2.resize(gray, (IMG_SIZE, IMG_SIZE)),
        cv2.COLOR_GRAY2RGB,
    )

    # Dla Qwena
    pil  = PILImage.fromarray(rgb)
    proc = processor.image_processor(images=[pil], return_tensors="pt")
    pv   = proc["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
    thw  = proc["image_grid_thw"].to(DEC_DEVICE)

    # Dla U-Neta
    img_tensor = unet_transform(image=rgb)["image"].unsqueeze(0).to(DEC_DEVICE)

    return pv, thw, img_tensor


# ══════════════════════════════════════════════════════════════════════════════
# TTA — Test-Time Augmentation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_with_tta(qwen, unet, pv, thw, img_tensor, use_tta: bool = True):
    """
    Run Qwen once, then U-Net on original + H/V/HV flips (optional TTA).

    Qwen mask is binarised (>0.5) before the U-Net — same as training.
    """
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        qwen_prob = qwen(pv, thw).float()    # (1, 1, 512, 512)  ciągłe

    # Binaryzacja — zgodność z treningiem (kluczowe!)
    qwen_bin = (qwen_prob > 0.5).float()     # (1, 1, 512, 512)  0/1

    if not use_tta:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logit = unet(torch.cat([img_tensor, qwen_bin], dim=1))
        return torch.sigmoid(logit), qwen_prob

    augmentations = [
        (False, False),
        (True,  False),
        (False, True),
        (True,  True),
    ]
    preds = []
    for flip_h, flip_v in augmentations:
        img = img_tensor.clone()
        qm  = qwen_bin.clone()     # używamy binarnej we wszystkich wariantach TTA

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

    return torch.stack(preds).mean(0), qwen_prob   # (1,1,512,512), (1,1,512,512)


# ══════════════════════════════════════════════════════════════════════════════
# Inferencja jednego obrazu
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_single(img_path: Path, qwen, unet, processor,
                 threshold: float = 0.5, use_tta: bool = False):
    """
    Zwraca:
        qwen_mask  (512, 512) uint8  0/255
        unet_mask  (512, 512) uint8  0/255
        qwen_prob  (512, 512) float32  0-1   (ciągłe prawdopodobieństwo Qwena)
        unet_prob  (512, 512) float32  0-1   (ciągłe prawdopodobieństwo U-Neta)
    """
    pv, thw, img_tensor = preprocess_image(img_path, processor)

    unet_prob_t, qwen_prob_t = predict_with_tta(
        qwen, unet, pv, thw, img_tensor, use_tta=use_tta
    )

    qwen_prob = qwen_prob_t[0, 0].cpu().float().numpy()
    unet_prob = unet_prob_t[0, 0].cpu().float().numpy()

    qwen_mask = (qwen_prob > threshold).astype(np.uint8) * 255
    unet_mask = (unet_prob > threshold).astype(np.uint8) * 255

    return qwen_mask, unet_mask, qwen_prob, unet_prob


# ══════════════════════════════════════════════════════════════════════════════
# Metryki
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(pred_bin, gt_bin, smooth=1e-6):
    p  = pred_bin.astype(float).ravel()
    t  = gt_bin.astype(float).ravel()
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()
    return {
        "dice":      round((2*tp + smooth) / (2*tp + fp + fn + smooth), 4),
        "iou":       round((tp + smooth) / (tp + fp + fn + smooth), 4),
        "precision": round((tp + smooth) / (tp + fp + smooth), 4),
        "recall":    round((tp + smooth) / (tp + fn + smooth), 4),
        "px_acc":    round((tp + tn) / (tp + tn + fp + fn + smooth), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Panel wizualizacyjny — 3 kolumny na jednym obrazie
# ══════════════════════════════════════════════════════════════════════════════

def make_panel(img_path: Path, qwen_mask, unet_mask,
               gt_mask=None, qwen_m=None, unet_m=None) -> np.ndarray:
    """
    Panel 3 kolumn (BGR):

      Gdy GT dostępne:
        [Qwen overlay] | [Qwen+UNet overlay] | [Ground Truth]

        Overlay na oryginalnym XCA:
          zielony  (0, 200,   0) = TP  — dobrze wykryte naczynie
          czerwony (0,   0, 200) = FP  — fałszywy alarm
          niebieski(200,  0,   0) = FN  — pominięte naczynie
        Metryki (Dice / Recall) w subtitle każdej kolumny.

      Gdy GT niedostępne:
        [XCA oryginal] | [Qwen maska] | [Qwen+UNet maska]
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    header_h = 40   # trochę więcej miejsca na dwie linie tekstu

    def to_bgr(img_gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

    def make_col(bgr_img: np.ndarray, title: str, subtitle: str = "") -> np.ndarray:
        hdr = np.zeros((header_h, IMG_SIZE, 3), dtype=np.uint8)
        cv2.putText(hdr, title,    (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(hdr, subtitle, (4, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, ( 80, 220,  80), 1, cv2.LINE_AA)
        return np.vstack([hdr, bgr_img])

    def build_overlay(mask_bin: np.ndarray, gt_bin: np.ndarray) -> np.ndarray:
        """
        Nakłada kolorowy TP/FP/FN na szary XCA.
        mask_bin, gt_bin: bool (H, W)
        """
        base    = to_bgr(gray)
        overlay = base.copy()
        overlay[ mask_bin &  gt_bin] = (  0, 200,   0)   # TP — zielony
        overlay[ mask_bin & ~gt_bin] = (  0,   0, 200)   # FP — czerwony
        overlay[~mask_bin &  gt_bin] = (200,   0,   0)   # FN — niebieski
        return cv2.addWeighted(base, 0.40, overlay, 0.60, 0)

    if gt_mask is not None:
        gt_bin   = (gt_mask   > 127)
        qwen_bin = (qwen_mask > 127)
        unet_bin = (unet_mask > 127)

        qwen_ov = build_overlay(qwen_bin, gt_bin)
        unet_ov = build_overlay(unet_bin, gt_bin)
        gt_vis  = to_bgr((gt_bin.astype(np.uint8) * 255))

        qwen_sub = (f"Dice {qwen_m['dice']:.3f}  Rec {qwen_m['recall']:.3f}"
                    if qwen_m else "ziel=TP  czerw=FP  nieb=FN")
        unet_sub = (f"Dice {unet_m['dice']:.3f}  Rec {unet_m['recall']:.3f}"
                    if unet_m else "ziel=TP  czerw=FP  nieb=FN")

        cols = [
            make_col(qwen_ov, "Qwen (wstepna)",       qwen_sub),
            make_col(unet_ov, "Qwen+UNet (finalna)",   unet_sub),
            make_col(gt_vis,  "Ground Truth"),
        ]
    else:
        # Bez GT — pokazujemy XCA + obie binarne maski
        cols = [
            make_col(to_bgr(gray),      "XCA (oryginal)"),
            make_col(to_bgr(qwen_mask), "Qwen (wstepna)"),
            make_col(to_bgr(unet_mask), "Qwen+UNet (finalna)"),
        ]

    return np.hstack(cols)   # (H+header, W*3, 3) BGR


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print(f"CUDA dostępne: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name}  ({vram:.1f} GB)")

    # ── Załaduj modele ─────────────────────────────────────────────────────
    qwen, unet, processor, encoder_name = load_models(args.qwen_ckpt, args.unet_ckpt)

    # ── Zbierz listę obrazów ───────────────────────────────────────────────
    input_path = Path(args.input)
    if input_path.is_dir():
        img_paths = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
        print(f"\nZnaleziono {len(img_paths)} obrazów w {input_path}/")
    elif input_path.is_file():
        img_paths = [input_path]
        print(f"\nJeden obraz: {input_path}")
    else:
        raise FileNotFoundError(f"--input nie istnieje: {input_path}")

    if not img_paths:
        raise RuntimeError("Brak obrazów PNG/JPG.")

    # ── Katalogi wyjściowe ─────────────────────────────────────────────────
    out_dir    = Path(args.out_dir)
    masks_out  = out_dir / "masks"
    qwen_out   = out_dir / "qwen_masks"
    panels_out = out_dir / "panels"
    for d in (masks_out, qwen_out, panels_out):
        d.mkdir(parents=True, exist_ok=True)

    if args.save_probs:
        (out_dir / "unet_probs").mkdir(parents=True, exist_ok=True)
        (out_dir / "qwen_probs").mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir) if args.masks_dir else None

    all_qwen_metrics = []
    all_unet_metrics = []
    per_image        = []

    use_tta = args.tta
    print(f"\nInferencja: {len(img_paths)} obrazów  "
          f"| threshold={args.threshold}  | TTA={'tak' if use_tta else 'nie'}")

    for img_path in tqdm(img_paths):
        stem = img_path.stem

        qwen_mask, unet_mask, qwen_prob, unet_prob = infer_single(
            img_path, qwen, unet, processor,
            threshold=args.threshold, use_tta=use_tta,
        )

        # Zapisz binarne maski (grayscale — OK)
        cv2.imwrite(str(masks_out / f"{stem}.png"), unet_mask)
        cv2.imwrite(str(qwen_out  / f"{stem}.png"), qwen_mask)

        # Mapy prawdopodobieństwa (opcjonalnie)
        if args.save_probs:
            cv2.imwrite(str(out_dir / "unet_probs" / f"{stem}.png"),
                        (unet_prob * 255).astype(np.uint8))
            cv2.imwrite(str(out_dir / "qwen_probs" / f"{stem}.png"),
                        (qwen_prob * 255).astype(np.uint8))

        # Metryki + panel
        gt_mask = None
        qm = um = None
        if masks_dir is not None:
            gt_path = masks_dir / f"{stem}.png"
            if gt_path.exists():
                gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt_bin  = (gt_mask > 127).astype(np.uint8)
                qm = compute_metrics(qwen_mask // 255, gt_bin)
                um = compute_metrics(unet_mask // 255, gt_bin)
                all_qwen_metrics.append(qm)
                all_unet_metrics.append(um)
                per_image.append({"stem": stem, "qwen": qm, "unet": um})

        # Panel jest teraz BGR — kolory zachowane
        panel = make_panel(img_path, qwen_mask, unet_mask, gt_mask, qm, um)
        cv2.imwrite(str(panels_out / f"{stem}_panel.png"), panel)

    # ── Podsumowanie ───────────────────────────────────────────────────────
    print(f"\nMaski (U-Net)  → {masks_out}/")
    print(f"Maski (Qwen)   → {qwen_out}/")
    print(f"Panele         → {panels_out}/")

    if all_unet_metrics:
        def avg(lst, key):
            return round(float(np.mean([m[key] for m in lst])), 4)

        keys = ["dice", "iou", "precision", "recall", "px_acc"]

        summary = {
            "n_images":   len(all_unet_metrics),
            "threshold":  args.threshold,
            "tta":        use_tta,
            "encoder":    encoder_name,
            "qwen_only":  {k: avg(all_qwen_metrics, k) for k in keys},
            "qwen_unet":  {k: avg(all_unet_metrics,  k) for k in keys},
            "per_image":  per_image,
        }

        print("\n" + "=" * 54)
        print("WYNIKI TESTOWE")
        print(f"  n={len(all_unet_metrics)}  threshold={args.threshold}"
              f"  TTA={'tak' if use_tta else 'nie'}")
        print(f"{'Metryka':<12}  {'Qwen':>8}  {'Qwen+UNet':>10}  {'Δ':>8}")
        print("─" * 54)
        for k in keys:
            q = summary["qwen_only"][k]
            u = summary["qwen_unet"][k]
            d = u - q
            arrow = "↑" if d > 0.0001 else ("↓" if d < -0.0001 else "=")
            print(f"{k:<12}  {q:>8.4f}  {u:>10.4f}  {arrow}{abs(d):.4f}")
        print("=" * 54)

        out_json = out_dir / "metrics.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nMetryki → {out_json}")

    print("\nGotowe!")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencja: Qwen3-VL → U-Net refinement (segmentacja naczyń ARCADE)"
    )
    parser.add_argument("--qwen-ckpt", required=True,
                        help="Ścieżka do checkpointu Qwena")
    parser.add_argument("--unet-ckpt", required=True,
                        help="Ścieżka do checkpointu U-Neta (qwen_unet_best.pth)")
    parser.add_argument("--input", required=True,
                        help="Obraz PNG lub folder z obrazami")
    parser.add_argument("--out-dir", default="results/infer",
                        help="Folder wyjściowy (default: results/infer)")
    parser.add_argument("--masks-dir", default=None,
                        help="(opcjonalnie) Folder z GT maskami — liczy metryki")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Próg binaryzacji (default: 0.5; użyj wartości z metrics.json)")
    parser.add_argument("--tta", action="store_true",
                        help="Włącz Test-Time Augmentation (4x flip, ~+1%% Dice)")
    parser.add_argument("--save-probs", action="store_true",
                        help="Zapisz mapy prawdopodobieństwa 0-255 PNG")
    main(parser.parse_args())