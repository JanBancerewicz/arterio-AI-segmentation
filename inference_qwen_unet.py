"""
infer_qwen_unet.py  —  Inferencja: Qwen3-VL → U-Net refinement
===============================================================
Wczytuje wytrenowane checkpointy i produkuje maski dla obrazów.

UŻYCIE:

  # Jeden obraz
  python infer_qwen_unet.py \
      --qwen-ckpt  checkpoints/qwen_seg_best_LoRA_deepstack.pth \
      --unet-ckpt  checkpoints/qwen_unet_best.pth \
      --input      data/syntax/test/images/001.png \
      --out-dir    results/infer

  # Cały folder obrazów
  python infer_qwen_unet.py \
      --qwen-ckpt  checkpoints/qwen_seg_best_LoRA_deepstack.pth \
      --unet-ckpt  checkpoints/qwen_unet_best.pth \
      --input      data/syntax/test/images/ \
      --out-dir    results/infer

  # Z ground truth (liczy Dice, IoU itd.)
  python infer_qwen_unet.py \
      --qwen-ckpt  checkpoints/qwen_seg_best_LoRA_deepstack.pth \
      --unet-ckpt  checkpoints/qwen_unet_best.pth \
      --input      data/syntax/test/images/ \
      --masks-dir  data/masks/test/ \
      --out-dir    results/infer

WYJŚCIE (w --out-dir):
  masks/          ← binarne maski PNG (0/255) z U-Neta
  qwen_masks/     ← binarne maski PNG z samego Qwena (do porównania)
  panels/         ← panele: XCA | Qwen | Qwen+UNet | GT (jeśli podano maski)
  metrics.json    ← metryki (jeśli podano --masks-dir)
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
DEC_DEVICE    = "cuda:0"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ══════════════════════════════════════════════════════════════════════════════
# Architektury  (muszą być identyczne jak w train_qwen_unet.py)
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


def build_unet_refiner() -> nn.Module:
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,   # nie ładuj ImageNet — załadujemy nasz ckpt
        in_channels     = 3,
        classes         = 1,
        activation      = None,
    )
    # Znajdź pierwszą Conv2d i zamień na 4-kanałową
    first_conv = None
    first_conv_name = None
    for name, m in model.encoder.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv      = m
            first_conv_name = name
            break

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
    print(f"\nŁadowanie Qwen3-VL...")
    vl_model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype         = torch.bfloat16,
        device_map    = "auto",
        low_cpu_mem_usage = True,
    )

    ckpt_q = torch.load(qwen_ckpt, map_location=DEC_DEVICE)
    cfg    = ckpt_q.get("config", {})

    use_lora      = cfg.get("use_lora", False) or ("lora_state" in ckpt_q)
    use_deepstack = cfg.get("use_deepstack", True)

    if use_lora and "lora_state" in ckpt_q:
        print("  Aplikuję LoRA...")
        lora_r = cfg.get("lora_r") or 16
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

    vc      = vl_model.config.vision_config if not use_lora else \
              vl_model.base_model.model.config.vision_config
    decoder = SegDecoder(in_channels=vc.out_hidden_size).to(DEC_DEVICE)
    decoder.load_state_dict(ckpt_q["decoder"])

    fusion = None
    if use_deepstack and ckpt_q.get("fusion") is not None:
        fusion = DeepStackFusion(n_sources=4).to(DEC_DEVICE)
        fusion.load_state_dict(ckpt_q["fusion"])

    qwen = QwenMaskPredictor(
        vl_model     = vl_model,
        decoder      = decoder,
        fusion       = fusion,
        spatial_merge= vc.spatial_merge_size,
        patch_size   = vc.patch_size,
        img_size     = IMG_SIZE,
    ).eval()
    print("  Qwen załadowany i zamrożony.")

    print(f"\nŁadowanie U-Net z: {unet_ckpt}")
    unet     = build_unet_refiner().to(DEC_DEVICE)
    ckpt_u   = torch.load(unet_ckpt, map_location=DEC_DEVICE)
    unet.load_state_dict(ckpt_u["unet"])
    unet.eval()
    val_dice = ckpt_u.get("val_dice", "?")
    print(f"  U-Net załadowany. Best val Dice podczas treningu: {val_dice}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return qwen, unet, processor


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
    rgb  = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Dla Qwena
    pil  = PILImage.fromarray(cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)))
    proc = processor.image_processor(images=[pil], return_tensors="pt")
    pv   = proc["pixel_values"].to(DEC_DEVICE, dtype=torch.bfloat16)
    thw  = proc["image_grid_thw"].to(DEC_DEVICE)

    # Dla U-Neta
    aug        = unet_transform(image=cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)))
    img_tensor = aug["image"].unsqueeze(0).to(DEC_DEVICE)  # (1, 3, 512, 512)

    return pv, thw, img_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Inferencja jednego obrazu
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer_single(img_path: Path, qwen, unet, processor, threshold: float = 0.5):
    """
    Zwraca:
        qwen_mask  (512, 512) uint8  0/255
        unet_mask  (512, 512) uint8  0/255
        qwen_prob  (512, 512) float32  0-1  (mapa prawdopodobieństwa)
        unet_prob  (512, 512) float32  0-1
    """
    pv, thw, img_tensor = preprocess_image(img_path, processor)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        qwen_prob_t = qwen(pv, thw)                                     # (1,1,512,512)

    qwen_prob_t = qwen_prob_t.float()
    unet_input  = torch.cat([img_tensor, qwen_prob_t], dim=1)           # (1,4,512,512)
    unet_logits = unet(unet_input)
    unet_prob_t = torch.sigmoid(unet_logits)

    qwen_prob = qwen_prob_t[0, 0].cpu().numpy()
    unet_prob = unet_prob_t[0, 0].cpu().numpy()

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
# Panel wizualizacyjny
# ══════════════════════════════════════════════════════════════════════════════

def make_panel(img_path: Path, qwen_mask, unet_mask, gt_mask=None,
               qwen_metrics=None, unet_metrics=None):
    """
    Zwraca panel numpy (H, W) lub (H, W, 3):
        XCA | Qwen | Qwen+UNet | GT (jeśli dostępne)
    """
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    cols = [gray, qwen_mask, unet_mask]
    labels = ["XCA (oryginal)", "Qwen (wstepna)", "Qwen+UNet (finalna)"]

    if gt_mask is not None:
        gt_vis = cv2.resize((gt_mask > 127).astype(np.uint8) * 255, (IMG_SIZE, IMG_SIZE))
        cols.append(gt_vis)
        labels.append("Ground Truth")

    # Konwertuj na BGR żeby dodać kolorowe etykiety
    cols_bgr = [cv2.cvtColor(c, cv2.COLOR_GRAY2BGR) for c in cols]

    # Dodaj metryki do etykiet
    if qwen_metrics and len(cols_bgr) > 1:
        labels[1] += f"\nDice {qwen_metrics['dice']:.3f}"
    if unet_metrics and len(cols_bgr) > 2:
        labels[2] += f"\nDice {unet_metrics['dice']:.3f}"

    # Nagłówek nad każdą kolumną
    header_h = 30
    panels_with_header = []
    for col_bgr, label in zip(cols_bgr, labels):
        header = np.zeros((header_h, IMG_SIZE, 3), dtype=np.uint8)
        first_line = label.split("\n")[0]
        second_line = label.split("\n")[1] if "\n" in label else ""
        cv2.putText(header, first_line, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        if second_line:
            cv2.putText(header, second_line, (4, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1)
        panels_with_header.append(np.vstack([header, col_bgr]))

    return np.hstack(panels_with_header)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    print(f"CUDA: {torch.cuda.is_available()}")
    device = torch.device(DEC_DEVICE)

    # Wczytaj modele
    qwen, unet, processor = load_models(args.qwen_ckpt, args.unet_ckpt)

    # Zbierz listę obrazów
    input_path = Path(args.input)
    if input_path.is_dir():
        img_paths = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
        print(f"\nZnaleziono {len(img_paths)} obrazów w {input_path}")
    elif input_path.is_file():
        img_paths = [input_path]
    else:
        raise FileNotFoundError(f"--input nie istnieje: {input_path}")

    if not img_paths:
        raise RuntimeError("Brak obrazów PNG/JPG w podanym katalogu.")

    # Katalogi wyjściowe
    out_dir   = Path(args.out_dir)
    masks_out = out_dir / "masks"        # finalne maski U-Neta
    qwen_out  = out_dir / "qwen_masks"   # maski samego Qwena
    panels_out= out_dir / "panels"       # panele porównawcze
    for d in (masks_out, qwen_out, panels_out):
        d.mkdir(parents=True, exist_ok=True)

    masks_dir = Path(args.masks_dir) if args.masks_dir else None

    all_qwen_metrics = []
    all_unet_metrics = []

    print(f"\nInferencja na {len(img_paths)} obrazach...")
    for img_path in tqdm(img_paths):
        stem = img_path.stem

        # Inferencja
        qwen_mask, unet_mask, qwen_prob, unet_prob = infer_single(
            img_path, qwen, unet, processor, threshold=args.threshold
        )

        # Zapisz maski
        cv2.imwrite(str(masks_out / f"{stem}.png"), unet_mask)
        cv2.imwrite(str(qwen_out  / f"{stem}.png"), qwen_mask)

        # Opcjonalnie: mapa prawdopodobieństwa (float → 8-bit)
        if args.save_probs:
            cv2.imwrite(str(out_dir / "unet_probs" / f"{stem}.png"),
                        (unet_prob * 255).astype(np.uint8))

        # Metryki i panel (jeśli mamy GT)
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

        # Panel
        panel = make_panel(img_path, qwen_mask, unet_mask, gt_mask, qm, um)
        cv2.imwrite(str(panels_out / f"{stem}_panel.png"), panel)

    print(f"\nMaski (U-Net)  → {masks_out}/")
    print(f"Maski (Qwen)   → {qwen_out}/")
    print(f"Panele         → {panels_out}/")

    # Podsumowanie metryk
    if all_unet_metrics:
        def avg(lst, key):
            return round(float(np.mean([m[key] for m in lst])), 4)

        keys = ["dice", "iou", "precision", "recall", "px_acc"]
        summary = {
            "n_images": len(all_unet_metrics),
            "threshold": args.threshold,
            "qwen_only": {k: avg(all_qwen_metrics, k) for k in keys},
            "qwen_unet":  {k: avg(all_unet_metrics,  k) for k in keys},
        }

        print("\n" + "=" * 50)
        print("WYNIKI TESTOWE")
        print(f"{'Metryka':<12}  {'Qwen':>8}  {'Qwen+UNet':>10}  {'Δ':>7}")
        print("─" * 50)
        for k in keys:
            q = summary["qwen_only"][k]
            u = summary["qwen_unet"][k]
            d = u - q
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
            print(f"{k:<12}  {q:>8.4f}  {u:>10.4f}  {arrow}{abs(d):.4f}")
        print("=" * 50)

        out_json = out_dir / "metrics.json"
        with open(out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nMetryki → {out_json}")

    print("\nGotowe!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencja: Qwen3-VL → U-Net refinement"
    )
    parser.add_argument(
        "--qwen-ckpt", required=True,
        help="checkpoints/qwen_seg_best_LoRA_deepstack.pth"
    )
    parser.add_argument(
        "--unet-ckpt", required=True,
        help="checkpoints/qwen_unet_best.pth"
    )
    parser.add_argument(
        "--input", required=True,
        help="Ścieżka do obrazu PNG lub folderu z obrazami"
    )
    parser.add_argument(
        "--out-dir", default="results/infer",
        help="Folder wyjściowy (default: results/infer)"
    )
    parser.add_argument(
        "--masks-dir", default=None,
        help="(opcjonalnie) Folder z GT maskami — liczy metryki"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Próg binaryzacji maski (default: 0.5)"
    )
    parser.add_argument(
        "--save-probs", action="store_true",
        help="Zapisz też mapy prawdopodobieństwa (0-255 PNG)"
    )
    main(parser.parse_args())