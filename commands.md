# Workflow — commands

Assumptions: repo root, active `.venv`, GPU, access to `Qwen/Qwen3-VL-8B-Instruct` on HF.


(`DataParallel` in `train_unet.py` uses all visible GPUs.)

---

## 0. Quick smoke (optional)

To check that the environment comes up — do not train from scratch. Use existing checkpoints and a single image (step 5, smoke).

---

## 1. Masks from COCO

Rasterizes SYNTAX polygons into binary masks (+ a few overlays for preview).

```bash
python convert_mask.py --data-root data --vis 5 --seed 42
```

- `--data-root data` — directory containing `syntax/`
- `--vis 5` — 5 overlays per split; enough for a sanity check
- `--seed 42` — same image selection across runs

Output: `data/masks/{train,val,test}/`, optionally `data/mask_vis/`.  

---

## 2. Baseline U-Net

Classic U-Net (ResNet-34). Best Dice so far: ~0.796.

```bash
CUDA_VISIBLE_DEVICES=0 python train_unet.py --data-root data --epochs 50 --batch-size 8
```

- `--epochs 50` — number of training epochs
- `--batch-size 8` — default and baseline value; on OOM drop to `4`

LR (`1e-4`) and encoder (`resnet34`) are hardcoded in the script.  
Output: `checkpoints/unet_best.pth`, `results/unet_metrics.json`.

---

## 3. Qwen3-VL + SegDecoder

Decoder (and optionally LoRA) on Qwen3-VL embeddings. Script: `train_qwen_seg_new.py`.

### 3a. Frozen + DeepStack

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_qwen_seg_new.py --no-lora --deepstack \
  --epochs 30 --batch-size 1 --lr 1e-4 --lora-r 16 --num-workers 2
```

### 3b. LoRA + DeepStack (main variant → teacher for the hybrid)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_qwen_seg_new.py --lora --deepstack \
  --epochs 30 --batch-size 1 --lr 1e-4 --lora-r 16 --num-workers 2
```

- `--no-lora` / `--lora` — required, mutually exclusive; LoRA performed better (~0.76 Dice)
- `--deepstack` — fusion of several ViT layers; better than last hidden alone
- `--epochs 30` — default; successful runs used this
- `--batch-size 1` — 8B model; larger batch usually will not fit
- `--lr 1e-4` — default; LoRA internally gets 0.1×
- `--lora-r 16` — default from the successful LoRA run
- `--num-workers 2` — DataLoader without unnecessary CPU load

Output: `checkpoints/qwen_seg_best_{frozen|LoRA}_deepstack.pth` + `results/qwen_metrics_*.json`.  
For the hybrid, **3b** is enough; 3a is only for comparison.

---

## 4. Hybrid Qwen → U-Net (v4)

Qwen from the LoRA checkpoint is frozen; the U-Net refiner is trained.

```bash
CUDA_VISIBLE_DEVICES=0,1 python qwen_unet_pipeline.py \
  --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --encoder resnet34 \
  --epochs 50 \
  --batch-size 2 \
  --lr 1e-5 \
  --pos-weight 20.0 \
  --max-dilation 9 \
  --min-dilation 1 \
  --conn-weight 0.3 \
  --qwen-weight 0.2 \
  --warmup-conn 5 \
  --ema-start 5 \
  --patience 10 \
  --num-workers 2
```

- `--qwen-ckpt …LoRA_deepstack.pth` — teacher; same as in 3b
- `--encoder resnet34` — same as baseline / v4 metrics
- `--epochs 50` — full v4 run
- `--batch-size 2` — default; more often hurts VRAM
- `--lr 1e-5` — lower than baseline (you refine, not train from scratch)
- `--pos-weight 20` — few vessel pixels relative to background
- `--max-dilation 9` — curriculum start ≈ thickness of Qwen masks
- `--min-dilation 1` — end = raw thin GT
- `--conn-weight 0.3` — penalty for vessel gaps
- `--qwen-weight 0.2` — do not erase what Qwen was confident about
- `--warmup-conn 5` — connectivity only after a few epochs
- `--ema-start 5` — EMA from epoch 5
- `--patience 10` — early stopping

Output: `checkpoints/qwen_unet_best.pth`, `results/qwen_unet_v4_metrics.json`  
(the script picks the threshold on val itself).

---

## 5. Inference

**Full test:**

```bash
CUDA_VISIBLE_DEVICES=0 python inference_qwen_unet.py \
  --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --unet-ckpt checkpoints/qwen_unet_best.pth \
  --input data/syntax/test/images/ \
  --masks-dir data/masks/test/ \
  --out-dir results/infer \
  --threshold 0.375 \
  --tta
```

**Smoke (1 image):**

```bash
CUDA_VISIBLE_DEVICES=0 python inference_qwen_unet.py \
  --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --unet-ckpt checkpoints/qwen_unet_best.pth \
  --input data/syntax/test/images/1.png \
  --masks-dir data/masks/test/ \
  --out-dir results/infer_smoke \
  --threshold 0.375
```

- `--qwen-ckpt` / `--unet-ckpt` — must match each other (same teacher as in step 4)
- `--input` — folder or a single PNG
- `--masks-dir data/masks/test/` — without this there is no Dice/IoU, only masks
- `--out-dir results/infer` — output directory
- `--threshold 0.375` — threshold from v4; do not blindly use the CLI default `0.35`
- `--tta` — 4× flip, slightly better, ~4× slower; unnecessary for smoke

---

## 6. Metrics / plots

### 6a. U-Net from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model unet \
  --checkpoint checkpoints/unet_best.pth \
  --split test --batch-size 8 --data-root data
```

- `--model unet` — SMP U-Net path
- `--checkpoint …unet_best.pth` — baseline for comparison
- `--split test` — official split
- `--batch-size 8` — same as training

### 6b. Qwen from checkpoint (`train_qwen_seg_new`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py --model qwen \
  --checkpoint checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --split test --batch-size 1
```

- `--batch-size 1` — 8B VL; with default `8` the script will drop to `1` anyway

### 6c. Table from existing JSONs (no GPU)

```bash
python evaluate.py --compare --results-dir results
```

Reads `results/*metrics*.json`, loads nothing onto GPU.

### 6d. Plots

```bash
python generate_plots.py              # PL → results/plots/  +  EN → results/plots_en/
python generate_plots.py --lang pl    # Polish only
python generate_plots.py --lang en    # English only (for git)
```

Reads JSONs from `results/`. The `plots_en/` folder is tracked; `plots/` stays local.
