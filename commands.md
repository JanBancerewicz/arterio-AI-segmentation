# Workflow — komendy

Założenia: root repo, aktywny `.venv`, GPU, dostęp do `Qwen/Qwen3-VL-8B-Instruct` na HF.


(`DataParallel` w `train_unet.py` wchodzi na wszystkie widoczne GPU.)

---

## 0. Szybki smoke (opcjonalnie)

Żeby sprawdzić, czy środowisko wstaje — nie trenuj od zera. Weź istniejące ckpty i jeden obraz (krok 5, smoke).

---

## 1. Maski z COCO

Rasteryzuje poligony SYNTAX do binarnych masek (+ kilka overlayów do podglądu).

```bash
python convert_mask.py --data-root data --vis 5 --seed 42
```

- `--data-root data` — katalog z `syntax/`
- `--vis 5` — 5 overlayów na split; wystarczy do sprawdzenia
- `--seed 42` — ten sam wybór obrazów przy kolejnych runach

Wyjście: `data/masks/{train,val,test}/`, opcjonalnie `data/mask_vis/`.  

---

## 2. Baseline U-Net

Klasyczny U-Net (ResNet-34). Dotychczas najlepszy Dice: ~0.796.

```bash
CUDA_VISIBLE_DEVICES=0 python train_unet.py --data-root data --epochs 50 --batch-size 8
```

- `--epochs 50` — liczba epok do treningu
- `--batch-size 8` — default i wartość z baseline; przy OOM zejdź do `4`

LR (`1e-4`) i encoder (`resnet34`) są hardcoded w skrypcie.  
Wyjście: `checkpoints/unet_best.pth`, `results/unet_metrics.json`.

---

## 3. Qwen3-VL + SegDecoder

Dekoder (i ewentualnie LoRA) na embeddingach Qwen3-VL. Skrypt: `train_qwen_seg_new.py`.

### 3a. Frozen + DeepStack

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_qwen_seg_new.py --no-lora --deepstack \
  --epochs 30 --batch-size 1 --lr 1e-4 --lora-r 16 --num-workers 2
```

### 3b. LoRA + DeepStack (główny wariant → teacher do hybrydy)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_qwen_seg_new.py --lora --deepstack \
  --epochs 30 --batch-size 1 --lr 1e-4 --lora-r 16 --num-workers 2
```

- `--no-lora` / `--lora` — wymagane, wykluczają się; LoRA wyszedł lepiej (~0.76 Dice)
- `--deepstack` — fusion kilku warstw ViT; lepsze niż sam last hidden
- `--epochs 30` — default, tyle trwały udane runy
- `--batch-size 1` — 8B model; większy batch zwykle nie wejdzie
- `--lr 1e-4` — default; LoRA wewnętrznie dostaje 0.1×
- `--lora-r 16` — default z udanego LoRA runu
- `--num-workers 2` — DataLoader bez zbędnego obciążania CPU

Wyjście: `checkpoints/qwen_seg_best_{frozen|LoRA}_deepstack.pth` + `results/qwen_metrics_*.json`.  
Do hybrydy wystarczy **3b**; 3a tylko do porównania.

---

## 4. Hybryda Qwen → U-Net (v4)

Qwen z checkpointu LoRA zamrożony; trenowany jest U-Net-refiner.

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

- `--qwen-ckpt …LoRA_deepstack.pth` — teacher; ten sam co w 3b
- `--encoder resnet34` — jak w baseline / v4 metrics
- `--epochs 50` — pełny run v4
- `--batch-size 2` — default; więcej często boli VRAM
- `--lr 1e-5` — niższe niż baseline (refinujesz, nie uczysz od zera)
- `--pos-weight 20` — mało pikseli naczyń względem tła
- `--max-dilation 9` — start curriculum ≈ grubość masek Qwena
- `--min-dilation 1` — koniec = surowe cienkie GT
- `--conn-weight 0.3` — kara za przerwy w naczyniach
- `--qwen-weight 0.2` — nie kasuj tego, czego Qwen był pewny
- `--warmup-conn 5` — connectivity dopiero po kilku epokach
- `--ema-start 5` — EMA od epoki 5
- `--patience 10` — early stopping

Wyjście: `checkpoints/qwen_unet_best.pth`, `results/qwen_unet_v4_metrics.json`  
(skrypt sam dobiera próg na val).

---

## 5. Inferencja

**Pełny test:**

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

**Smoke (1 obraz):**

```bash
CUDA_VISIBLE_DEVICES=0 python inference_qwen_unet.py \
  --qwen-ckpt checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --unet-ckpt checkpoints/qwen_unet_best.pth \
  --input data/syntax/test/images/1.png \
  --masks-dir data/masks/test/ \
  --out-dir results/infer_smoke \
  --threshold 0.375
```

- `--qwen-ckpt` / `--unet-ckpt` — muszą pasować do siebie (ten sam teacher co w kroku 4)
- `--input` — folder albo pojedynczy PNG
- `--masks-dir data/masks/test/` — bez tego nie ma Dice/IoU, tylko maski
- `--out-dir results/infer` — katalog wyjść
- `--threshold 0.375` — próg z v4; nie bierz ślepo defaultu CLI `0.35`
- `--tta` — 4× flip, trochę lepiej, ~4× wolniej; na smoke zbędne

---

## 6. Metryki / wykresy

### 6a. U-Net z checkpointu

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --model unet \
  --checkpoint checkpoints/unet_best.pth \
  --split test --batch-size 8 --data-root data
```

- `--model unet` — ścieżka SMP U-Net
- `--checkpoint …unet_best.pth` — baseline do porównania
- `--split test` — oficjalny split
- `--batch-size 8` — jak przy treningu

### 6b. Qwen z checkpointu (`train_qwen_seg_new`)

```bash
CUDA_VISIBLE_DEVICES=0,1 python evaluate.py --model qwen \
  --checkpoint checkpoints/qwen_seg_best_LoRA_deepstack.pth \
  --split test --batch-size 1
```

- `--batch-size 1` — 8B VL; przy defaultcie `8` skrypt i tak zejdzie do `1`

### 6c. Tabela z gotowych JSON-ów (bez GPU)

```bash
python evaluate.py --compare --results-dir results
```

Czyta `results/*metrics*.json`, nic nie ładuje na GPU.

### 6d. Wykresy

```bash
python generate_plots.py              # PL → results/plots/  +  EN → results/plots_en/
python generate_plots.py --lang pl    # tylko polskie
python generate_plots.py --lang en    # tylko angielskie (do gita)
```

Czyta JSON-y z `results/`. Folder `plots_en/` jest trackowany; `plots/` zostaje lokalny.
