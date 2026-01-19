# Inference Pipeline

## Struktura
- `pipeline.py` — ładowanie modelu Qwen2-VL + (opcjonalnie) LoRA, pojedyncza inferencja obrazu.
- `prompt.py` — jednolity prompt dla detekcji segmentów tętnic w skali 0–1000.
- `visualizer.py` — rysowanie ramek (denormalizacja 0–1000 → piksele) i zapisywanie podglądów.
- `validator.py` — wczytywanie GT (Qwen JSONL lub COCO JSON), dopasowanie i metryki (TP/FP/FN, precision/recall, mean IoU).
- `inference.py` — runner batchowy po folderze obrazów: zapis predykcji, wizualizacji i metryk.

## Konfiguracja
W `inference.py` (słownik `CONFIG`):
- `base_model` — ID modelu bazowego (np. `Qwen/Qwen2-VL-2B-Instruct`).
- `adapter` — ścieżka do folderu z LoRA (jeśli brak, działa na modelu bazowym).
- `input_dir` — folder z obrazami testowymi.
- `gt_annotations` — plik COCO JSON lub JSONL (Qwen) z etykietami.
- `output_dir` — gdzie zapisywać wyniki.
- `max_images` — limit obrazów (None = wszystkie).

## Wyniki
- `final_results/pred_{filename}` — wizualizacje z ramkami.
- `final_results/predictions.json` — predykcje w skali 0–1000 dla każdego pliku.
- `final_results/metrics.json` — metryki: podsumowanie + per-image (TP/FP/FN, precision, recall, mean IoU).
