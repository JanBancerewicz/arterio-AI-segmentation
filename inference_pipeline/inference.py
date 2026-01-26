import os
import json
from tqdm import tqdm
from pipeline import InferencePipeline
from visualizer import ArterioVisualizer
from validator import Validator
from prompt import ARTERY_DETECTION_PROMPT

"""
DOKUMENTACJA FORMATU DANYCH

1. PLIK WEJŚCIOWY (GT JSONL):
   Validator oczekuje pliku wygenerowanego przez 'prepare_test_dataset.py'.
   Struktura pojedynczego wiersza:
   {
       "id": "identity_123",
       "image_path": "123.png",
       "width": 512,
       "height": 512,
       # To jest lista surowych stringów, z których Regex wyciąga liczby
       "boxes_list": [
           "(ymin,xmin),(ymax,xmax)", 
           "(ymin,xmin),(ymax,xmax)"
       ]
   }

2. STANDARD WSPÓŁRZĘDNYCH W KODZIE:
   Wszystkie listy ramek (pred_boxes, gt_boxes) wewnątrz Python'a
   mają format krotek: (xmin, ymin, xmax, ymax).
   Zakres wartości: 0-1000 (int).
"""

CONFIG = {
    "base_model": "Qwen/Qwen3-VL-8B-Instruct",
    "adapter": "../arterio_checkpoints/checkpoint-432",
    "input_dir": "../data/syntax/val/images",
    "gt_annotations": "../data/val_bbox.jsonl",
    "output_dir": "final_results",
    "max_images": 2,
    "prompt": "Detect the coronary arteries. Return bounding boxes."
}


def run_full_test():
    """
    Główna pętla testowa. Łaczy:
    1. Model AI (InferencePipeline) - generuje predykcje.
    2. Validator - dostarcza poprawne odpowiedzi.
    3. Visualizer - nakłada boxy (stworzone zarówno przez lekarzy jak i przez model) na obrazki.
    """
    print("Uruchamiam pipeline testowy..")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    pipe = InferencePipeline(CONFIG["base_model"], CONFIG["adapter"])
    pipe.initialize_model()

    viz = ArterioVisualizer()
    validator = Validator(CONFIG["gt_annotations"])

    if not os.path.exists(CONFIG["input_dir"]):
        print("Błąd: Katalog wejściowy nie istnieje.")
        return

    all_files = [f for f in os.listdir(CONFIG["input_dir"]) if f.lower().endswith(('.png', '.jpg'))]
    files_to_process = viz.natural_sort(all_files)
    if CONFIG["max_images"]:
        files_to_process = files_to_process[:CONFIG["max_images"]]

    tp_tot = fp_tot = fn_tot = 0
    per_image_metrics = []

    raw_responses = []

    print(f"Przetwarzanie {len(files_to_process)} zdjęć z promptem: '{CONFIG['prompt']}'")

    for filename in tqdm(files_to_process):
        img_path = os.path.join(CONFIG["input_dir"], filename)

        # Pipe zwraca dwie rzeczy:
        # 1. pred_boxes: Czysta lista krotek [(x1,y1,x2,y2), ...]
        # 2. raw_text: Surowy string, który pozwala sprawdzić czy model nie zwraca wypowiedzi zamiast ramek)
        pred_boxes, raw_text = pipe.run_inference(img_path, CONFIG["prompt"])

        raw_responses.append({
            "filename": filename,
            "raw_output": raw_text,
            "parsed_boxes": pred_boxes
        })

        gt_boxes = validator.get_gt(filename)

        scores = validator.match_and_score(pred_boxes, gt_boxes)
        tp_tot += scores["tp"]
        fp_tot += scores["fp"]
        fn_tot += scores["fn"]

        per_image_metrics.append({
            "filename": filename,
            "tp": scores["tp"], "fp": scores["fp"], "fn": scores["fn"]
        })

        # Wizualizacja
        if pred_boxes or gt_boxes:
            res_img = viz.draw_bounding_boxes(img_path, pred_boxes, gt_boxes)
            if res_img:
                res_img.save(os.path.join(CONFIG["output_dir"], f"viz_{filename}"))

    precision = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 0
    recall = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Plik służy do debugowania zwracanych promptów i jest tylko tymczasowy (trzeba na wytrenowanym modelu to sprawdzić i usuwamy)
    raw_output_path = os.path.join(CONFIG["output_dir"], "model_raw_outputs.json")
    with open(raw_output_path, "w", encoding="utf-8") as f:
        # Zapisujemy to czytelnie (indent=2)
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)

    metrics_path = os.path.join(CONFIG["output_dir"], "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"precision": precision, "recall": recall, "f1": f1},
            "details": per_image_metrics
        }, f, indent=2)

    print("\n" + "=" * 30)
    print(f"RAPORT KOŃCOWY (Endergebnis):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    run_full_test()
