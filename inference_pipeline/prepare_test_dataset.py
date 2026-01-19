import json
import cv2
import numpy as np
import os


def normalize_bbox(bbox, width, height):
    """
    Ta funkcja jest odpowiedzialna za normalizację współrzędnych 0-1000.
    Zwraca format: "(ymin,xmin),(ymax,xmax)"
    """
    x1, y1, x2, y2 = bbox

    # Clamping
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Przeskalowanie i zamiana kolejności na (Y, X), czyli naturalnej formie
    # wyrażania współrzędnychboxów przez Qwen
    x1_norm = int((x1 / width) * 1000)
    y1_norm = int((y1 / height) * 1000)
    x2_norm = int((x2 / width) * 1000)
    y2_norm = int((y2 / height) * 1000)

    return f"({y1_norm},{x1_norm}),({y2_norm},{x2_norm})"


def prepare_test_dataset(json_path, output_jsonl):
    """
        Generuje plik "Golden Set" (Ground Truth) z surowych danych COCO.
        To jest punkt odniesienia do sprawdzania skuteczności modelu.

        WEJŚCIE:
            - json_path: Ścieżka do oryginalnego pliku z adnotacjami (np. val.json).
            - output_jsonl: Ścieżka, gdzie zapisany zostanie gotowy plik walidacyjny.

        DZIAŁANIE:
            1. Skrypt bierze segmentację (wielokąty) z danych treningowych.
            2. Grupuje wszystkie adnotacje (tętnice) przypisane do jednego zdjęcia.
            3. Oblicza z nich najmniejsze prostokąty (BBoxy), bo tego uczył się model.
            4. Normalizuje wszystko do standardu 0-1000 i formatu Qwena.
            5. Zapisuje czysty JSONL, w formacie: jeden rekord na zdjęcie (z listą wszystkich ramek w środku).
        """


    print(f"Generowanie zbioru testowego z: {json_path}...")

    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku JSON.")
        return

    images = {img['id']: img for img in coco['images']}

    # Grupowanie adnotacji
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    test_data = []

    for img_id, anns in img_anns.items():
        if img_id not in images:
            continue

        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']

        ground_truth_text = ""

        # Dodatkowo przechowywane są surowe liczby (łatwiejsze do debugowania)
        raw_boxes_norm = []

        valid_boxes_count = 0

        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    if len(seg) < 6: continue

                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    bx, by, bw, bh = cv2.boundingRect(poly)

                    if bw < 5 or bh < 5: continue

                    x2 = bx + bw
                    y2 = by + bh

                    # BBox w pikselach [x1, y1, x2, y2]
                    pixel_box = [bx, by, x2, y2]

                    # Tekst dla modelu (znormalizowany string)
                    bbox_string = normalize_bbox(pixel_box, w, h)
                    ground_truth_text += f"<box>{bbox_string}</box>"

                    # Dodajemy do listy, żeby mieć podgląd
                    raw_boxes_norm.append(bbox_string)
                    valid_boxes_count += 1


        if valid_boxes_count == 0:
            continue

        # --- Strutkura rekordu  ---
        entry = {
            "id": f"identity_{img_id}",
            "image_path": img_info['file_name'],
            "width": w,
            "height": h,
            "ground_truth_text": ground_truth_text,
            "boxes_list": raw_boxes_norm
        }
        test_data.append(entry)

    # Zapis do JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as out:
        for entry in test_data:
            json.dump(entry, out, ensure_ascii=False)
            out.write('\n')

    print(f"Zakończono! Utworzono {len(test_data)} wpisów testowych.")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    input_path = os.path.join(project_root, 'data', 'syntax', 'val', 'annotations', 'val.json')
    output_path = os.path.join(project_root, 'data', 'val_bbox.jsonl')

    print(f"Tworzenie datasetu w: {input_path}")

    prepare_test_dataset(
        input_path,
        output_path
    )