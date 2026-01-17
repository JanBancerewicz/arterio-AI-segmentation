import json
import numpy as np
import cv2

def normalize_bbox(bbox, width, height):
    """
    Przyjmuje bbox w formacie [x1, y1, x2, y2] (piksele).
    Zwraca string sformatowany dla Qwen2-VL: "(ymin,xmin),(ymax,xmax)"
    """
    # Rozpakowanie współrzędnych
    x1, y1, x2, y2 = bbox

    # Zabezpieczenie (Clamping) - żeby nie wyjść poza wymiary obrazka
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    # Normalizacja do 0-1000
    # UWAGA: Qwen wymaga kolejności (Y, X) !!!
    x1_norm = int((x1 / width) * 1000)
    y1_norm = int((y1 / height) * 1000)
    x2_norm = int((x2 / width) * 1000)
    y2_norm = int((y2 / height) * 1000)

    # Zwracamy gotowy fragment tekstu
    return f"({y1_norm},{x1_norm}),({y2_norm},{x2_norm})"

def prepare_dataset_multi_bbox(json_path, output_jsonl):
    print(f"Przetwarzanie (tryb Multi-BBox): {json_path}...")

    try:
        with open(json_path, 'r') as f:
            coco = json.load(f)
    except FileNotFoundError:
        print("Błąd: Nie znaleziono pliku JSON.")
        return

    images = {img['id']: img for img in coco['images']}

    # Grupowanie adnotacji po image_id
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    qwen_data = []

    for img_id, anns in img_anns.items():
        if img_id not in images:
            continue

        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']

        # Lista ramek dla tego jednego obrazka
        boxes_list_str = ""
        valid_boxes_count = 0

        for ann in anns:
            # Sprawdź czy adnotacja ma segmentację
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    if len(seg) < 6: continue # Ignoruj błędy/szumy

                    # 1. Tworzymy małą maskę/kontur tylko dla tego jednego segmentu
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)

                    # 2. Wyliczamy BBox (x, y, width, height)
                    bx, by, bw, bh = cv2.boundingRect(poly)

                    # Ignoruj bardzo małe skrawki
                    if bw < 5 or bh < 5:
                        continue

                    # 3. Konwersja z XYWH na XYXY (x2 = x1 + w)
                    x2 = bx + bw
                    y2 = by + bh

                    # 4. Normalizacja i formatowanie (funkcja zwraca string)
                    # Przekazujemy listę [x1, y1, x2, y2] oraz wymiary obrazu
                    bbox_string = normalize_bbox([bx, by, x2, y2], w, h)

                    # Dodaj do ciągu wynikowego
                    boxes_list_str += f"<box>{bbox_string}</box>"
                    valid_boxes_count += 1

        if valid_boxes_count == 0:
            continue

        # 5. Konstrukcja wpisu JSONL
        entry = {
            "id": f"identity_{img_id}",
            "image": img_info['file_name'],
            "conversations": [
                {
                    "from": "user",
                    "value": "Detect and segment the coronary arteries.\nPicture 1: <img>" + img_info['file_name'] + "</img>"
                },
                {
                    "from": "assistant",
                    "value": f"<ref>Coronary Arteries</ref>{boxes_list_str}"
                }
            ]
        }
        qwen_data.append(entry)

    with open(output_jsonl, 'w', encoding='utf-8') as out:
        for entry in qwen_data:
            json.dump(entry, out, ensure_ascii=False)
            out.write('\n')

    print(f"Gotowe! Zapisano {len(qwen_data)} przykładów.")

if __name__ == "__main__":
    prepare_dataset_multi_bbox(
        'data/syntax/train/annotations/train.json',
        'data/qwen_train_multi_bbox.jsonl'
    )