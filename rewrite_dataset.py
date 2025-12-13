import json
import numpy as np
import cv2

def normalize_bbox(x, y, w, h, img_w, img_h, max_val=1000):
    """
    Zamienia bbox xywh (piksele) na format x1,y1,x2,y2 (znormalizowany 0-1000).
    """
    x1 = int((x / img_w) * max_val)
    y1 = int((y / img_h) * max_val)
    x2 = int(((x + w) / img_w) * max_val)
    y2 = int(((y + h) / img_h) * max_val)

    # Clamp (zabezpieczenie, żeby nie wyjść poza 0-1000)
    x1 = max(0, min(max_val, x1))
    y1 = max(0, min(max_val, y1))
    x2 = max(0, min(max_val, x2))
    y2 = max(0, min(max_val, y2))

    return x1, y1, x2, y2

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

                    # 2. Wyliczamy ciasny BBox dla tego kawałka
                    bx, by, bw, bh = cv2.boundingRect(poly)

                    # Ignoruj bardzo małe skrawki (np. mniejsze niż 5x5 pikseli), żeby nie spamować modelu szumem
                    if bw < 5 or bh < 5:
                        continue

                    # 3. Normalizacja
                    nx1, ny1, nx2, ny2 = normalize_bbox(bx, by, bw, bh, w, h)

                    # Dodaj do ciągu wynikowego
                    boxes_list_str += f"<box>({nx1},{ny1}),({nx2},{ny2})</box>"
                    valid_boxes_count += 1

        if valid_boxes_count == 0:
            continue

        # 4. Konstrukcja wpisu JSONL
        # Output wygląda np. tak: <ref>Arteries</ref><box>...</box><box>...</box>
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

    print(f"Gotowe! Zapisano {len(qwen_data)} przykładów. Średnio ramek na obraz: {valid_boxes_count}")

if __name__ == "__main__":
    prepare_dataset_multi_bbox(
        'data/syntax/train/annotations/train.json',
        'data/qwen_train_multi_bbox.jsonl'
    )