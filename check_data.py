import json
import os
from PIL import Image, ImageDraw

# Ścieżki
JSONL_PATH = "data/qwen_train_multi_bbox.jsonl" # Upewnij się, że masz ten plik pobrany z serwera!
IMAGES_DIR = "data/syntax/train/images"         # Folder ze zdjęciami

def visualize_ground_truth():
    print("--- WERYFIKACJA DANYCH TRENINGOWYCH ---")

    # Wczytaj pierwszy wiersz z JSONL
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        line = f.readline()
        data = json.loads(line)

    image_name = os.path.basename(data['image'])
    image_path = os.path.join(IMAGES_DIR, image_name)

    # Wyciągnij odpowiedź asystenta (tam są koordynaty)
    conversations = data['conversations']
    assistant_msg = next(item for item in conversations if item['from'] == "assistant")['value']

    print(f"Obraz: {image_name}")
    print(f"RAW Target Text: {assistant_msg}")

    if not os.path.exists(image_path):
        print(f"Błąd: Nie mam obrazka {image_path} lokalnie, nie mogę narysować.")
        return

    # Parsowanie (prosty regex jak w teście)
    import re
    pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
    matches = re.findall(pattern, assistant_msg)

    # Rysowanie
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    print(f"Wymiary obrazka: {W}x{H}")

    for box in matches:
        y1_n, x1_n, y2_n, x2_n = map(int, box)

        # Denormalizacja (1000 -> px)
        x1 = (x1_n / 1000) * W
        y1 = (y1_n / 1000) * H
        x2 = (x2_n / 1000) * W
        y2 = (y2_n / 1000) * H

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    img.show()
    img.save("test_ground_truth.png")

if __name__ == "__main__":
    visualize_ground_truth()