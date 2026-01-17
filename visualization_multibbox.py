import json
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2  # Używamy OpenCV do wczytania rozmiaru obrazka
from matplotlib.patches import Rectangle

def parse_qwen_response(response_text):
    """
    Wyciąga etykietę i listę bboxów z tekstu odpowiedzi modelu Qwen.
    Format: <ref>Label</ref><box>(x1,y1),(x2,y2)</box>...
    """
    # 1. Wyciąganie nazwy kategorii (ref)
    ref_match = re.search(r'<ref>(.*?)</ref>', response_text)
    label = ref_match.group(1) if ref_match else "Object"

    # 2. Wyciąganie wszystkich boxów
    # Szukamy wzorca: <box>(liczba,liczba),(liczba,liczba)</box>
    pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
    matches = re.findall(pattern, response_text)

    boxes = []
    for m in matches:
        # Konwersja na inty: x1, y1, x2, y2
        coords = [int(val) for val in m]
        boxes.append(coords)

    return label, boxes

def visualize_qwen_jsonl(jsonl_path, image_folder, output_dir="viz_output", limit=5):
    """
    Wizualizuje dane z pliku JSONL (Multi-BBox) dla wielu obrazów.

    :param jsonl_path: Ścieżka do pliku .jsonl (np. qwen_train_bbox.jsonl)
    :param image_folder: Folder, w którym leżą prawdziwe zdjęcia (potrzebne do wymiarów)
    :param output_dir: Gdzie zapisać wyniki
    :param limit: Ile obrazów przetworzyć (0 = wszystkie)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Otwieranie pliku: {jsonl_path}")

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Ograniczenie liczby przetwarzanych obrazów
    if limit > 0:
        lines = lines[:limit]

    for i, line in enumerate(lines):
        data = json.loads(line)

        image_filename = data['image'] # np. "922.png" lub "images/922.png"

        # Pobieramy odpowiedź asystenta (tam są boxy)
        # Zakładamy, że conversations[1] to odpowiedź asystenta
        assistant_response = data['conversations'][1]['value']

        # Parsowanie tekstu
        label_text, normalized_boxes = parse_qwen_response(assistant_response)

        # --- PRZYGOTOWANIE OBRAZU ---
        full_image_path = os.path.join(image_folder, os.path.basename(image_filename))

        # Próba wczytania obrazu, aby poznać jego wymiary
        if os.path.exists(full_image_path):
            img = cv2.imread(full_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Matplotlib woli RGB
            h, w, _ = img.shape
        else:
            print(f"[{i}] Ostrzeżenie: Nie znaleziono obrazu {full_image_path}. Używam czarnego tła 512x512.")
            w, h = 512, 512
            img = np.zeros((h, w, 3), dtype=np.uint8)

        # --- RYSOWANIE ---
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)
        ax.set_title(f"Qwen Multi-BBox: {os.path.basename(image_filename)}")
        ax.axis('off')

        for box in normalized_boxes:
            nx1, ny1, nx2, ny2 = box

            # DENORMALIZACJA (kluczowy moment!)
            # Qwen daje 0-1000. Musimy to zamienić na piksele.
            x1 = (nx1 / 1000.0) * w
            y1 = (ny1 / 1000.0) * h
            x2 = (nx2 / 1000.0) * w
            y2 = (ny2 / 1000.0) * h

            # Oblicz szerokość i wysokość dla Rectangle
            rect_w = x2 - x1
            rect_h = y2 - y1

            # Rysowanie ramki
            rect = Rectangle(
                (x1, y1), rect_w, rect_h,
                edgecolor='lime',    # Jasna zieleń dla kontrastu
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(rect)

            # Opcjonalnie: etykieta nad ramką
            # Rysujemy tylko jeśli ramka jest wystarczająco duża, żeby nie zamazać obrazu
            if rect_w > 10 and rect_h > 10:
                ax.text(
                    x1, y1 - 5,
                    label_text,
                    color='lime',
                    fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1)
                )

        # Zapis do pliku
        output_filename = os.path.join(output_dir, f"vis_{os.path.basename(image_filename)}")
        plt.tight_layout()
        plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

        print(f"[{i+1}/{len(lines)}] Zapisano: {output_filename}")

# ==============================================================================
# KONFIGURACJA I URUCHOMIENIE
# ==============================================================================
if __name__ == "__main__":
    # 1. Gdzie jest Twój plik JSONL wygenerowany w poprzednim kroku?
    path_to_jsonl = 'data/qwen_train_multi_bbox.jsonl'

    # 2. Gdzie fizycznie leżą zdjęcia? (potrzebne do wczytania tła i wymiarów)
    # Jeśli skrypt nie znajdzie zdjęć, użyje czarnego tła.
    path_to_images = 'data/syntax/train/images'

    # 3. Uruchomienie (limit=10 wygeneruje 10 pierwszych zdjęć)
    visualize_qwen_jsonl(path_to_jsonl, path_to_images, limit=1)