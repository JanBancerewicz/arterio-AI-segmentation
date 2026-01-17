import torch
from PIL import Image, ImageDraw
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import re
import os

# ==========================================
# KONFIGURACJA
# ==========================================
# Ścieżka do Twojego wytrenowanego folderu
ADAPTER_PATH = "arterio_checkpoints"

# Ścieżka do zdjęcia, które chcesz przetestować (zmień na jakieś swoje!)
TEST_IMAGE_PATH = "data/syntax/test/images/1.png"

# Model bazowy (musi być ten sam co przy treningu)
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

def draw_boxes(image_path, boxes_norm):
    """
    Rysuje ramki na obrazie.
    boxes_norm: lista krotek [(y1, x1, y2, x2), ...] w skali 0-1000
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    print(f"Oryginalny rozmiar obrazu: {width}x{height}")

    for box in boxes_norm:
        # Qwen zwraca y1, x1, y2, x2 w skali 0-1000. Musimy to zamienić na piksele.
        # Format Qwen: (y_min, x_min, y_max, x_max)
        y1_n, x1_n, y2_n, x2_n = box

        x1 = (x1_n / 1000) * width
        y1 = (y1_n / 1000) * height
        x2 = (x2_n / 1000) * width
        y2 = (y2_n / 1000) * height

        # Rysowanie (zielona ramka, grubość 3)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

    return image

def main():
    print("--- Ładowanie modelu (to może chwilę potrwać) ---")

    # 1. Konfiguracja 4-bit (żeby model zmieścił się na laptopie)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Ładowanie Modelu Bazowego
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 3. Ładowanie Twoich Wag (LoRA)
    print(f"Ładowanie adaptera z: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Scalanie wag (opcjonalne, ale przyspiesza inferencję)
    # model = model.merge_and_unload()

    # 4. Ładowanie Processora (korzystamy z configu w Twoim folderze)
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    # ==========================================
    # INFERENCJA
    # ==========================================
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"BŁĄD: Nie znaleziono pliku {TEST_IMAGE_PATH}")
        return

    print(f"Analiza obrazu: {TEST_IMAGE_PATH}...")

    # Prompt systemowy (taki sam jak w treningu!)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE_PATH},
                {"type": "text", "text": "Detect and segment the coronary arteries."},
            ],
        }
    ]

    # Przygotowanie danych
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda") # Przeniesienie na GPU

    # Generowanie odpowiedzi
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    # Dekodowanie wyjścia (usuwamy input z outputu)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n--- ODPOWIEDŹ MODELU (RAW) ---")
    print(output_text)

    # ==========================================
    # PARSOWANIE I RYSOWANIE
    # ==========================================
    # Wyciąganie koordynatów z formatu <box>(y1,x1),(y2,x2)</box>
    # Regex szuka wzorca: (liczba,liczba),(liczba,liczba)
    pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
    matches = re.findall(pattern, output_text)

    boxes = []
    for match in matches:
        # BYŁO (zgodnie z dokumentacją Qwen):
        # y1, x1, y2, x2 = map(int, match)
        x1, y1, x2, y2 = map(int, match) # <--- ZAMIANA X z Y

        boxes.append((y1, x1, y2, x2)) # Dodajemy do listy, funkcja rysująca sobie poradzi

    print(f"\nWykryto {len(boxes)} segmentów.")

    if boxes:
        result_image = draw_boxes(TEST_IMAGE_PATH, boxes)

        # Zapis i wyświetlenie
        output_filename = "wynik_detekcji.png"
        result_image.save(output_filename)
        print(f"Zapisano wynik jako: {output_filename}")
        result_image.show() # Otwiera domyślną przeglądarkę zdjęć
    else:
        print("Nie wykryto żadnych obiektów (model zwrócił pusty tekst lub błędny format).")

if __name__ == "__main__":
    main()