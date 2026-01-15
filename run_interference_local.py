import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import re
from PIL import Image

# ==========================================
# KONFIGURACJA
# ==========================================

# Ścieżka do Twojego pobranego modelu 2B (zalecane na start)
# Upewnij się, że ta ścieżka istnieje. Jeśli nie, podaj po prostu "Qwen/Qwen2-VL-2B-Instruct" - pobierze brakujące pliki
MODEL_PATH = r"C:\Users\Jan Bancerewicz\.cache\huggingface\hub\models--Qwen--Qwen2-VL-2B-Instruct\snapshots\..."
# UWAGA: W folderze .cache często są same skróty (blobs).
# Jeśli to nie zadziała, wpisz po prostu nazwę repozytorium, a on skorzysta z cache automatycznie:
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

IMAGE_PATH = "data/syntax/train/images/922.png" # Podaj ścieżkę do jakiegoś zdjęcia testowego

# ==========================================
# 1. ŁADOWANIE MODELU
# ==========================================
print(f"Sprawdzanie CUDA: {torch.cuda.is_available()}")

# Ładujemy procesor (obsługuje obrazki i tekst)
# min_pixels i max_pixels sterują rozdzielczością - dla medycznych dajemy więcej
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28)

# Ładujemy model z kwantyzacją 4-bitową (oszczędza RAM)
try:
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    print("Ładowanie modelu w 4-bitach (tryb oszczędny)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    )
except ImportError:
    print("Brak bitsandbytes - ładowanie w fp16 (może zająć dużo VRAM)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

# ==========================================
# 2. PRZYGOTOWANIE DANYCH
# ==========================================

prompt_text = "Detect the coronary arteries region."

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": IMAGE_PATH,
            },
            {"type": "text", "text": prompt_text},
        ],
    }
]

# Przygotowanie wsadu dla modelu
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# ==========================================
# 3. GENEROWANIE (INFERENCJA)
# ==========================================
print("Generowanie odpowiedzi...")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)

# Dekodowanie wyjścia (usuwamy tokeny wejściowe z wyniku)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]

print("\n--- ODPOWIEDŹ MODELU ---")
print(output_text)
print("------------------------")

# ==========================================
# 4. WIZUALIZACJA (Szybki podgląd)
# ==========================================

# Parsowanie boxów z odpowiedzi <box>(y1,x1),(y2,x2)</box>
# UWAGA: Qwen2-VL domyślnie zwraca (y, x) w skali 1000! Trzeba uważać na kolejność.
boxes = []
pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
matches = re.findall(pattern, output_text)

if matches:
    img = Image.open(IMAGE_PATH)
    w, h = img.size

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    ax = plt.gca()

    for match in matches:
        # Qwen output: (y1, x1), (y2, x2) normalized 0-1000
        ny1, nx1, ny2, nx2 = map(int, match)

        # Denormalizacja
        x1 = (nx1 / 1000) * w
        y1 = (ny1 / 1000) * h
        x2 = (nx2 / 1000) * w
        y2 = (ny2 / 1000) * h

        rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
        ax.text(x1, y1, "Detection", color='white', bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.title(f"Wynik surowego modelu Qwen2-VL-2B\nPrompt: {prompt_text}")
    plt.show()
else:
    print("Nie znaleziono ramek w odpowiedzi modelu.")