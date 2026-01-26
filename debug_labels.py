import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw
import os
import re
import json

# --- KONFIGURACJA ---
MODEL_PATH = "output/qwen3_vl_coronary_finetune/final_model"
VAL_JSON_PATH = "dataset/qwen_format/test.json"
IMAGE_NAME = "103.png" 
IMAGE_PATH = os.path.join("data/syntax/test/images", IMAGE_NAME)

# --- FUNKCJE POMOCNICZE ---

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_ground_truth(json_path, img_name):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for entry in data:
        if entry["image"] == img_name:
            text = entry["conversations"][1]["value"]
            return [[int(x) for x in m] for m in re.findall(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", text)]
    return []

def check_training_data_stats():
    with open(VAL_JSON_PATH, 'r') as f:
        data = json.load(f)
    
    box_counts = []
    for entry in data:
        text = entry["conversations"][1]["value"]
        boxes = re.findall(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", text)
        box_counts.append(len(boxes))
    
    print(f"\n📈 Statystyki danych treningowych:")
    print(f"  Min boxów: {min(box_counts)}")
    print(f"  Max boxów: {max(box_counts)}")
    print(f"  Średnio: {sum(box_counts)/len(box_counts):.2f}")

# --- PROCES INFERENCJI ---

print("🚀 Ładowanie modelu...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

messages = [{"role": "user", "content": [{"type": "image", "image": IMAGE_PATH}, {"type": "text", "text": "\nDetect coronary vessels in the angiography image."}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

print("🧠 Analiza obrazu...")
with torch.no_grad():
    generated_ids = model.generate(
    **inputs, 
    max_new_tokens=2048,  # Więcej miejsca
    do_sample=False,       # Deterministyczne
    num_beams=1,           # Bez beam search
    temperature=1.0,
    repetition_penalty=1.0  # Bez kary za powtórzenia
) # ZWIĘKSZONE Z 512
    output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

# --- DIAGNOSTYKA ---
print("\n" + "="*60)
print("🔍 DIAGNOSTYKA KOMPLETNA")
print("="*60)

# 1. Sprawdź surowe wyjście
print(f"\n1️⃣ Surowy tekst modelu:\n{output_text}\n")

# 2. Sprawdź ile boxów wykryto
pred_boxes = re.findall(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", output_text)
print(f"2️⃣ Liczba wykrytych boxów przez model: {len(pred_boxes)}")
for i, box in enumerate(pred_boxes, 1):
    print(f"   Box {i}: ({box[0]},{box[1]}),({box[2]},{box[3]})")

# 3. Porównaj z ground truth
gt_boxes = get_ground_truth(VAL_JSON_PATH, IMAGE_NAME)
print(f"\n3️⃣ Liczba boxów w ground truth: {len(gt_boxes)}")
for i, box in enumerate(gt_boxes, 1):
    print(f"   GT Box {i}: {box}")

# 4. Sprawdź długość generacji
print(f"\n4️⃣ Długość wygenerowanego tekstu: {len(output_text)} znaków")
print(f"5️⃣ Liczba tokenów w input: {inputs.input_ids.shape[1]}")
print(f"6️⃣ Liczba tokenów w output: {generated_ids.shape[1]}")
print(f"7️⃣ Wygenerowano nowych tokenów: {generated_ids.shape[1] - inputs.input_ids.shape[1]}")

# 5. Statystyki danych treningowych
check_training_data_stats()

print("="*60)

# --- WIZUALIZACJA ---

def visualize_results(img_path, pred_text, json_path, img_name):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    raw_boxes = re.findall(r"\((\d+),(\d+)\),\((\d+),(\d+)\)", pred_text)
    
    print(f"\n🎨 Wizualizacja na obrazie {width}x{height}px")
    
    # 1. Rysujemy Predykcję (Zielony)
    for i, box in enumerate(raw_boxes, 1):
        x1 = float(box[0]) * width / 1000.0
        y1 = float(box[1]) * height / 1000.0
        x2 = float(box[2]) * width / 1000.0
        y2 = float(box[3]) * height / 1000.0
        
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 10), f"MODEL {i}", fill="green")
        
        print(f"  Predykcja {i}: surowe={box} → piksele=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    # 2. Rysujemy Ground Truth (Niebieski)
    gt_boxes = get_ground_truth(json_path, img_name)
    for i, g_box in enumerate(gt_boxes, 1):
        gx1 = float(g_box[0]) * width / 1000.0
        gy1 = float(g_box[1]) * height / 1000.0
        gx2 = float(g_box[2]) * width / 1000.0
        gy2 = float(g_box[3]) * height / 1000.0
        draw.rectangle([gx1, gy1, gx2, gy2], outline="blue", width=2)
        draw.text((gx1, gy1 - 25), f"GT {i}", fill="blue")

    img.save("diagnostic_output.png")
    print(f"\n✅ Zapisano wizualizację: diagnostic_output.png")

visualize_results(IMAGE_PATH, output_text, VAL_JSON_PATH, IMAGE_NAME)