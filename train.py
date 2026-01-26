import os

# 0) Optymalizacja pamięci CUDA - musi być na samym początku
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import re
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# --- KONFIGURACJA ---
MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct" # Używamy architektury Qwen2-VL wspieranej przez Transformers
TRAIN_IMAGE_FOLDER = "data/syntax/train/images"
VAL_IMAGE_FOLDER = "data/syntax/val/images"
OUTPUT_DIR = "output/qwen3_vl_coronary_finetune"

# 1) Funkcje pomocnicze
def find_img_path(example):
    p1 = os.path.join(VAL_IMAGE_FOLDER, example["image"])
    p2 = os.path.join(TRAIN_IMAGE_FOLDER, example["image"])
    if os.path.exists(p1): return p1
    if os.path.exists(p2): return p2
    return None

# 2) Ładowanie i filtrowanie datasetu
data_files = {"train": "dataset/qwen_format/train.json", "validation": "dataset/qwen_format/val.json"}
full_dataset = load_dataset("json", data_files=data_files)

dataset = {
    "train": full_dataset["train"].filter(lambda x: find_img_path(x) is not None),
    "validation": full_dataset["validation"].filter(lambda x: find_img_path(x) is not None)
}

# 3) Procesor - KLUCZOWA OPTYMALIZACJA PAMIĘCI
# Ograniczamy rozdzielczość, aby zmieścić się w 16GB VRAM
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True,
    min_pixels=128 * 28 * 28, 
    max_pixels=448 * 28 * 28  # Zmniejszone, aby uniknąć OOM podczas walidacji
)

# 4) Model w 4-bitach
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
    attn_implementation="sdpa"
)

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, # Zmniejszone z 32 dla oszczędności pamięci
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 5) Data Collator
class Qwen3VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        messages_list = []
        for ex in examples:
            img_path = find_img_path(ex)
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": img_path}, {"type": "text", "text": ex['conversations'][0]['value']}]
                },
                {"role": "assistant", "content": ex['conversations'][1]['value']}
            ]
            messages_list.append(messages)
        
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages_list]
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        batch = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        
        labels = batch["input_ids"].clone()
        for i, msg in enumerate(messages_list):
            prompt_text = self.processor.apply_chat_template([msg[0]], tokenize=False, add_generation_prompt=True)
            prompt_ids = self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            labels[i, :len(prompt_ids)] = -100 
        
        batch["labels"] = labels
        return batch

# 6) Argumenty treningu z poprawkami na OOM
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8, # Symuluje batch size 8
    num_train_epochs=3,
    learning_rate=1e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    gradient_checkpointing=True, # Obowiązkowe przy 16GB
    remove_unused_columns=False,
    eval_accumulation_steps=1, # Czyści pamięć po każdym kroku ewaluacji
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=Qwen3VLDataCollator(processor)
)

print("Start treningu (zoptymalizowanego pod 16GB)...")
trainer.train()

# 7) Zapis
trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

# Generowanie wykresu Loss
history = trainer.state.log_history
train_loss = [x["loss"] for x in history if "loss" in x]
eval_loss = [x["eval_loss"] for x in history if "eval_loss" in x]

plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss')
if eval_loss:
    # Dopasowanie kroków dla eval_loss
    steps_eval = [x["step"] for x in history if "eval_loss" in x]
    plt.plot(steps_eval, eval_loss, label='Eval Loss', marker='o')

plt.title('Training Progress - Coronary Vessel Detection')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_plot.png"))
print(f"Gotowe. Model zapisany w {OUTPUT_DIR}")