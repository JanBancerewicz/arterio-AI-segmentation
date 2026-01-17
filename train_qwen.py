import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 1. KONFIGURACJA SPEED RUN (Szybki Trening)
# ==============================================================================
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_JSONL = "data/qwen_train_multi_bbox.jsonl"
IMAGE_FOLDER = "data/syntax/train/images"
OUTPUT_DIR = "arterio_checkpoints"

# --- PARAMETRY PRĘDKOŚCI ---
BATCH_SIZE = 8

# Zmniejszamy akumulację, bo mamy duży batch.
# Dzięki temu wagi aktualizują się częściej (w czasie zegarowym).
GRAD_ACCUMULATION = 1

LEARNING_RATE = 2e-4
EPOCHS = 4
MAX_PIXELS = 768 * 28 * 28 # Zmniejszenie z 1024 na 768 drastycznie przyspieszy (mniej tokenów obrazu)

# ==============================================================================
# 2. DATASET
# ==============================================================================
class ArterioDataset(Dataset):
    def __init__(self, jsonl_path, image_folder, processor):
        self.data = []
        self.image_folder = image_folder
        self.processor = processor

        print(f"Ładowanie datasetu z: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"Załadowano {len(self.data)} przykładów.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        filename = os.path.basename(item['image'])
        image_path = os.path.join(self.image_folder, filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        messages = item['conversations']
        formatted_messages = []

        for msg in messages:
            role = msg['from']
            content = msg['value']
            if role == 'user':
                new_content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": content.replace(f"<img>{item['image']}</img>", "")}
                ]
                formatted_messages.append({"role": "user", "content": new_content})
            else:
                formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

        text = self.processor.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(formatted_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else None,
            "image_grid_thw": inputs["image_grid_thw"] if "image_grid_thw" in inputs else None,
            "labels": inputs["input_ids"].squeeze(0),
        }

# ==============================================================================
# 3. COLLATOR
# ==============================================================================
def data_collator(features):
    first = features[0]
    batch = {}

    if "input_ids" in first:
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=151643
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )

    if "pixel_values" in first and first["pixel_values"] is not None:
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

    return batch

# ==============================================================================
# 4. SILNIK TRENINGOWY PRO
# ==============================================================================
def main():
    print(f"--- Inicjalizacja Silnika Treningowego (TRYB PRO) ---")

    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=MAX_PIXELS)

    # Konfiguracja 4-bit (Mimo mocnego serwera, 4-bit pozwala na większy batch size)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Ładowanie modelu: {MODEL_ID}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2"
    )

    model = prepare_model_for_kbit_training(model)

    # --- KONFIGURACJA LORA  ---
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=[],
    )
    model = get_peft_model(model, peft_config)

    print("Statystyki trenowalnych parametrów (Większy model LoRA):")
    model.print_trainable_parameters()

    train_dataset = ArterioDataset(DATA_JSONL, IMAGE_FOLDER, processor)

    # --- ARGUMENTY TRENINGOWE ---
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,

        # Optymalizacja pod GPU
        bf16=True,                  # Używamy BFloat16 (wymaga Ampere lub nowszej, np. RTX 30xx/40xx)
        fp16=False,                 # Wyłączamy stare fp16 na rzecz bf16
        gradient_checkpointing=True, # Oszczędza VRAM, pozwala na większy batch/rozdzielczość
        dataloader_num_workers=4,   # Szybsze ładowanie danych (CPU -> GPU)

        # Scheduler
        lr_scheduler_type="cosine", # Lepsza zbieżność niż "linear"
        warmup_ratio=0.03,          # 3% czasu na rozgrzewkę

        # Logowanie i zapis
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(">>> ROZPOCZYNAM TRENING (10 epok, LoRA r=64) <<<")
    trainer.train()

    print(f"Zapisywanie modelu do {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(">>> ZAKOŃCZONO SUKCESEM <<<")

if __name__ == "__main__":
    main()