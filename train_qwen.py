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
# 1. KONFIGURACJA (Settings)
# ==============================================================================
# Ścieżki (Dostosowane pod strukturę na serwerze)
# Zakładamy, że wrzucisz folder Qwen rozpakowany, lub model pobierze się sam
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_JSONL = "train.jsonl"               # Plik w tym samym folderze co skrypt
IMAGE_FOLDER = "images"                  # Folder z obrazami obok skryptu
OUTPUT_DIR = "arterio_checkpoints"       # Tu wylądują wyniki

# Hiperparametry
BATCH_SIZE = 1           # 1 zdjęcie na GPU (bezpieczne dla 16-32GB VRAM przy dużych obrazach)
GRAD_ACCUMULATION = 16   # Zbieramy gradienty z 16 kroków (efektywny batch = 16)
LEARNING_RATE = 2e-4     # Standard dla QLoRA
EPOCHS = 3               # Ilość przejść przez cały zbiór
MAX_PIXELS = 1024 * 28 * 28 # Zwiększone dla medycyny (ok. 1000x1000px)

# ==============================================================================
# 2. DATASET (Ładowanie Danych)
# ==============================================================================
class ArterioDataset(Dataset):
    def __init__(self, jsonl_path, image_folder, processor):
        self.data = []
        self.image_folder = image_folder
        self.processor = processor

        # Wczytanie pliku JSONL do pamięci
        print(f"Ładowanie datasetu z: {jsonl_path}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        print(f"Załadowano {len(self.data)} przykładów.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 1. Obsługa obrazu
        # Ścieżka w JSONL może być np. "images/922.png" lub po prostu "922.png"
        # Musimy to skleić z folderem bazowym na serwerze
        filename = os.path.basename(item['image'])
        image_path = os.path.join(self.image_folder, filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        # 2. Przygotowanie konwersacji
        # Kopiujemy strukturę rozmowy z JSONL, ale podmieniamy ścieżkę do obrazka
        # na obiekt zrozumiały dla processora Qwen
        messages = item['conversations']
        formatted_messages = []

        for msg in messages:
            role = msg['from']
            content = msg['value']

            if role == 'user':
                # Szukamy znacznika <img>...</img> i zamieniamy go na obiekt obrazu
                # Zakładamy, że w preprocessingu przygotowaliśmy format:
                # "Text... Picture 1: <img>filename.png</img>"

                # Dla uproszczenia w Qwen-VL Processorze podajemy listę:
                new_content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": content.replace(f"<img>{item['image']}</img>", "")}
                ]
                formatted_messages.append({"role": "user", "content": new_content})
            else:
                # Odpowiedź asystenta (czysty tekst z BBoxami)
                formatted_messages.append({"role": "assistant", "content": [{"type": "text", "text": content}]})

        # 3. Przetwarzanie przez Processor (Tekst -> Tokeny, Obraz -> Tensory)
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
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0), # Model uczy się przewidywać własne wejście
        }

# ==============================================================================
# 3. COLLATOR (Pakowanie batcha)
# ==============================================================================
def data_collator(features):
    # Qwen wymaga specyficznego łączenia danych (obrazki mają różne rozmiary gridu)
    first = features[0]
    batch = {}

    # Paddujemy tekst do najdłuższego w batchu
    if "input_ids" in first:
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=151643 # pad_token_id Qwena
        )
        batch["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100 # -100 ignoruje stratę na paddingu
        )

    # Obrazy łączymy w jedną listę (flatten)
    if "pixel_values" in first:
        batch["pixel_values"] = torch.cat([f["pixel_values"] for f in features], dim=0)
        batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

    return batch

# ==============================================================================
# 4. SILNIK TRENINGOWY (Main Engine)
# ==============================================================================
def main():
    print(f"--- Inicjalizacja Silnika Treningowego ---")

    # 1. Processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=MAX_PIXELS)

    # 2. Model z Kwantyzacją (QLoRA Setup)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"Ładowanie modelu: {MODEL_ID}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Zamrażanie wag bazowych
    model = prepare_model_for_kbit_training(model)

    # 3. Konfiguracja Adapterów LoRA
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

    print("Statystyki trenowalnych parametrów:")
    model.print_trainable_parameters()

    # 4. Ładowanie Danych
    train_dataset = ArterioDataset(DATA_JSONL, IMAGE_FOLDER, processor)

    # 5. Konfiguracja Trainera
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,                # Używamy Mixed Precision (szybciej na NVIDIA)
        logging_steps=5,          # Loguj co 5 kroków
        save_strategy="epoch",    # Zapisuj co epokę
        save_total_limit=1,       # Trzymaj tylko ostatni checkpoint
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 6. START PĘTLI
    print(">>> ROZPOCZYNAM TRENING <<<")
    trainer.train()

    # 7. Zapis wyników
    print(f"Zapisywanie modelu do {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(">>> ZAKOŃCZONO SUKCESEM <<<")

if __name__ == "__main__":
    main()