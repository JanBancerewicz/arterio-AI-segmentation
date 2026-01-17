import torch
import os
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

class InferencePipeline:
    def __init__(self, base_model_id, adapter_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model_id = base_model_id
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None

    def initialize_model(self):
        """Konfiguruje i ładuje model wraz z adapterem LoRA."""
        load_params = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }

        if self.device == "cuda":
            load_params["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
            )

        print(f"[*] Inicjalizacja modelu na: {self.device.upper()}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.base_model_id, **load_params
        )

      
        if os.path.exists(self.adapter_path):
            print(f"[+] Załadowano wagi adaptera z: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.processor = AutoProcessor.from_pretrained(self.adapter_path)
        else:
            print("[!] Brak adaptera. Praca na modelu bazowym.")
            self.processor = AutoProcessor.from_pretrained(self.base_model_id)

    def run_inference(self, image_path, prompt):
        """Przetwarza obraz i zwraca listę krotek (y1, x1, y2, x2)."""
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text], images=image_inputs, padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        
        trimmed_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

      
        pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
        matches = re.findall(pattern, output_text)
        return [tuple(map(int, m)) for m in matches]