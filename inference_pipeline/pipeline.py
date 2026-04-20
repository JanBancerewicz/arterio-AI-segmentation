import torch
import os
import re
import tempfile
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
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
        """
        Funkcja odpowiada za inicjalizację modelu.
        Ładuje model bazowy, kwantyzację (jeśli GPU) i nakłada adapter LoRA.
        Przypisuje wartości do zmiennych self.model i self.processor
        """

        print(f"[*] Inicjalizacja modelu na: {self.device.upper()}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        ) if self.device == "cuda" else None

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.base_model_id, quantization_config=bnb_config, device_map="auto"
        )

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"[+] Ładowanie adaptera: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.processor = AutoProcessor.from_pretrained(self.base_model_id)

    def run_inference(self, image_path, prompt):
        """
        Wysyła zdjęcie do modelu i odbiera surową odpowiedź.

        WEJŚCIE:
            - image_path (str): Ścieżka do pliku, np. "data/images/1.png"
            - prompt (str): Tekst, np. "Detect and segment the coronary arteries."

        WYJŚCIE:
            - parsed_boxes (List[tuple]): Lista krotek w standardzie [(x1, y1, x2, y2), ...]
            - raw_output (str): Surowa odpowiedź modelu do debugowania.
        """
        filename = os.path.basename(image_path)

        # Qwen VL wymaga RGB — obrazy angiograficzne są grayscale (mode L), konwertujemy
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Format zgodny z treningiem (qwen_train_bbox.jsonl):
        # user: "Detect the coronary arteries region.\nPicture 1: <img>1.png</img>"
        prompt_with_img = f"{prompt}\nPicture 1: <img>{filename}</img>"

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_with_img}
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, *rest = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # do_sample=False = Greedy Decoding, deterministyczny i dobry do detekcji
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # Skrócenie inputu (prompta) — zostawiamy TYLKO odpowiedź asystenta
        # skip_special_tokens=False żeby zachować <box> tagi używane przez model
        trimmed_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=False)[0]

        parsed_boxes = self._parse_box_output(output_text)

        return parsed_boxes, output_text

    def _parse_box_output(self, text):
        print(f"\n[DEBUG RAW OUTPUT]: {text}")

        # Model trenowany na formacie: <ref>Coronary Arteries</ref><box>(x1,y1),(x2,y2)</box>
        # Dane treningowe (qwen_train_bbox.jsonl) używają kolejności x,y
        box_tags = re.findall(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>', text)

        if not box_tags:
            # Fallback: szukamy czterech liczb w nawiasach
            box_tags = re.findall(r'\((\d+),(\d+)\),\((\d+),(\d+)\)', text)

        boxes = []
        for match in box_tags:
            # Format treningowy: x1, y1, x2, y2
            x1, y1, x2, y2 = map(int, match)

            def clip(v): return max(0, min(1000, v))
            cx1, cy1, cx2, cy2 = clip(x1), clip(y1), clip(x2), clip(y2)

            if cx1 < cx2 and cy1 < cy2:
                boxes.append((cx1, cy1, cx2, cy2))

        if not boxes:
            print("[!] Nie znaleziono żadnych poprawnych boxów w tekście!")

        return boxes
