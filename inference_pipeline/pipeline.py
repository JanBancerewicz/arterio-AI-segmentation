import torch
import os
import re
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import json


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

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.base_model_id, quantization_config=bnb_config, device_map="auto"
        )

        if self.adapter_path and os.path.exists(self.adapter_path):
            print(f"[+] Ładowanie adaptera (niem. Adapter wird geladen): {self.adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

        self.processor = AutoProcessor.from_pretrained(self.base_model_id)

    def run_inference(self, image_path, prompt):
        """
        Wysyła zdjęcie do modelu i odbiera surową odpowiedź.

        WEJŚCIE:
            - image_path (str): Ścieżka do pliku, np. "data/images/1.png"
            - prompt (str): Tekst, np. "Detect the coronary arteries."

        WYJŚCIE:
            - parsed_boxes (List[tuple]): Lista krotek w standardzie [(x1, y1, x2, y2), ...]
              Przykład: [(100, 200, 150, 250), (500, 500, 600, 600)]
              [!] To są już "czyste" dane, po naprawie kolejności X/Y.

            - raw_output (str): Surowa odpowiedź modelu do debugowania.
              Przykład: "Sure! <box>(200,100),(250,150)</box>"
        """

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, *rest = process_vision_info(messages)

        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            # Inferencja
            # do_sample=False zapewnia Greedy Decoding, który jest najbardziej deterministyczny i dobry do detekcji
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        # Skrócienie inputu (prompta) w celu uzyskania samej odpowiedzi asystenta
        trimmed_ids = [out[len(ins):] for ins, out in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

        parsed_boxes = self._parse_box_output(output_text)

        return parsed_boxes, output_text

    def _parse_box_output(self, text):
        """
            Tłumaczy "standard Qwena" na "standard dalszych algorytmów".

            WEJŚCIE:
                - text (str): Surowy string z modelu.
                  Może zawierać tekst, tagi <box> lub same liczby.
                  Np.: "Found: (ymin, xmin), (ymax, xmax)"

            PROCES:
                1. Znajdź wszystkie liczby integer w tekście.
                2. Grupuj czwórkami: [y1, x1, y2, x2].
                3. Zamień kolejność na: [x1, y1, x2, y2].
                4. Odsiej wartości spoza zakresu 0-1000.

            WYJŚCIE:
                - boxes (List[tuple]): [(xmin, ymin, xmax, ymax), ...]
                  Jeśli nic nie znajdzie -> zwraca pustą listę [].
            """

        import re

        # Wypisywanie do konsoli w celu debugowania formatu
        print(f"\n[DEBUG RAW OUTPUT]: {text}")

        # Strategia wyekstraktowania listy boxów z wypowiedzi zwrotnej modelu polega
        # na wyszukiwania wszystkich grup 4 liczb i zapisywaniu ich jako koordynaty boxów
        # Regex szuka: liczba, separator, liczba, separator, liczba, separator, liczba
        # Separatorem ([\D]+) może być przecinek, spacja, nawias, cokolwiek co nie jest cyfrą.
        # Takie podejście jest odporne na format (działa dla JSON, XML i <box>), ale trzyma liczby w grupach.
        matches = re.findall(r"(\d+)[\D]+(\d+)[\D]+(\d+)[\D]+(\d+)", text)

        boxes = []

        for match in matches:
            y1, x1, y2, x2 = map(int, match)

            # Clamping
            # Naprawienie drobnych błędów modelu poprzez przycięcie współrzędnych do zakresu (0, 1000)
            def clip(val):
                return max(0, min(1000, val))

            x1 = clip(x1)
            y1 = clip(y1)
            x2 = clip(x2)
            y2 = clip(y2)

            # Dodatkowe zabezpieczenie:
            # Jeśli ramka ma zerową powierzchnię (x1==x2 lub y1==y2), to najprawdopodbniej jest to błąd
            # i należy taką ramkę pominąć
            if x1 >= x2 or y1 >= y2:
                continue

            boxes.append((x1, y1, x2, y2))

        if not boxes:
            print("Nie znaleziono żadnych boxów!")

        return boxes