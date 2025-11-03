import os
import json
from datasets import Dataset
from PIL import Image
from typing import Dict, Any, List, Union


IMAGE_FOLDER = "data/syntax/train/images" 
JSON_FILE = "data/syntax/train/annotations/train.json"


def group_annotations(image_folder: str, json_file: str) -> List[Dict[str, Any]]:
    """
    Wczytuje plik COCO JSON, grupuje adnotacje pod ID obrazu i zwraca surową listę przykładów.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: Plik JSON nie znaleziony: {json_file}")
        return []
        
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    image_map = {img['id']: img for img in coco_data.get('images', [])}
    examples = []
    
    for image_id, ann_list in annotations_by_image.items():
        img_info = image_map.get(image_id)
        if not img_info or not ann_list:
            continue
            
        img_path = os.path.join(image_folder, img_info['file_name'])
        
        output_data = {
            "image_id": image_id,
            "width": img_info['width'],
            "height": img_info['height'],
            "annotations": ann_list
        }
        
        examples.append({
            "image_path": img_path,
            "segmentation_data": output_data
        })
        
    return examples


def format_qwen_vl_data(item: Dict[str, Any], processor: Any) -> Dict[str, Union[str, None]]:
    """
    Formatuje parę Obraz (PIL) - JSON (tekst) w ciąg konwersacji Qwen-VL.
    Ta funkcja jest przeznaczona do użycia z metodą .map() obiektu Hugging Face Dataset.
    """
    image_path = item["image_path"]
    segmentation_json = item["segmentation_data"]
    
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        # Ten błąd jest już częściowo obsłużony przez group_annotations
        return {"text": None}
    
    try:
        json_str = json.dumps(segmentation_json, indent=2, ensure_ascii=False)
        assistant_response = f"```json\n{json_str}\n```"
    except Exception:
        return {"text": None}

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": "Wygeneruj kompletne adnotacje segmentacji zwężeń naczyń wieńcowych w formacie COCO JSON."}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_response}
            ]
        }
    ]

    try:
        text_with_image_tokens = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False 
        )
    except Exception:
        return {"text": None}

    return {"text": text_with_image_tokens}


def load_multimodal_dataset(processor: Any, max_seq_length: int) -> Dataset:
    """
    Główna funkcja ładująca, która pobiera dane, formatuje je i zwraca gotowy obiekt Dataset.
    """
    print("Krok Dataloader: Ładowanie surowych danych i grupowanie adnotacji...")
    raw_examples = group_annotations(IMAGE_FOLDER, JSON_FILE)
    raw_dataset = Dataset.from_list(raw_examples)

    print("Krok Dataloader: Mapowanie i formatowanie do konwersacji Qwen-VL...")
    
    # Przekazujemy max_seq_length, aby był dostępny dla procesów mapowania, 
    # chociaż jego główna rola jest w SFTTrainer.
    dataset = raw_dataset.map(
        lambda item: format_qwen_vl_data(item, processor), 
        remove_columns=raw_dataset.column_names, 
        num_proc=os.cpu_count() or 2,
        batched=False
    ).filter(lambda x: x["text"] is not None)

    dataset = dataset.shuffle(seed=42)
    return dataset