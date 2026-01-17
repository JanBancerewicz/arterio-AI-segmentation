import json
import re
import os

class Validator:
    def __init__(self, gt_jsonl_path):
        self.gt_map = {}
        print(f"Ładowanie Ground Truth z: {gt_jsonl_path} ...")
        
        if not os.path.exists(gt_jsonl_path):
            print(f"BŁĄD: Nie znaleziono pliku {gt_jsonl_path}")
            return

        with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    image_filename = data.get('image')
                    
                    assistant_text = ""
                    for msg in data.get('conversations', []):
                        if msg['from'] == 'assistant':
                            assistant_text = msg['value']
                            break
                    
                    boxes = self._parse_qwen_boxes(assistant_text)
                    
                    if image_filename and boxes:
                        self.gt_map[image_filename] = boxes
                        
                except json.JSONDecodeError:
                    continue

        print(f"Załadowano etykiety dla {len(self.gt_map)} obrazów.")

    def _parse_qwen_boxes(self, text):
        pattern = r"\((\d+),(\d+)\),\((\d+),(\d+)\)"
        matches = re.findall(pattern, text)
        
        boxes = []
        for m in matches:
            y1, x1, y2, x2 = map(int, m)
            boxes.append([x1, y1, x2, y2])
            
        return boxes

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        
        if unionArea == 0: return 0.0
        return interArea / unionArea

    def evaluate(self, image_path_or_name, ai_pred_boxes):
        filename = os.path.basename(image_path_or_name) 

        if filename not in self.gt_map:
            print(f"Brak etykiety (GT) dla pliku: {filename}")
            return None 

        gt_boxes = self.gt_map[filename]
        max_iou = 0.0
        
        if not ai_pred_boxes:
            return {"iou": 0.0, "match": False, "gt": gt_boxes, "pred": []}

        for gt in gt_boxes:
            for pred in ai_pred_boxes:
                iou = self.calculate_iou(gt, pred)
                if iou > max_iou:
                    max_iou = iou
        
        is_match = max_iou > 0.5
        
        return {
            "filename": filename,
            "iou": max_iou,
            "match": is_match,
            "gt": gt_boxes,   
            "pred": ai_pred_boxes 
        }