import json
import os
import re

class Validator:
    """
    Klasa odpowiedzialna za wczytanie współrzędnych wzorowych boxów (Ground Truth) i ocenę wyników modelu.
    Porównuje output modelu z tym co zaznaczył człowiek.

    Podstawowy wskaźnik skuteczności:
    - IoU (Intersection over Union) - "procent nakładania się ramek"

    Klasyfikacja wyników:
    - TP (True Positive) – Model narysował ramkę i ona pokrywa się z ramką lekarza (IoU > 0.5)
    - FP (False Positive) – Fałszywy Alarm: Model narysował ramkę, ale pod spodem nic nie ma (albo IoU < 0.5)
    - FN (False Negative) – Przeoczenie: Lekarz zaznaczył tętnicę, a model jej nie wykrył (brak ramki w tym miejscu)

    Metryki końcowe:
    - Precision (Precyzja)
    - Recall (Czułość)
    - F1-Score: Średnia harmoniczna z obu powyższych
    """

    def __init__(self, gt_jsonl_path):
        self.ground_truth = {}
        # Wywołujemy ładowanie
        self.load_gt(gt_jsonl_path)

    def load_gt(self, path):
     with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            filename = os.path.basename(data['image_path'])
            
            clean_boxes = []
            for box_str in data.get('boxes_list', []):
                # Format val_bbox.jsonl: "(x1,y1),(x2,y2)"
                nums = re.findall(r'\d+', box_str)
                if len(nums) == 4:
                    x1, y1, x2, y2 = map(int, nums)
                    clean_boxes.append((x1, y1, x2, y2))
            
            self.ground_truth[filename] = clean_boxes
    def get_gt(self, filename):
        """Pobiera poprawne ramki dla danego pliku"""
        basename = os.path.basename(filename)
        return self.ground_truth.get(basename, [])

    # ... reszta metod (calculate_iou, match_and_score) bez zmian ...
    def calculate_iou(self, boxA, boxB):
        """
        Obliczenie w jakiej części nakładają się prostokąty (IoU).
        Zwraca od 0.0 (brak styku) do 1.0 (identyczne).
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0

    def match_and_score(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        """
        Algorytm Greedy Matching.
        Dla każdej ramki modelu szukamy najlepiej pasującej ramki z Ground Truth.

        Zwraca: TP (Trafienia), FP (Szum), FN (Przeoczenia).
        """
        tp = 0
        fp = 0
        fn = 0
        matched_gt_indices = set()

        for p_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt_indices: continue
                iou = self.calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt_indices)
        return {"tp": tp, "fp": fp, "fn": fn}