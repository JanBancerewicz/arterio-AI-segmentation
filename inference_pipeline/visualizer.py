import re
from PIL import Image, ImageDraw, ImageFont


class ArterioVisualizer:
    def __init__(self):
        self.colors = {"gt": "lime", "pred": "red"}

    def natural_sort(self, l):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def draw_bounding_boxes(self, img_path, pred_boxes, gt_boxes=None):
        """
        pred_boxes: lista tupli (x1, y1, x2, y2) [0-1000]
        gt_boxes: lista tupli (x1, y1, x2, y2) [0-1000]
        """
        try:
            image = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            w, h = image.size

            # Helper do konwersji 0-1000 -> pixele
            def scale_box(b):
                return [
                    (b[0] / 1000) * w, (b[1] / 1000) * h,
                    (b[2] / 1000) * w, (b[3] / 1000) * h
                ]

            # 1. Rysuj Ground Truth (Zielone) - na spodzie
            if gt_boxes:
                for box in gt_boxes:
                    sb = scale_box(box)
                    draw.rectangle(sb, outline=self.colors["gt"], width=3)

            # 2. Rysuj Predykcje (Czerwone) - na wierzchu
            if pred_boxes:
                for box in pred_boxes:
                    sb = scale_box(box)
                    draw.rectangle(sb, outline=self.colors["pred"], width=2)
                    # Opcjonalnie: mały napis
                    # draw.text((sb[0], sb[1]-10), "AI", fill="red")

            return image
        except Exception as e:
            print(f"Błąd rysowania: {e}")
            return None