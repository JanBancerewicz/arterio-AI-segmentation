import os
import re
from PIL import Image, ImageDraw

class ArterioVisualizer:
    @staticmethod
    def natural_sort(file_list):
        """Sortuje listę plików numerycznie (1, 2, 10 zamiast 1, 10, 2)."""
        return sorted(file_list, key=lambda f: int(re.sub(r'\D', '', f)) if re.sub(r'\D', '', f) else 0)

    @staticmethod
    def draw_bounding_boxes(image_path, boxes, color="green", width=3):
        """Otwiera obraz, rysuje ramki i zwraca obiekt Image."""
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for (y1_n, x1_n, y2_n, x2_n) in boxes:
            
            x1, y1 = (x1_n / 1000) * w, (y1_n / 1000) * h
            x2, y2 = (x2_n / 1000) * w, (y2_n / 1000) * h
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        return img