import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

def visualize_coco_annotations_with_bbox_and_labels(json_path, image_id, output_filename="annotated_image_final.png"):
    """
    Wizualizuje adnotacje (segmentacje) oraz ramki ograniczające (bbox) z opisami.

    :param json_path: Ścieżka do pliku COCO JSON.
    :param image_id: ID obrazu do wizualizacji.
    :param output_filename: Nazwa pliku wyjściowego.
    """
    try:
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku JSON pod ścieżką: {json_path}")
        return

    image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
    if image_info is None:
        print(f"Błąd: Nie znaleziono obrazu o ID: {image_id}")
        return

    file_name = image_info['file_name']
    img_width = image_info['width']
    img_height = image_info['height']

    # 1. Mapowanie ID kategorii na nazwy
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Używamy czarnego obrazu zastępczego (zastąp własnym plikiem PNG)
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)
    ax.set_title(f"Segmenty i BBox dla {file_name} (ID: {image_id})")
    ax.axis('off')

    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    # Słownik do śledzenia kolorów dla różnych kategorii (opcjonalne)
    # colors = plt.cm.get_cmap('hsv', len(categories)) 
    
    for ann in annotations:
        category_id = ann.get('category_id')
        category_name = categories.get(category_id, f"ID: {category_id}")
        bbox = ann.get('bbox') # [x, y, w, h]

        # 2. Rysowanie segmentacji (Poligon)
        if 'segmentation' in ann and ann['segmentation'] and isinstance(ann['segmentation'][0], list):
            for segment_coords in ann['segmentation']:
                points = np.array(segment_coords).reshape(-1, 2)
                
                # Rysowanie segmentu
                patch = Polygon(
                    points, 
                    closed=True, 
                    edgecolor='lime', 
                    facecolor=(0, 1, 0, 0.2), # Bardzo przezroczyste wypełnienie
                    linewidth=1
                )
                ax.add_patch(patch)
        
        # 3. Rysowanie Ramki Ograniczającej (BBox) i Opisu
        if bbox:
            x, y, w, h = bbox
            
            # Rysowanie prostokąta (bbox)
            rect = Rectangle(
                (x, y), w, h,
                edgecolor='blue', # Kolor ramki
                facecolor='none',
                linewidth=2
            )
            ax.add_patch(rect)
            
            # Dodawanie etykiety (opis)
            ax.text(
                x, y - 5, # Pozycja: lekko nad ramką (y - 5)
                category_name, 
                color='white', 
                fontsize=8,
                # Styl tła etykiety
                bbox=dict(facecolor='blue', alpha=0.7, edgecolor='none', pad=1)
            )

    # Zapisanie wizualizacji
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    return output_filename

# ==============================================================================
# PRZYKŁAD UŻYCIA ULEPSZONEJ FUNKCJI DLA OBRAZU 922:
# output_file = visualize_coco_annotations_with_bbox_and_labels('data/syntax/train/annotations/train.json', image_id=5)
output_file = visualize_coco_annotations_with_bbox_and_labels('data/syntax/test/annotations/test.json', image_id=1)
print(f"Plik wyjściowy z segmentami, bbox i etykietami: {output_file}")