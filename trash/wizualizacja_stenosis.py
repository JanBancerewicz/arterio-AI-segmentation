import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from PIL import Image

def visualize_real_image(json_path, images_dir, target_id=None, save_path="wynik.png"):
    """
    Wizualizuje adnotacje COCO na prawdziwym obrazie z dysku.
    """
    # 1. Wczytanie JSON
    if not os.path.exists(json_path):
        print(f"❌ Błąd: Nie znaleziono pliku JSON: {json_path}")
        return

    with open(json_path, 'r') as f:
        coco = json.load(f)

    # 2. Mapowanie kategorii (ID -> Nazwa)
    # Tworzymy słownik np. {26: 'stenosis', 1: '1', ...}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # 3. Wybór obrazu
    if target_id is None:
        # Jeśli nie podano ID, bierzemy pierwszy z listy, który ma adnotacje
        # (szukamy obrazu, który faktycznie ma przypisane jakieś bbox/segmentacje)
        annotated_ids = set(ann['image_id'] for ann in coco['annotations'])
        valid_images = [img for img in coco['images'] if img['id'] in annotated_ids]

        if not valid_images:
            print("❌ Brak obrazów z adnotacjami w tym pliku JSON.")
            return

        img_info = valid_images[0]
        print(f"ℹ️ Automatyczny wybór obrazu ID: {img_info['id']}")
    else:
        # Szukamy konkretnego ID
        img_info = next((img for img in coco['images'] if img['id'] == target_id), None)
        if img_info is None:
            print(f"❌ Nie znaleziono obrazu o ID {target_id} w pliku JSON.")
            return

    image_id = img_info['id']
    file_name = img_info['file_name']

    # 4. Budowanie ścieżki do obrazu
    full_image_path = os.path.join(images_dir, file_name)

    if not os.path.exists(full_image_path):
        print(f"❌ Błąd: Plik obrazu nie istnieje: {full_image_path}")
        print(f"   Sprawdź, czy folder 'images_dir' jest poprawny.")
        return

    # 5. Wczytanie obrazu
    try:
        image = Image.open(full_image_path).convert("RGB")
    except Exception as e:
        print(f"❌ Błąd otwierania obrazu: {e}")
        return

    # 6. Rysowanie
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    plt.axis('off')
    plt.title(f"Plik: {file_name} (ID: {image_id}) | Dataset: {os.path.basename(images_dir)}")

    # Pobranie adnotacji dla tego obrazu
    annotations = [ann for ann in coco['annotations'] if ann['image_id'] == image_id]

    # Kolory: Stenosis na czerwono/różowo, anatomia (1-16) na niebiesko/zielono
    colors_anatomy = list(mcolors.TABLEAU_COLORS.values())

    print(f"✅ Znaleziono {len(annotations)} obiektów na zdjęciu.")

    for i, ann in enumerate(annotations):
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, 'unknown')
        bbox = ann.get('bbox', []) # [x, y, w, h]
        segmentation = ann.get('segmentation', [])

        # Ustalanie koloru: Stenosis (ID 26) wyróżniamy!
        if cat_name == 'stenosis':
            color = '#FF00FF' # Magenta / Fuksja dla stenozy
            line_width = 3
            alpha_fill = 0.4
        else:
            # Anatomia - losowy kolor z palety, ale stały dla danego ID
            color = colors_anatomy[cat_id % len(colors_anatomy)]
            line_width = 1
            alpha_fill = 0.15

        # A. Rysowanie Segmentacji (Polygon)
        if segmentation:
            for seg in segmentation:
                poly = [ (seg[i], seg[i+1]) for i in range(0, len(seg), 2) ]
                poly_patch = patches.Polygon(poly, closed=True, edgecolor=color, facecolor=color, alpha=alpha_fill)
                ax.add_patch(poly_patch)

        # B. Rysowanie BBox
        if bbox:
            x, y, w, h = bbox
            rect = patches.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Etykieta
            ax.text(x, y - 5, cat_name, color='white', fontsize=9, fontweight='bold',
                    bbox=dict(facecolor=color, edgecolor='none', alpha=0.8, pad=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"💾 Zapisano wynik w pliku: {save_path}")
    plt.close()

# ============================================================================
# KONFIGURACJA - TU ZMIENIASZ KATALOGI
# ============================================================================

# OPCJA 1: Sprawdźmy zbiór TRAIN (tam gdzie jest 676)
visualize_real_image(
    json_path='../data/stenosis/train/annotations/train.json',
    images_dir='../data/stenosis/train/images',
    target_id=676,  # Możesz wpisać None, żeby wybrał automat
    save_path="../wizualizacja_train_676.png"
)

# OPCJA 2: Sprawdźmy zbiór TEST (tam gdzie są pliki 1.png, 10.png)
# Patrząc na Twój snippet test.json, ID 1 to plik "1.png" i ma stenozę (cat_id 26)!
visualize_real_image(
    json_path='../data/stenosis/test/annotations/test.json',
    images_dir='../data/stenosis/test/images',
    target_id=1,    # Sprawdzamy ID 1
    save_path="../wizualizacja_test_1.png"
)