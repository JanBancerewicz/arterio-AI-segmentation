import os
import cv2
import numpy as np

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

IMAGES_DIR = os.path.join(PROJECT_ROOT, 'data_processing', 'syntax', 'engine', 'images')
MASKS_DIR = os.path.join(PROJECT_ROOT, 'data_processing', 'unet', 'masks', 'syntax', 'engine')

OUTPUT_VIS_DIR = os.path.join(PROJECT_ROOT, 'data_processing', 'unet', 'visualizations')
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


def visualize_masks(num_images_to_check=10):

    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

    if not mask_files:
        print("WARNING: Not found any masks in {}".format(MASKS_DIR))
        return

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(50, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]

    count = 0

    for mask_name in mask_files:
        if count >= num_images_to_check:
            break

        img_path = os.path.join(IMAGES_DIR, mask_name)
        mask_path = os.path.join(MASKS_DIR, mask_name)

        if not os.path.exists(img_path):
            print(f"WARNING: Original not found for {mask_name}; continuing")
            continue

        original_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        colored_mask = np.zeros_like(original_img)

        unique_classes = np.unique(mask)

        for cls in unique_classes:
            if cls == 0:
                continue

            colored_mask[mask == cls] = colors[cls].tolist()

        alpha = 0.7
        beta = 0.5
        blended = cv2.addWeighted(original_img, alpha, colored_mask, beta, 0)

        out_path = os.path.join(OUTPUT_VIS_DIR, f"check_{mask_name}")
        cv2.imwrite(out_path, blended)
        count += 1

    print(f"OUTPUT: Visualizations in: {OUTPUT_VIS_DIR}")

if __name__ == '__main__':
    visualize_masks(10)