import json
import os
import numpy as np
import cv2

# CONFIG
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UNET_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(UNET_DIR)

JSON_FILE = os.path.join(PROJECT_ROOT, 'data', 'syntax', 'val', 'annotations', 'val.json')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'data', 'syntax', 'val', 'images')
OUTPUT_MASKS_DIR = os.path.join(PROJECT_ROOT, 'data', 'unet', 'masks', 'syntax', 'val')

# How many images to process / 'None' to process all
MAX_IMAGES_TO_PROCESS = None


def generate_masks_from_coco():
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)

    with open(JSON_FILE, 'r') as f:
        coco_data = json.load(f)

    all_images = coco_data['images']

    if MAX_IMAGES_TO_PROCESS is not None:
        print(f"MAX IMAGES: {MAX_IMAGES_TO_PROCESS}")
        selected_images = all_images[:MAX_IMAGES_TO_PROCESS]
    else:
        selected_images = all_images

    images_info = {img['id']: img for img in selected_images}

    valid_image_ids = set(images_info.keys())

    masks_dict = {}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']

        if image_id not in valid_image_ids:
            continue

        category_id = ann['category_id']
        segmentation = ann['segmentation']

        if image_id not in masks_dict:
            img_width = images_info[image_id]['width']
            img_height = images_info[image_id]['height']
            masks_dict[image_id] = np.zeros((img_height, img_width), dtype=np.uint8)

        current_mask = masks_dict[image_id]

        for poly in segmentation:
            poly_np = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(current_mask, [poly_np], color=(category_id))

    saved_count = 0
    for image_id, mask in masks_dict.items():
        file_name = images_info[image_id]['file_name']

        mask_filename = os.path.splitext(file_name)[0] + '.png'
        mask_path = os.path.join(OUTPUT_MASKS_DIR, mask_filename)

        cv2.imwrite(mask_path, mask)
        saved_count += 1

    print(f"OUTPUT: Generated and saved {saved_count} masks in folder: {OUTPUT_MASKS_DIR}")


if __name__ == "__main__":
    generate_masks_from_coco()