import json
import random
from pathlib import Path
from PIL import Image

def rescale_bbox(bbox, img_w, img_h):
    # COCO format: [x_min, y_min, width, height] w pikselach
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h

    # Skalowanie bezpośrednio do 0-1000
    return (
        int(round(x_min / img_w * 1000)),
        int(round(y_min / img_h * 1000)),
        int(round(x_max / img_w * 1000)),
        int(round(y_max / img_h * 1000))
    )
def load_coco_annotations(ann_path):
    with open(ann_path, "r") as f:
        return json.load(f)

def build_dataset(images_dir, ann_path, output_path, split_ratio=0.8, bbox_scale=1024):
    ann = load_coco_annotations(ann_path)

    id2file = {img["id"]: img["file_name"] for img in ann["images"]}

    boxes_by_image = {}
    for obj in ann["annotations"]:
        img_id = obj["image_id"]
        boxes_by_image.setdefault(img_id, []).append(obj["bbox"])

    data = []
    for img_id, file_name in id2file.items():
        img_path = Path(images_dir) / file_name
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        w, h = img.size

        bboxes = boxes_by_image.get(img_id, [])
        if not bboxes:
            continue

        boxes_str = ""
        for bbox in bboxes:
            # USUNIĘTO bbox_scale=bbox_scale
            x1, y1, x2, y2 = rescale_bbox(bbox, w, h) 
            boxes_str += f"<ref>coronary vessel</ref><box>({x1},{y1}),({x2},{y2})</box>"
        entry = {
            "image": file_name,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDetect coronary vessels in the angiography image."
                },
                {
                    "from": "assistant",
                    "value": boxes_str
                }
            ]
        }

        data.append(entry)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zapisujemy wszystko do jednego pliku dataset.json
    final_output = output_dir / "test.json"
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    build_dataset(
        images_dir="data/syntax/test/images",
        ann_path="data/syntax/test/annotations/test.json",
        output_path="dataset/qwen_format"
    )
