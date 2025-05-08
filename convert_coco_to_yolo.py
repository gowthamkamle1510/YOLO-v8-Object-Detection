import os
import json
from tqdm import tqdm

# === CONFIG ===
image_dir = "coco_yolo/images/train"
label_dir = "coco_yolo/labels/train"
annotation_file = "coco/annotations/instances_train2017.json"

os.makedirs(label_dir, exist_ok=True)

# === Load image filenames you want labels for ===
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
image_ids = set()

print("🔍 Mapping filenames to image IDs...")
with open(annotation_file, "r") as f:
    data = json.load(f)
    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    filename_to_image_id = {v: k for k, v in image_id_to_filename.items()}
    for fname in image_files:
        if fname in filename_to_image_id:
            image_ids.add(filename_to_image_id[fname])

# === Build category mapping ===
categories = {cat["id"]: idx for idx, cat in enumerate(data["categories"])}  # COCO ID → YOLO ID

print("✍️ Converting annotations to YOLO format...")
for ann in tqdm(data["annotations"]):
    img_id = ann["image_id"]
    if img_id not in image_ids:
        continue

    bbox = ann["bbox"]
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2

    # Get image dimensions
    image_info = next(img for img in data["images"] if img["id"] == img_id)
    iw, ih = image_info["width"], image_info["height"]

    # Normalize
    x_center /= iw
    y_center /= ih
    w /= iw
    h /= ih

    class_id = categories[ann["category_id"]]

    label_file = os.path.join(label_dir, image_info["file_name"].replace(".jpg", ".txt"))
    with open(label_file, "a") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print("✅ Conversion complete!")
