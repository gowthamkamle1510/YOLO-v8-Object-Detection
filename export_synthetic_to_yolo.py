import os
import torch
import torchvision.transforms as T
from PIL import Image

# === CONFIGURATION ===
SYNTHETIC_PT_PATH = "synthetic_step100.pt"
OUT_IMG_DIR = "coco_synthetic_yolo/images/train"
OUT_LABEL_DIR = "coco_synthetic_yolo/labels/train"
IMAGE_SIZE = 64  # as defined during training

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# === Load synthetic dataset ===
synthetic_data = torch.load(SYNTHETIC_PT_PATH)

# === Normalization to denormalized pixel range
unnormalize = T.Normalize(
    mean=[-0.0 / 1.0, -0.0 / 1.0, -0.0 / 1.0],
    std=[1 / 1.0, 1 / 1.0, 1 / 1.0]
)

to_pil = T.ToPILImage()

image_count = 0

for class_id, images in synthetic_data.items():
    for i, img_tensor in enumerate(images):
        img_tensor = img_tensor.detach().cpu()
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_pil = to_pil(img_tensor)

        filename = f"{class_id}_{i}.jpg"
        filepath = os.path.join(OUT_IMG_DIR, filename)
        img_pil.save(filepath)

        # Create dummy YOLO label: center in image, fixed size
        label_path = os.path.join(OUT_LABEL_DIR, filename.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            # Format: <class_id> <x_center> <y_center> <width> <height>
            f.write(f"{class_id} 0.5 0.5 0.4 0.4\n")

        image_count += 1

print(f"\n✅ Exported {image_count} synthetic images and labels.")
