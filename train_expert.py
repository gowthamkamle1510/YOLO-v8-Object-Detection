import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CocoDetection
import torchvision.transforms as transforms
from tqdm import tqdm
import sys

# Add current folder to sys.path to import models.py
sys.path.append(os.path.dirname(__file__))
from models import SimpleCNN

# === CONFIGURATION ===
COCO_IMG_DIR = "coco/train2017"
COCO_ANN_FILE = "coco/annotations/instances_train2017.json"

USE_SUBSET = False   # ✅ Set to True if you want quick testing on smaller set
NUM_IMAGES = 4000    # Used only if USE_SUBSET is True

EPOCHS = 20          # 🔁 Increased for full dataset
BATCH_SIZE = 32
SAVE_EVERY = 2       # ✅ Save checkpoints every N epochs
SAVE_DIR = "dataset_distillation/experts"

os.makedirs(SAVE_DIR, exist_ok=True)

# === IMAGE TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize COCO images to smaller size
    transforms.ToTensor()
])

# === LOAD COCO DATASET ===
full_dataset = CocoDetection(root=COCO_IMG_DIR, annFile=COCO_ANN_FILE, transform=transform)

# === MAP COCO CATEGORY IDS TO INDEX 0–79 ===
category_id_to_index = {
    cat_id: idx for idx, (cat_id, cat) in enumerate(full_dataset.coco.cats.items())
}

# === DATA SUBSET OR FULL ===
if USE_SUBSET:
    indices = random.sample(range(len(full_dataset)), NUM_IMAGES)
    dataset = Subset(full_dataset, indices)
else:
    dataset = full_dataset

# === DATALOADER WITH CUSTOM COLLATE ===
def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# === MODEL, LOSS, OPTIMIZER ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=80).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# === TRAINING LOOP ===
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch}"):
        inputs = torch.stack(inputs).to(device)

        # Convert COCO category_id to index (0–79)
        labels = torch.tensor([
            category_id_to_index[t[0]['category_id']]
            for t in targets if len(t) > 0 and t[0]['category_id'] in category_id_to_index
        ]).to(device)

        if len(labels) != len(inputs):
            continue  # skip invalid batches

        # Forward + backward + update
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[Epoch {epoch}] Loss: {running_loss / len(loader):.4f}")

    # === Save checkpoint every N epochs or final epoch ===
    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
        checkpoint_path = os.path.join(SAVE_DIR, f"expert_epoch{epoch}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved checkpoint: {checkpoint_path}")
