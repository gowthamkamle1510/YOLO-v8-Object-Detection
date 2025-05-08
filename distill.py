import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import SimpleCNN

# === CONFIGURATION ===
NUM_CLASSES = 80
SYN_IMAGES_PER_CLASS = 30       # 🔁 You can adjust to 50, 100, etc.
IMAGE_SIZE = (3, 64, 64)
TOTAL_STEPS = 100                # 🔁 You can increase to 300 or more
LEARNING_RATE = 0.01
EXPERT_CKPT_DIR = "dataset_distillation/experts"
SAVE_PATH = "synthetic_step100.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Create Synthetic Data ===
synthetic_data = {
    cls: torch.randn((SYN_IMAGES_PER_CLASS, *IMAGE_SIZE), requires_grad=True, device=device)
    for cls in range(NUM_CLASSES)
}

# === Load Expert Models ===
expert_checkpoints = sorted([
    os.path.join(EXPERT_CKPT_DIR, ckpt) for ckpt in os.listdir(EXPERT_CKPT_DIR)
    if ckpt.endswith(".pt")
])

# === Optimization Setup ===
optimizer = optim.SGD([p for imgs in synthetic_data.values() for p in [imgs]], lr=LEARNING_RATE)

# === Training Loop (Match Expert Trajectories) ===
for step in range(1, TOTAL_STEPS + 1):
    total_loss = 0.0
    optimizer.zero_grad()

    for class_id, syn_imgs in synthetic_data.items():
        # Load corresponding expert model
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        expert_idx = step % len(expert_checkpoints)
        model.load_state_dict(torch.load(expert_checkpoints[expert_idx], map_location=device))
        model.eval()

        # Synthetic labels: all same class
        labels = torch.full((SYN_IMAGES_PER_CLASS,), class_id, dtype=torch.long, device=device)

        # Forward pass
        outputs = model(syn_imgs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        # Accumulate loss
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    print(f"Step [{step}/{TOTAL_STEPS}] - Total Loss: {total_loss.item():.4f}")

# === Save Synthetic Data ===
torch.save(synthetic_data, SAVE_PATH)
print(f"\n✅ Saved synthetic dataset to {SAVE_PATH}")
