import torch
import torch.nn as nn
import torch.nn.functional as F
from load_teacher import load_teacher_model
from load_student import load_student_model
from kd_loss import FeatureDistillationLoss
from hooks import feature_maps, register_hook
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader

# =========================
# Hyperparameters
# =========================
EPOCHS = 10
LAMBDA_KD = 0.3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =========================
# Load Teacher and Student
# =========================
teacher = load_teacher_model().to(DEVICE)
student = load_student_model("/Users/mahidharreddypatukuri/yolov8_image_detector/runs/detect/train/weights/best.pt")

# =========================
# Register Hooks for Feature Extraction
# =========================
register_hook(teacher.backbone.body, 'layer3', 'teacher_feat')
register_hook(student.model.model, '4', 'student_feat')  # Adjust if needed

# =========================
# Define Optimizer, KD Loss, and Projection Layer
# =========================
optimizer = torch.optim.Adam(student.model.parameters(), lr=1e-4)
kd_loss_fn = FeatureDistillationLoss()

# Projection layer to align student feature channels to teacher's
proj_layer = nn.Conv2d(64, 1024, kernel_size=1).to(DEVICE)

# =========================
# DataLoader Setup
# =========================
def collate_fn(batch):
    return tuple(zip(*batch))

transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

coco_dataset = CocoDetection(
    root='/Users/mahidharreddypatukuri/yolov8_image_detector/coco/train2017',
    annFile='/Users/mahidharreddypatukuri/yolov8_image_detector/coco/annotations/instances_train2017.json',
    transform=transform
)

dataloader = DataLoader(
    coco_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# Training Loop
# =========================
for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, targets in dataloader:
        imgs = torch.stack(list(imgs)).to(DEVICE)

        # Forward pass
        teacher(imgs)
        student_preds = student.model(imgs)

        # Extract features
        t_feat = feature_maps['teacher_feat']
        s_feat = feature_maps['student_feat']

        # Project student features to match teacher channels
        s_feat_projected = proj_layer(s_feat)

        # Align spatial dimensions if needed
        if t_feat.shape[2:] != s_feat_projected.shape[2:]:
            t_feat = F.adaptive_avg_pool2d(t_feat, s_feat_projected.shape[2:])

        # Compute KD Loss
        distill_loss = kd_loss_fn(s_feat_projected, t_feat)

        # Placeholder for YOLOv8 detection loss (to implement properly later)
        detection_loss = torch.tensor(0.0).to(DEVICE)

        # Total loss
        loss = detection_loss + LAMBDA_KD * distill_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Total Loss: {total_loss:.4f}")

# =========================
# Save KD-trained Student Model
# =========================
student.model.save("/Users/mahidharreddypatukuri/yolov8_image_detector/runs/kd_training/kd_best.pt")