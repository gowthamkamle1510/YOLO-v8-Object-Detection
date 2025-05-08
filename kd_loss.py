import torch.nn as nn

class FeatureDistillationLoss(nn.Module):
    def __init__(self):
        super(FeatureDistillationLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_features, teacher_features):
        return self.mse(student_features, teacher_features.detach())
