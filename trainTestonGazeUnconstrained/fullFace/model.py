import torch
import torch.nn as nn
import torchvision.models as tv

class model(nn.Module):
    def __init__(self):
        super().__init__()
        alex = tv.alexnet(weights="DEFAULT")      # torchvision≥0.13
        self.backbone = alex.features

        self.attn = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.ReLU(True),
            nn.Conv2d(256, 256, 1), nn.ReLU(True),
            nn.Conv2d(256,   1, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 13 * 13, 4096), nn.ReLU(True),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Linear(4096, 2)   # yaw, pitch (rad)
        )

    def forward(self, x):
        feat = self.backbone(x["face"])
        feat = feat * self.attn(feat)
        feat = torch.flatten(feat, 1)
        return self.head(feat)
