# affnet_model.py  (维度完全匹配版)
# ----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Squeeze-and-Excitation ----------
class SELayer(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.avg(x))


# ---------- Adaptive Group Normalization ----------
class AGN(nn.Module):
    def __init__(self, channels: int, factor_dim: int, num_groups: int = 8):
        super().__init__()
        self.gn  = nn.GroupNorm(num_groups, channels, affine=False)
        self.map = nn.Linear(factor_dim, channels)     # factor → γ
    def forward(self, x, factor):
        gamma = self.map(factor).unsqueeze(-1).unsqueeze(-1)  # (N,C,1,1)
        return self.gn(x) * gamma


# ---------- 辅助 Conv Block ----------
def conv_bn(cin, cout, k, s=1, p=0):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, p, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


# ---------- Eye Encoder ----------
class EyeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3,   32, 3, 2, 1),   # 56×56
            conv_bn(32,  64, 3, 2, 1),   # 28×28
            conv_bn(64, 128, 3, 2, 1),   # 14×14
            SELayer(128, 16))
    def forward(self, eye, _):
        return self.net(eye)             # (N,128,14,14)


# ---------- Face Encoder ----------
class FaceImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(3,   32, 3, 2, 1),   # 112×112
            conv_bn(32,  64, 3, 2, 1),   # 56×56
            conv_bn(64, 128, 3, 2, 1),   # 28×28
            conv_bn(128,256, 3, 2, 1),   # 14×14
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True))
    def forward(self, face):
        return self.net(face)            # (N,128)


# ---------- AFF-Net ----------
class AFFNet(nn.Module):
    """
    输入:
      eyeL/eyeR : (N,3,112,112)
      face      : (N,3,224,224)
      rects     : (N,12)   (0~1)
    输出:
      gaze      : (N,2)    (0~1)
    """
    def __init__(self):
        super().__init__()
        self.eyeModel  = EyeImageModel()
        self.faceModel = FaceImageModel()

        # Merge 双眼
        self.merge1 = nn.Sequential(
            SELayer(256, 16),
            nn.Conv2d(256, 64, 3, 2, 1))      # 14×14 → 7×7
        self.mergeAGN = AGN(channels=64, factor_dim=192, num_groups=8)
        self.merge2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(64,16))

        # FC 分支
        self.eyesFC = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),        # 3136 → 128
            nn.LeakyReLU(inplace=True))

        self.rectsFC = nn.Sequential(
            nn.Linear(12, 64),  nn.LeakyReLU(),
            nn.Linear(64, 96),  nn.LeakyReLU(),
            nn.Linear(96,128),  nn.LeakyReLU(),
            nn.Linear(128,64),  nn.LeakyReLU())

        self.fc = nn.Sequential(
            nn.Linear(320, 128),               # 128 (eyes) +128 (face)+64 (rects)
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
            nn.Sigmoid())                      # gaze 0~1

    def forward(self, eyeL, eyeR, face, rects):
        # 人脸 & rect
        xFace = self.faceModel(face)           # (N,128)
        xRect = self.rectsFC(rects)            # (N,64)
        factor = torch.cat([xFace, xRect], 1)  # (N,192)

        # 双眼
        eL = self.eyeModel(eyeL, factor)
        eR = self.eyeModel(eyeR, factor)
        e  = torch.cat([eL, eR], 1)            # (N,256,14,14)
        e  = self.merge1(e)                    # (N,64,7,7)
        e  = self.mergeAGN(e, factor)
        e  = self.merge2(e)
        e  = e.view(e.size(0), -1)             # (N,3136)
        e  = self.eyesFC(e)                    # (N,128)

        feat = torch.cat([e, xFace, xRect], 1) # (N,320)
        gaze = self.fc(feat)                   # (N,2)
        return gaze
