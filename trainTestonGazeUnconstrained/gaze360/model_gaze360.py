# gaze360_model.py ---------------------------------------------------
import torch, math, torch.nn as nn
from model_origin import GazeStatic, GazeLSTM      # 原始文件已有
# --------------------------------------------------
def angles_to_vector(yaw, pitch):
    gx = torch.sin(yaw) * torch.cos(pitch)
    gy = -torch.sin(pitch)
    gz = -torch.cos(yaw) * torch.cos(pitch)
    return torch.stack([gx, gy, gz], dim=1)

class Gaze360(nn.Module):
    """
    backbone: 'static' (单帧) | 'lstm' (7 帧序列)
    forward(..., return_angles=False):
      • static 模式  : face_img → (3D 向量) or (angles,var)
      • lstm  模式   : seq_img  → (3D 向量) or (angles,var)
        - seq_img shape: [B, 7, 3, H, W]
    """
    def __init__(self, backbone='static'):
        super().__init__()
        if backbone == 'static':
            self.core = GazeStatic()
            self.is_seq = False
        elif backbone == 'lstm':
            self.core = GazeLSTM()
            self.is_seq = True
        else:
            raise ValueError('backbone must be static | lstm')

    def forward(self, x, return_angles=False):
        # 对于 LSTM，x 已是 [B,7,3,H,W]；Static 为 [B,3,H,W]
        angles, var = self.core({"face": x} if not self.is_seq else x)  # (B,2),(B,2)
        if return_angles:
            return angles, var
        else:
            yaw, pitch = angles[:, 0], angles[:, 1]
            return angles_to_vector(yaw, pitch)       # (B,3)
