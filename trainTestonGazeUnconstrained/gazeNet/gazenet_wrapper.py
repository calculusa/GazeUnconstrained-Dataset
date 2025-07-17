import torch, math
from model_gazenet import model as _RawGazeNet        # 你的 VGG16 网络

def angles_to_vector(yaw, pitch):
    gx = torch.sin(yaw) * torch.cos(pitch)
    gy = -torch.sin(pitch)
    gz = -torch.cos(yaw) * torch.cos(pitch)
    return torch.stack([gx,gy,gz], dim=1)

class GazeNetWrapper(_RawGazeNet):
    """
    forward(feat_dict, return_vector=False):
      False → yaw,pitch  (B,2)
      True  → 3D 单位向量 (B,3)
    """
    def forward(self, inputs, return_vector=False):
        ang = super().forward(inputs)           # (B,2)
        if return_vector:
            return angles_to_vector(ang[:,0], ang[:,1])
        return ang
