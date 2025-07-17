# affnet_dataset.py
# ----------------------------------------------------------
# 兼容 AFF-Net 输入：
#   • face 224×224  RGB
#   • left/right 112×112  RGB
#   • rects 12-dim bbox  (float32, 0-1)
#   • gaze  2-dim 归一化屏幕坐标 (float32, 0-1)
# ----------------------------------------------------------

import os, json, numpy as np, torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class AFFNetDataset(Dataset):
    def __init__(self, root_dir, include_pids=None):
        super().__init__()
        self.root = root_dir
        self.face_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.eye_tf  = T.Compose([T.Resize((112, 112)), T.ToTensor()])

        self.samples = []
        pids = include_pids or sorted(os.listdir(root_dir))
        for pid in pids:
            ann_f = os.path.join(root_dir, pid, "annotations.json")
            if not os.path.isfile(ann_f): continue
            with open(ann_f, "r") as f:
                ann = json.load(f)

            for frame, info in ann.items():
                base = os.path.join(root_dir, pid, "images")
                face_p  = os.path.join(base, "face",      frame)
                left_p  = os.path.join(base, "left_eye",  frame)
                right_p = os.path.join(base, "right_eye", frame)
                if not (os.path.exists(face_p) and os.path.exists(left_p) and os.path.exists(right_p)):
                    continue

                fb, lb, rb = info["face_bbox"], info["left_eye_bbox"], info["right_eye_bbox"]
                rects = np.array(fb + lb + rb, dtype=np.float32)
                norm  = np.array([1080,1080,1920,1920]*3, dtype=np.float32)   # (y1,y2,x1,x2)×3
                rects = rects / norm                                           # → [0,1]

                self.samples.append({
                    "face": face_p,
                    "left": left_p,
                    "right": right_p,
                    "rects": rects,                                            # float32
                    "gaze":  np.array(info["screen_gaze"], dtype=np.float32) / np.array([1920.0, 1080.0], dtype=np.float32)
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        face  = self.face_tf(Image.open(s["face"]).convert("RGB")).float()     # tensor float32
        left  = self.eye_tf (Image.open(s["left"]).convert("RGB")).float()
        right = self.eye_tf (Image.open(s["right"]).convert("RGB")).float()

        rects = torch.tensor(s["rects"], dtype=torch.float32)   # (12,) float32
        gaze  = torch.tensor(s["gaze"],  dtype=torch.float32)   # (2,)  float32

        return {"face": face, "left": left, "right": right, "rects": rects, "gaze": gaze}
