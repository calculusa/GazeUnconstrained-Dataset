import os, json, math, torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

# ───────── transforms ─────────
_tf = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ───────── Dataset ────────────
class GazeDataset(Dataset):
    """
    读取你的自制 annotations.json，并把
    • yaw, pitch 从度 → 弧度
    • 同时全部乘 -1 使其符号与 MPIIFaceGaze 对齐
      （MPII: yaw>0 表示向左看，pitch>0 表示向下看）
    """
    def __init__(self, label_paths, img_root):
        self.recs = []
        paths = label_paths if isinstance(label_paths, list) else [label_paths]

        for p in paths:
            ann_path = os.path.join(p, "annotations.json")
            img_dir  = os.path.join(p, "images", "face")
            if not os.path.isfile(ann_path):
                continue

            with open(ann_path) as f:
                ann = json.load(f)

            for fname, meta in ann.items():
                ga = meta.get("gaze_angles")          # [yaw_deg, pitch_deg]
                if ga is None:
                    continue
                img_path = os.path.join(img_dir, fname)
                if not os.path.isfile(img_path):
                    continue

                yaw   = -math.radians(ga[0])          # ← 修正：乘 -1
                pitch = -math.radians(ga[1])          # ← 修正：乘 -1
                self.recs.append((img_path,
                                  torch.tensor([yaw, pitch], dtype=torch.float32)))

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, idx):
        img_path, gaze = self.recs[idx]
        img = _tf(Image.open(img_path).convert("RGB"))
        name = os.path.basename(img_path)
        return {"face": img, "name": name}, gaze


# ───────── Dataloader 工具函数 ─────────
def txtload(label_paths, img_root, bs,
            shuffle=True, num_workers=2, **_):
    """
    与原 train/test 脚本兼容的包装器
    """
    return DataLoader(GazeDataset(label_paths, img_root),
                      batch_size=bs,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=True)
