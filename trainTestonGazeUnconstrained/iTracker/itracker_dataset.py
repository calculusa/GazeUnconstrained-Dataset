import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np


class ITrackerDataset(Dataset):
    """
    数据加载器：
    - 自动遍历 processed_dataset/pXX/
    - 将 2D gaze 像素坐标归一化到 [0,1]
    - 返回 face、left_eye、right_eye、face_grid、gaze
    """

    def __init__(self, root_dir, include_pids=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        self.samples = []

        all_pids = sorted(os.listdir(root_dir))
        pids = include_pids if include_pids else all_pids

        for pid in pids:
            participant_path = os.path.join(root_dir, pid)
            ann_path = os.path.join(participant_path, "annotations.json")
            if not os.path.exists(ann_path):
                continue

            with open(ann_path, "r") as f:
                annotations = json.load(f)

            for frame_name, ann in annotations.items():
                face_p = os.path.join(participant_path, "images/face", frame_name)
                left_p = os.path.join(participant_path, "images/left_eye", frame_name)
                right_p = os.path.join(participant_path, "images/right_eye", frame_name)
                if not (os.path.exists(face_p) and os.path.exists(left_p) and os.path.exists(right_p)):
                    continue

                sample = {
                    "face_path": face_p,
                    "left_eye_path": left_p,
                    "right_eye_path": right_p,
                    "face_grid": np.array(ann["face_grid"]).reshape(25, 25).astype(np.float32),
                    # 归一化到 [0,1]
                    "gaze": np.array(ann["screen_gaze"], dtype=np.float32) / np.array([1920.0, 1080.0], dtype=np.float32),
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        face = self.transform(Image.open(s["face_path"]).convert("RGB"))
        left = self.transform(Image.open(s["left_eye_path"]).convert("RGB"))
        right = self.transform(Image.open(s["right_eye_path"]).convert("RGB"))

        face_grid = torch.tensor(s["face_grid"]).unsqueeze(0)  # [1,25,25]
        gaze = torch.tensor(s["gaze"])  # [2] in [0,1]

        return {
            "image_face": face,
            "image_left": left,
            "image_right": right,
            "face_grid": face_grid,
            "gaze": gaze,
        }
