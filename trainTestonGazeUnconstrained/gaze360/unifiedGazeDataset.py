# unified_gaze_dataset.py  —— 仅展示修改后的 __init__ & 辅助函数
import os, json, glob, math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def angles_to_vector(yaw, pitch):
    gx = math.sin(yaw) * math.cos(pitch)
    gy = -math.sin(pitch)
    gz = -math.cos(yaw) * math.cos(pitch)
    return [gx, gy, gz]

class UnifiedGazeDataset(Dataset):
    def __init__(self, root, mode='3d', eye_mode='face',
                 transform=None, include_pids=None, exclude_pids=None):
        self.root, self.mode, self.eye_mode = root, mode, eye_mode
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.samples = []

        # -------- 1. 找所有 annotations.json --------
        anno_files = glob.glob(os.path.join(root, 'p*', 'annotations.json'))
        if not anno_files and os.path.isfile(os.path.join(root, 'annotations.json')):
            anno_files = [os.path.join(root, 'annotations.json')]
        if not anno_files:
            raise FileNotFoundError(f'No annotations.json under {root}')

        # -------- 2. 读取每个文件 --------
        for af in sorted(anno_files):
            pid = os.path.basename(os.path.dirname(af)) if 'p' in af else ''
            if include_pids and pid not in include_pids: continue
            if exclude_pids and pid in exclude_pids:    continue

            with open(af, 'r') as f:
                ann = json.load(f)

            # 2-A 结构: {"frame0000.png": {...}}  (最常见)
            if all(isinstance(v, dict) for v in ann.values()):
                for fname, meta in ann.items():
                    self._add_sample(pid, fname, meta)
            # 2-B 结构: {"p00":[{frame_name:...}, ...]}  (集中式)
            elif all(isinstance(v, list) for v in ann.values()):
                for spid, recs in ann.items():
                    if include_pids and spid not in include_pids: continue
                    if exclude_pids and spid in exclude_pids:    continue
                    for meta in recs:
                        fname = meta['frame_name']
                        self._add_sample(spid, fname, meta)
            # 2-C 结构: [ {...}, {...} ]  (列表)
            elif isinstance(ann, list):
                for meta in ann:
                    fname = meta['frame_name']
                    self._add_sample(pid, fname, meta)
            else:
                raise ValueError(f'Unrecognized annotation format in {af}')

    # -------- 3. 生成单条样本 --------
    def _add_sample(self, pid, fname, meta):
        # ---- 标签选择 ----
        gvec = meta.get('gaze_vector')
        if gvec is None or any(v is None for v in gvec):
            gang = meta.get('gaze_angles')      # [yaw, pitch]
            if gang and all(a is not None for a in gang):
                gvec = angles_to_vector(gang[0], gang[1])
            else:
                return  # 没有 3D 信息，跳过
        if self.mode == '2d' and meta.get('screen_gaze') is None:
            return  # 2D 任务却缺少 screen_gaze，跳过

        self.samples.append({
            'pid'  : pid,
            'face' : os.path.join(self.root, pid, 'images', 'face',      fname),
            'left' : os.path.join(self.root, pid, 'images', 'left_eye',  fname),
            'right': os.path.join(self.root, pid, 'images', 'right_eye', fname),
            'gaze_vector': gvec,
            'screen_gaze': meta.get('screen_gaze')
        })

    # -------- 4. 其余函数保持不变 --------
    def __len__(self): return len(self.samples)

    def _img(self, p): return self.transform(Image.open(p).convert('RGB'))

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.eye_mode == 'face':
            x = self._img(s['face'])
        elif self.eye_mode == 'both':
            x = torch.cat([self._img(s['left']), self._img(s['right'])], dim=0)
        else:
            raise ValueError('eye_mode must be "face" or "both"')

        if self.mode == '3d':
            y = torch.tensor(s['gaze_vector'], dtype=torch.float32)
        else:
            y = torch.tensor(s['screen_gaze'], dtype=torch.float32)
        return x, y
