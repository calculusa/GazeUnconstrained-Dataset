# gaze360_lstm_dataset.py  (robust version)
import os, re, bisect
from unifiedGazeDataset import UnifiedGazeDataset
from PIL import Image
import torch

class Gaze360SequenceDataset(UnifiedGazeDataset):
    """
    返回 7 帧序列:
        seq = [t-3, t-2, …, t, …, t+3]
    若某帧缺失，则取最近存在的帧（向内收缩）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- 预扫描每个被试目录的实际帧 index 列表 ---
        self.frame_pat = re.compile(r'frame(\d+)\.png$')
        self.subject_frames = {}              # pid -> sorted idx list
        for s in self.samples:
            pid = s['pid']
            if pid not in self.subject_frames:
                face_dir = os.path.join(self.root, pid, 'images', 'face')
                idxs = [int(self.frame_pat.search(f).group(1))
                        for f in os.listdir(face_dir)
                        if self.frame_pat.match(f)]
                self.subject_frames[pid] = sorted(idxs)

            # 将当前样本对应帧 idx 记录
            s['idx'] = int(self.frame_pat.search(s['face']).group(1))

    # ---------- 加载一张图 ----------
    def _load(self, pid, fid):
        fname = f'frame{fid:04d}.png'
        path  = os.path.join(self.root, pid, 'images', 'face', fname)
        return self.transform(Image.open(path).convert('RGB'))

    # ---------- 取最近存在帧 ----------
    def _nearest_idx(self, idx_list, target):
        pos = bisect.bisect_left(idx_list, target)
        if pos == 0:
            return idx_list[0]
        if pos == len(idx_list):
            return idx_list[-1]
        before = idx_list[pos - 1]
        after  = idx_list[pos]
        return before if abs(target - before) <= abs(after - target) else after

    # ---------- Dataset 核心 ----------
    def __getitem__(self, idx):
        s   = self.samples[idx]
        pid = s['pid']
        cur = s['idx']
        idx_list = self.subject_frames[pid]

        seq_imgs = []
        for off in range(-3, 4):              # t-3 … t+3
            want = cur + off
            real = self._nearest_idx(idx_list, want)
            seq_imgs.append(self._load(pid, real))

        seq = torch.stack(seq_imgs, dim=0)    # [7,3,H,W]

        label = torch.tensor(s['gaze_vector'] if self.mode == '3d'
                             else s['screen_gaze'], dtype=torch.float32)
        return seq, label
