# gazenet_dataset.py  (final robust version)  ───────────────────────
import os, json, glob, torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class GazeNetDataset(Dataset):
    """
    读取:
      • eye 图像 (左眼)      : 3×36×60 RGB
      • head_pose            : [yaw, pitch]  (2 维弧度)
      • label (优先顺序)     : gaze_angles[yaw,pitch]  若缺则 gaze_vector[3]
    注释布局自动适配:
      1) root/annotations.json
      2) root/pXX/annotations.json
    支持 LOSO: include_pids / exclude_pids
    """
    def __init__(self, root, include_pids=None, exclude_pids=None):
        self.root = root
        self.Teye = transforms.Compose([
            transforms.Resize((36, 60)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

        # -------- 找到所有注释文件 --------
        anno_files = []
        root_json = os.path.join(root, 'annotations.json')
        if os.path.isfile(root_json):
            anno_files.append(root_json)
        else:
            anno_files.extend(glob.glob(os.path.join(root, 'p*', 'annotations.json')))
        if not anno_files:
            raise FileNotFoundError(f'No annotations.json under {root}')

        self.samples = []
        for af in sorted(anno_files):
            pid_base = os.path.basename(os.path.dirname(af)) if 'p' in af else ''

            with open(af, 'r') as f:
                ann = json.load(f)

            # ① {"frame0000.png": {...}}
            if isinstance(next(iter(ann.values())), dict):
                pid = pid_base
                if not self._pid_ok(pid, include_pids, exclude_pids): continue
                for fname, meta in ann.items():
                    self._add_sample(pid, fname, meta)
            # ② {"p00":[ {...}, ... ]}
            elif isinstance(next(iter(ann.values())), list):
                for pid, recs in ann.items():
                    if not self._pid_ok(pid, include_pids, exclude_pids): continue
                    for meta in recs:
                        self._add_sample(pid, meta['frame_name'], meta)
            # ③ [ {...}, {...} ]
            elif isinstance(ann, list):
                pid = pid_base
                if not self._pid_ok(pid, include_pids, exclude_pids): continue
                for meta in ann:
                    self._add_sample(pid, meta['frame_name'], meta)
            else:
                raise ValueError(f'Unknown annotation format in {af}')

    # -------- PID 过滤 --------
    @staticmethod
    def _pid_ok(pid, inc, exc):
        if inc and pid not in inc:  return False
        if exc and pid in exc:      return False
        return True

    # -------- 加入单条样本 --------
    def _add_sample(self, pid, fname, meta):
        # (1) 检查有效 3-D 标签
        angles = meta.get('gaze_angles')
        vec    = meta.get('gaze_vector')
        if (angles is None or any(a is None for a in angles)) and \
           (vec is None or any(v is None for v in vec)):
            return  # 无 3-D 标签，跳过

        # (2) 检查眼图存在
        eye_path = os.path.join(self.root, pid, 'images', 'left_eye', fname)
        if not os.path.isfile(eye_path): return

        self.samples.append({
            'eye' : eye_path,
            'head_pose'  : torch.tensor(meta['head_pose'][:2], dtype=torch.float32),
            'gaze_angles': angles,
            'gaze_vector': vec
        })

    # -------- Dataset API --------
    def __len__(self): return len(self.samples)

    def _load_eye(self, p):
        return self.Teye(Image.open(p).convert('RGB'))

    def __getitem__(self, idx):
        s = self.samples[idx]
        eye = self._load_eye(s['eye'])
        hp  = s['head_pose']
        if s['gaze_angles'] and all(a is not None for a in s['gaze_angles']):
            label = torch.tensor(s['gaze_angles'], dtype=torch.float32)   # [yaw,pitch]
        else:
            label = torch.tensor(s['gaze_vector'], dtype=torch.float32)   # [gx,gy,gz]
        return {'eye': eye, 'head_pose': hp}, label
