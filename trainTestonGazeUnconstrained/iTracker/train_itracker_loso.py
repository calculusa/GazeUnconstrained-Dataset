#!/usr/bin/env python
# train_itracker.py
# ----------------------------------------------------------
#  使用示例:
#    python train_itracker.py --config config.yaml --val_pid p00
# ----------------------------------------------------------

import os, sys, argparse, yaml, logging, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_itracker import ITrackerModel          # 你的模型
from itracker_dataset import ITrackerDataset  # 你的数据集

# ---------- 日志工具 ---------- #
def setup_logger(save_dir):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%m-%d %H:%M:%S"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(save_dir, "train.log"))
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt, handlers=handlers)

# ---------- 欧几里得距离 ---------- #
def euclid(pred, tgt):  # (N,2) → float
    return torch.norm(pred - tgt, dim=1).mean().item()

# ---------- 训练 & 验证 ---------- #
def train_epoch(model, loader, crit, optim_, device):
    model.train()
    loss_sum = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        face  = batch["image_face"].to(device)
        left  = batch["image_left"].to(device)
        right = batch["image_right"].to(device)
        grid  = batch["face_grid"].to(device)
        gaze  = batch["gaze"].to(device)        # 归一化 [0,1]

        optim_.zero_grad()
        out   = model({"face": face, "left": left, "right": right, "grid": grid})
        loss  = crit(out, gaze)
        loss.backward()
        optim_.step()
        loss_sum += loss.item() * face.size(0)

    return loss_sum / len(loader.dataset)

def validate(model, loader, crit, device, w_cm, h_cm):
    model.eval()
    loss_sum = dist_sum = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val  ", leave=False):
            face  = batch["image_face"].to(device)
            left  = batch["image_left"].to(device)
            right = batch["image_right"].to(device)
            grid  = batch["face_grid"].to(device)
            gaze  = batch["gaze"].to(device)

            out   = model({"face": face, "left": left, "right": right, "grid": grid})
            loss  = crit(out, gaze)

            # 像素 → 厘米
            out_cm  = out  * torch.tensor([w_cm, h_cm], device=device)
            gaze_cm = gaze * torch.tensor([w_cm, h_cm], device=device)
            dist_cm = euclid(out_cm, gaze_cm)

            loss_sum += loss.item() * face.size(0)
            dist_sum += dist_cm     * face.size(0)

    N = len(loader.dataset)
    return loss_sum / N, dist_sum / N        # 归一化 MSE, cm

# ---------- 主函数 ---------- #
def main(cfg_path, val_pid):
    # 1. 读取配置 ------------------------------------------------
    cfg = yaml.safe_load(open(cfg_path))
    root     = cfg["data"]["processed_dataset_path"]
    bs       = cfg["train"]["batch_size"]
    epochs   = cfg["train"]["epoch"]
    save_ev  = cfg["train"]["save_every"]
    save_dir = cfg["train"]["save_path"]
    Wcm, Hcm = cfg["screen"]["width_cm"], cfg["screen"]["height_cm"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir)
    logging.info(f"Device: {device}")

    # 2. 划分 LOSO ----------------------------------------------
    all_pids = sorted([p for p in os.listdir(root) if p.startswith("p")])
    if val_pid not in all_pids:
        raise ValueError(f"{val_pid} not found. Available: {all_pids}")
    train_p = [p for p in all_pids if p != val_pid]
    logging.info(f"Train PIDs: {train_p} | Val PID: {val_pid}")

    train_ds = ITrackerDataset(root, include_pids=train_p)
    val_ds   = ITrackerDataset(root, include_pids=[val_pid])
    train_ld = DataLoader(train_ds, bs, shuffle=True,  num_workers=4)
    val_ld   = DataLoader(val_ds,   bs, shuffle=False, num_workers=4)

    # 3. 模型/优化器 --------------------------------------------
    model = ITrackerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    pid_dir = os.path.join(save_dir, val_pid); os.makedirs(pid_dir, exist_ok=True)

    # 4. 早停参数初始化 ------------------------------------------
    best_dist = float('inf')
    patience  = 5     # 若连续 5 epoch 无提升则停止
    stale     = 0

    # 5. 训练循环 ----------------------------------------------
    for ep in range(1, epochs + 1):
        tloss = train_epoch(model, train_ld, criterion, optimizer, device)
        vloss, vdist = validate(model, val_ld, criterion, device, Wcm, Hcm)

        logging.info(f"Epoch {ep:02d}/{epochs} | "
                     f"TrainLoss {tloss:.4f} | ValLoss {vloss:.4f} | ValDist {vdist:.2f} cm")

        # 定期 checkpoint
        if ep % save_ev == 0:
            torch.save(model.state_dict(), os.path.join(pid_dir, f"Iter_{ep}_model.pt"))

        # ---------- 早停逻辑 ----------
        if vdist < best_dist:             # 新纪录
            best_dist = vdist
            stale = 0
            torch.save(model.state_dict(), os.path.join(pid_dir, "best_model.pt"))
            logging.info(f"  ↳ New best!  ValDist={vdist:.2f} cm  (model saved)")
        else:
            stale += 1
            logging.info(f"  ↳ No improvement for {stale} epoch(s)")

        if stale >= patience:
            logging.info(f"Early stop at epoch {ep} (best ValDist {best_dist:.2f} cm)")
            break

    logging.info("Training finished.")

# ---------- CLI ---------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True, help="Path to config.yaml")
    ap.add_argument("--val_pid", required=True, help="Participant ID, e.g. p00")
    args = ap.parse_args()
    main(args.config, args.val_pid)
