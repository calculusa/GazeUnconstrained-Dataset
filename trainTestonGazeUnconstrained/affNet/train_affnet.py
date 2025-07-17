#!/usr/bin/env python
# train_affnet.py  （支持 --gpu 选卡 + 早停 + 日志）
# -----------------------------------------------------
#  用法示例：
#    python train_affnet.py --config config_affnet.yaml --val_pid p00 --gpu 1
# -----------------------------------------------------

import os, sys, argparse, yaml, logging
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from affnet_dataset import AFFNetDataset
from model_affnet   import AFFNet


# ---------- 实用函数 ---------- #
def euclid(a, b):
    """均值欧氏距离 (N,2) → float"""
    return torch.norm(a - b, dim=1).mean().item()

def setup_logger(save_dir):
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        datefmt="%m-%d %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(save_dir, "train.log"))])

# ---------- 单 epoch ---------- #
def train_epoch(net, loader, crit, opt, dev):
    net.train(); sm = 0
    for b in tqdm(loader, desc="Train", leave=False):
        f, l, r, rect, g = (b["face"].to(dev), b["left"].to(dev),
                            b["right"].to(dev), b["rects"].to(dev),
                            b["gaze"].to(dev))
        opt.zero_grad()
        out = net(l, r, f, rect)
        loss = crit(out, g)
        loss.backward(); opt.step()
        sm += loss.item() * f.size(0)
    return sm / len(loader.dataset)

def validate(net, loader, crit, dev, Wcm, Hcm):
    net.eval(); ls = ds = 0
    with torch.no_grad():
        for b in tqdm(loader, desc="Val  ", leave=False):
            f, l, r, rect, g = (b["face"].to(dev), b["left"].to(dev),
                                b["right"].to(dev), b["rects"].to(dev),
                                b["gaze"].to(dev))
            out  = net(l, r, f, rect)
            loss = crit(out, g)
            out_cm  = out * torch.tensor([Wcm, Hcm], device=dev)
            g_cm    = g   * torch.tensor([Wcm, Hcm], device=dev)
            dist_cm = euclid(out_cm, g_cm)
            ls += loss.item() * f.size(0)
            ds += dist_cm     * f.size(0)
    N = len(loader.dataset)
    return ls / N, ds / N


# ---------- 主入口 ---------- #
def main(cfg_path, val_pid, gpu_idx):
    # ---- 读配置 ----
    cfg = yaml.safe_load(open(cfg_path))
    root     = cfg["data"]["processed_dataset_path"]
    bs       = cfg["train"]["batch_size"]
    epochs   = cfg["train"]["epoch"]
    save_ev  = cfg["train"]["save_every"]
    save_dir = cfg["train"]["save_path"]
    Wcm, Hcm = cfg["screen"]["width_cm"], cfg["screen"]["height_cm"]

    # ---- 指定 GPU ----
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir)
    logging.info(f"Use device: {device}")

    # ---- 数据划分 ----
    pids = sorted([p for p in os.listdir(root) if p.startswith("p")])
    if val_pid not in pids:
        raise ValueError(f"{val_pid} not in dataset {pids}")
    train_p = [p for p in pids if p != val_pid]
    logging.info(f"Train {train_p} | Val {val_pid}")

    tr_ds = AFFNetDataset(root, train_p);   tr_ld = DataLoader(tr_ds, bs, shuffle=True,  num_workers=4)
    va_ds = AFFNetDataset(root, [val_pid]); va_ld = DataLoader(va_ds, bs, shuffle=False, num_workers=4)

    # ---- 网络 / 优化 ----
    net = AFFNet().to(device)
    crit = nn.MSELoss()
    opt  = optim.Adam(net.parameters(), lr=1e-4)

    pid_dir = os.path.join(save_dir, val_pid); os.makedirs(pid_dir, exist_ok=True)

    # ---- 早停 ----
    best = float("inf"); patience = 5; stale = 0

    for ep in range(1, epochs + 1):
        tl = train_epoch(net, tr_ld, crit, opt, device)
        vl, vd = validate(net, va_ld, crit, device, Wcm, Hcm)
        logging.info(f"Ep {ep:02d}/{epochs} | Train {tl:.4f} | Val {vl:.4f} | {vd:.2f} cm")

        # 周期性 checkpoint
        if ep % save_ev == 0:
            torch.save(net.state_dict(), f"{pid_dir}/Iter_{ep}.pt")

        # 早停监控
        if vd < best:
            best = vd; stale = 0
            torch.save(net.state_dict(), f"{pid_dir}/best_model.pt")
            logging.info(f"  ↳ New best! {best:.2f} cm")
        else:
            stale += 1
            logging.info(f"  ↳ No improvement ({stale}/{patience})")

        if stale >= patience:
            logging.info(f"Early stop @ {ep}. Best {best:.2f} cm")
            break


# ---------- CLI ---------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",  required=True)
    ap.add_argument("--val_pid", required=True, help="e.g. p00")
    ap.add_argument("--gpu",     type=int, default=0,
                    help="GPU index to use, e.g. 0 / 1 / 2 ...")
    args = ap.parse_args()
    main(args.config, args.val_pid, args.gpu)
