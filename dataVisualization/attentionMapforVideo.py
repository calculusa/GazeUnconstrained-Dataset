#!/usr/bin/env python3
"""

将多名参与者在指定帧 ±delta 秒内的 gaze 点绘制到截图上。
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import cv2, pytz
from datetime import datetime
from pathlib import Path
from itertools import cycle

# =============== 用户可修改 ===============
IMG_PATH      = "backgroundVideoFrame.png"          # 背景截图
MAP_CSV       = "time_mapping.csv"   # 对照表
DELTA         = 0.5                  # 时间窗口 ± 秒
TIME_FMT      = "%Y-%m-%d %H:%M:%S"  # 可读时间格式
LOCAL_TZ = pytz.timezone("Europe/London")   # 伦敦时区（含夏令时自动处理）# 录制机器所在时区
NORMALIZED    = True   # True: gaze 坐标 ∈[0,1]；False: 已是像素
DRAW_HEATMAP  = True  # True 叠加 KDE 热图
OUT_PNG       = "gaze_overlay_heatmap.pdf"
# =========================================

# ---------- 颜色循环 ----------
COLORS = cycle(sns.color_palette("tab10"))

def human2unix(tstr: str) -> float:
    """将 'YYYY-MM-DD HH:MM:SS' 转为 Unix 秒"""
    dt = datetime.strptime(tstr.strip(), TIME_FMT)
    if LOCAL_TZ:
        dt = LOCAL_TZ.localize(dt)
    return dt.timestamp()

def find_cols(df, keyword):
    """返回第一个包含 keyword（忽略大小写空格）的列名"""
    matches = [c for c in df.columns if keyword.lower() in c.lower().replace(" ", "")]
    if not matches:
        raise ValueError(f"找不到列 {keyword}*")
    return matches[0]

def load_gaze(csv_path, t_frame, delta):
    df = pd.read_csv(csv_path)
    t_col = find_cols(df, "timestamp")
    # 自动把微秒转秒（>1e12 视为微秒）
    if df[t_col].abs().mean() > 1e12:
        df[t_col] /= 1e6

    lx = find_cols(df, "leftgazex")
    ly = find_cols(df, "leftgazey")
    rx = find_cols(df, "rightgazex")
    ry = find_cols(df, "rightgazey")

    win = df.query(f"{t_col} >= @t_frame-{delta} and {t_col} <= @t_frame+{delta}")
    x = (win[lx].values + win[rx].values) / 2
    y = (win[ly].values + win[ry].values) / 2
    return x, y

def main():
    img = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    mapping = pd.read_csv(MAP_CSV)
    all_xy = []
    plt.figure(figsize=(12, 7))
    plt.imshow(img)

    for (_, row), color in zip(mapping.iterrows(), COLORS):
        pid   = str(row["participant_id"])
        path  = Path(row["csv_path"])
        tstr  = str(row["frame_human_datetime"])
        t_abs = human2unix(tstr)

        x, y = load_gaze(path, t_abs, DELTA)
        if NORMALIZED:
            x *= W;  y *= H
        plt.scatter(x, y, s=40, c=[color], edgecolors="w",
                    linewidths=0.5, alpha=0.9, label=f"{pid} ({len(x)})")
        all_xy.append(np.vstack([x, y]).T)

    if DRAW_HEATMAP and len(all_xy):
        xy = np.vstack(all_xy)
        sns.kdeplot(x=xy[:, 0], y=xy[:, 1], cmap="viridis", 
                    fill=True, alpha=0.4, bw_adjust=0.5,
                    thresh=0.01, levels=100)

    plt.legend(loc="upper right")
    plt.axis("off"); plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    print(f"✅ Saved: {OUT_PNG}")

if __name__ == "__main__":
    main()
