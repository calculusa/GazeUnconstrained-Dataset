import os
import cv2
import pandas as pd

# Params—adjust to your paths:
GAZE_CSV    = "/Users/**/Documents/Tobii/TobiiProSDK/64/rawData/gaze_data_*.csv"
VIDEO_FILE  = "/Users/**/Documents/Tobii/TobiiProSDK/64/rawData/face_video_*.mp4"
OUT_DIR     = "/Users/**/Documents/Tobii/TobiiProSDK/64/processedDataStep1/extracted_frames_*"

# 1) Load gaze data
df = pd.read_csv(GAZE_CSV)
video_start_ts = df['AbsoluteTimestamp'].iloc[0]

# 2) Open video and read metadata
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video {VIDEO_FILE}")

fps          = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video loaded: {total_frames} frames at {fps:.2f} fps")

# 3) Pre‐load every frame into memory
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

os.makedirs(OUT_DIR, exist_ok=True)

# 4) For each gaze row, compute and clamp frame index
for idx, row in df.iterrows():
    abs_ts = row['AbsoluteTimestamp']
    offset = abs_ts - video_start_ts       # seconds since start
    frame_idx = int(round(offset * fps))    # desired frame

    # clamp to valid range
    if frame_idx < 0:
        frame_idx = 0
    elif frame_idx >= total_frames:
        frame_idx = total_frames - 1

    frame = frames[frame_idx]
    # name like frame_0000_1702381234.567.png
    # fname = os.path.join(OUT_DIR, f"frame_{idx:04d}_{abs_ts:.3f}.png")
    fname = os.path.join(OUT_DIR, f"frame{idx:04d}.png")
    cv2.imwrite(fname, frame)

print("Done extracting frames.")    
