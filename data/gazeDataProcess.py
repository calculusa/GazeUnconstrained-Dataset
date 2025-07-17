import pandas as pd
import numpy as np
import math

# 1) Load your raw gaze CSV
in_csv = "/Users/**/Documents/Tobii/TobiiProSDK/64/rawData/gaze_data_*.csv"
df = pd.read_csv(in_csv)

# 2) Helper to parse "(x, y, z)"
def parse_triplet(s):
    if not isinstance(s, str):
        return [np.nan, np.nan, np.nan]
    s = s.strip().lstrip('(').rstrip(')')
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        return [np.nan, np.nan, np.nan]
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except:
            out.append(np.nan)
    return out

# 3) Unpack the 3D columns
df[['LOrigin_X','LOrigin_Y','LOrigin_Z']] = (
    df['Left Gaze Origin'].apply(parse_triplet).tolist()
)
df[['ROrigin_X','ROrigin_Y','ROrigin_Z']] = (
    df['Right Gaze Origin'].apply(parse_triplet).tolist()
)
df[['LPoint_X','LPoint_Y','LPoint_Z']] = (
    df['Left Gaze Point (User Coord)'].apply(parse_triplet).tolist()
)
df[['RPoint_X','RPoint_Y','RPoint_Z']] = (
    df['Right Gaze Point (User Coord)'].apply(parse_triplet).tolist()
)

# 4) Compute the annotations
img_w, img_h = 1920, 1080

records = []
for idx, row in df.iterrows():
    # 2D pixel gaze
    x2d = ((row['Left Gaze X'] + row['Right Gaze X'])/2)*img_w
    y2d = ((row['Left Gaze Y'] + row['Right Gaze Y'])/2)*img_h

    # 3D vectors
    oL = np.array([row.LOrigin_X, row.LOrigin_Y, row.LOrigin_Z])
    pL = np.array([row.LPoint_X,   row.LPoint_Y,   row.LPoint_Z])
    vL = pL - oL
    if np.linalg.norm(vL)==0: vL=np.array([0,0,1])
    else: vL/=np.linalg.norm(vL)

    oR = np.array([row.ROrigin_X, row.ROrigin_Y, row.ROrigin_Z])
    pR = np.array([row.RPoint_X,   row.RPoint_Y,   row.RPoint_Z])
    vR = pR - oR
    if np.linalg.norm(vR)==0: vR=np.array([0,0,1])
    else: vR/=np.linalg.norm(vR)

    v3d = vL + vR
    if np.linalg.norm(v3d)==0: v3d=np.array([0,0,1])
    else: v3d/=np.linalg.norm(v3d)

    # yaw & pitch
    yaw   = math.degrees(math.atan2(v3d[0], v3d[2]))
    pitch = math.degrees(math.atan2(v3d[1], math.hypot(v3d[0], v3d[2])))

    records.append({
        'frame': idx,    # integer for now
        'x2d':   x2d,
        'y2d':   y2d,
        'v3d_x': v3d[0],
        'v3d_y': v3d[1],
        'v3d_z': v3d[2],
        'yaw':   yaw,
        'pitch': pitch
    })

# 5) Build DataFrame and zero‑pad the frame column
ann_df = pd.DataFrame(records)
ann_df['frame'] = ann_df['frame'].astype(int).astype(str).str.zfill(4)

# Verify the first few
print(ann_df['frame'].head())

# 6) Save to CSV
out_csv = (
    "/Users/**/Documents/Tobii/TobiiProSDK/64/processedDataStep1/"
    "gaze_annotations_*.csv"
)
ann_df.to_csv(out_csv, index=False)
print(f"Saved annotations to {out_csv}")

# after your ann_df.to_csv(…) call:
with open(out_csv, 'r') as f:
    for _ in range(5):
        print(f.readline().rstrip())

