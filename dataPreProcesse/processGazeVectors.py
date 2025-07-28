import pandas as pd
import numpy as np
import math

# === 1. Load Tobii raw data ===
in_csv = "/Users/**/Documents/Tobii/TobiiProSDK/64/rawData/gaze_data_**.csv"  # 修改为你的路径
df = pd.read_csv(in_csv)

# === 2. Parse string triplet (x, y, z) into float list ===
def parse_triplet(s):
    if not isinstance(s, str):
        return [np.nan, np.nan, np.nan]
    s = s.strip().lstrip('(').rstrip(')')
    parts = [p.strip() for p in s.split(',')]
    if len(parts) != 3:
        return [np.nan, np.nan, np.nan]
    try:
        return [float(parts[0]), float(parts[1]), float(parts[2])]
    except:
        return [np.nan, np.nan, np.nan]

df[['LOrigin_X','LOrigin_Y','LOrigin_Z']] = df['Left Gaze Origin'].apply(parse_triplet).tolist()
df[['LPoint_X','LPoint_Y','LPoint_Z']] = df['Left Gaze Point (User Coord)'].apply(parse_triplet).tolist()
df[['ROrigin_X','ROrigin_Y','ROrigin_Z']] = df['Right Gaze Origin'].apply(parse_triplet).tolist()
df[['RPoint_X','RPoint_Y','RPoint_Z']] = df['Right Gaze Point (User Coord)'].apply(parse_triplet).tolist()

# === 3. Image resolution ===
img_w, img_h = 1920, 1080  # 修改为你的屏幕像素

# === 4. Main loop ===
records = []

for idx, row in df.iterrows():
    oL = np.array([row['LOrigin_X'], row['LOrigin_Y'], row['LOrigin_Z']])
    pL = np.array([row['LPoint_X'], row['LPoint_Y'], row['LPoint_Z']])
    oR = np.array([row['ROrigin_X'], row['ROrigin_Y'], row['ROrigin_Z']])
    pR = np.array([row['RPoint_X'], row['RPoint_Y'], row['RPoint_Z']])

    valid_L = not np.any(np.isnan(oL)) and not np.any(np.isnan(pL))
    valid_R = not np.any(np.isnan(oR)) and not np.any(np.isnan(pR))

    if valid_L and valid_R:
        origin = (oL + oR) / 2
        point = (pL + pR) / 2
    elif valid_L:
        origin, point = oL, pL
    elif valid_R:
        origin, point = oR, pR
    else:
        records.append({
            'frame': str(idx).zfill(4),
            'x2d': np.nan,
            'y2d': np.nan,
            'screen_gaze_x': np.nan,
            'screen_gaze_y': np.nan,
            'gaze_vector_x': np.nan,
            'gaze_vector_y': np.nan,
            'gaze_vector_z': np.nan,
            'yaw_deg': np.nan,
            'pitch_deg': np.nan
        })
        continue

    # Gaze direction
    vector = point - origin
    norm = np.linalg.norm(vector)
    gaze_vector = vector / norm if norm > 0 else [np.nan, np.nan, np.nan]

    x, y, z = gaze_vector
    yaw = math.degrees(math.atan2(x, -z))
    pitch = math.degrees(math.asin(y))

    # Average gaze position on screen
    lx, ly = row.get('Left Gaze X'), row.get('Left Gaze Y')
    rx, ry = row.get('Right Gaze X'), row.get('Right Gaze Y')

    if not np.isnan(lx) and not np.isnan(rx):
        gx, gy = (lx + rx)/2, (ly + ry)/2
    elif not np.isnan(lx):
        gx, gy = lx, ly
    elif not np.isnan(rx):
        gx, gy = rx, ry
    else:
        gx, gy = np.nan, np.nan

    x2d = gx * img_w if not np.isnan(gx) else np.nan
    y2d = gy * img_h if not np.isnan(gy) else np.nan

    records.append({
        'frame': str(idx).zfill(4),
        'x2d': x2d,
        'y2d': y2d,
        'screen_gaze_x': gx,
        'screen_gaze_y': gy,
        'gaze_vector_x': x,
        'gaze_vector_y': y,
        'gaze_vector_z': z,
        'yaw_deg': yaw,
        'pitch_deg': pitch
    })

# === 5. Save CSV ===
df_out = pd.DataFrame(records)
out_csv = "gaze_vectors_output_full_p13.csv"
df_out.to_csv(out_csv, index=False)
print(f"✅ Saved to: {out_csv}")
