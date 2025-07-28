# coding: utf-8
"""
多进程批量预处理 gaze 数据集
"""

import os, cv2, dlib, json, numpy as np, pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ─────────── 1. 路径配置 ───────────
CSV_PATH       = "/Users/**/Documents/Tobii/TobiiProSDK/64/finalDataset/gaze_annotations_p13.csv"
IMAGE_DIR      = "/Users/**/Documents/Tobii/TobiiProSDK/64/finalDataset/p13"
OUTPUT_DIR     = "/Users/**/Documents/Tobii/TobiiProSDK/64/processed_dataset/p13"
PREDICTOR_PATH = "/Users/**/Documents/Tobii/TobiiProSDK/64/finalDataset/shape_predictor_68_face_landmarks.dat"

# ─────────── 2. 创建输出文件夹 ─────
for sub in ["", "/images/face", "/images/left_eye", "/images/right_eye"]:
    os.makedirs(OUTPUT_DIR + sub, exist_ok=True)

# ─────────── 3. 常量 ───────────────
HEADPOSE_IDX = [30, 8, 45, 36, 54, 48]
LEFT_EYE_IDX = list(range(36, 42))
RIGHT_EYE_IDX= list(range(42, 48))
GRID_SIZE    = 25

# ─────────── 4. 读取 CSV → 预填注释 ─
df = pd.read_csv(CSV_PATH)
annotations = {}
for _, row in df.iterrows():
    fname = f"frame{int(row['frame']):04d}.png"
    annotations[fname] = {
        "face_bbox": None,
        "left_eye_bbox": None,
        "right_eye_bbox": None,
        "head_pose": None,
        "face_grid": None,
        "gaze_vector": ([float(row['gaze_vector_x']), float(row['gaze_vector_y']), float(row['gaze_vector_z'])]
                        if not pd.isna(row['gaze_vector_x']) else None),
        "screen_gaze": ([float(row['screen_gaze_x']),  float(row['screen_gaze_y'])]
                        if not pd.isna(row['screen_gaze_x']) else None),
        "gaze_angles": ([float(row['yaw_deg']),  float(row['pitch_deg'])]
                        if not pd.isna(row['yaw']) else None)
    }

# ─────────── 5. face_grid 工具函数 ─
def generate_face_grid(bbox, img_shape, g=GRID_SIZE):
    grid = np.zeros((g, g), dtype=int)
    h, w = img_shape[:2]
    fx, fy, fw, fh = bbox
    if fw<=0 or fh<=0:
        return grid.flatten().tolist()
    cx = (fx + fw/2) / w
    cy = (fy + fh/2) / h
    gx, gy = int(cx * g), int(cy * g)
    if 0 <= gx < g and 0 <= gy < g:
        grid[gy][gx] = 1
    return grid.flatten().tolist()

# ─────────── 6. 单帧处理函数 ───────
def process_frame(fname):
    if fname not in annotations or not fname.endswith(".png"):
        return ("SKIP", fname)

    path = os.path.join(IMAGE_DIR, fname)
    img  = cv2.imread(path)
    if img is None:
        return ("SKIP", fname)
    h, w = img.shape[:2]

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    dets = detector(img, 0)
    if len(dets)==0:
        return ("SKIP", fname)

    shape = predictor(img, dets[0])
    lm = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    # bbox helper
    def bbox(idxs, mar=10):
        pts = np.array([lm[i] for i in idxs])
        x1, y1 = pts.min(0) - mar
        x2, y2 = pts.max(0) + mar
        x1, y1 = max(x1,0), max(y1,0)
        x2, y2 = min(x2,w), min(y2,h)
        return int(x1), int(y1), int(x2-x1), int(y2-y1)

    # head‑pose
    img_pts = np.ascontiguousarray(np.array([lm[i] for i in HEADPOSE_IDX], dtype=np.float64))
    mdl_pts = np.array([[0,0,0],[0,-330,-65],[225,170,-135],[-225,170,-135],[150,-150,-125],[-150,-150,-125]],
                       dtype=np.float64)
    cam_mtx = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    try:
        ok, rvec, _ = cv2.solvePnP(mdl_pts, img_pts, cam_mtx,
                                   np.zeros((4,1),dtype=np.float64),
                                   flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return ("SKIP", fname)
        yaw,pitch,roll = R.from_matrix(cv2.Rodrigues(rvec)[0]).as_euler('zyx',degrees=True)
    except cv2.error:
        return ("SKIP", fname)

    # crop
    fx, fy, fw, fh = bbox(range(68),20)
    lx, ly, lw, lh = bbox(LEFT_EYE_IDX,5)
    rx, ry, rw, rh = bbox(RIGHT_EYE_IDX,5)

    cv2.imwrite(f"{OUTPUT_DIR}/images/face/{fname}",      img[fy:fy+fh, fx:fx+fw])
    cv2.imwrite(f"{OUTPUT_DIR}/images/left_eye/{fname}",  img[ly:ly+lh, lx:lx+lw])
    cv2.imwrite(f"{OUTPUT_DIR}/images/right_eye/{fname}", img[ry:ry+rh, rx:rx+rw])

    ent = annotations[fname].copy()
    ent.update({
        "face_bbox":   [fx,fy,fw,fh],
        "left_eye_bbox":[lx,ly,lw,lh],
        "right_eye_bbox":[rx,ry,rw,rh],
        "head_pose":   [float(yaw), float(pitch), float(roll)],
        "face_grid":   generate_face_grid([fx,fy,fw,fh], img.shape)
    })
    return (fname, ent)

# ─────────── 7. 多进程批量处理 ───
if __name__ == "__main__":
    files = sorted(os.listdir(IMAGE_DIR))

    skipped, processed = [], {}
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_frame, files), total=len(files)):
            if res[0] == "SKIP":
                skipped.append(res[1])
            else:
                fname, entry = res
                processed[fname] = entry

    # 保存成功帧
    with open(f"{OUTPUT_DIR}/annotations.json", "w") as f:
        json.dump(processed, f, indent=2)

    # 保存跳过帧列表
    if skipped:
        with open(f"{OUTPUT_DIR}/skipped_frames.txt", "w") as f:
            f.write("\n".join(skipped))

    # 终端打印
    print(f"\n✅ 完成！成功 {len(processed)} 帧，跳过 {len(skipped)} 帧。")
    if skipped:
        print("跳过的帧示例:", skipped[:20])
