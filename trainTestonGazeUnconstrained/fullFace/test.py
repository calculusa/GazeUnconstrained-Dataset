import os, sys, yaml, math, json, importlib, numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
import model

# ---------- util ----------
def gazeto3d(yaw_pitch):
    yaw, pitch = yaw_pitch
    return np.array([
        -math.cos(pitch)*math.sin(yaw),
        -math.sin(pitch),
        -math.cos(pitch)*math.cos(yaw)
    ], dtype=np.float32)

def angular(v1, v2):
    return math.degrees(
        math.acos(
            max(-1., min(np.dot(v1, v2) /
                         (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6), 1.))
        )
    )

def list_clean(path):
    """返回去掉隐藏文件的排序列表"""
    return sorted([f for f in os.listdir(path) if not f.startswith('.')])

# ---------- main ----------
def test(cfg_file, val_id):
    cfg_all = yaml.load(open(cfg_file), Loader=yaml.FullLoader)
    rdr_name= cfg_all["reader"]
    dataloader = importlib.import_module(rdr_name)

    tcfg      = cfg_all["test"]
    img_root  = tcfg["data"]["image"]
    label_dir = tcfg["data"]["label"]

    folders   = list_clean(label_dir)           # 修复 .DS_Store
    val_id    = int(val_id)
    assert 0 <= val_id < len(folders), "val_id out of range"
    test_set  = folders[val_id]
    print(f"[Test Set]  {test_set}")

    ds = dataloader.txtload(
            os.path.join(label_dir, test_set),
            img_root,
            bs := 32, shuffle=False, num_workers=4, header=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- checkpoint 路径 ----------
    mdl_dir = os.path.join(tcfg["load"]["load_path"], "checkpoint", test_set)
    mdl_name= tcfg["load"]["model_name"]

    # cfg 里若缺 begin/end/steps，则根据目录自动推断
    begin = tcfg["load"].get("begin_step")
    end   = tcfg["load"].get("end_step")
    step  = tcfg["load"].get("steps")

    if None in (begin, end, step):
        iters = [int(f.split('_')[1])
                 for f in list_clean(mdl_dir)
                 if f.startswith("Iter_") and f.endswith(f"_{mdl_name}.pt")]
        assert iters, f"No checkpoints found in {mdl_dir}"
        iters.sort()
        begin, end = iters[0], iters[-1]
        step = iters[1]-iters[0] if len(iters)>1 else 1
        print(f"(auto) use checkpoints {iters}")

    # ---------- evaluation ----------
    for epoch in range(begin, end+1, step):
        ckpt = os.path.join(mdl_dir, f"Iter_{epoch}_{mdl_name}.pt")
        if not os.path.isfile(ckpt):
            print("⨯  missing", ckpt); continue

        net = model.model().to(device)
        net.load_state_dict(torch.load(ckpt, map_location=device))
        net.eval()

        acc, cnt = 0., 0
        pred_all, gt_all = [], []
        with torch.no_grad():
            for data, gt in tqdm(ds):
                imgs = {"face": data["face"].to(device)}
                gts  = gt.numpy()
                preds= net(imgs).cpu().numpy()

                for p,g in zip(preds,gts):
                    acc += angular(gazeto3d(p), gazeto3d(g))
                    cnt += 1
                pred_all.append(preds); gt_all.append(gts)

        mae = acc/cnt
        print(f"[Epoch {epoch}]  MAE: {mae:.2f}°  ({cnt} samples)")
        np.save(f"pred_iter{epoch}.npy", np.vstack(pred_all))
        np.save(f"gt_iter{epoch}.npy",   np.vstack(gt_all))

        # optional log file
        eval_dir = os.path.join(tcfg["load"]["load_path"], "evaluation", test_set)
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, f"{epoch}.txt"), "a") as f:
            f.write(f"{mae:.4f}\n")

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv)!=3:
        print("usage: python test.py config.yaml val_id(0-14)"); sys.exit(1)
    test(sys.argv[1], sys.argv[2])
