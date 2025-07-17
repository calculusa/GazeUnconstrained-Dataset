import os, sys, yaml, importlib, torch, torch.nn as nn, torch.optim as optim, model

def safe_dirs(root):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith('.'))

def train(cfg_file, val_id=0):
    cfg_all = yaml.safe_load(open(cfg_file))
    reader_mod = importlib.import_module(cfg_all["reader"])
    tcfg = cfg_all["train"]

    img_root   = tcfg["data"]["image"]
    label_root = tcfg["data"]["label"]
    folds      = safe_dirs(label_root)
    val_fold   = folds[val_id]
    train_flds = [f for f in folds if f != val_fold]

    loader = reader_mod.txtload([os.path.join(label_root, f) for f in train_flds],
                                img_root, tcfg["params"]["batch_size"])

    save_dir = os.path.join(tcfg["save"]["save_path"], "checkpoint", val_fold)
    os.makedirs(save_dir, exist_ok=True)
    logf = open(os.path.join(save_dir, "train_log.txt"), "w")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.model().to(dev).train()
    loss_fn = getattr(nn, tcfg["params"].get("loss", "SmoothL1Loss"))()
    opt  = optim.Adam(net.parameters(), lr=tcfg["params"]["lr"], betas=(0.9,0.95))
    sched= optim.lr_scheduler.StepLR(opt, tcfg["params"]["decay_step"], tcfg["params"]["decay"])

    for ep in range(1, tcfg["params"]["epoch"]+1):
        for i,(x,y) in enumerate(loader):
            x["face"], y = x["face"].to(dev), y.to(dev)
            loss = loss_fn(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if i%20==0:
                msg=f"Epoch {ep:02d} | Batch {i:04d} | Loss {loss.item():.4f}"
                print(msg); logf.write(msg+"\n"); logf.flush()
        ckpt=os.path.join(save_dir, f"Iter_{ep}_{tcfg['save']['model_name']}.pt")
        torch.save(net.state_dict(), ckpt); print(f"[Saved] {ckpt}")
    logf.close()

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv)>1 else "config.yaml"
    vid = int(sys.argv[2]) if len(sys.argv)>2 else 0
    train(cfg, vid)
