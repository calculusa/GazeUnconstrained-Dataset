# train_gaze360.py ---------------------------------------------------
import argparse, os, yaml, csv, math, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gaze360_dataset       import Gaze360Dataset as StaticDataset
from gaze360_lstm_dataset  import Gaze360SequenceDataset as SeqDataset
from model_gaze360         import Gaze360, angles_to_vector
from model_origin               import PinBallLoss
# ---------- 函数同前 (angular_error_deg / vector_to_angles) ----------
def angular_error_deg(g_pred, g_true):
    eps = (g_pred * g_true).sum(dim=1) / (
        g_pred.norm(dim=1) * g_true.norm(dim=1)
    )
    eps = eps.clamp(-1.0, 1.0)
    return torch.acos(eps) * 180.0 / math.pi
def vector_to_angles(g):
    yaw   = torch.atan2(g[:,0], -g[:,2])
    pitch = -torch.asin(g[:,1])
    return torch.stack([yaw, pitch], dim=1)
# --------------------------------------------------------------------
def run_once(cfg, val_pid, device):
    backbone = cfg['model'].get('backbone', 'static').lower()
    use_lstm = (backbone == 'lstm')

    DS = SeqDataset if use_lstm else StaticDataset
    tr_ds = DS(cfg['data']['root'], mode='3d', eye_mode='face',
               exclude_pids=[val_pid])
    va_ds = DS(cfg['data']['root'], mode='3d', eye_mode='face',
               include_pids=[val_pid])

    tr_ld = DataLoader(tr_ds, batch_size=cfg['train']['batch_size'],
                       shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg['train']['batch_size'],
                       shuffle=False, num_workers=4, pin_memory=True)

    model = Gaze360(backbone=backbone).to(device)
    use_pinball = str(cfg['train'].get('loss', '')).lower().startswith('pinball')
    crit = PinBallLoss() if use_pinball else nn.MSELoss()

    opt   = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=cfg['train']['step_size'], gamma=cfg['train']['gamma'])

    save_dir = cfg['save']['dir']; os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f'log_{val_pid}.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_ang_deg'])

    best = 1e9
    for ep in range(cfg['train']['epochs']):
        # ---- train ----
        model.train(); run_loss = 0
        pbar = tqdm(tr_ld, desc=f'[PID {val_pid}] {ep+1}/{cfg["train"]["epochs"]}',
                    unit='batch', leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            if use_pinball:
                angles, var = model(x, return_angles=True)
                y_ang = vector_to_angles(y)
                loss  = crit(angles, y_ang, var)
            else:
                loss  = crit(model(x), y)

            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()
        train_loss = run_loss / len(tr_ld.dataset)

        # ---- validate ----
        model.eval(); tot_ang = 0
        with torch.no_grad():
            for x, y in va_ld:
                x, y = x.to(device), y.to(device)
                if use_pinball:
                    angles, _ = model(x, return_angles=True)
                    g_pred = angles_to_vector(angles[:,0], angles[:,1])
                else:
                    g_pred = model(x)
                ang = angular_error_deg(g_pred, y).mean()
                tot_ang += ang.item() * x.size(0)
        val_ang = tot_ang / len(va_ld.dataset)

        print(f'[PID {val_pid}] Ep {ep+1:03d} | train={train_loss:.4f} | val={val_ang:.2f}°')
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep+1,f'{train_loss:.6f}',f'{val_ang:.6f}'])
        if val_ang < best:
            best = val_ang
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f'best_{val_pid}.pth'))
    print(f'✓ PID={val_pid} best val_ang={best:.2f}° | log→{log_path}\n')
# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--val_pid', required=True)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    run_once(yaml.safe_load(open(args.config,'r')), args.val_pid, dev)
if __name__ == '__main__':
    main()
