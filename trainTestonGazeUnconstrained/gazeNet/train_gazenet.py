import argparse, os, yaml, csv, math, torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from gazenet_dataset  import GazeNetDataset
from gazenet_wrapper  import GazeNetWrapper, angles_to_vector

# ----- metric -----
def angular_error_deg(v_pred, v_true):
    cos = (v_pred * v_true).sum(dim=1) / (v_pred.norm(dim=1)*v_true.norm(dim=1))
    cos = cos.clamp(-1,1)
    return torch.acos(cos)*180.0/math.pi

def vector_to_angles(v):
    yaw = torch.atan2(v[:,0], -v[:,2])
    pitch = -torch.asin(v[:,1])
    return torch.stack([yaw,pitch],dim=1)

# ----- single split -----
def run_once(cfg, val_pid, device):
    tr_ds = GazeNetDataset(cfg['data']['root'], exclude_pids=[val_pid])
    va_ds = GazeNetDataset(cfg['data']['root'], include_pids=[val_pid])

    tr_ld = DataLoader(tr_ds, batch_size=cfg['train']['batch_size'],
                       shuffle=True, num_workers=4, pin_memory=True)
    va_ld = DataLoader(va_ds, batch_size=cfg['train']['batch_size'],
                       shuffle=False, num_workers=4, pin_memory=True)

    model = GazeNetWrapper().to(device)
    crit  = nn.MSELoss()
    opt   = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    sched = torch.optim.lr_scheduler.StepLR(opt,
                step_size=cfg['train']['step_size'], gamma=cfg['train']['gamma'])

    save_dir = cfg['save']['dir']; os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir,f'log_{val_pid}.csv')
    with open(log_path,'w',newline='') as f:
        csv.writer(f).writerow(['epoch','train_loss','val_ang_deg'])

    best=1e9
    for ep in range(cfg['train']['epochs']):
        # --- train ---
        model.train(); tot=0
        pbar = tqdm(tr_ld, desc=f'[{val_pid}] {ep+1}/{cfg["train"]["epochs"]}',
                    unit='batch', leave=False)
        for inputs, y in pbar:
            inputs = {k:v.to(device) for k,v in inputs.items()}
            y = y.to(device)

            v_pred = model(inputs, return_vector=True)
            if y.shape[1]==2: y = angles_to_vector(y[:,0],y[:,1])
            loss = crit(v_pred, y)

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*y.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()
        train_loss = tot/len(tr_ld.dataset)

        # --- val ---
        model.eval(); tot_ang=0
        with torch.no_grad():
            for inputs, y in va_ld:
                inputs = {k:v.to(device) for k,v in inputs.items()}
                y = y.to(device)
                v_pred = model(inputs, return_vector=True)
                if y.shape[1]==2: y = angles_to_vector(y[:,0],y[:,1])
                ang = angular_error_deg(v_pred, y).mean()
                tot_ang += ang.item()*y.size(0)
        val_ang = tot_ang/len(va_ld.dataset)

        print(f'[{val_pid}] Ep {ep+1:03d} | train={train_loss:.4f} | val={val_ang:.2f}°')
        with open(log_path,'a',newline='') as f:
            csv.writer(f).writerow([ep+1,f'{train_loss:.6f}',f'{val_ang:.6f}'])
        if val_ang<best:
            best=val_ang
            torch.save(model.state_dict(),
                       os.path.join(save_dir,f'best_{val_pid}.pth'))
    print(f'✓ PID={val_pid} best={best:.2f}° | log→{log_path}')

# ----- CLI -----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--val_pid',required=True)
    ap.add_argument('--gpu',default='0')
    args = ap.parse_args()

    torch.backends.cudnn.benchmark=True
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    cfg = yaml.safe_load(open(args.config))
    run_once(cfg,args.val_pid,device)

if __name__=='__main__':
    main()
