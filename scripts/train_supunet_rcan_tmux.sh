#!/usr/bin/env bash
set -euo pipefail

SESSION="wf2p_baselines_$(date +%Y%m%d_%H%M%S)"
PROJ="/home/jilab/anna_OS_ML/PKL-DiffusionDenoising"
PY="$PROJ/.venv/bin/python"
[ -x "$PY" ] || PY="python"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

DATA_DIR="$PROJ/data/real_microscopy"
OUT_ROOT="$PROJ/outputs"
CKPT_DIR="$PROJ/checkpoints"
mkdir -p "$OUT_ROOT" "$CKPT_DIR" "$PROJ/logs"

# Write SupUNet train+infer script to file
cat > "$PROJ/scripts/_sup_unet_train_infer.py" <<'PY'
import os, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel
from PIL import Image
import numpy as np
import tifffile

PROJ=Path('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising')
CKPT_DIR=PROJ/'checkpoints'
OUT_ROOT=PROJ/'outputs'
DATA_DIR=PROJ/'data/real_microscopy'
RUN=f"sup_unet_{time.strftime('%Y%m%d_%H%M%S')}"
CKPT_PATH=CKPT_DIR/f"{RUN}_sup_unet.pt"
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS=16

class SupervisedUNet(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        ch = base_channels
        self.enc1 = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU())
        self.down1 = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
        self.enc2 = nn.Sequential(nn.Conv2d(ch, ch*2, 3, padding=1), nn.SiLU(), nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.SiLU())
        self.down2 = nn.Conv2d(ch*2, ch*2, 4, stride=2, padding=1)
        self.enc3 = nn.Sequential(nn.Conv2d(ch*2, ch*4, 3, padding=1), nn.SiLU(), nn.Conv2d(ch*4, ch*4, 3, padding=1), nn.SiLU())
        self.up2 = nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2, padding=1)
        self.dec2 = nn.Sequential(nn.Conv2d(ch*4, ch*2, 3, padding=1), nn.SiLU(), nn.Conv2d(ch*2, ch*2, 3, padding=1), nn.SiLU())
        self.up1 = nn.ConvTranspose2d(ch*2, ch, 4, stride=2, padding=1)
        self.dec1 = nn.Sequential(nn.Conv2d(ch*2, ch, 3, padding=1), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.SiLU())
        self.out_conv = nn.Conv2d(ch, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x); d1 = self.down1(e1)
        e2 = self.enc2(d1); d2 = self.down2(e2)
        e3 = self.enc3(d2)
        u2 = self.up2(e3); c2 = torch.cat([u2, e2], dim=1); d2 = self.dec2(c2)
        u1 = self.up1(d2); c1 = torch.cat([u1, e1], dim=1); d1 = self.dec1(c1)
        return self.out_conv(d1)

transform = IntensityToModel(minIntensity=0, maxIntensity=255)
train_ds = RealPairsDataset(data_dir=str(DATA_DIR), split='train', transform=transform, image_size=256)
val_ds   = RealPairsDataset(data_dir=str(DATA_DIR), split='val',   transform=transform, image_size=256)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = SupervisedUNet(64).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
best_val = float('inf'); patience=15; bad=0
max_epochs=200

CKPT_DIR.mkdir(parents=True, exist_ok=True)
for epoch in range(1, max_epochs+1):
    model.train()
    tbar = tqdm(train_loader, desc=f'SupUNet Train {epoch}/{max_epochs}', leave=False)
    for x_2p, y_wf in tbar:
        x_2p = x_2p.to(DEVICE); y_wf = y_wf.to(DEVICE)
        pred = model(y_wf)
        loss = F.l1_loss(pred, x_2p)
        opt.zero_grad(); loss.backward(); opt.step()
        tbar.set_postfix(l1=f'{float(loss):.4f}')
    model.eval(); val_loss=0.0; nb=0
    with torch.no_grad():
        for x_2p, y_wf in val_loader:
            x_2p = x_2p.to(DEVICE); y_wf = y_wf.to(DEVICE)
            pred = model(y_wf)
            val_loss += float(F.l1_loss(pred, x_2p)); nb += 1
    val_loss /= max(nb,1)
    scheduler.step(val_loss)
    print(f"[VAL] epoch {epoch}: {val_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")
    if val_loss < best_val - 1e-5:
        best_val = val_loss; bad = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f'[CKPT] saved {CKPT_PATH}')
    else:
        bad += 1
        if bad >= patience:
            print('[EARLY STOP] patience reached'); break

from pkl_dg.baselines.unet_supervised import infer_supervised_unet
infer_out = OUT_ROOT / 'tmp_eval_png' / RUN
infer_supervised_unet(
    {'run_name': RUN, 'device': DEVICE, 'paths': {'checkpoints': str(CKPT_DIR), 'outputs': str(OUT_ROOT)}, 'data': {'data_dir': str(DATA_DIR), 'image_size': 256, 'min_intensity': 0, 'max_intensity': 255}, 'model_channels': 64},
    CKPT_PATH,
    PROJ/'data/real_microscopy/splits/test/wf',
    infer_out
)
print('[DONE] SupUNet training+inference complete:', infer_out)
PY

# Write RCAN train+infer script to file
cat > "$PROJ/scripts/_rcan_train_infer.py" <<'PY'
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel
import tifffile
from PIL import Image

PROJ=Path('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising')
DATA_DIR=PROJ/'data/real_microscopy'
CKPT_DIR=PROJ/'checkpoints'
OUT_ROOT=PROJ/'outputs'
RUN=f"rcan_{time.strftime('%Y%m%d_%H%M%S')}"
CKPT_PATH=CKPT_DIR/f"{RUN}.pt"
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS=16
BATCH_SIZE=8

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCAB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size//2),
            CALayer(n_feat, reduction)
        )
    def forward(self, x):
        res = self.body(x)
        return x + res

class TinyRCAN(nn.Module):
    def __init__(self, n_feat=64, n_blocks=12):
        super().__init__()
        self.head = nn.Conv2d(1, n_feat, 3, padding=1)
        self.body = nn.Sequential(*[RCAB(n_feat) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(n_feat, 1, 3, padding=1)
    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        return self.tail(res)

transform = IntensityToModel(minIntensity=0, maxIntensity=255)
train_ds = RealPairsDataset(data_dir=str(DATA_DIR), split='train', transform=transform, image_size=256)
val_ds   = RealPairsDataset(data_dir=str(DATA_DIR), split='val',   transform=transform, image_size=256)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = TinyRCAN(64, n_blocks=12).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True)
best_val = float('inf'); patience=15; bad=0
max_epochs=200
CKPT_DIR.mkdir(parents=True, exist_ok=True)

for epoch in range(1, max_epochs+1):
    model.train()
    tbar = tqdm(train_loader, desc=f'RCAN Train {epoch}/{max_epochs}', leave=False)
    for x_2p, y_wf in tbar:
        x_2p = x_2p.to(DEVICE); y_wf = y_wf.to(DEVICE)
        pred = model(y_wf)
        loss = F.l1_loss(pred, x_2p)
        opt.zero_grad(); loss.backward(); opt.step()
        tbar.set_postfix(l1=f'{float(loss):.4f}')
    model.eval(); val_loss=0.0; nb=0
    with torch.no_grad():
        for x_2p, y_wf in val_loader:
            x_2p = x_2p.to(DEVICE); y_wf = y_wf.to(DEVICE)
            pred = model(y_wf)
            val_loss += float(F.l1_loss(pred, x_2p)); nb += 1
    val_loss /= max(nb,1)
    scheduler.step(val_loss)
    print(f"[VAL] epoch {epoch}: {val_loss:.6f}  lr={opt.param_groups[0]['lr']:.2e}")
    if val_loss < best_val - 1e-5:
        best_val = val_loss; bad = 0
        torch.save(model.state_dict(), CKPT_PATH)
        print(f'[CKPT] saved {CKPT_PATH}')
    else:
        bad += 1
        if bad >= patience:
            print('[EARLY STOP] patience reached'); break

def read_img(p: Path):
    if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        return np.array(Image.open(p))
    return tifffile.imread(str(p))

def to_uint8(a: np.ndarray):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo: lo, hi = float(a.min()), float(a.max())
    if hi > lo: a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255).astype(np.uint8)

test_wf = PROJ/'data/real_microscopy/splits/test/wf'
out_dir = OUT_ROOT/'tmp_eval_png'/RUN
out_dir.mkdir(parents=True, exist_ok=True)

model.load_state_dict(torch.load(str(CKPT_PATH), map_location=DEVICE)); model.eval()
tiles = sorted(list(test_wf.glob('frame_*_patch_*.png'))) or sorted(list(test_wf.glob('frame_*_patch_*.tif')))
for p in tqdm(tiles, desc='RCAN Inference'):
    wf = read_img(p).astype(np.float32)
    ten = torch.from_numpy(wf)
    if ten.ndim == 2: ten = ten.unsqueeze(0).unsqueeze(0)
    ten = transform(ten.to(DEVICE))
    with torch.no_grad():
        pred = model(ten)
        out = transform.inverse(pred).squeeze().detach().cpu().numpy().astype(np.float32)
    tifffile.imwrite(str(out_dir/f"{p.stem}_reconstructed.tif"), out)
    Image.fromarray(to_uint8(out)).save(str(out_dir/f"{p.stem}_reconstructed.png"))
print('[DONE] RCAN training+inference complete:', out_dir)
PY

# Launch tmux session and windows
tmux new-session -d -s "$SESSION" -n "sup_unet" "$PY $PROJ/scripts/_sup_unet_train_infer.py"
tmux new-window -t "$SESSION":2 -n "rcan" "$PY $PROJ/scripts/_rcan_train_infer.py"

echo "tmux session created: $SESSION"
echo "Attach with: tmux attach -t $SESSION"


