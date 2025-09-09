import os
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
from PIL import Image
import tifffile

from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel


class SupervisedUNet(nn.Module):
    """Simple supervised U-Net for WF→2P direct mapping.

    Input: WF (1 channel)
    Output: 2P (1 channel)
    Trained with L1 + optional MS-SSIM (not included to keep deps minimal).
    """

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        d2 = self.down2(e2)
        e3 = self.enc3(d2)
        u2 = self.up2(e3)
        c2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(c2)
        u1 = self.up1(d2)
        c1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(c1)
        out = self.out_conv(d1)
        return out


def train_supervised_unet(cfg: Dict[str, Any]) -> Path:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    run_name = cfg.get("run_name", f"sup_unet_{OmegaConf.get_type(cfg) is dict}")

    transform = IntensityToModel(
        minIntensity=float(cfg["data"]["min_intensity"]),
        maxIntensity=float(cfg["data"]["max_intensity"]),
    )

    train_ds = RealPairsDataset(
        data_dir=str(cfg["data"]["data_dir"]),
        split="train",
        transform=transform,
        image_size=int(cfg["data"]["image_size"]),
    )
    val_ds = RealPairsDataset(
        data_dir=str(cfg["data"]["data_dir"]),
        split="val",
        transform=transform,
        image_size=int(cfg["data"]["image_size"]),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]))
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]))

    model = SupervisedUNet(base_channels=int(cfg.get("model_channels", 64))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]))

    best_val = float("inf")
    ckpt_dir = Path(str(cfg["paths"]["checkpoints"]))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{run_name}_sup_unet.pt"

    max_epochs = int(cfg["training"]["max_epochs"])
    lam_l1 = float(cfg.get("l1_weight", 1.0))
    for epoch in range(max_epochs):
        model.train()
        tbar = tqdm(train_loader, desc=f"SupUNet Train {epoch+1}/{max_epochs}", leave=False)
        for x_2p, y_wf in tbar:
            x_2p = x_2p.to(device)
            y_wf = y_wf.to(device)
            pred = model(y_wf)
            loss = lam_l1 * F.l1_loss(pred, x_2p)
            opt.zero_grad(); loss.backward(); opt.step()
            tbar.set_postfix({"l1": f"{float(loss):.4f}"})

        # val
        model.eval(); val_loss = 0.0; nb = 0
        with torch.no_grad():
            for x_2p, y_wf in val_loader:
                x_2p = x_2p.to(device)
                y_wf = y_wf.to(device)
                pred = model(y_wf)
                val_loss += float(F.l1_loss(pred, x_2p))
                nb += 1
        val_loss = val_loss / max(nb, 1)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt)

    return ckpt


def infer_supervised_unet(cfg: Dict[str, Any], ckpt_path: Path, input_dir: Path, out_dir: Path) -> None:
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    transform = IntensityToModel(
        minIntensity=float(cfg["data"]["min_intensity"]),
        maxIntensity=float(cfg["data"]["max_intensity"]),
    )

    model = SupervisedUNet(base_channels=int(cfg.get("model_channels", 64))).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)

    # List WF tiles
    tiles = sorted(list(input_dir.glob("frame_*_patch_*.png")))
    # Fallback to tif
    if len(tiles) == 0:
        tiles = sorted(list(input_dir.glob("frame_*_patch_*.tif")))

    # Tile-wise inference and save
    for p in tqdm(tiles, desc="SupUNet Inference"):
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            wf = np.array(Image.open(p))
        else:
            wf = tifffile.imread(str(p))
        ten = torch.from_numpy(wf).float()
        if ten.ndim == 2:
            ten = ten.unsqueeze(0).unsqueeze(0)
        ten = transform(ten).to(device)
        with torch.no_grad():
            pred = model(ten)
            out = transform.inverse(pred).squeeze().detach().cpu().numpy().astype(np.float32)
        tifffile.imwrite(str(out_dir / f"{p.stem}_reconstructed.tif"), out)
        Image.fromarray(np.clip(out / float(cfg["data"]["max_intensity"]) * 255.0, 0, 255).astype(np.uint8)).save(
            str(out_dir / f"{p.stem}_reconstructed.png")
        )

    # Stitch FOVs similarly to scripts/inference.py logic
    try:
        import re
        def _read_img(path: Path) -> np.ndarray:
            ext = path.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg"):
                return np.array(Image.open(path))
            return tifffile.imread(str(path))
        def _to_uint8(a: np.ndarray) -> np.ndarray:
            a = a.astype(np.float32)
            lo, hi = np.percentile(a, (1, 99))
            if hi <= lo:
                lo, hi = float(a.min()), float(a.max())
            if hi > lo:
                a = (a - lo) / (hi - lo)
            a = np.clip(a, 0.0, 1.0)
            return (a * 255).astype(np.uint8)
        def _infer_grid_side(ids):
            if not ids: return 0
            max_idx = max(ids)
            side = int(round(np.sqrt(max_idx + 1)))
            return side if side * side == (max_idx + 1) else int(np.ceil(np.sqrt(len(ids))))
        def _stitch(tiles: dict[int, Path], grid_side: int, patch_size: Optional[int] = None) -> np.ndarray:
            if len(tiles) == 0: raise ValueError("No tiles")
            if patch_size is None:
                s = _read_img(next(iter(tiles.values())))
                if s.ndim > 2: s = s[..., 0]
                ph, pw = s.shape[:2]
            else:
                ph = pw = int(patch_size)
            stride = ph // 2 if grid_side > 8 else ph
            H = W = 0
            for idx in tiles.keys():
                r = idx // grid_side; c = idx % grid_side
                y0 = r * stride; x0 = c * stride
                H = max(H, y0 + ph); W = max(W, x0 + pw)
            accum = np.zeros((H, W), dtype=np.float32)
            weight = np.zeros((H, W), dtype=np.float32)
            for idx, path in tiles.items():
                r = idx // grid_side; c = idx % grid_side
                y0 = r * stride; x0 = c * stride
                tile = _read_img(path)
                if tile.ndim > 2: tile = tile[..., 0]
                tile = tile.astype(np.float32)
                h, w = tile.shape
                accum[y0:y0+h, x0:x0+w] += tile
                weight[y0:y0+h, x0:x0+w] += 1.0
            out = np.zeros_like(accum)
            m = weight > 0
            out[m] = accum[m] / weight[m]
            return out.astype(np.float32)

        patt = re.compile(r"^frame_(\d{3})_patch_(\d{3})_reconstructed$")
        pred_tiles_by_frame = {}
        for p in out_dir.iterdir():
            if not p.is_file():
                continue
            stem = p.stem
            m = patt.match(stem)
            if m:
                frame_id = m.group(1)
                tile_id = int(m.group(2))
                pred_tiles_by_frame.setdefault(frame_id, {})[tile_id] = p

        run_name = str(cfg.get("run_name", "sup_unet_run"))
        fov_out_dir = Path(str(cfg["paths"]["outputs"])) / "tmp_eval_png" / run_name
        fov_out_dir.mkdir(parents=True, exist_ok=True)

        wf_dir = input_dir
        gt_dir = input_dir.parent / "2p"
        for frame_id, pred_tiles in sorted(pred_tiles_by_frame.items()):
            ids = sorted(pred_tiles.keys())
            grid_side = _infer_grid_side(ids)
            pred_fov = _stitch({k: pred_tiles[k] for k in ids}, grid_side)
            pred_u8 = _to_uint8(pred_fov)
            wf_tiles = {k: (wf_dir / f"frame_{frame_id}_patch_{k:03d}.png") for k in ids if (wf_dir / f"frame_{frame_id}_patch_{k:03d}.png").exists()}
            gt_tiles = {k: (gt_dir / f"frame_{frame_id}_patch_{k:03d}.png") for k in ids if gt_dir.exists() and (gt_dir / f"frame_{frame_id}_patch_{k:03d}.png").exists()}
            wf_u8 = None; gt_u8 = None
            if len(wf_tiles) > 0:
                wf_u8 = _to_uint8(_stitch(wf_tiles, grid_side))
            if len(gt_tiles) > 0:
                gt_u8 = _to_uint8(_stitch(gt_tiles, grid_side))
            panels = []
            if wf_u8 is not None: panels.append(wf_u8)
            panels.append(pred_u8)
            if gt_u8 is not None: panels.append(gt_u8)
            strip = np.concatenate(panels, axis=1)
            Image.fromarray(strip).save(str(fov_out_dir / f"frame_{frame_id}_wf_pred_gt_fov.png"))
    except Exception:
        pass


