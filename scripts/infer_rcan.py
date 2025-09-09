#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
from typing import Dict

import numpy as np
import tifffile
from PIL import Image
from tqdm import tqdm

from pkl_dg.baselines import RCANWrapper


def to_uint8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        return np.array(Image.open(path))
    return tifffile.imread(str(path))


def infer_tiles(rcan: RCANWrapper, input_dir: Path, out_dir: Path, normalize: bool,
                min_intensity: float, max_intensity: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tiles = sorted(list(input_dir.glob("frame_*_patch_*.png")))
    if len(tiles) == 0:
        tiles = sorted(list(input_dir.glob("frame_*_patch_*.tif")))
    for p in tqdm(tiles, desc="RCAN inference"):
        wf = read_image(p).astype(np.float32)
        if wf.ndim == 3:
            wf = wf.mean(axis=-1)
        if normalize:
            wf = np.clip((wf - min_intensity) / max(1e-6, (max_intensity - min_intensity)), 0.0, 1.0)
        pred = rcan.infer(wf)
        # If normalized, map back to intensity
        if normalize:
            pred = np.clip(pred, 0.0, 1.0) * (max_intensity - min_intensity) + min_intensity
        pred = pred.astype(np.float32)
        tifffile.imwrite(str(out_dir / f"{p.stem}_reconstructed.tif"), pred)
        Image.fromarray(to_uint8(pred)).save(out_dir / f"{p.stem}_reconstructed.png")


def stitch_fov(pred_dir: Path, wf_dir: Path, gt_dir: Path, fov_out_dir: Path) -> None:
    patt = re.compile(r"^frame_(\d{3})_patch_(\d{3})_reconstructed$")
    pred_tiles_by_frame: Dict[str, Dict[int, Path]] = {}
    for p in pred_dir.iterdir():
        if not p.is_file():
            continue
        m = patt.match(p.stem)
        if m:
            frame_id = m.group(1)
            tile_id = int(m.group(2))
            pred_tiles_by_frame.setdefault(frame_id, {})[tile_id] = p

    def _read_img(path: Path) -> np.ndarray:
        ext = path.suffix.lower()
        if ext in (".png", ".jpg", ".jpeg"):
            return np.array(Image.open(path))
        return tifffile.imread(str(path))

    def _infer_grid(ids):
        if not ids:
            return 0
        mx = max(ids)
        side = int(round(np.sqrt(mx + 1)))
        return side if side * side == (mx + 1) else int(np.ceil(np.sqrt(len(ids))))

    def _stitch(tiles: Dict[int, Path], grid_side: int) -> np.ndarray:
        if len(tiles) == 0:
            raise ValueError("No tiles")
        sample = _read_img(next(iter(tiles.values())))
        if sample.ndim > 2:
            sample = sample[..., 0]
        ph, pw = sample.shape[:2]
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
        return out

    fov_out_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, pred_tiles in sorted(pred_tiles_by_frame.items()):
        ids = sorted(pred_tiles.keys())
        grid_side = _infer_grid(ids)
        pred_fov = _stitch({k: pred_tiles[k] for k in ids}, grid_side)
        pred_u8 = to_uint8(pred_fov)
        wf_tiles = {k: (wf_dir / f"frame_{frame_id}_patch_{k:03d}.png") for k in ids if (wf_dir / f"frame_{frame_id}_patch_{k:03d}.png").exists()}
        gt_tiles = {k: (gt_dir / f"frame_{frame_id}_patch_{k:03d}.png") for k in ids if gt_dir.exists() and (gt_dir / f"frame_{frame_id}_patch_{k:03d}.png").exists()}
        panels = []
        if len(wf_tiles) > 0:
            wf_fov = _stitch(wf_tiles, grid_side)
            panels.append(to_uint8(wf_fov))
        panels.append(pred_u8)
        if len(gt_tiles) > 0:
            gt_fov = _stitch(gt_tiles, grid_side)
            panels.append(to_uint8(gt_fov))
        strip = np.concatenate(panels, axis=1)
        Image.fromarray(strip).save(fov_out_dir / f"frame_{frame_id}_wf_pred_gt_fov.png")


def main():
    ap = argparse.ArgumentParser(description="RCAN inference on WF tiles with FOV stitching")
    ap.add_argument("--checkpoint", required=True, help="Path to RCAN checkpoint")
    ap.add_argument("--input-dir", required=True, help="Directory with WF tiles (frame_XXX_patch_YYY.*)")
    ap.add_argument("--output-dir", required=True, help="Directory to write predictions")
    ap.add_argument("--outputs-root", default="outputs", help="Root outputs dir for FOV previews")
    ap.add_argument("--run-name", default="rcan_run", help="Run name for FOV previews")
    ap.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    ap.add_argument("--normalize", action="store_true", help="Normalize input to [0,1] using min/max intensity")
    ap.add_argument("--min-intensity", type=float, default=0.0)
    ap.add_argument("--max-intensity", type=float, default=255.0)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    rcan = RCANWrapper(checkpoint_path=args.checkpoint, device=args.device)

    infer_tiles(rcan, input_dir, out_dir, args.normalize, args.min_intensity, args.max_intensity)

    fov_out_dir = Path(args.outputs_root) / "tmp_eval_png" / args.run_name
    wf_dir = input_dir
    gt_dir = input_dir.parent / "2p"
    stitch_fov(out_dir, wf_dir, gt_dir, fov_out_dir)


if __name__ == "__main__":
    main()


