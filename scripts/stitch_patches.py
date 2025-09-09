#!/usr/bin/env python3
import os
import re
import sys
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import tifffile


FNAME_RE = re.compile(r"^frame_(\d{3})_patch_(\d{3})_reconstructed\.(?:png|tif|tiff)$")


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
    else:
        return tifffile.imread(str(path))


def infer_grid_side_from_ref(ref_tiles: Dict[int, Path]) -> Optional[int]:
    if not ref_tiles:
        return None
    # Prefer perfect square of max index + 1; fallback to ceil of count
    max_idx = max(ref_tiles.keys())
    side = int(round(np.sqrt(max_idx + 1)))
    if side * side == (max_idx + 1):
        return side
    # fallback
    return int(np.ceil(np.sqrt(len(ref_tiles))))


def stitch_frame(tiles: Dict[int, Path], patch_size: Optional[int] = None, grid_side: Optional[int] = None) -> np.ndarray:
    # Determine grid side
    if len(tiles) == 0:
        raise ValueError("No tiles provided for stitching")
    if grid_side is None:
        max_idx = max(tiles.keys())
        grid_side = int(round(np.sqrt(max_idx + 1)))
        if grid_side * grid_side != (max_idx + 1):
            grid_side = int(np.ceil(np.sqrt(len(tiles))))

    # Read one tile to get patch size
    if patch_size is None:
        any_tile = next(iter(tiles.values()))
        sample = read_image(any_tile)
        if sample.ndim > 2:
            sample = sample[..., 0]
        ph, pw = sample.shape[:2]
    else:
        ph = pw = int(patch_size)

    # Heuristic stride: if more than 8x8 tiles, assume 50% overlap, else no overlap
    stride = ph // 2 if grid_side > 8 else ph

    # Compute canvas size using tile positions
    # First pass to compute required H, W
    H = 0
    W = 0
    for idx in tiles.keys():
        r = idx // grid_side
        c = idx % grid_side
        y0 = r * stride
        x0 = c * stride
        H = max(H, y0 + ph)
        W = max(W, x0 + pw)

    accum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    for idx, path in tiles.items():
        r = idx // grid_side
        c = idx % grid_side
        y0 = r * stride
        x0 = c * stride
        tile = read_image(path)
        if tile.ndim > 2:
            tile = tile[..., 0]
        tile = tile.astype(np.float32)
        h, w = tile.shape
        accum[y0:y0+h, x0:x0+w] += tile
        weight[y0:y0+h, x0:x0+w] += 1.0

    mask = weight > 0
    out = np.zeros_like(accum)
    out[mask] = accum[mask] / weight[mask]
    return out.astype(np.float32)


def gather_tiles_by_frame(pred_dir: Path) -> Dict[str, Dict[int, Path]]:
    frames: Dict[str, Dict[int, Path]] = {}
    for p in pred_dir.iterdir():
        if not p.is_file():
            continue
        m = FNAME_RE.match(p.name)
        if not m:
            continue
        frame_id = m.group(1)
        tile_id = int(m.group(2))
        frames.setdefault(frame_id, {})[tile_id] = p
    return frames


def gather_ref_tiles(frame_id: str, ref_dir: Path) -> Dict[int, Path]:
    tiles: Dict[int, Path] = {}
    patt = re.compile(rf"^frame_{frame_id}_patch_(\d{{3}})\.(?:png|tif|tiff)$")
    if not ref_dir.exists():
        return tiles
    for p in ref_dir.iterdir():
        if not p.is_file():
            continue
        m = patt.match(p.name)
        if m:
            tiles[int(m.group(1))] = p
    return tiles


def concat_panels(images: List[np.ndarray]) -> np.ndarray:
    # Normalize each to uint8 consistently
    panels = [to_uint8(img) for img in images]
    # Ensure same height; pad/crop if needed
    h = min(img.shape[0] for img in panels)
    panels = [img[:h, :img.shape[1]] for img in panels]
    return np.concatenate(panels, axis=1)


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: stitch_patches.py <pred_wf_dir> [wf_dir] [tp_dir] [out_dir] [pred_2p_dir]\n"
              "  pred_wf_dir: directory with frame_XXX_patch_YYY_reconstructed.(png|tif) tiles from WF inference\n"
              "  wf_dir (optional): directory with frame_XXX_patch_YYY.png WF tiles (default: real_pairs/wf)\n"
              "  tp_dir (optional): directory with frame_XXX_patch_YYY.png 2P GT tiles (default: real_pairs/2p)\n"
              "  out_dir (optional): output directory (default: pred_wf_dir)\n"
              "  pred_2p_dir (optional): directory with frame_XXX_patch_YYY_reconstructed.(png|tif) tiles from 2P inference")
        sys.exit(1)

    pred_wf_dir = Path(sys.argv[1])
    wf_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/real_microscopy/real_pairs/wf")
    tp_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("data/real_microscopy/real_pairs/2p")
    out_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else pred_wf_dir
    pred_2p_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else None

    out_dir.mkdir(parents=True, exist_ok=True)

    frames_tiles_wf = gather_tiles_by_frame(pred_wf_dir)
    if len(frames_tiles_wf) == 0:
        print(f"No predicted tiles found in {pred_wf_dir}")
        sys.exit(1)

    frames_tiles_2p = {}
    if pred_2p_dir is not None and pred_2p_dir.exists():
        frames_tiles_2p = gather_tiles_by_frame(pred_2p_dir)

    count = 0
    for frame_id, pred_tiles_wf in sorted(frames_tiles_wf.items()):
        # Use WF grid as reference for all stitchings
        wf_tiles = gather_ref_tiles(frame_id, wf_dir)
        ref_side = infer_grid_side_from_ref(wf_tiles) or infer_grid_side_from_ref(pred_tiles_wf)

        # Stitch predicted (WF-based)
        pred_wf_fov = stitch_frame(pred_tiles_wf, grid_side=ref_side)
        Image.fromarray(to_uint8(pred_wf_fov)).save(out_dir / f"frame_{frame_id}_predwf_fov.png")

        # Stitch WF and 2P using same index set (best alignment)
        tp_tiles = gather_ref_tiles(frame_id, tp_dir)

        wf_fov = None
        tp_fov = None
        if len(wf_tiles) > 0:
            wf_fov = stitch_frame({k: wf_tiles[k] for k in sorted(pred_tiles_wf.keys()) if k in wf_tiles}, grid_side=ref_side)
            Image.fromarray(to_uint8(wf_fov)).save(out_dir / f"frame_{frame_id}_wf_fov.png")
        if len(tp_tiles) > 0:
            tp_fov = stitch_frame({k: tp_tiles[k] for k in sorted(pred_tiles_wf.keys()) if k in tp_tiles}, grid_side=ref_side)
            Image.fromarray(to_uint8(tp_fov)).save(out_dir / f"frame_{frame_id}_2p_fov.png")

        # Stitch predicted (2P-based) if provided
        pred2p_fov = None
        if frame_id in frames_tiles_2p:
            pred_tiles_2p = frames_tiles_2p[frame_id]
            pred2p_fov = stitch_frame(pred_tiles_2p, grid_side=ref_side)
            Image.fromarray(to_uint8(pred2p_fov)).save(out_dir / f"frame_{frame_id}_pred2p_fov.png")

        # Save 4-panel strip if possible: [WF | Pred(WF) | 2P | Pred(2P)]
        panels: List[np.ndarray] = []
        if wf_fov is not None:
            panels.append(wf_fov)
        panels.append(pred_wf_fov)
        if tp_fov is not None:
            panels.append(tp_fov)
        if pred2p_fov is not None:
            panels.append(pred2p_fov)
        if len(panels) >= 2:
            strip = concat_panels(panels)
            Image.fromarray(strip).save(out_dir / f"frame_{frame_id}_wf_predwf_2p_pred2p_fov.png")

        count += 1

    print(f"Stitched {count} frame(s) into full-FOV PNGs under {out_dir}")


if __name__ == "__main__":
    main()


