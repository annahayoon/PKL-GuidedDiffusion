#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

from pkl_dg.data.zarr_io import write_zarr_patches


def _load_png_map(dir_path: Path) -> Dict[str, np.ndarray]:
    files = sorted(list(dir_path.glob("*.png")))
    return {f.name: np.array(Image.open(f)).astype(np.float32) for f in files}


def _sample_values(images: List[np.ndarray], max_pixels: int = 2_000_000, rng: np.random.Generator = np.random.default_rng(0)) -> np.ndarray:
    if not images:
        return np.array([], dtype=np.float32)
    flat = np.concatenate([im.reshape(-1) for im in images]).astype(np.float32)
    if flat.size <= max_pixels:
        return flat
    idx = rng.choice(flat.size, size=max_pixels, replace=False)
    return flat[idx]


def _robust_quantiles(values: np.ndarray, num_bins: int = 2048) -> np.ndarray:
    vals = values[np.isfinite(values)]
    if vals.size == 0:
        return np.linspace(0.0, 1.0, num_bins)
    qs = np.linspace(0.0, 1.0, num_bins)
    qv = np.quantile(vals, qs)
    return np.maximum.accumulate(qv)


def _compute_shared_reference(wf_vals: np.ndarray, tp_vals: np.ndarray, num_bins: int = 2048) -> np.ndarray:
    wf_q = _robust_quantiles(wf_vals, num_bins)
    tp_q = _robust_quantiles(tp_vals, num_bins)
    ref = np.mean(np.stack([wf_q, tp_q], axis=0), axis=0)
    return np.maximum.accumulate(ref)


def _compute_mapping_to_reference(source_vals: np.ndarray, ref_q: np.ndarray, num_bins: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    src_q = _robust_quantiles(source_vals, num_bins)
    return src_q, ref_q


def _apply_piecewise_linear(image: np.ndarray, src_points: np.ndarray, ref_points: np.ndarray) -> np.ndarray:
    return np.interp(image.astype(np.float32), src_points, ref_points).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Build Zarr patches from WF/2P PNG pairs with optional quantile matching")
    ap.add_argument("--data-dir", required=True, type=str, help="Root containing splits/<train|val>/{wf,2p}")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], type=str)
    ap.add_argument("--out", required=True, type=str, help="Output Zarr directory (e.g., .../train.zarr)")
    ap.add_argument("--tile-size", default=256, type=int)
    ap.add_argument("--overlap", default=32, type=int)
    ap.add_argument("--enable-quantile-matching", action="store_true")
    ap.add_argument("--num-bins", default=2048, type=int)
    ap.add_argument("--sample-pixels", default=2_000_000, type=int, help="Pixels sampled per modality to build CDFs")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    wf_dir = data_dir / "splits" / args.split / "wf"
    tp_dir = data_dir / "splits" / args.split / "2p"
    assert wf_dir.exists() and tp_dir.exists(), f"Missing {wf_dir} or {tp_dir}"

    wf_map = _load_png_map(wf_dir)
    tp_map = _load_png_map(tp_dir)
    names = sorted(set(wf_map.keys()) & set(tp_map.keys()))
    if not names:
        raise SystemExit("No matching WF/2P PNG file pairs found.")

    x_images = [tp_map[n] for n in names]
    y_images = [wf_map[n] for n in names]

    if args.enable_quantile_matching:
        rng = np.random.default_rng(0)
        wf_vals = _sample_values(y_images, max_pixels=int(args.sample_pixels), rng=rng)
        tp_vals = _sample_values(x_images, max_pixels=int(args.sample_pixels), rng=rng)
        ref_q = _compute_shared_reference(wf_vals, tp_vals, num_bins=int(args.num_bins))
        wf_src, wf_ref = _compute_mapping_to_reference(wf_vals, ref_q, num_bins=int(args.num_bins))
        tp_src, tp_ref = _compute_mapping_to_reference(tp_vals, ref_q, num_bins=int(args.num_bins))

        # Apply mapping per image
        y_images = [_apply_piecewise_linear(im, wf_src, wf_ref) for im in y_images]
        x_images = [_apply_piecewise_linear(im, tp_src, tp_ref) for im in x_images]

    # Write Zarr patches (x=2P clean, y=WF noisy)
    write_zarr_patches(x_images, y_images, args.out, tile_size=int(args.tile_size), overlap=int(args.overlap))
    print(f"[OK] Wrote Zarr patches to {args.out} with N={len(x_images)} images")


if __name__ == "__main__":
    main()


