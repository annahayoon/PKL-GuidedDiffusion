#!/usr/bin/env python3
import os
import sys
import glob
from typing import Tuple

import numpy as np
from PIL import Image
import tifffile


def to_uint8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, (1, 99))
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = np.clip(a, 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def read_tif(path: str) -> np.ndarray:
    return tifffile.imread(path)


def main() -> None:
    if len(sys.argv) < 5:
        print('usage: make_previews.py <wf_tif_dir> <pred_dir> <gt_tif_dir> <out_dir> [max_n]')
        sys.exit(1)

    wf_dir = sys.argv[1]
    pred_dir = sys.argv[2]
    gt_dir = sys.argv[3]
    out_dir = sys.argv[4]
    max_n = int(sys.argv[5]) if len(sys.argv) > 5 else 24

    os.makedirs(out_dir, exist_ok=True)
    wf_files = sorted(glob.glob(os.path.join(wf_dir, '*.tif')))
    if max_n > 0:
        wf_files = wf_files[:max_n]

    count = 0
    for wf_path in wf_files:
        stem = os.path.splitext(os.path.basename(wf_path))[0]
        # Support both tif and png reconstructions
        pred_path = os.path.join(pred_dir, f'{stem}_reconstructed.tif')
        if not os.path.exists(pred_path):
            alt = os.path.join(pred_dir, f'{stem}_reconstructed.png')
            if os.path.exists(alt):
                pred_path = alt
        if not os.path.exists(pred_path):
            continue
        wf = to_uint8(read_tif(wf_path))
        pr = to_uint8(read_tif(pred_path))
        panels = [wf, pr]
        gt_path = os.path.join(gt_dir, f'{stem}.tif')
        if os.path.exists(gt_path):
            gt = to_uint8(read_tif(gt_path))
            panels.append(gt)
        strip = np.concatenate(panels, axis=1)
        Image.fromarray(strip).save(os.path.join(out_dir, f'{stem}_wf_pred_gt.png'))
        count += 1

    print(f'Generated {count} preview(s) in {out_dir}')


if __name__ == '__main__':
    main()


