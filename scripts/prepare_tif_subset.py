#!/usr/bin/env python3
import os
import sys
import glob
from typing import List

import numpy as np
from PIL import Image
import tifffile


def list_pngs(directory: str, max_n: int) -> List[str]:
    files = sorted(glob.glob(os.path.join(directory, '*.png')))
    if max_n > 0:
        files = files[:max_n]
    return files


def save_tif(array: np.ndarray, out_path: str) -> None:
    array = array.astype(np.float32)
    tifffile.imwrite(out_path, array)


def main() -> None:
    if len(sys.argv) < 6:
        print(
            'usage: prepare_tif_subset.py <src_wf_png_dir> <src_2p_png_dir> <out_wf_tif_dir> <out_2p_tif_dir> <max_n>'
        )
        sys.exit(1)

    src_wf = sys.argv[1]
    src_2p = sys.argv[2]
    out_wf = sys.argv[3]
    out_2p = sys.argv[4]
    max_n = int(sys.argv[5])

    os.makedirs(out_wf, exist_ok=True)
    os.makedirs(out_2p, exist_ok=True)

    wf_pngs = list_pngs(src_wf, max_n)
    converted = 0
    for wf_path in wf_pngs:
        base = os.path.basename(wf_path)
        stem, _ = os.path.splitext(base)
        gt_png = os.path.join(src_2p, base)

        wf_img = Image.open(wf_path)
        wf_arr = np.array(wf_img)
        save_tif(wf_arr, os.path.join(out_wf, f'{stem}.tif'))

        if os.path.exists(gt_png):
            gt_img = Image.open(gt_png)
            gt_arr = np.array(gt_img)
            save_tif(gt_arr, os.path.join(out_2p, f'{stem}.tif'))
        converted += 1

    print(f'Converted {converted} WF images to TIF under {out_wf}')
    print(f'GT TIFs (if available) under {out_2p}')


if __name__ == '__main__':
    main()


