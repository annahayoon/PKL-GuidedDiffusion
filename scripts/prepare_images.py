#!/usr/bin/env python
import argparse
from pathlib import Path

from pkl_dg.data.downloaders import prepare_image_folders


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate raw images into train/val folders")
    parser.add_argument("--raw-dirs", type=str, nargs="+", help="List of raw dataset directories")
    parser.add_argument("--out-dir", type=str, default="data/images", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Training split ratio")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_roots = [Path(p) for p in args.raw_dirs]
    out = prepare_image_folders(raw_roots, args.out_dir, train_ratio=args.train_ratio)
    print(f"[OK] Prepared images at: {out}")


if __name__ == "__main__":
    main()


