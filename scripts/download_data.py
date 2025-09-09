#!/usr/bin/env python
import argparse
from pathlib import Path

from pkl_dg.data.downloaders import (
    download_imagenet_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets (ImageNet subset)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--skip-imagenet", action="store_true", help="Skip ImageNet subset download")
    # BioTISR and Allen Observatory removed
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_imagenet:
        out = download_imagenet_subset(raw_root)
        print(f"[OK] ImageNet subset ready at: {out}")

    # No additional datasets


if __name__ == "__main__":
    main()


