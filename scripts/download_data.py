#!/usr/bin/env python
import argparse
from pathlib import Path

from pkl_dg.data.downloaders import (
    download_imagenet_subset,
    download_biotisr,
    download_allen_observatory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets (ImageNet subset, BioTISR placeholder, Allen Observatory guidance)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--skip-imagenet", action="store_true", help="Skip ImageNet subset download")
    parser.add_argument("--biotisr-url", type=str, default=None, help="Optional direct URL to BioTISR archive")
    parser.add_argument("--skip-biotisr", action="store_true", help="Skip BioTISR")
    parser.add_argument("--skip-allen", action="store_true", help="Skip Allen Observatory setup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_imagenet:
        out = download_imagenet_subset(raw_root)
        print(f"[OK] ImageNet subset ready at: {out}")

    if not args.skip_biotisr:
        out = download_biotisr(raw_root, url=args.biotisr_url)
        print(f"[OK] BioTISR root ready at: {out}")

    if not args.skip_allen:
        out = download_allen_observatory(raw_root)
        print(f"[OK] Allen Observatory root ready at: {out}")


if __name__ == "__main__":
    main()


