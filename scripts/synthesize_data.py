#!/usr/bin/env python
import argparse
from pathlib import Path
import hashlib

import torch
from tqdm import tqdm

from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel


def parse_args():
    parser = argparse.ArgumentParser(description="Synthesize WF/2P-like training pairs to disk")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory containing natural/microscopy images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for pairs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-intensity", type=float, default=0)
    parser.add_argument("--max-intensity", type=float, default=1000)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_x = out_root / args.mode / "x2p"
    out_y = out_root / args.mode / "ywf"
    out_x.mkdir(parents=True, exist_ok=True)
    out_y.mkdir(parents=True, exist_ok=True)

    # Minimal PSF and forward model
    psf = PSF()
    forward_model = ForwardModel(psf=psf.to_torch(device="cpu"), background=0.0, device="cpu")

    transform = IntensityToModel(minIntensity=args.min_intensity, maxIntensity=args.max_intensity)

    dataset = SynthesisDataset(
        source_dir=args.source_dir,
        forward_model=forward_model,
        transform=transform,
        image_size=args.image_size,
        mode=args.mode,
    )

    # Build stable names based on source image paths to ensure idempotency
    source_paths = dataset.image_paths
    for idx in tqdm(range(len(dataset)), desc="Synthesizing data"):
        src = str(source_paths[idx])
        try:
            abs_path = str(Path(src).resolve())
        except Exception:
            abs_path = src
        stem = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]
        x_path = out_x / f"x_{stem}.pt"
        y_path = out_y / f"y_{stem}.pt"
        if x_path.exists() and y_path.exists():
            continue
        x2p, ywf = dataset[idx]
        if not x_path.exists():
            torch.save(x2p, x_path)
        if not y_path.exists():
            torch.save(ywf, y_path)

    print(f"[OK] Wrote {len(dataset)} pairs under: {out_root}")


if __name__ == "__main__":
    main()


