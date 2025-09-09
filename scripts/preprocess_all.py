#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from pkl_dg.data.downloaders import (
    download_imagenet_subset,
    prepare_image_folders,
)


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end dataset preprocessing")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Project root")
    parser.add_argument("--data-dir", type=str, default=None, help="Data root (defaults to <root>/data)")
    # BioTISR/Allen options removed
    parser.add_argument("--skip-download", action="store_true", help="Skip downloads and use existing raw data")
    parser.add_argument("--skip-synthesize", action="store_true", help="Skip synthesis stage")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-intensity", type=float, default=0)
    parser.add_argument("--max-intensity", type=float, default=1000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--sample", type=int, default=0, help="If >0, subsample N images per split for a fast check")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    data_root = Path(args.data_dir) if args.data_dir else (root / "data")
    raw_root = data_root / "raw"
    images_root = data_root

    raw_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        imagenet_root = download_imagenet_subset(raw_root)
        print(f"[OK] Imagenette subset at: {imagenet_root}")

        # Only Imagenet subset is downloaded here

    # Aggregate all available raw sources
    raw_dirs = []
    imagenet_dir = raw_root / "imagenet_subset" / "imagenette2-320"
    if imagenet_dir.exists():
        raw_dirs += [imagenet_dir / "train", imagenet_dir / "val"]
    # Only include imagenet-derived dirs

    prepare_image_folders([Path(p) for p in raw_dirs], images_root, train_ratio=args.train_ratio)
    print(f"[OK] Aggregated raw images into: {images_root}")

    if args.sample > 0:
        # Optionally keep only a subset per split for quick validation
        for split in tqdm(["train", "val"], desc="Subsampling splits"):
            split_dir = images_root / split / "classless"
            if split_dir.exists():
                keep = sorted(list(split_dir.glob("*")))[: args.sample]
                files_to_remove = sorted(split_dir.glob("*"))[args.sample :]
                for f in tqdm(files_to_remove, desc=f"Removing {split} files", leave=False):
                    try:
                        f.unlink()
                    except Exception:
                        pass
                print(f"[OK] Subsampled {split} to {len(keep)} files")

    if not args.skip_synthesize:
        # Use existing CLI to synthesize and save .pt pairs
        synth_script = root / "scripts" / "synthesize_data.py"
        for split in tqdm(["train", "val"], desc="Synthesizing data"):
            source_dir = images_root / split / "classless"
            if not source_dir.exists():
                print(f"[WARN] Missing source dir: {source_dir}")
                continue
            out_dir = data_root / "synth"
            cmd = (
                f"python3 {synth_script} --source-dir {source_dir} --output-dir {out_dir} "
                f"--image-size {args.image_size} --min-intensity {args.min_intensity} "
                f"--max-intensity {args.max_intensity} --mode {split}"
            )
            print(f"[RUN] {cmd}")
            ret = os.system(cmd)
            if ret != 0:
                raise SystemExit(f"Synthesis failed for {split} with code {ret}")

    # Verify counts and a sample load
    synth_root = data_root / "synth"
    total_x, total_y = 0, 0
    for split in tqdm(["train", "val"], desc="Verifying counts"):
        x_dir = synth_root / split / "x2p"
        y_dir = synth_root / split / "ywf"
        x_count = len(list(x_dir.glob("*.pt"))) if x_dir.exists() else 0
        y_count = len(list(y_dir.glob("*.pt"))) if y_dir.exists() else 0
        total_x += x_count
        total_y += y_count
        print(f"[OK] {split}: x2p={x_count}, ywf={y_count}")

    # Try loading one sample tensor for sanity
    sample_path = None
    for split in ["train", "val"]:
        cand = list((synth_root / split / "x2p").glob("*.pt"))
        if cand:
            sample_path = cand[0]
            break
    if sample_path is not None:
        t = torch.load(sample_path, map_location="cpu")
        print(f"[OK] Sample tensor loaded: shape={tuple(t.shape)} dtype={t.dtype}")
    else:
        print("[WARN] No synthesized tensors found to load.")

    print(f"[DONE] Synthesized totals: x2p={total_x}, ywf={total_y}")


if __name__ == "__main__":
    main()


