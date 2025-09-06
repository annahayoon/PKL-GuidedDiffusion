#!/usr/bin/env python3
"""
Create frame-based train/val/test splits for existing real microscopy data.

This script can be used to re-create splits from already processed data
without having to re-process the raw microscopy files.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def create_frame_based_splits(data_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                             random_seed: int = 42):
    """Create train/val/test splits from processed data, grouping by frames.
    
    Args:
        data_dir: Directory containing real_pairs/wf and real_pairs/2p subdirectories
        train_ratio: Fraction of frames for training (default: 0.8)
        val_ratio: Fraction of frames for validation (default: 0.1)
        random_seed: Random seed for reproducible splits (default: 42)
    """
    print("Creating frame-based training splits...")
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Process WF/2P pairs
    pairs_dir = data_dir / "real_pairs"
    if not pairs_dir.exists():
        raise ValueError(f"Data directory not found: {pairs_dir}")
        
    wf_dir = pairs_dir / "wf"
    tp_dir = pairs_dir / "2p"
    
    if not wf_dir.exists() or not tp_dir.exists():
        raise ValueError(f"WF or 2P directories not found: {wf_dir}, {tp_dir}")
    
    wf_files = sorted(list(wf_dir.glob("*.png")))
    tp_files = sorted(list(tp_dir.glob("*.png")))
    
    if len(wf_files) == 0 or len(tp_files) == 0:
        raise ValueError("No PNG files found in WF or 2P directories")
    
    assert len(wf_files) == len(tp_files), f"Mismatch in WF/2P file counts: {len(wf_files)} vs {len(tp_files)}"
    
    # Group files by frame number
    frame_groups = defaultdict(list)
    
    for wf_file, tp_file in zip(wf_files, tp_files):
        # Extract frame number from filename (e.g., "frame_000_patch_001.png" -> 0)
        try:
            frame_num = int(wf_file.name.split('_')[1])
            frame_groups[frame_num].append((wf_file, tp_file))
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse frame number from {wf_file.name}, skipping")
            continue
    
    if len(frame_groups) == 0:
        raise ValueError("No valid frame groups found. Check filename format (expected: frame_XXX_patch_YYY.png)")
    
    # Get unique frame numbers and shuffle them (not individual patches)
    frame_numbers = sorted(frame_groups.keys())
    np.random.shuffle(frame_numbers)
    
    n_frames = len(frame_numbers)
    n_train_frames = int(n_frames * train_ratio)
    n_val_frames = int(n_frames * val_ratio)
    n_test_frames = n_frames - n_train_frames - n_val_frames
    
    train_frames = frame_numbers[:n_train_frames]
    val_frames = frame_numbers[n_train_frames:n_train_frames + n_val_frames]
    test_frames = frame_numbers[n_train_frames + n_val_frames:]
    
    # Count total patches for reporting
    train_patches = sum(len(frame_groups[f]) for f in train_frames)
    val_patches = sum(len(frame_groups[f]) for f in val_frames)
    test_patches = sum(len(frame_groups[f]) for f in test_frames)
    
    print(f"Frame-based split (seed={random_seed}):")
    print(f"  Frames: {n_train_frames} train, {n_val_frames} val, {n_test_frames} test")
    print(f"  Patches: {train_patches} train, {val_patches} val, {test_patches} test")
    print(f"  Train frames: {sorted(train_frames)}")
    print(f"  Val frames: {sorted(val_frames)}")
    print(f"  Test frames: {sorted(test_frames)}")
    
    # Remove existing splits directory if it exists
    splits_dir = data_dir / "splits"
    if splits_dir.exists():
        print(f"Removing existing splits directory: {splits_dir}")
        import shutil
        shutil.rmtree(splits_dir)
    
    # Create split directories
    for split in ["train", "val", "test"]:
        for modality in ["wf", "2p"]:
            split_dir = splits_dir / split / modality
            split_dir.mkdir(parents=True, exist_ok=True)
    
    # Assign frames to splits and copy files
    frame_to_split = {}
    for frame in train_frames:
        frame_to_split[frame] = "train"
    for frame in val_frames:
        frame_to_split[frame] = "val"
    for frame in test_frames:
        frame_to_split[frame] = "test"
    
    # Copy all patches for each frame to the appropriate split
    for frame_num, file_pairs in tqdm(frame_groups.items(), desc="Creating splits"):
        split = frame_to_split[frame_num]
        
        for wf_src, tp_src in file_pairs:
            wf_dst = splits_dir / split / "wf" / wf_src.name
            tp_dst = splits_dir / split / "2p" / tp_src.name
            
            # Create symlinks to save space
            wf_dst.symlink_to(wf_src.absolute())
            tp_dst.symlink_to(tp_src.absolute())
    
    # Save split metadata
    metadata = {
        "split_type": "frame_based",
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "total_frames": n_frames,
        "train_frames": n_train_frames,
        "val_frames": n_val_frames,
        "test_frames": n_test_frames,
        "total_patches": len(wf_files),
        "train_patches": train_patches,
        "val_patches": val_patches,
        "test_patches": test_patches,
        "train_frame_list": sorted(train_frames),
        "val_frame_list": sorted(val_frames),
        "test_frame_list": sorted(test_frames)
    }
    
    with open(splits_dir / "split_metadata.txt", "w") as f:
        f.write("# Frame-based train/val/test split metadata\n")
        f.write("# This ensures patches from the same frame stay in the same split\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nFrame-based splits created successfully!")
    print(f"Split metadata saved to: {splits_dir / 'split_metadata.txt'}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Create frame-based splits for real microscopy data")
    parser.add_argument("--data-dir", default="./data/real_microscopy",
                       help="Directory containing real_pairs subdirectory")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Fraction of frames for training (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                       help="Fraction of frames for validation (default: 0.1)")
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducible splits (default: 42)")
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio cannot exceed 1.0")
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise ValueError(f"Data directory does not exist: {data_dir}")
    
    print(f"Creating frame-based splits for data in: {data_dir}")
    print(f"Split ratios - Train: {args.train_ratio:.1f}, Val: {args.val_ratio:.1f}, Test: {test_ratio:.1f}")
    print(f"Random seed: {args.random_seed}")
    
    metadata = create_frame_based_splits(
        data_dir=data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_seed=args.random_seed
    )
    
    print("\nSplit creation complete!")


if __name__ == "__main__":
    main()
