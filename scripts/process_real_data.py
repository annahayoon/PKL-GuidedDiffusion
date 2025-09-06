#!/usr/bin/env python3
"""
Process real microscopy data into usable format for PKL diffusion training.

Handles:
1. WF/2P paired data (tp_reg.tif and wf.tif)
2. Bead z-stack data with and without adaptive optics
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


def normalize_to_uint8(data: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """Normalize data to uint8 with percentile clipping."""
    # Clip extreme values
    low, high = np.percentile(data, [100 - percentile_clip, percentile_clip])
    data_clipped = np.clip(data, low, high)
    
    # Normalize to [0, 255]
    data_norm = (data_clipped - data_clipped.min()) / (data_clipped.max() - data_clipped.min())
    return (data_norm * 255).astype(np.uint8)


def extract_patches(image: np.ndarray, patch_size: int = 256, stride: int = 128, 
                   min_intensity_threshold: float = 0.1) -> list:
    """Extract patches from large images with overlap."""
    patches = []
    h, w = image.shape
    
    for y in tqdm(range(0, h - patch_size + 1, stride), desc="Extracting patches", leave=False):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Skip patches that are too dark (likely background)
            if patch.mean() > min_intensity_threshold * image.max():
                patches.append(patch)
    
    return patches


def process_wf_2p_pairs(wf_path: str, tp_path: str, output_dir: Path, 
                       patch_size: int = 256) -> None:
    """Process WF/2P paired data into training format."""
    print("Processing WF/2P paired data...")
    
    # Load data
    wf_stack = tifffile.imread(wf_path)  # Shape: (51, 2048, 2048)
    tp_stack = tifffile.imread(tp_path)  # Shape: (51, 2048, 2048)
    
    print(f"WF stack shape: {wf_stack.shape}")
    print(f"2P stack shape: {tp_stack.shape}")
    
    # Create output directories
    wf_dir = output_dir / "real_pairs" / "wf"
    tp_dir = output_dir / "real_pairs" / "2p"
    wf_dir.mkdir(parents=True, exist_ok=True)
    tp_dir.mkdir(parents=True, exist_ok=True)
    
    patch_count = 0
    
    for frame_idx in tqdm(range(wf_stack.shape[0]), desc="Processing frames"):
        wf_frame = wf_stack[frame_idx]
        tp_frame = tp_stack[frame_idx]
        
        # Normalize frames
        wf_norm = normalize_to_uint8(wf_frame)
        tp_norm = normalize_to_uint8(tp_frame)
        
        # Extract patches
        wf_patches = extract_patches(wf_norm, patch_size)
        tp_patches = extract_patches(tp_norm, patch_size)
        
        # Save corresponding patches
        min_patches = min(len(wf_patches), len(tp_patches))
        for patch_idx in tqdm(range(min_patches), desc="Saving patches", leave=False):
            wf_patch = wf_patches[patch_idx]
            tp_patch = tp_patches[patch_idx]
            
            # Save patches
            wf_filename = f"frame_{frame_idx:03d}_patch_{patch_idx:03d}.png"
            tp_filename = f"frame_{frame_idx:03d}_patch_{patch_idx:03d}.png"
            
            Image.fromarray(wf_patch).save(wf_dir / wf_filename)
            Image.fromarray(tp_patch).save(tp_dir / tp_filename)
            
            patch_count += 1
    
    print(f"Processed {patch_count} WF/2P patch pairs")
    
    # Save metadata
    metadata = {
        "source_wf": str(wf_path),
        "source_2p": str(tp_path),
        "total_frames": wf_stack.shape[0],
        "patch_size": patch_size,
        "total_patches": patch_count,
        "original_shape": f"{wf_stack.shape[1]}x{wf_stack.shape[2]}",
        "data_type": "wf_2p_pairs"
    }
    
    with open(output_dir / "real_pairs" / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def process_bead_data(beads_dir: str, output_dir: Path, z_projection: str = "max") -> None:
    """Process bead z-stack data."""
    print("Processing bead data...")
    
    beads_path = Path(beads_dir)
    bead_files = list(beads_path.glob("*.tif"))
    
    # Create output directory
    bead_output_dir = output_dir / "beads"
    bead_output_dir.mkdir(parents=True, exist_ok=True)
    
    for bead_file in tqdm(bead_files, desc="Processing bead files"):
        print(f"Processing {bead_file.name}...")
        
        # Load z-stack
        bead_stack = tifffile.imread(str(bead_file))  # Shape: (241, 256, 256)
        print(f"Bead stack shape: {bead_stack.shape}")
        
        # Determine output name
        if "after_AO" in bead_file.name:
            output_name = "bead_with_AO"
        else:
            output_name = "bead_no_AO"
        
        # Create z-projection
        if z_projection == "max":
            projection = np.max(bead_stack, axis=0)
        elif z_projection == "mean":
            projection = np.mean(bead_stack, axis=0)
        elif z_projection == "sum":
            projection = np.sum(bead_stack, axis=0)
        else:
            raise ValueError(f"Unknown projection type: {z_projection}")
        
        # Normalize and save projection
        projection_norm = normalize_to_uint8(projection)
        Image.fromarray(projection_norm).save(bead_output_dir / f"{output_name}_projection.png")
        
        # Save middle slice as well
        middle_slice = bead_stack[bead_stack.shape[0] // 2]
        middle_norm = normalize_to_uint8(middle_slice)
        Image.fromarray(middle_norm).save(bead_output_dir / f"{output_name}_middle_slice.png")
        
        # Save multiple z-slices for analysis
        z_slices_dir = bead_output_dir / f"{output_name}_slices"
        z_slices_dir.mkdir(exist_ok=True)
        
        # Save every 10th slice
        for z_idx in tqdm(range(0, bead_stack.shape[0], 10), desc="Saving z-slices", leave=False):
            slice_data = bead_stack[z_idx]
            slice_norm = normalize_to_uint8(slice_data)
            z_pos_um = z_idx * 0.1  # 0.1 μm steps
            Image.fromarray(slice_norm).save(
                z_slices_dir / f"z_{z_pos_um:.1f}um_slice_{z_idx:03d}.png"
            )
    
    # Save bead metadata
    metadata = {
        "source_dir": str(beads_dir),
        "z_step_um": 0.1,
        "total_z_slices": 241,
        "z_range_um": f"0.0 - {240 * 0.1:.1f}",
        "xy_size": "256x256",
        "projection_type": z_projection,
        "data_type": "bead_z_stacks"
    }
    
    with open(bead_output_dir / "metadata.txt", "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")


def create_training_splits(output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Create train/val/test splits from processed data, grouping by frames."""
    print("Creating frame-based training splits...")
    
    # Process WF/2P pairs
    pairs_dir = output_dir / "real_pairs"
    if pairs_dir.exists():
        wf_files = sorted(list((pairs_dir / "wf").glob("*.png")))
        tp_files = sorted(list((pairs_dir / "2p").glob("*.png")))
        
        assert len(wf_files) == len(tp_files), "Mismatch in WF/2P file counts"
        
        # Group files by frame number
        from collections import defaultdict
        frame_groups = defaultdict(list)
        
        for wf_file, tp_file in zip(wf_files, tp_files):
            # Extract frame number from filename (e.g., "frame_000_patch_001.png" -> 0)
            frame_num = int(wf_file.name.split('_')[1])
            frame_groups[frame_num].append((wf_file, tp_file))
        
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
        
        print(f"Frame-based split:")
        print(f"  Frames: {n_train_frames} train, {n_val_frames} val, {n_test_frames} test")
        print(f"  Patches: {train_patches} train, {val_patches} val, {test_patches} test")
        print(f"  Train frames: {sorted(train_frames)}")
        print(f"  Val frames: {sorted(val_frames)}")
        print(f"  Test frames: {sorted(test_frames)}")
        
        # Create split directories
        for split in ["train", "val", "test"]:
            for modality in ["wf", "2p"]:
                split_dir = output_dir / f"splits/{split}/{modality}"
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
                wf_dst = output_dir / f"splits/{split}/wf" / wf_src.name
                tp_dst = output_dir / f"splits/{split}/2p" / tp_src.name
                
                # Create symlinks to save space
                if wf_dst.exists():
                    wf_dst.unlink()
                if tp_dst.exists():
                    tp_dst.unlink()
                    
                wf_dst.symlink_to(wf_src.absolute())
                tp_dst.symlink_to(tp_src.absolute())


def main():
    parser = argparse.ArgumentParser(description="Process real microscopy data")
    parser.add_argument("--wf-path", default="../data_wf_tp/wf.tif", 
                       help="Path to WF data")
    parser.add_argument("--tp-path", default="../data_wf_tp/tp_reg.tif",
                       help="Path to 2P data") 
    parser.add_argument("--beads-dir", default="../data_wf_tp/beads",
                       help="Directory with bead data")
    parser.add_argument("--output-dir", default="./data/real_microscopy",
                       help="Output directory for processed data")
    parser.add_argument("--patch-size", type=int, default=256,
                       help="Size of extracted patches")
    parser.add_argument("--z-projection", choices=["max", "mean", "sum"], 
                       default="max", help="Z-projection method for beads")
    parser.add_argument("--create-splits", action="store_true",
                       help="Create train/val/test splits")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process WF/2P pairs
    if os.path.exists(args.wf_path) and os.path.exists(args.tp_path):
        process_wf_2p_pairs(args.wf_path, args.tp_path, output_dir, args.patch_size)
    else:
        print(f"Warning: WF/2P data not found at {args.wf_path} or {args.tp_path}")
    
    # Process bead data
    if os.path.exists(args.beads_dir):
        process_bead_data(args.beads_dir, output_dir, args.z_projection)
    else:
        print(f"Warning: Bead data not found at {args.beads_dir}")
    
    # Create splits if requested
    if args.create_splits:
        create_training_splits(output_dir)
    
    print(f"\nProcessing complete! Data saved to: {output_dir}")
    print(f"Directory structure:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    main()
