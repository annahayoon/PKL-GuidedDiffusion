#!/usr/bin/env python3
"""
Create the final denoised_frame010.png using ALL 225 denoised patches.
No original WF patches in the middle panel - only denoised reconstructions.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import re
import tifffile
from typing import Dict, Tuple
from scipy.ndimage import gaussian_filter

def normalize_to_uint8(data: np.ndarray, percentile_clip: float = 99.5) -> np.ndarray:
    """Normalize data to uint8 range with percentile clipping."""
    data_norm = np.clip(data, 0, np.percentile(data, percentile_clip))
    if data_norm.max() > data_norm.min():
        data_norm = (data_norm - data_norm.min()) / (data_norm.max() - data_norm.min())
    return (data_norm * 255).astype(np.uint8)

def reconstruct_complete_patches(patches: Dict[int, np.ndarray], 
                               original_image: np.ndarray,
                               patch_size: int = 256, 
                               stride: int = 128) -> np.ndarray:
    """
    Reconstruct image using ALL patches with seamless blending.
    """
    
    # Calculate expected grid dimensions
    patches_y = (original_image.shape[0] - patch_size) // stride + 1
    patches_x = (original_image.shape[1] - patch_size) // stride + 1
    
    print(f"Expected grid: {patches_y} rows × {patches_x} cols = {patches_y * patches_x} patches")
    print(f"Available patches: {len(patches)}")
    
    # Initialize canvas
    canvas = np.zeros_like(original_image, dtype=np.float32)
    weight_map = np.zeros_like(original_image, dtype=np.float32)
    
    # Process all patches
    for patch_id, patch_data in patches.items():
        row = patch_id // patches_x
        col = patch_id % patches_x
        
        # Calculate position in the original image
        y_start = row * stride
        x_start = col * stride
        y_end = y_start + patch_size
        x_end = x_start + patch_size
        
        # Create feathering weights for seamless blending
        patch_weight = np.ones((patch_size, patch_size), dtype=np.float32)
        
        # Feather the edges to create smooth blending
        feather_size = stride // 2  # Feather over half the overlap region
        
        # Top edge feathering
        if row > 0:
            patch_weight[:feather_size, :] *= np.linspace(0, 1, feather_size)[:, np.newaxis]
        
        # Bottom edge feathering
        if row < patches_y - 1:
            patch_weight[-feather_size:, :] *= np.linspace(1, 0, feather_size)[:, np.newaxis]
        
        # Left edge feathering
        if col > 0:
            patch_weight[:, :feather_size] *= np.linspace(0, 1, feather_size)[np.newaxis, :]
        
        # Right edge feathering
        if col < patches_x - 1:
            patch_weight[:, -feather_size:] *= np.linspace(1, 0, feather_size)[np.newaxis, :]
        
        # Add patch to canvas
        canvas[y_start:y_end, x_start:x_end] += patch_data * patch_weight
        weight_map[y_start:y_end, x_start:x_end] += patch_weight
    
    # Normalize by weights
    mask = weight_map > 0
    canvas[mask] = canvas[mask] / weight_map[mask]
    
    # Apply slight smoothing to reduce any remaining artifacts
    canvas = gaussian_filter(canvas, sigma=0.5)
    
    return canvas

def load_patches(directory: Path, pattern: str) -> Dict[int, np.ndarray]:
    """Load patches from directory and return as dictionary indexed by patch number."""
    patches = {}
    
    for patch_file in directory.glob(pattern):
        # Extract patch number from filename
        match = re.search(r'patch_(\d+)', patch_file.name)
        if match:
            patch_num = int(match.group(1))
            img = Image.open(patch_file)
            patches[patch_num] = np.array(img, dtype=np.float32)
    
    return patches

def main():
    # Paths
    base_dir = Path("/home/jilab/anna_OS_ML/PKL-DiffusionDenoising")
    complete_patches_dir = base_dir / "data/real_microscopy/complete_patches"
    wf_dir = complete_patches_dir / "wf"
    gt_dir = complete_patches_dir / "2p"
    denoised_dir = base_dir / "outputs/frame010_complete_inference"
    
    # Load original frames for normalization reference
    print("Loading original frames...")
    wf_stack = tifffile.imread('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/../data_wf_tp/wf.tif')
    tp_stack = tifffile.imread('/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/../data_wf_tp/tp_reg.tif')
    
    wf_original = wf_stack[10]  # frame_010
    tp_original = tp_stack[10]   # frame_010
    
    # Normalize original frames
    wf_original_norm = normalize_to_uint8(wf_original)
    tp_original_norm = normalize_to_uint8(tp_original)
    
    print("Loading complete patches...")
    
    # Load ALL WF patches
    wf_patches = load_patches(wf_dir, "frame_010_patch_*.png")
    print(f"Loaded {len(wf_patches)} WF patches")
    
    # Load ALL GT patches
    gt_patches = load_patches(gt_dir, "frame_010_patch_*.png")
    print(f"Loaded {len(gt_patches)} GT patches")
    
    # Load ALL denoised patches (now complete!)
    denoised_patches = load_patches(denoised_dir, "frame_010_patch_*_reconstructed.png")
    print(f"Loaded {len(denoised_patches)} denoised patches")
    
    # Verify we have all 225 denoised patches
    if len(denoised_patches) != 225:
        print(f"ERROR: Expected 225 denoised patches, got {len(denoised_patches)}")
        return
    
    # Reconstruct using complete patches
    print("Reconstructing WF patches with complete coverage...")
    wf_full = reconstruct_complete_patches(wf_patches, wf_original_norm.astype(np.float32))
    wf_full_u8 = normalize_to_uint8(wf_full)
    
    print("Reconstructing GT patches with complete coverage...")
    gt_full = reconstruct_complete_patches(gt_patches, tp_original_norm.astype(np.float32))
    gt_full_u8 = normalize_to_uint8(gt_full)
    
    print("Reconstructing denoised patches with complete coverage...")
    denoised_full = reconstruct_complete_patches(denoised_patches, wf_original_norm.astype(np.float32))
    denoised_full_u8 = normalize_to_uint8(denoised_full)
    
    # Create composite: WF | Denoised | GT
    print("Creating final composite...")
    composite = np.concatenate([wf_full_u8, denoised_full_u8, gt_full_u8], axis=1)
    
    # Save as the final denoised_frame010.png
    output_path = base_dir / "denoised_frame010.png"
    Image.fromarray(composite).save(output_path)
    print(f"Saved final denoised_frame010.png to {output_path}")
    print(f"Composite size: {composite.shape}")
    print(f"Each panel size: {composite.shape[0]}x{composite.shape[1]//3}")
    
    print("Done! Final denoised_frame010.png created with ALL 225 denoised patches.")

if __name__ == "__main__":
    main()
