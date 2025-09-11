#!/usr/bin/env python3
"""
Full FOV baseline comparison script.
Reconstructs complete frames from patches and compares WF | PKL | L2 | Anscombe | GT.
Based on patch_denoised_inference.py but with multiple guidance methods.
"""

import os
import numpy as np
from pathlib import Path
import re
import tifffile
from typing import Dict, Tuple
from scipy.ndimage import gaussian_filter
from PIL import Image
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.psf_estimator import build_psf_bank
from pkl_dg.guidance.pkl import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel


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
    """Reconstruct image using ALL patches with seamless blending."""
    
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


def _load_model_and_sampler(cfg: DictConfig, guidance_type: str, ddim_steps: int = 100):
    """Load model and create sampler for specific guidance type."""
    device = str(cfg.experiment.device)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    if use_conditioning and conditioning_type == "wf" and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2
    unet = DenoisingUNet(model_cfg)
    ddpm = DDPMTrainer(model=unet, config=OmegaConf.to_container(cfg.training, resolve=True))
    checkpoint_path = Path(str(cfg.inference.checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(state_dict, strict=False)
    ddpm.eval().to(device)
    
    # Forward model
    forward_model = None
    try:
        phys_cfg = cfg.physics
        use_psf = bool(getattr(phys_cfg, "use_psf", False))
        background = float(getattr(phys_cfg, "background", 0.0))
        if use_psf:
            psf_path = getattr(phys_cfg, "psf_path", None)
            use_bead = bool(getattr(phys_cfg, "use_bead_psf", False))
            if use_bead:
                beads_dir = str(getattr(phys_cfg, "beads_dir", ""))
                bank = build_psf_bank(beads_dir)
                mode = getattr(phys_cfg, "bead_mode", None)
                if mode is None:
                    psf_t = bank.get("with_AO", bank.get("no_AO"))
                else:
                    psf_t = bank.get(str(mode), next(iter(bank.values())))
                psf = psf_t.to(device=device, dtype=torch.float32)
                if psf.ndim == 2:
                    psf = psf.unsqueeze(0).unsqueeze(0)
            elif psf_path is not None:
                psf = PSF(psf_path=str(psf_path)).to_torch(device=device)
            else:
                psf = PSF().to_torch(device=device)
            read_noise_sigma = float(getattr(phys_cfg, "read_noise_sigma", 0.0))
            forward_model = ForwardModel(psf=psf, background=background, device=device, read_noise_sigma=read_noise_sigma)
    except Exception:
        forward_model = None
    
    # Guidance
    if guidance_type == "pkl":
        guidance = PKLGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    elif guidance_type == "l2":
        guidance = L2Guidance()
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
    
    # Schedule
    schedule_cfg = getattr(cfg.guidance, "schedule", {})
    schedule = AdaptiveSchedule(
        lambda_base=float(getattr(cfg.guidance, "lambda_base", 0.1)),
        T_threshold=int(getattr(schedule_cfg, "T_threshold", 800)),
        epsilon_lambda=float(getattr(schedule_cfg, "epsilon_lambda", 1e-3)),
        T_total=int(cfg.training.num_timesteps),
    )
    
    # Transform
    transform = IntensityToModel(
        min_intensity=float(cfg.data.min_intensity),
        max_intensity=float(cfg.data.max_intensity),
    )
    
    # Sampler
    sampler = DDIMSampler(
        model=ddpm,
        forward_model=forward_model,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=int(cfg.training.num_timesteps),
        ddim_steps=ddim_steps,
        eta=float(cfg.inference.eta),
        use_autocast=bool(getattr(cfg.inference, "use_autocast", True)),
    )
    
    return sampler


def process_patches_with_guidance(sampler, patches: Dict[int, np.ndarray], device: str) -> Dict[int, np.ndarray]:
    """Process patches through the sampler and return denoised patches."""
    denoised_patches = {}
    conditioning_type = str(getattr(sampler.model.training, "use_conditioning", False)).lower()
    
    for patch_id, patch_data in patches.items():
        # Convert to tensor
        y = torch.from_numpy(patch_data).float().to(device)
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        
        # Build conditioner if needed
        try:
            cond = sampler.transform(y) if conditioning_type == "wf" else None
        except Exception:
            cond = None
        
        # Sample
        with torch.no_grad():
            pred = sampler.sample(y, tuple(y.shape), device=device, verbose=False, conditioner=cond)
            out = pred.squeeze().detach().cpu().numpy().astype(np.float32)
        
        denoised_patches[patch_id] = out
    
    return denoised_patches


def evaluate_fov_baselines(cfg: DictConfig) -> None:
    """Run full FOV baseline comparison."""
    device = str(cfg.experiment.device)
    
    # Paths
    base_dir = Path("/home/jilab/anna_OS_ML/PKL-DiffusionDenoising")
    complete_patches_dir = base_dir / "data/real_microscopy/complete_patches"
    wf_dir = complete_patches_dir / "wf"
    gt_dir = complete_patches_dir / "2p"
    output_dir = Path(str(cfg.inference.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Verify we have all 225 patches
    if len(wf_patches) != 225 or len(gt_patches) != 225:
        print(f"ERROR: Expected 225 patches, got WF: {len(wf_patches)}, GT: {len(gt_patches)}")
        return
    
    # Reconstruct WF and GT frames
    print("Reconstructing WF frame...")
    wf_full = reconstruct_complete_patches(wf_patches, wf_original_norm.astype(np.float32))
    wf_full_u8 = normalize_to_uint8(wf_full)
    
    print("Reconstructing GT frame...")
    gt_full = reconstruct_complete_patches(gt_patches, tp_original_norm.astype(np.float32))
    gt_full_u8 = normalize_to_uint8(gt_full)
    
    # Process with different guidance methods
    guidance_methods = ["pkl", "l2", "anscombe"]
    ddim_steps = int(cfg.inference.ddim_steps)
    
    results = {}
    
    for method in guidance_methods:
        print(f"Processing patches with {method.upper()} guidance ({ddim_steps} DDIM steps)...")
        
        # Load sampler for this guidance method
        sampler = _load_model_and_sampler(cfg, method, ddim_steps)
        
        # Process all patches
        denoised_patches = process_patches_with_guidance(sampler, wf_patches, device)
        
        # Reconstruct full frame
        print(f"Reconstructing {method.upper()} frame...")
        denoised_full = reconstruct_complete_patches(denoised_patches, wf_original_norm.astype(np.float32))
        denoised_full_u8 = normalize_to_uint8(denoised_full)
        
        results[method] = denoised_full_u8
    
    # Create composite: WF | PKL | L2 | Anscombe | GT
    print("Creating final composite...")
    composite_parts = [wf_full_u8]
    for method in guidance_methods:
        composite_parts.append(results[method])
    composite_parts.append(gt_full_u8)
    
    composite = np.concatenate(composite_parts, axis=1)
    
    # Save composite
    output_path = output_dir / f"fov_baseline_comparison_steps_{ddim_steps}.png"
    Image.fromarray(composite).save(output_path)
    print(f"Saved FOV baseline comparison to {output_path}")
    print(f"Composite size: {composite.shape}")
    print(f"Each panel size: {composite.shape[0]}x{composite.shape[1]//5}")
    
    # Save individual frames
    for method in guidance_methods:
        frame_path = output_dir / f"fov_{method}_steps_{ddim_steps}.png"
        Image.fromarray(results[method]).save(frame_path)
        print(f"Saved {method.upper()} frame to {frame_path}")
    
    print(f"Done! FOV baseline comparison completed with {ddim_steps} DDIM steps.")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    evaluate_fov_baselines(cfg)


if __name__ == "__main__":
    main()
