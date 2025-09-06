import os
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import tifffile
from tqdm import tqdm

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel


def run_inference(cfg: DictConfig) -> List[Path]:
    """Run guided diffusion inference and return saved file paths."""

    # Device
    device = str(cfg.experiment.device)

    # Load model and wrap in trainer (to get buffers/noise schedule)
    unet = DenoisingUNet(OmegaConf.to_container(cfg.model, resolve=True))
    ddpm = DDPMTrainer(
        model=unet,
        config=OmegaConf.to_container(cfg.training, resolve=True),
    )

    checkpoint_path = Path(str(cfg.inference.checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(state_dict)
    ddpm.eval()
    ddpm.to(device)

    # Select network (EMA if available)
    if getattr(ddpm, "use_ema", False):
        model_for_sampling = ddpm.ema_model
    else:
        model_for_sampling = ddpm.model

    # Forward model with optimizations
    psf = PSF(getattr(cfg.physics, "psf_path", None))
    
    # Enable FFT pre-computation for common sizes
    common_sizes = getattr(cfg.physics, "common_sizes", None)
    if common_sizes is None:
        # Default optimized sizes for microscopy
        common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
    
    forward_model = ForwardModel(
        psf=psf.to_torch(device=device),
        background=float(cfg.physics.background),
        device=device,
        common_sizes=common_sizes,
    )

    # Guidance
    guidance_type = str(getattr(cfg.guidance, "type", "pkl"))
    epsilon = float(getattr(cfg.guidance, "epsilon", 1e-6))
    if guidance_type == "pkl":
        guidance = PKLGuidance(epsilon=epsilon)
    elif guidance_type == "l2":
        guidance = L2Guidance()
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")

    # Schedule
    lambda_base = float(getattr(cfg.guidance, "lambda_base", 0.1))
    schedule_cfg = getattr(cfg.guidance, "schedule", {})
    T_threshold = int(getattr(schedule_cfg, "T_threshold", 800))
    epsilon_lambda = float(getattr(schedule_cfg, "epsilon_lambda", 1e-3))
    schedule = AdaptiveSchedule(
        lambda_base=lambda_base,
        T_threshold=T_threshold,
        epsilon_lambda=epsilon_lambda,
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
        ddim_steps=int(cfg.inference.ddim_steps),
        eta=float(cfg.inference.eta),
        use_autocast=bool(getattr(cfg.inference, "use_autocast", True)),
    )

    # IO
    input_dir = Path(str(cfg.inference.input_dir))
    output_dir = Path(str(cfg.inference.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))

    saved_paths: List[Path] = []
    for img_path in tqdm(image_paths, desc="Inference"):
        y_np = tifffile.imread(str(img_path))
        y = torch.from_numpy(y_np).float().to(device)
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        shape = tuple(y.shape)
        reconstruction = sampler.sample(y, shape, device=device, verbose=False)
        out = reconstruction.squeeze().detach().cpu().numpy().astype(np.float32)
        output_path = output_dir / f"{img_path.stem}_reconstructed.tif"
        tifffile.imwrite(str(output_path), out)
        saved_paths.append(output_path)

    return saved_paths


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def inference(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    inference()


