import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Optional, Dict, Any

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint file in the directory."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    
    # Look for latest_checkpoint.pt first
    latest_ckpt = checkpoint_path / "latest_checkpoint.pt"
    if latest_ckpt.exists():
        return str(latest_ckpt)
    
    # Look for epoch-based checkpoints
    checkpoint_files = list(checkpoint_path.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None
    
    # Sort by epoch number and return the latest
    checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(checkpoint_files[-1])


def load_checkpoint(checkpoint_path: str, device: str = "cuda") -> Dict[str, Any]:
    """Load checkpoint data from file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_config(cfg: DictConfig) -> DDPMTrainer:
    """Create model and trainer from configuration."""
    # Create transform
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(cfg.data.max_intensity))
    else:
        transform = IntensityToModel(
            minIntensity=float(cfg.data.min_intensity),
            maxIntensity=float(cfg.data.max_intensity),
        )

    # Create model
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    if use_conditioning and conditioning_type == "wf" and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2  # x_t + WF conditioner
    unet = DenoisingUNet(model_cfg)

    # Create trainer with Colab optimizations
    training_config = OmegaConf.to_container(cfg.training, resolve=True)
    
    # Add Colab-specific training config
    if hasattr(cfg, 'colab_optimizations'):
        colab_opts = cfg.colab_optimizations
        training_config.update({
            'use_diffusers_scheduler': colab_opts.get('use_diffusers_scheduler', True),
            'scheduler_type': colab_opts.get('scheduler_type', 'dpm_solver'),
            'use_karras_sigmas': colab_opts.get('use_karras_sigmas', True),
        })
    
    ddpm_trainer = DDPMTrainer(
        model=unet,
        config=training_config,
        transform=transform,
    )
    
    return ddpm_trainer


def resume_training_from_checkpoint(cfg: DictConfig, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Resume training from a checkpoint."""
    checkpoint_dir = str(cfg.paths.checkpoints)
    
    # Find checkpoint if not provided
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            print("❌ No checkpoint found. Starting training from scratch.")
            return {}
    
    # Load checkpoint
    device = cfg.experiment.device if torch.cuda.is_available() and cfg.experiment.device == "cuda" else "cpu"
    checkpoint_data = load_checkpoint(checkpoint_path, device)
    
    # Create model
    ddpm_trainer = create_model_from_config(cfg)
    ddpm_trainer.to(device)
    
    # Load model state
    ddpm_trainer.load_state_dict(checkpoint_data['model_state_dict'])
    
    print(f"✅ Model loaded from epoch {checkpoint_data.get('epoch', 'unknown')}")
    print(f"📊 Last validation loss: {checkpoint_data.get('last_val_loss', 'unknown')}")
    print(f"🔢 Global step: {checkpoint_data.get('global_step', 'unknown')}")
    
    return {
        'trainer': ddpm_trainer,
        'checkpoint_data': checkpoint_data,
        'resume_epoch': checkpoint_data.get('epoch', 0),
        'resume_step': checkpoint_data.get('global_step', 0),
        'last_val_loss': checkpoint_data.get('last_val_loss', None),
    }


def setup_resume_optimizer(cfg: DictConfig, checkpoint_data: Dict[str, Any]) -> tuple:
    """Setup optimizer and scheduler from checkpoint."""
    # Create trainer to get optimizer configuration
    ddpm_trainer = create_model_from_config(cfg)
    optim_or_pair = ddpm_trainer.configure_optimizers()
    
    if isinstance(optim_or_pair, tuple) or isinstance(optim_or_pair, list):
        if isinstance(optim_or_pair[0], list):
            optimizer = optim_or_pair[0][0]
        else:
            optimizer = optim_or_pair[0]
        scheduler = None
        if len(optim_or_pair) > 1 and optim_or_pair[1] is not None:
            scheduler = optim_or_pair[1][0] if isinstance(optim_or_pair[1], list) else optim_or_pair[1]
    else:
        optimizer = optim_or_pair
        scheduler = None
    
    # Load optimizer state if available
    if 'optimizer_state_dict' in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        print("✅ Optimizer state restored")
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint_data:
        scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
        print("✅ Scheduler state restored")
    
    return optimizer, scheduler


def setup_resume_scaler(cfg: DictConfig, checkpoint_data: Dict[str, Any]) -> Optional[torch.cuda.amp.GradScaler]:
    """Setup gradient scaler from checkpoint."""
    use_amp = bool(getattr(cfg.experiment, "mixed_precision", False)) and torch.cuda.is_available()
    if not use_amp:
        return None
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    # Load scaler state if available
    if 'scaler_state_dict' in checkpoint_data:
        scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
        print("✅ Gradient scaler state restored")
    
    return scaler


@hydra.main(version_base=None, config_path="../configs", config_name="config_colab")
def main(cfg: DictConfig):
    """Main function to resume training from checkpoint."""
    print("🔄 Resuming training from checkpoint...")
    
    # Resume from checkpoint
    resume_info = resume_training_from_checkpoint(cfg)
    
    if not resume_info:
        print("❌ No checkpoint found. Please run training from scratch first.")
        return
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_resume_optimizer(cfg, resume_info['checkpoint_data'])
    
    # Setup gradient scaler
    scaler = setup_resume_scaler(cfg, resume_info['checkpoint_data'])
    
    print(f"🚀 Ready to resume training from epoch {resume_info['resume_epoch']}")
    print(f"📈 Last validation loss: {resume_info['last_val_loss']}")
    
    # Import and run the training function with resume info
    from train_diffusion_colab import run_training_colab
    
    # Modify config to start from resume epoch
    cfg.training.resume_from_checkpoint = True
    cfg.training.resume_epoch = resume_info['resume_epoch']
    cfg.training.resume_step = resume_info['resume_step']
    
    # Run training
    run_training_colab(cfg)


if __name__ == "__main__":
    main()
