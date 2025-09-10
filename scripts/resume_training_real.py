#!/usr/bin/env python3
"""
Resume training script for real microscopy data from checkpoint.
"""

import os
from typing import Optional, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F
import numpy as np
import random
import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from PIL import Image
import numpy as np

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel
from pkl_dg.data.zarr_io import ZarrPatchesDataset


def run_resume_training_real(cfg: DictConfig, checkpoint_path: str, start_epoch: int = 0) -> DDPMTrainer:
    """Resume training routine for real microscopy data from checkpoint."""

    # Set seed
    seed = int(cfg.experiment.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize W&B (optional)
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name,
        )

    # Setup paths
    data_dir = str(cfg.data.data_dir)
    # Force checkpoint directory to be real_run1 for continuous tensorboard logging
    checkpoint_dir = "/home/jilab/anna_OS_ML/PKL-DiffusionDenoising/checkpoints/real_run1"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create transform: VST options
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(cfg.data.max_intensity))
    elif noise_model == "poisson_gaussian":
        gat_cfg = getattr(cfg.data, "gat", {})
        transform = GeneralizedAnscombeToModel(
            maxIntensity=float(cfg.data.max_intensity),
            alpha=float(getattr(gat_cfg, "alpha", 1.0)),
            mu=float(getattr(gat_cfg, "mu", 0.0)),
            sigma=float(getattr(gat_cfg, "sigma", 0.0)),
        )
    else:
        transform = IntensityToModel(
            minIntensity=float(cfg.data.min_intensity),
            maxIntensity=float(cfg.data.max_intensity),
        )

    # Create datasets - either Zarr patches or file pairs
    use_zarr = bool(getattr(cfg.data, "use_zarr", False))
    if use_zarr:
        zarr_train = os.path.join(data_dir, "zarr", "train.zarr")
        zarr_val = os.path.join(data_dir, "zarr", "val.zarr")
        train_dataset = ZarrPatchesDataset(zarr_train, transform=transform)
        val_dataset = ZarrPatchesDataset(zarr_val, transform=transform)
    else:
        # using real microscopy data from PNG pairs
        train_dataset = RealPairsDataset(
            data_dir=data_dir,
            split="train",
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="train",
        )

        val_dataset = RealPairsDataset(
            data_dir=data_dir,
            split="val", 
            transform=transform,
            image_size=int(cfg.data.image_size),
            mode="val",
        )

    print(f"Training dataset: {len(train_dataset)} pairs")
    print(f"Validation dataset: {len(val_dataset)} pairs")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        pin_memory=True,
    )

    # Create model - optionally increase in_channels to include conditioner (WF)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", True))
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    if use_conditioning and conditioning_type == "wf" and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2  # x_t (2P) + WF conditioner
    unet = DenoisingUNet(model_cfg)

    # Create trainer module
    ddpm_trainer = DDPMTrainer(
        model=unet,
        config=OmegaConf.to_container(cfg.training, resolve=True),
        transform=transform,
    )
    
    # Select device
    device = (
        cfg.experiment.device if torch.cuda.is_available() and cfg.experiment.device == "cuda" else "cpu"
    )
    ddpm_trainer.to(device)
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in ddpm_trainer.parameters()):,}")

    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ddpm_trainer.load_state_dict(checkpoint, strict=False)
    print(f"Successfully loaded checkpoint from epoch {start_epoch}")
    
    # Try to load additional training states if available
    optimizer_state_path = checkpoint_path.replace("_trainer.pt", "_optimizer.pt")
    scheduler_state_path = checkpoint_path.replace("_trainer.pt", "_scheduler.pt")
    scaler_state_path = checkpoint_path.replace("_trainer.pt", "_scaler.pt")
    
    optimizer_state_loaded = False
    scheduler_state_loaded = False
    scaler_state_loaded = False

    # TensorBoard writer - log to logs/real_run1/real_run1 (nested subdirectory like original training)
    logs_dir = str(cfg.paths.logs) if hasattr(cfg, "paths") and hasattr(cfg.paths, "logs") else os.path.join(os.getcwd(), "logs")
    log_path = os.path.join(logs_dir, "real_run1", "real_run1")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    # Optimizer and optional scheduler
    optim_or_pair: Union[torch.optim.Optimizer, Tuple] = ddpm_trainer.configure_optimizers()
    if isinstance(optim_or_pair, tuple) or isinstance(optim_or_pair, list):
        # Lightning-style return: ([opt], [sched]) or (opt, sched)
        if len(optim_or_pair) == 2:
            optimizer, scheduler = optim_or_pair
            if isinstance(optimizer, list):
                optimizer = optimizer[0]
            if isinstance(scheduler, list):
                scheduler = scheduler[0]
        else:
            optimizer = optim_or_pair[0]
            scheduler = None
    else:
        optimizer = optim_or_pair
        scheduler = None
    
    # Try to load optimizer state if available
    if os.path.exists(optimizer_state_path):
        try:
            optimizer_state = torch.load(optimizer_state_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
            optimizer_state_loaded = True
            print(f"✅ Loaded optimizer state from {optimizer_state_path}")
        except Exception as e:
            print(f"⚠️  Failed to load optimizer state: {e}")
    
    # Try to load scheduler state if available
    if scheduler is not None and os.path.exists(scheduler_state_path):
        try:
            scheduler_state = torch.load(scheduler_state_path, map_location=device)
            scheduler.load_state_dict(scheduler_state)
            scheduler_state_loaded = True
            print(f"✅ Loaded scheduler state from {scheduler_state_path}")
        except Exception as e:
            print(f"⚠️  Failed to load scheduler state: {e}")
    
    if not optimizer_state_loaded:
        print("⚠️  Starting with fresh optimizer state")
    if scheduler is not None and not scheduler_state_loaded:
        print("⚠️  Starting with fresh scheduler state")

    # Mixed precision
    use_amp = bool(getattr(cfg.experiment, "mixed_precision", False))
    scaler = GradScaler(enabled=use_amp)
    
    # Try to load scaler state if available
    if use_amp and os.path.exists(scaler_state_path):
        try:
            scaler_state = torch.load(scaler_state_path, map_location=device)
            scaler.load_state_dict(scaler_state)
            scaler_state_loaded = True
            print(f"✅ Loaded scaler state from {scaler_state_path}")
        except Exception as e:
            print(f"⚠️  Failed to load scaler state: {e}")
    
    if use_amp and not scaler_state_loaded:
        print("⚠️  Starting with fresh scaler state")

    # Training parameters
    max_epochs = int(cfg.training.max_epochs)
    grad_clip_val = float(getattr(cfg.training, "gradient_clip", 0.0))
    accumulate_grad_batches = int(getattr(cfg.training, "accumulate_grad_batches", 1))
    
    # Resume from start_epoch
    best_val_loss = float('inf')
    global_step = start_epoch * len(train_loader)
    
    # Early stopping parameters
    early_stopping_patience = int(getattr(cfg.training, "early_stopping_patience", 10))
    early_stopping_counter = 0
    min_delta = float(getattr(cfg.training, "early_stopping_min_delta", 0.0))

    def _save_samples(epoch: int):
        """Save sample images during training."""
        try:
            ddpm_trainer.eval()
            with torch.no_grad():
                # Get a batch from validation set
                sample_batch = next(iter(val_loader))
                x_0_sample, y_wf_sample = sample_batch
                x_0_sample = x_0_sample[:4].to(device)  # Take first 4 samples
                y_wf_sample = y_wf_sample[:4].to(device)
                
                # Generate samples
                if use_conditioning and conditioning_type == "wf":
                    samples = ddpm_trainer.sample(batch_size=4, cond=y_wf_sample)
                else:
                    samples = ddpm_trainer.sample(batch_size=4)
                
                # Save samples
                samples_dir = os.path.join(checkpoint_dir, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                for i in range(4):
                    # Convert from [-1, 1] to [0, 255]
                    sample_img = ((samples[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    gt_img = ((x_0_sample[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    wf_img = ((y_wf_sample[i].cpu().numpy().transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8)
                    
                    # Save as PNG
                    Image.fromarray(sample_img).save(os.path.join(samples_dir, f"epoch_{epoch:03d}_sample_{i}.png"))
                    Image.fromarray(gt_img).save(os.path.join(samples_dir, f"epoch_{epoch:03d}_gt_{i}.png"))
                    Image.fromarray(wf_img).save(os.path.join(samples_dir, f"epoch_{epoch:03d}_wf_{i}.png"))
        except Exception as e:
            print(f"Warning: Could not save samples: {e}")

    # Training loop - resume from start_epoch
    for epoch in range(start_epoch, max_epochs):
        ddpm_trainer.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        optimizer.zero_grad(set_to_none=True)
        
        max_steps_this_epoch = int(getattr(cfg.training, "steps_per_epoch", 0))
        for batch_idx, batch in enumerate(progress):
            x_0, y_wf = batch  # x_0 is clean 2P, y_wf is noisy WF
            x_0 = x_0.to(device, non_blocking=True)
            y_wf = y_wf.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                # Sample timesteps
                t = torch.randint(0, ddpm_trainer.num_timesteps, (x_0.shape[0],), device=device)
                noise = torch.randn_like(x_0)
                x_t = ddpm_trainer.q_sample(x_0, t, noise)
                # Predict noise with WF conditioning
                if use_conditioning and conditioning_type == "wf":
                    predicted_noise = ddpm_trainer.model(x_t, t, cond=y_wf)
                else:
                    predicted_noise = ddpm_trainer.model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)
                # Optional supervised x0 loss
                sup_w = float(getattr(cfg.training, "supervised_x0_weight", 0.0))
                if sup_w > 0:
                    alpha_t = ddpm_trainer.alphas_cumprod[t].view(-1, 1, 1, 1)
                    sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
                    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
                    x0_hat = (x_t - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t
                    loss = loss + sup_w * F.l1_loss(x0_hat, x_0)

            # Backward
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if grad_clip_val and grad_clip_val > 0.0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddpm_trainer.parameters(), max_norm=grad_clip_val)

            step_now = (batch_idx + 1) % accumulate_grad_batches == 0
            if step_now:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_train_loss += float(loss.detach().item())
            num_train_batches += 1
            global_step += 1

            # TensorBoard step logging
            writer.add_scalar("train/loss", float(loss.detach().item()), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            progress.set_postfix({"loss": f"{float(loss.detach().item()):.4f}"})

            # Stop at steps_per_epoch if configured
            if max_steps_this_epoch > 0 and (batch_idx + 1) >= max_steps_this_epoch:
                break

        avg_train_loss = epoch_train_loss / max(1, num_train_batches)

        # Validation
        ddpm_trainer.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [val]", leave=False)):
                x_0, y_wf = batch
                x_0 = x_0.to(device, non_blocking=True)
                y_wf = y_wf.to(device, non_blocking=True)
                
                # Validation loss computation
                t = torch.randint(0, ddpm_trainer.num_timesteps, (x_0.shape[0],), device=device)
                noise = torch.randn_like(x_0)
                x_t = ddpm_trainer.q_sample(x_0, t, noise)
                if use_conditioning and conditioning_type == "wf":
                    predicted_noise = ddpm_trainer.model(x_t, t, cond=y_wf)
                else:
                    predicted_noise = ddpm_trainer.model(x_t, t)
                loss = F.mse_loss(predicted_noise, noise)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)

        # Scheduler step
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Logging
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)
        
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "lr": optimizer.param_groups[0]["lr"]
            })

        # Save samples like ddpm.py and per-epoch weights
        _save_samples(epoch + 1)
        epoch_ckpt_prefix = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}")
        torch.save(ddpm_trainer.state_dict(), f"{epoch_ckpt_prefix}_trainer.pt")
        try:
            torch.save(ddpm_trainer.model.state_dict(), f"{epoch_ckpt_prefix}_model.pt")
        except Exception:
            pass
        if hasattr(ddpm_trainer, "ema_model") and ddpm_trainer.ema_model is not None:
            try:
                torch.save(ddpm_trainer.ema_model.state_dict(), f"{epoch_ckpt_prefix}_ema_model.pt")
            except Exception:
                pass
        
        # Save optimizer, scheduler, and scaler states for proper resuming
        try:
            torch.save(optimizer.state_dict(), f"{epoch_ckpt_prefix}_optimizer.pt")
        except Exception:
            pass
        if scheduler is not None:
            try:
                torch.save(scheduler.state_dict(), f"{epoch_ckpt_prefix}_scheduler.pt")
            except Exception:
                pass
        if use_amp:
            try:
                torch.save(scaler.state_dict(), f"{epoch_ckpt_prefix}_scaler.pt")
            except Exception:
                pass

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Save best model and check early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0  # Reset counter
            torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/best_model.pt")
            # Also save training states for best model
            try:
                torch.save(optimizer.state_dict(), f"{checkpoint_dir}/best_optimizer.pt")
            except Exception:
                pass
            if scheduler is not None:
                try:
                    torch.save(scheduler.state_dict(), f"{checkpoint_dir}/best_scheduler.pt")
                except Exception:
                    pass
            if use_amp:
                try:
                    torch.save(scaler.state_dict(), f"{checkpoint_dir}/best_scaler.pt")
                except Exception:
                    pass
            print(f"New best validation loss: {best_val_loss:.4f}")
        else:
            early_stopping_counter += 1
            print(f"Validation loss did not improve. Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Save epoch checkpoint (every epoch)
        epoch_ckpt_prefix = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}")
        torch.save(ddpm_trainer.state_dict(), f"{epoch_ckpt_prefix}_trainer.pt")
        if hasattr(ddpm_trainer, "ema_model") and ddpm_trainer.ema_model is not None:
            try:
                torch.save(ddpm_trainer.ema_model.state_dict(), f"{epoch_ckpt_prefix}_ema_model.pt")
            except Exception:
                pass

        # Check early stopping
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best validation loss: {best_val_loss:.4f}")
            break

    # Save final model
    torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")
    # Also save final training states
    try:
        torch.save(optimizer.state_dict(), f"{checkpoint_dir}/final_optimizer.pt")
    except Exception:
        pass
    if scheduler is not None:
        try:
            torch.save(scheduler.state_dict(), f"{checkpoint_dir}/final_scheduler.pt")
        except Exception:
            pass
    if use_amp:
        try:
            torch.save(scaler.state_dict(), f"{checkpoint_dir}/final_scaler.pt")
        except Exception:
            pass

    # Close writers
    try:
        writer.flush()
        writer.close()
    except Exception:
        pass

    if cfg.wandb.mode != "disabled":
        wandb.finish()

    return ddpm_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config_real")
def main(cfg: DictConfig):
    """Resume training from checkpoint."""
    # Get checkpoint path and start epoch from config overrides
    checkpoint_path = None
    start_epoch = 0
    
    # Look for checkpoint and start_epoch in the config overrides
    if hasattr(cfg, 'resume_checkpoint'):
        checkpoint_path = str(cfg.resume_checkpoint)
    if hasattr(cfg, 'resume_start_epoch'):
        start_epoch = int(cfg.resume_start_epoch)
    
    # If not found in config, try to extract from command line
    import sys
    for i, arg in enumerate(sys.argv):
        if arg.startswith('+resume_checkpoint='):
            checkpoint_path = arg.split('=', 1)[1]
        elif arg.startswith('+resume_start_epoch='):
            start_epoch = int(arg.split('=', 1)[1])
    
    if checkpoint_path is None:
        print("Error: resume_checkpoint must be specified")
        print("Usage: python scripts/resume_training_real.py resume_checkpoint=/path/to/checkpoint.pt resume_start_epoch=299")
        return
    
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    print(f"Starting from epoch: {start_epoch}")
    
    # Run training
    trainer = run_resume_training_real(cfg, checkpoint_path, start_epoch)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
