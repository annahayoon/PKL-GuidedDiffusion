#!/usr/bin/env python3
"""
Training script for real microscopy data.
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

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel


def run_training_real(cfg: DictConfig) -> DDPMTrainer:
    """Training routine for real microscopy data."""

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
    checkpoint_dir = str(cfg.paths.checkpoints)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create transform (respect codebase camelCase constructor args)
    transform = IntensityToModel(
        minIntensity=float(cfg.data.min_intensity),
        maxIntensity=float(cfg.data.max_intensity),
    )

    # Create datasets - using real microscopy data
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

    # Create model
    unet = DenoisingUNet(OmegaConf.to_container(cfg.model, resolve=True))

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

    # TensorBoard writer
    logs_dir = str(cfg.paths.logs) if hasattr(cfg, "paths") and hasattr(cfg.paths, "logs") else os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logs_dir, f"{cfg.experiment.name}"))

    # Optimizer and optional scheduler
    optim_or_pair: Union[torch.optim.Optimizer, Tuple] = ddpm_trainer.configure_optimizers()
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

    # AMP scaler
    use_amp = bool(getattr(cfg.experiment, "mixed_precision", False)) and device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    max_epochs = int(cfg.training.max_epochs)
    accumulate_grad_batches = int(getattr(cfg.training, "accumulate_grad_batches", 1))
    grad_clip_val = float(getattr(cfg.training, "gradient_clip", 0.0))

    global_step = 0
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(max_epochs):
        ddpm_trainer.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(progress):
            x_0, y_wf = batch  # x_0 is clean 2P, y_wf is noisy WF
            x_0 = x_0.to(device, non_blocking=True)
            y_wf = y_wf.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                # Train diffusion model on clean images - get loss directly
                # Sample timesteps
                t = torch.randint(0, ddpm_trainer.num_timesteps, (x_0.shape[0],), device=device)
                
                # Add noise to clean images
                noise = torch.randn_like(x_0)
                x_t = ddpm_trainer.q_sample(x_0, t, noise)
                
                # Predict noise
                predicted_noise = ddpm_trainer.model(x_t, t)
                
                # Compute loss
                loss = F.mse_loss(predicted_noise, noise)

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

        print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/best_model.pt")
            print(f"New best validation loss: {best_val_loss:.4f}")

        # Save epoch checkpoint
        if (epoch + 1) % 10 == 0 or epoch == max_epochs - 1:
            epoch_ckpt_prefix = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}")
            torch.save(ddpm_trainer.state_dict(), f"{epoch_ckpt_prefix}_trainer.pt")
            if hasattr(ddpm_trainer, "ema_model") and ddpm_trainer.ema_model is not None:
                try:
                    torch.save(ddpm_trainer.ema_model.state_dict(), f"{epoch_ckpt_prefix}_ema_model.pt")
                except Exception:
                    pass

    # Save final model
    torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")

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
def train_real(cfg: DictConfig):
    """Hydra entrypoint for real data training."""
    run_training_real(cfg)


if __name__ == "__main__":
    train_real()
