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
from PIL import Image
import numpy as np

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel, GeneralizedAnscombeToModel
from pkl_dg.data.zarr_io import ZarrPatchesDataset


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
        print("✅ Using Zarr format for faster data loading")
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
        print("✅ Using PNG format for data loading")

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

    # Create samples output dir
    samples_dir = os.path.join(str(cfg.paths.outputs), "samples", cfg.experiment.name)
    os.makedirs(samples_dir, exist_ok=True)

    def _save_samples(epoch_idx: int, num_rows: int = 2, num_cols: int = 8) -> None:
        """Generate and save a grid of samples like ddpm.py during training."""
        try:
            ddpm_trainer.eval()
            with torch.no_grad():
                num_images = num_rows * num_cols
                H = int(getattr(cfg.data, "image_size", 128))
                W = H
                samples = ddpm_trainer.ddpm_sample(num_images=num_images, image_shape=(1, H, W), use_ema=True)
                # Convert from model domain to intensity via transform
                samples = transform.inverse(samples.clamp(-1, 1)).cpu().numpy()
                # Build grid
                grid_h = num_rows * H
                grid_w = num_cols * W
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                for i in range(num_images):
                    r = i // num_cols
                    c = i % num_cols
                    img = samples[i, 0]
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
                grid_img = (grid * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_img).save(os.path.join(samples_dir, f"epoch_{epoch_idx:03d}.png"))
        except Exception:
            # Be robust to sampling failures; continue training
            pass

    def _save_validation_comparison(epoch_idx: int, val_batch, num_samples: int = 3) -> None:
        """Save validation comparison: WF input | Predicted | 2P ground truth."""
        try:
            ddpm_trainer.eval()
            with torch.no_grad():
                x_0, y_wf = val_batch
                x_0 = x_0[:num_samples].to(device)
                y_wf = y_wf[:num_samples].to(device)
                
                # Generate predictions using faster scheduler-based sampling
                if use_conditioning and conditioning_type == "wf":
                    predictions = ddpm_trainer.sample_with_scheduler(
                        shape=(num_samples, *x_0.shape[1:]),
                        num_inference_steps=50,  # Fast sampling for validation
                        use_ema=True,
                        conditioner=y_wf
                    )
                else:
                    predictions = ddpm_trainer.sample_with_scheduler(
                        shape=(num_samples, *x_0.shape[1:]),
                        num_inference_steps=50,  # Fast sampling for validation
                        use_ema=True
                    )
                
                # Convert to intensity domain
                x_0_int = transform.inverse(x_0.clamp(-1, 1)).cpu().numpy()
                y_wf_int = transform.inverse(y_wf.clamp(-1, 1)).cpu().numpy()
                pred_int = transform.inverse(predictions.clamp(-1, 1)).cpu().numpy()
                
                # Create comparison grid: WF | Predicted | 2P GT
                H, W = x_0_int.shape[2], x_0_int.shape[3]
                grid_h = num_samples * H
                grid_w = 3 * W  # 3 columns: WF, Predicted, 2P GT
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                
                for i in range(num_samples):
                    # WF input
                    grid[i*H:(i+1)*H, 0:W] = y_wf_int[i, 0]
                    # Predicted
                    grid[i*H:(i+1)*H, W:2*W] = pred_int[i, 0]
                    # 2P ground truth
                    grid[i*H:(i+1)*H, 2*W:3*W] = x_0_int[i, 0]
                
                grid_img = (grid * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_img).save(os.path.join(samples_dir, f"validation_epoch_{epoch_idx:03d}.png"))
                
                # Also log to TensorBoard
                writer.add_image(f"validation/epoch_{epoch_idx}", grid_img, epoch_idx, dataformats='HWC')
                
        except Exception as e:
            print(f"Warning: Could not save validation comparison: {e}")
            pass

    # Training loop
    for epoch in range(max_epochs):
        ddpm_trainer.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        optimizer.zero_grad(set_to_none=True)
        
        steps_per_epoch = getattr(cfg.training, "steps_per_epoch", None)
        max_steps_this_epoch = int(steps_per_epoch) if steps_per_epoch is not None else len(train_loader)
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

        # Save samples and validation comparisons (every 20 epochs)
        if (epoch + 1) % 20 == 0:
            _save_samples(epoch + 1)
            
            # Save validation comparison (WF | Predicted | 2P GT)
            try:
                # Get a validation batch for comparison
                val_batch = next(iter(val_loader))
                _save_validation_comparison(epoch + 1, val_batch)
            except Exception as e:
                print(f"Warning: Could not generate validation comparison: {e}")
        
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
        print(f"📊 Progress: {epoch+1}/{max_epochs} epochs ({((epoch+1)/max_epochs)*100:.1f}%)")
        print(f"💾 Latest checkpoint: epoch_{epoch+1:03d}_trainer.pt")
        if (epoch + 1) % 10 == 0:
            print(f"🖼️ Validation images saved: validation_epoch_{epoch+1:03d}.png")
        print(f"📈 TensorBoard logs: {cfg.paths.logs}")
        print("-" * 60)

        # Save best model and associated states
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/best_model.pt")
            
            # Save best optimizer state
            try:
                torch.save(optimizer.state_dict(), f"{checkpoint_dir}/best_optimizer.pt")
            except Exception:
                pass
            
            # Save best scheduler state
            if scheduler is not None:
                try:
                    torch.save(scheduler.state_dict(), f"{checkpoint_dir}/best_scheduler.pt")
                except Exception:
                    pass
            
            # Save best scaler state
            if use_amp:
                try:
                    torch.save(scaler.state_dict(), f"{checkpoint_dir}/best_scaler.pt")
                except Exception:
                    pass
            
            print(f"🏆 New best model saved! Val loss: {avg_val_loss:.4f}")

        # Save epoch checkpoint (every epoch)
        epoch_ckpt_prefix = os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}")
        torch.save(ddpm_trainer.state_dict(), f"{epoch_ckpt_prefix}_trainer.pt")
        if hasattr(ddpm_trainer, "ema_model") and ddpm_trainer.ema_model is not None:
            try:
                torch.save(ddpm_trainer.ema_model.state_dict(), f"{epoch_ckpt_prefix}_ema_model.pt")
            except Exception:
                pass

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
def train_real(cfg: DictConfig):
    """Hydra entrypoint for real data training."""
    run_training_real(cfg)


if __name__ == "__main__":
    train_real()
