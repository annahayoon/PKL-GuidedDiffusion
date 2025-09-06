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
from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel


def run_training(cfg: DictConfig) -> DDPMTrainer:
    """Core training routine separated from Hydra for testability.

    Note: Uses a lightweight loop without PyTorch Lightning to avoid heavy optional
    dependencies during tests. It still constructs datasets, model, trainer object,
    saves a final checkpoint, and returns the trainer with noise schedule buffers set.
    """

    # Set seed
    seed = int(cfg.experiment.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize W&B (optional)
    logger = None
    if cfg.wandb.mode != "disabled":
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.experiment.name,
        )

    # Setup paths
    data_dir = str(cfg.paths.data)
    checkpoint_dir = str(cfg.paths.checkpoints)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create forward model for data synthesis
    psf = PSF(getattr(cfg.physics, "psf_path", None))
    forward_model = ForwardModel(
        psf=psf.to_torch(device="cpu"),
        background=float(cfg.physics.background),
        device="cpu",
    )

    # Create transform (respect codebase camelCase constructor args)
    transform = IntensityToModel(
        minIntensity=float(cfg.data.min_intensity),
        maxIntensity=float(cfg.data.max_intensity),
    )

    # Create datasets
    train_dataset = SynthesisDataset(
        source_dir=f"{data_dir}/train",
        forward_model=forward_model,
        transform=transform,
        image_size=int(cfg.data.image_size),
        mode="train",
    )

    val_dataset = SynthesisDataset(
        source_dir=f"{data_dir}/val",
        forward_model=forward_model,
        transform=transform,
        image_size=int(cfg.data.image_size),
        mode="val",
    )

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

    # Create trainer module (LightningModule-like API but we won't use Lightning's Trainer)
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

    # TensorBoard writer
    logs_dir = str(cfg.paths.logs) if hasattr(cfg, "paths") and hasattr(cfg.paths, "logs") else os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(logs_dir, f"{cfg.experiment.name}"))

    # Optimizer and optional scheduler
    optim_or_pair: Union[torch.optim.Optimizer, Tuple] = ddpm_trainer.configure_optimizers()
    if isinstance(optim_or_pair, tuple) or isinstance(optim_or_pair, list):
        # Lightning-style return: ([opt], [sched]) or (opt, sched)
        if isinstance(optim_or_pair[0], list):
            optimizer = optim_or_pair[0][0]
        else:
            optimizer = optim_or_pair[0]
        scheduler = None
        if len(optim_or_pair) > 1 and optim_or_pair[1] is not None:
            scheduler = optim_or_pair[1][0] if isinstance(optim_or_pair[1], list) else optim_or_pair[1]
    else:
        optimizer = optim_or_pair  # type: ignore[assignment]
        scheduler = None

    # AMP scaler
    use_amp = bool(getattr(cfg.experiment, "mixed_precision", False)) and device == "cuda"
    scaler = GradScaler(enabled=use_amp)

    max_epochs = int(cfg.training.max_epochs)
    accumulate_grad_batches = int(getattr(cfg.training, "accumulate_grad_batches", 1))
    grad_clip_val = float(getattr(cfg.training, "gradient_clip", 0.0))

    global_step = 0
    last_val_loss: Optional[float] = None

    # Training loop with progress and end-of-epoch checkpoints
    for epoch in range(max_epochs):
        ddpm_trainer.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(progress):
            x_0, _ = batch
            x_0 = x_0.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                loss = ddpm_trainer.training_step((x_0, None), batch_idx)

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

            # TensorBoard step logging and tqdm postfix
            current_loss = float(loss.detach().item())
            current_lr = float(optimizer.param_groups[0]["lr"])
            writer.add_scalar("train/loss", current_loss, global_step)
            writer.add_scalar("train/lr", current_lr, global_step)
            postfix = {"loss": f"{current_loss:.4f}", "lr": f"{current_lr:.2e}"}
            if last_val_loss is not None:
                postfix["val"] = f"{last_val_loss:.4f}"
            progress.set_postfix(postfix)

        avg_train_loss = epoch_train_loss / max(1, num_train_batches)

        # Validation
        ddpm_trainer.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [val]", leave=False)
            for batch_idx, batch in enumerate(val_progress):
                x_0, _ = batch
                x_0 = x_0.to(device, non_blocking=True)
                loss = ddpm_trainer.validation_step((x_0, None), batch_idx)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1
                running_avg = val_loss_accum / max(1, val_batches)
                val_progress.set_postfix({"val_loss": f"{running_avg:.4f}"})

        avg_val_loss = val_loss_accum / max(1, val_batches)
        last_val_loss = avg_val_loss

        # Scheduler step (epoch-level)
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                # Be lenient with scheduler differences during tests
                pass

        # Epoch-level TensorBoard
        writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("epoch/val_loss", avg_val_loss, epoch + 1)

        # End-of-epoch checkpoints (save trainer, model, and EMA if present)
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

    # Save final model
    torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")

    # Close TensorBoard writer
    try:
        writer.flush()
        writer.close()
    except Exception:
        pass

    if cfg.wandb.mode != "disabled":
        wandb.finish()

    return ddpm_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    """Hydra entrypoint. Wraps the core training routine."""
    run_training(cfg)


if __name__ == "__main__":
    train()


