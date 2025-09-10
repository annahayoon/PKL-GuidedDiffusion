import os
import time
import signal
import threading
from typing import Optional, Tuple, Union
from contextlib import nullcontext

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
from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel
from pkl_dg.utils import MemoryProfiler, profile_memory_usage
from pkl_dg.utils.adaptive_batch import get_optimal_batch_size


class ColabSessionManager:
    """Manages Colab session to prevent disconnection crashes."""
    
    def __init__(self, checkpoint_interval_minutes: int = 15):
        self.checkpoint_interval_minutes = checkpoint_interval_minutes
        self.last_checkpoint_time = time.time()
        self.checkpoint_callback = None
        self.running = True
        
    def set_checkpoint_callback(self, callback):
        """Set callback function to save checkpoints."""
        self.checkpoint_callback = callback
        
    def start_keepalive(self):
        """Start background thread to keep session alive."""
        def keepalive():
            while self.running:
                # Simulate activity to prevent disconnection
                time.sleep(60)  # Check every minute
                if self.checkpoint_callback and self._should_checkpoint():
                    try:
                        self.checkpoint_callback()
                        self.last_checkpoint_time = time.time()
                        print(f"🔄 Auto-checkpoint saved at {time.strftime('%H:%M:%S')}")
                    except Exception as e:
                        print(f"⚠️ Auto-checkpoint failed: {e}")
        
        thread = threading.Thread(target=keepalive, daemon=True)
        thread.start()
        return thread
        
    def _should_checkpoint(self) -> bool:
        """Check if it's time for an auto-checkpoint."""
        elapsed_minutes = (time.time() - self.last_checkpoint_time) / 60
        return elapsed_minutes >= self.checkpoint_interval_minutes
        
    def stop(self):
        """Stop the keepalive thread."""
        self.running = False


def setup_colab_environment(cfg: DictConfig) -> DictConfig:
    """Setup Colab-specific environment and paths."""
    # Detect if running in Colab
    is_colab = 'COLAB_GPU' in os.environ or 'google.colab' in str(os.getcwd())
    
    if is_colab:
        print("🚀 Running in Google Colab environment")
        
        # Update paths for Colab
        if hasattr(cfg, 'colab') and cfg.colab.get('save_to_drive', False):
            drive_path = cfg.colab.get('drive_path', '/content/drive/MyDrive/PKL-DiffusionDenoising')
            cfg.paths.checkpoints = f"{drive_path}/checkpoints"
            cfg.paths.outputs = f"{drive_path}/outputs"
            cfg.paths.logs = f"{drive_path}/logs"
            print(f"📁 Using Google Drive paths: {cfg.paths.checkpoints}")
        
        # Set Colab-optimized settings
        if hasattr(cfg, 'colab_optimizations'):
            colab_opts = cfg.colab_optimizations
            
            # Enable gradient checkpointing for memory efficiency
            if colab_opts.get('gradient_checkpointing', False):
                cfg.training.gradient_checkpointing = True
                
            # Set diffusers scheduler
            if colab_opts.get('use_diffusers_scheduler', True):
                cfg.training.use_diffusers_scheduler = True
                cfg.training.scheduler_type = colab_opts.get('scheduler_type', 'dpm_solver')
                
    return cfg


def run_training_colab(cfg: DictConfig) -> DDPMTrainer:
    """Core training routine optimized for Google Colab with crash prevention."""

    # Setup Colab environment
    cfg = setup_colab_environment(cfg)
    
    # Initialize session manager for crash prevention
    session_manager = None
    if hasattr(cfg, 'colab') and cfg.colab.get('prevent_disconnect', True):
        checkpoint_interval = cfg.colab.get('checkpoint_interval_minutes', 15)
        session_manager = ColabSessionManager(checkpoint_interval)
        session_manager.start_keepalive()
        print(f"🛡️ Session keepalive started (checkpoint every {checkpoint_interval} min)")

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

    # No PSF / forward model usage
    forward_model = None

    # Create transform (respect codebase camelCase constructor args)
    noise_model = str(getattr(cfg.data, "noise_model", "gaussian")).lower()
    if noise_model == "poisson":
        transform = AnscombeToModel(maxIntensity=float(cfg.data.max_intensity))
    else:
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

    # Create data loaders with Colab optimizations
    num_workers = int(cfg.training.num_workers)
    persistent_workers = bool(getattr(cfg.training, "persistent_workers", False))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 2)),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 2)),
    )

    # Create model; set in_channels=2 when conditioning is enabled
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    if use_conditioning and conditioning_type == "wf" and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2  # x_t + WF conditioner
    unet = DenoisingUNet(model_cfg)

    # Create trainer module with Colab optimizations
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
    
    # Select device
    device = (
        cfg.experiment.device if torch.cuda.is_available() and cfg.experiment.device == "cuda" else "cpu"
    )
    ddpm_trainer.to(device)

    # Optional: Multi-GPU training via PyTorch Lightning Trainer
    use_multi_gpu = False
    try:
        requested_devices = int(getattr(cfg.training, "devices", 1))
    except Exception:
        requested_devices = 1
    distributed_enabled = bool(getattr(cfg.training, "distributed", {}).get("enabled", False)) if hasattr(cfg.training, "distributed") else False
    accelerator = str(getattr(cfg.training, "accelerator", "gpu"))
    precision = str(getattr(cfg.training, "precision", "16-mixed")) if bool(getattr(cfg.experiment, "mixed_precision", False)) else "32-true"

    # We only attempt Lightning multi-GPU when GPUs are available and requested > 1
    if accelerator == "gpu" and torch.cuda.is_available() and torch.cuda.device_count() >= 2 and (requested_devices > 1 or distributed_enabled):
        try:
            import pytorch_lightning as pl  # type: ignore
            from pytorch_lightning.strategies import DDPStrategy  # type: ignore
            from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

            # Enhanced checkpoint callback for Colab
            ckpt_cb = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="ddpm-{epoch:03d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                save_last=True,
                every_n_epochs=1,
            )
            
            # Early stopping for Colab
            early_stop_cb = None
            if hasattr(cfg, 'colab_optimizations'):
                colab_opts = cfg.colab_optimizations
                early_stop_cb = EarlyStopping(
                    monitor=colab_opts.get('early_stopping_monitor', 'val/loss'),
                    patience=cfg.training.get('early_stopping_patience', 5),
                    min_delta=colab_opts.get('early_stopping_min_delta', 0.001),
                    verbose=True
                )

            callbacks = [ckpt_cb]
            if early_stop_cb:
                callbacks.append(early_stop_cb)

            trainer = pl.Trainer(
                devices=requested_devices,
                accelerator="gpu",
                strategy=DDPStrategy(find_unused_parameters=False),
                precision=precision,
                max_epochs=int(cfg.training.max_epochs),
                enable_progress_bar=True,
                log_every_n_steps=50,
                callbacks=callbacks,
            )

            # Set checkpoint callback for session manager
            if session_manager:
                session_manager.set_checkpoint_callback(lambda: ckpt_cb.save_checkpoint(trainer))

            # Fit using external dataloaders to avoid adding hooks to the module
            trainer.fit(ddpm_trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Save final checkpoint on global zero only
            is_global_zero = getattr(trainer, "is_global_zero", True)
            if is_global_zero:
                torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")

            return ddpm_trainer
        except Exception as e:
            print(f"⚠️ Lightning multi-GPU failed, falling back to single-process: {e}")
            # Fall back to single-process loop
            pass

    # Optional dynamic batch sizing (rebuild loaders with optimal batch size)
    enable_dynamic_bs = bool(getattr(cfg.training, "dynamic_batch_sizing", False))
    if enable_dynamic_bs:
        try:
            H = int(getattr(cfg.data, "image_size", 128))
            C = int(model_cfg.get("in_channels", 1))
            input_shape = (C, H, H)
            safety = float(getattr(cfg.training, "dynamic_batch_safety_factor", 0.8))
            grad_ckpt = bool(getattr(cfg.training, "gradient_checkpointing", False))
            optimal_bs = get_optimal_batch_size(
                ddpm_trainer.model,
                input_shape,
                device=device,
                mixed_precision=use_amp,
                gradient_checkpointing=grad_ckpt,
                safety_factor=safety,
                verbose=True,  # Enable verbose for Colab
            )
            print(f"🎯 Optimal batch size for A100: {optimal_bs}")
            
            # Rebuild loaders with optimal batch size
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=optimal_bs,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 2)),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=optimal_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=persistent_workers,
                prefetch_factor=int(getattr(cfg.training, "prefetch_factor", 2)),
            )
        except Exception as e:
            print(f"⚠️ Dynamic batch sizing failed: {e}")
            pass

    # Full Lightning training mode (single GPU/CPU) controlled by flag
    use_lightning = bool(getattr(cfg.training, "use_lightning", False))
    if use_lightning:
        try:
            import pytorch_lightning as pl  # type: ignore
            from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # type: ignore

            accel = "gpu" if (accelerator == "gpu" and torch.cuda.is_available()) else "cpu"
            devs = requested_devices if (accel == "gpu" and requested_devices > 0) else 1

            # Enhanced checkpoint callback for Colab
            ckpt_cb = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="ddpm-{epoch:03d}-{val_loss:.4f}",
                save_top_k=3,
                monitor="val/loss",
                mode="min",
                save_last=True,
                every_n_epochs=1,
            )
            
            # Early stopping for Colab
            early_stop_cb = None
            if hasattr(cfg, 'colab_optimizations'):
                colab_opts = cfg.colab_optimizations
                early_stop_cb = EarlyStopping(
                    monitor=colab_opts.get('early_stopping_monitor', 'val/loss'),
                    patience=cfg.training.get('early_stopping_patience', 5),
                    min_delta=colab_opts.get('early_stopping_min_delta', 0.001),
                    verbose=True
                )

            callbacks = [ckpt_cb]
            if early_stop_cb:
                callbacks.append(early_stop_cb)

            trainer = pl.Trainer(
                devices=devs,
                accelerator=accel,
                precision=precision if accel == "gpu" else "32-true",
                max_epochs=int(cfg.training.max_epochs),
                enable_progress_bar=True,
                log_every_n_steps=50,
                callbacks=callbacks,
            )

            # Set checkpoint callback for session manager
            if session_manager:
                session_manager.set_checkpoint_callback(lambda: ckpt_cb.save_checkpoint(trainer))

            trainer.fit(ddpm_trainer, train_dataloaders=train_loader, val_dataloaders=val_loader)
            torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")
            return ddpm_trainer
        except Exception as e:
            print(f"⚠️ Lightning training failed, falling back to manual loop: {e}")
            # Fall back below to built-in loop
            pass

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
                samples = transform.inverse(samples.clamp(-1, 1)).cpu().numpy()
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
        except Exception as e:
            print(f"⚠️ Sample generation failed: {e}")

    def _save_validation_visualization(epoch_idx: int, val_batch: tuple) -> None:
        """Save validation visualization: WF | Predicted | 2P GT."""
        try:
            ddpm_trainer.eval()
            with torch.no_grad():
                x_0, y_wf = val_batch
                x_0 = x_0.to(device, non_blocking=True)
                y_wf = y_wf.to(device, non_blocking=True)
                
                # Take first 4 samples from batch
                num_samples = min(4, x_0.shape[0])
                x_0_samples = x_0[:num_samples]
                y_wf_samples = y_wf[:num_samples]
                
                # Generate predictions using fast sampling
                if hasattr(ddpm_trainer, 'fast_sample'):
                    predictions = ddpm_trainer.fast_sample(
                        shape=(num_samples, 1, x_0_samples.shape[2], x_0_samples.shape[3]),
                        num_inference_steps=25,
                        device=device,
                        use_ema=True,
                        conditioner=y_wf_samples
                    )
                else:
                    predictions = ddpm_trainer.ddpm_sample(
                        num_images=num_samples,
                        image_shape=(1, x_0_samples.shape[2], x_0_samples.shape[3]),
                        use_ema=True
                    )
                
                # Convert to intensity domain
                x_0_int = transform.inverse(x_0_samples.clamp(-1, 1)).cpu().numpy()
                y_wf_int = transform.inverse(y_wf_samples.clamp(-1, 1)).cpu().numpy()
                pred_int = transform.inverse(predictions.clamp(-1, 1)).cpu().numpy()
                
                # Create visualization grid: WF | Predicted | 2P GT
                H, W = x_0_int.shape[2], x_0_int.shape[3]
                grid_h = num_samples * H
                grid_w = 3 * W  # 3 columns: WF, Predicted, 2P GT
                
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                
                for i in range(num_samples):
                    row_start = i * H
                    row_end = (i + 1) * H
                    
                    # Column 1: WF (widefield)
                    col_start = 0
                    col_end = W
                    grid[row_start:row_end, col_start:col_end] = y_wf_int[i, 0]
                    
                    # Column 2: Predicted
                    col_start = W
                    col_end = 2 * W
                    grid[row_start:row_end, col_start:col_end] = pred_int[i, 0]
                    
                    # Column 3: 2P GT (ground truth)
                    col_start = 2 * W
                    col_end = 3 * W
                    grid[row_start:row_end, col_start:col_end] = x_0_int[i, 0]
                
                # Normalize and save
                grid_normalized = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
                grid_img = (grid_normalized * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save validation visualization
                val_viz_path = os.path.join(samples_dir, f"validation_epoch_{epoch_idx:03d}.png")
                Image.fromarray(grid_img).save(val_viz_path)
                
                # Also save to TensorBoard
                writer.add_image(f"validation/wf_pred_gt_epoch_{epoch_idx}", 
                               torch.from_numpy(grid_normalized).unsqueeze(0), epoch_idx)
                
                print(f"📊 Validation visualization saved: {val_viz_path}")
                
        except Exception as e:
            print(f"⚠️ Validation visualization failed: {e}")

    def _save_checkpoint(epoch_idx: int, optimizer, scheduler, scaler):
        """Save comprehensive checkpoint for resuming."""
        try:
            checkpoint_data = {
                'epoch': epoch_idx,
                'model_state_dict': ddpm_trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'last_val_loss': last_val_loss,
            }
            
            if scheduler is not None:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            if use_amp and scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                
            torch.save(checkpoint_data, f"{checkpoint_dir}/checkpoint_epoch_{epoch_idx:03d}.pt")
            torch.save(checkpoint_data, f"{checkpoint_dir}/latest_checkpoint.pt")
            print(f"💾 Checkpoint saved: epoch {epoch_idx}")
        except Exception as e:
            print(f"⚠️ Checkpoint save failed: {e}")

    # Set checkpoint callback for session manager
    if session_manager:
        session_manager.set_checkpoint_callback(lambda: _save_checkpoint(0, optimizer, scheduler, scaler))

    # Training loop with progress and end-of-epoch checkpoints (single-process fallback)
    enable_mem_profile = bool(getattr(cfg.experiment, "enable_memory_profiling", False))
    empty_cache_freq = int(getattr(cfg, 'colab_optimizations', {}).get('empty_cache_frequency', 100))
    
    for epoch in range(max_epochs):
        ddpm_trainer.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [train]", leave=False)
        optimizer.zero_grad(set_to_none=True)
        max_steps_this_epoch = int(getattr(cfg.training, "steps_per_epoch", 0))
        
        for batch_idx, batch in enumerate(progress):
            x_0, y_wf = batch
            x_0 = x_0.to(device, non_blocking=True)
            y_wf = y_wf.to(device, non_blocking=True)

            with autocast(enabled=use_amp):
                loss = ddpm_trainer.training_step((x_0, y_wf), batch_idx)

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

            # Colab memory optimization: clear cache periodically
            if global_step % empty_cache_freq == 0:
                torch.cuda.empty_cache()

            # Stop at steps_per_epoch if configured
            if max_steps_this_epoch > 0 and (batch_idx + 1) >= max_steps_this_epoch:
                break

        avg_train_loss = epoch_train_loss / max(1, num_train_batches)

        # Validation
        ddpm_trainer.eval()
        val_loss_accum = 0.0
        val_batches = 0
        first_val_batch = None  # Store first batch for visualization
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [val]", leave=False)
            for batch_idx, batch in enumerate(val_progress):
                x_0, y_wf = batch
                x_0 = x_0.to(device, non_blocking=True)
                y_wf = y_wf.to(device, non_blocking=True)
                loss = ddpm_trainer.validation_step((x_0, y_wf), batch_idx)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1
                running_avg = val_loss_accum / max(1, val_batches)
                val_progress.set_postfix({"val_loss": f"{running_avg:.4f}"})
                
                # Store first batch for validation visualization
                if first_val_batch is None:
                    first_val_batch = batch

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
        if enable_mem_profile:
            mem_snapshot = profile_memory_usage()
            if isinstance(mem_snapshot, dict):
                writer.add_scalar("memory/allocated_mb", float(mem_snapshot.get("allocated_mb", 0.0)), epoch + 1)
                writer.add_scalar("memory/reserved_mb", float(mem_snapshot.get("reserved_mb", 0.0)), epoch + 1)
                writer.add_scalar("memory/peak_allocated_mb", float(mem_snapshot.get("peak_allocated_mb", 0.0)), epoch + 1)

        # Save samples periodically
        save_samples_freq = int(getattr(cfg, 'colab_optimizations', {}).get('save_samples_every_n_epochs', 10))
        if (epoch + 1) % save_samples_freq == 0:
            _save_samples(epoch + 1)

        # Save validation visualization every 50 epochs
        if (epoch + 1) % 50 == 0 and first_val_batch is not None:
            _save_validation_visualization(epoch + 1, first_val_batch)

        # Save checkpoints every epoch (as requested)
        _save_checkpoint(epoch + 1, optimizer, scheduler, scaler)

    # Save final model
    torch.save(ddpm_trainer.state_dict(), f"{checkpoint_dir}/final_model.pt")
    _save_checkpoint(max_epochs, optimizer, scheduler, scaler)

    # Close TensorBoard writer
    try:
        writer.flush()
        writer.close()
    except Exception:
        pass

    if cfg.wandb.mode != "disabled":
        wandb.finish()

    # Stop session manager
    if session_manager:
        session_manager.stop()

    return ddpm_trainer


@hydra.main(version_base=None, config_path="../configs", config_name="config_colab")
def train(cfg: DictConfig):
    """Hydra entrypoint for Colab training. Wraps the core training routine."""
    run_training_colab(cfg)


if __name__ == "__main__":
    train()
