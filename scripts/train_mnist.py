#!/usr/bin/env python3
"""
Train DDPM on MNIST (grayscale) with optional W&B logging.

Uses our DenoisingUNet and DDPMTrainer, resizing MNIST to 32x32 and scaling to [-1, 1].
"""

import os
import argparse
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from PIL import Image
import numpy as np

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel
from pkl_dg.physics.noise import PoissonNoise, GaussianBackground


def build_mnist_loaders(batch_size: int, num_workers: int, image_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    # Use HuggingFace datasets to avoid system _lzma dependency entirely
    from datasets import load_dataset
    import numpy as np

    ds_train = load_dataset("mnist", split="train")
    ds_test = load_dataset("mnist", split="test")

    class _HFWrapper(Dataset):
        def __init__(self, hf_ds, size):
            self.hf_ds = hf_ds
            self.size = size
        def __len__(self):
            return len(self.hf_ds)
        def __getitem__(self, idx):
            item = self.hf_ds[int(idx)]
            img = item["image"].convert("L").resize((self.size, self.size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            ten = torch.from_numpy(arr).unsqueeze(0)
            label = int(item.get("label", 0))
            return ten, label

    train_set = _HFWrapper(ds_train, image_size)
    val_set = _HFWrapper(ds_test, image_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="pkl-diffusion-mnist")
    parser.add_argument("--name", type=str, default="mnist-ddpm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--simulate-wf", action="store_true", help="Generate WF-like from MNIST via Poisson+background and use as conditioner")
    parser.add_argument("--poisson-gain", type=float, default=1.0)
    parser.add_argument("--bg-std", type=float, default=0.5)
    parser.add_argument("--use-domain-cond", action="store_true", help="Append domain one-hot channel(s) as conditioner", default=True)
    args = parser.parse_args()
    # Enforce 128x128 regardless of user input
    args.image_size = 128

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if args.wandb:
        wandb.init(project=args.project, name=args.name, config=vars(args))

    # Data
    train_loader, val_loader = build_mnist_loaders(args.batch_size, args.num_workers, args.image_size)

    # Transform to model domain: use Poisson-friendly Anscombe VST
    to_model = AnscombeToModel(maxIntensity=1.0)

    # Model config (small UNet for 32x32, 1 channel)
    # Configure UNet in_channels based on conditioning
    domain_channels = 1 if args.use_domain_cond else 0  # MNIST domain only
    cond_channels = (1 if args.simulate_wf else 0) + domain_channels
    in_channels = 1 + cond_channels
    model_cfg = {
        "sample_size": args.image_size,
        "in_channels": in_channels,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": [64, 128, 256],
        "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    }
    unet = DenoisingUNet(model_cfg)

    # Trainer
    trainer_cfg = {
        "learning_rate": args.lr,
        "weight_decay": 1e-6,
        "use_scheduler": False,
        "num_timesteps": args.timesteps,
        "beta_schedule": "cosine",
        "use_ema": args.use_ema,
        "use_conditioning": cond_channels > 0,
        "supervised_x0_weight": 0.0,
        "max_epochs": args.max_epochs,
    }
    ddpm = DDPMTrainer(model=unet, config=trainer_cfg, transform=to_model)
    ddpm.to(device)

    # Optimizer
    optim_or_pair: Union[torch.optim.Optimizer, Tuple] = ddpm.configure_optimizers()
    if isinstance(optim_or_pair, tuple) or isinstance(optim_or_pair, list):
        optimizer = optim_or_pair[0][0] if isinstance(optim_or_pair[0], list) else optim_or_pair[0]
        scheduler = None
    else:
        optimizer = optim_or_pair  # type: ignore
        scheduler = None

    # Writer
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    scaler = GradScaler(enabled=(device == "cuda"))

    global_step = 0
    for epoch in range(args.max_epochs):
        ddpm.train()
        progress = tqdm(train_loader, desc=f"MNIST Epoch {epoch+1}/{args.max_epochs} [train]", leave=False)
        for batch_idx, (imgs, _labels) in enumerate(progress):
            imgs = imgs.to(device, non_blocking=True)
            # Optionally simulate WF-like measurement via Poisson + background
            if args.simulate_wf:
                # Build 2P-like clean intensity
                x_min = imgs.amin(dim=(1,2,3), keepdim=True)
                x_max = imgs.amax(dim=(1,2,3), keepdim=True)
                x2p = (imgs - x_min) / torch.clamp(x_max - x_min, min=1e-8)
                x2p = torch.pow(x2p, 0.8)
                x2p = x2p * 500.0 + 10.0
                y_wf = PoissonNoise.add_noise(x2p, gain=float(args.poisson_gain))
                y_wf = GaussianBackground.add_background(y_wf, mean=0.0, std=float(args.bg_std))
                y_wf = torch.clamp(y_wf, min=0.0)
                x_0 = x2p
                # Domain one-hot channels (MNIST only -> 1 channel all-ones)
                dom = torch.ones((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]), device=device) if args.use_domain_cond else None
                cond_parts = [y_wf]
                if dom is not None:
                    cond_parts.append(dom)
                cond = torch.cat(cond_parts, dim=1)
            else:
                x_0 = imgs
                cond = None
                if args.use_domain_cond:
                    cond = torch.ones((imgs.shape[0], 1, imgs.shape[2], imgs.shape[3]), device=device)

            with autocast(enabled=(device == "cuda")):
                if cond is None:
                    # Provide dummy zero conditioner to match API
                    cond_tensor = torch.zeros_like(x_0)
                else:
                    cond_tensor = cond
                loss = ddpm.training_step((to_model(x_0), to_model(cond_tensor) if cond_tensor is not None else to_model(cond_tensor)), batch_idx)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            cur = float(loss.detach().item())
            writer.add_scalar("train/loss", cur, global_step)
            if args.wandb:
                wandb.log({"train/loss": cur, "step": global_step})
            progress.set_postfix({"loss": f"{cur:.4f}"})

        # Simple val loop
        ddpm.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, _labels in tqdm(val_loader, desc=f"MNIST Epoch {epoch+1}/{args.max_epochs} [val]", leave=False):
                imgs = imgs.to(device)
                x_0 = to_model(imgs)
                zeros_cond = torch.zeros_like(x_0)
                loss = ddpm.validation_step((x_0, zeros_cond), 0)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1
        avg_val = val_loss_accum / max(1, val_batches)
        writer.add_scalar("val/loss", avg_val, epoch + 1)
        if args.wandb:
            wandb.log({"val/loss": avg_val, "epoch": epoch + 1})

        # Optional scheduler step
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass

        # Save per-epoch samples like ddpm.py
        try:
            ddpm.eval()
            with torch.no_grad():
                num_rows, num_cols = 2, 8
                num_images = num_rows * num_cols
                H = args.image_size
                W = args.image_size
                samples = ddpm.ddpm_sample(num_images=num_images, image_shape=(1, H, W), use_ema=True)
                samples = to_model.inverse(samples.clamp(-1, 1)).cpu().numpy()
                grid_h = num_rows * H
                grid_w = num_cols * W
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                for i in range(num_images):
                    r = i // num_cols
                    c = i % num_cols
                    img = samples[i, 0]
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
                out_dir = os.path.join("outputs", "samples", args.name)
                os.makedirs(out_dir, exist_ok=True)
                grid_img = (grid * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_img).save(os.path.join(out_dir, f"epoch_{epoch+1:03d}.png"))
        except Exception:
            pass

    try:
        writer.flush(); writer.close()
    except Exception:
        pass

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()


