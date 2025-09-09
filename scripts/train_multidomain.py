#!/usr/bin/env python3
"""
Train a single diffusion model over multiple domains: MNIST, synthetic microscopy-like (ImageNet-like), and real WF/2P pairs.

- Uses HF datasets for MNIST (no torchvision/lzma).
- Uses SynthesisDataset for synthetic pairs and RealPairsDataset for real pairs.
- Model is configured with in_channels=2 to allow WF conditioning; for MNIST and synthesis, pass zeros conditioner.
- Per-epoch sample grids are saved to outputs/samples/{experiment}/epoch_XXX.png.
"""

import os
import argparse
from typing import Tuple, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from tqdm import tqdm
import wandb

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.noise import PoissonNoise, GaussianBackground


DOMAIN_WF = 0
DOMAIN_2P = 1
DOMAIN_MNIST = 2
DOMAIN_CIFAR = 3
DOMAIN_IMAGENET = 4


class HFMNIST(Dataset):
    def __init__(self, split: str, image_size: int):
        from datasets import load_dataset
        self.ds = load_dataset("mnist", split=split)
        self.size = image_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = item["image"].convert("L").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        # Create 2P-like clean intensity (match synthesis scaling)
        x_min = float(x.min()); x_max = float(x.max())
        x2p = (x - x_min) / max(x_max - x_min, 1e-8)
        x2p = torch.pow(x2p, 0.8)
        x2p = x2p * 500.0 + 10.0
        # Poisson + small Gaussian background to simulate WF measurement
        y_wf = PoissonNoise.add_noise(x2p, gain=1.0)
        y_wf = GaussianBackground.add_background(y_wf, mean=0.0, std=0.5)
        y_wf = torch.clamp(y_wf, min=0.0)
        return x2p, y_wf, DOMAIN_MNIST


class HFCIFAR10(Dataset):
    def __init__(self, split: str, image_size: int):
        from datasets import load_dataset
        self.ds = load_dataset("cifar10", split=split)
        self.size = image_size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[int(idx)]
        img = item["img"].convert("L").resize((self.size, self.size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)
        # 2P-like scaling
        x_min = float(x.min()); x_max = float(x.max())
        x2p = (x - x_min) / max(x_max - x_min, 1e-8)
        x2p = torch.pow(x2p, 0.8)
        x2p = x2p * 500.0 + 10.0
        # Poisson + Gaussian background
        y_wf = PoissonNoise.add_noise(x2p, gain=1.0)
        y_wf = GaussianBackground.add_background(y_wf, mean=0.0, std=0.5)
        y_wf = torch.clamp(y_wf, min=0.0)
        return x2p, y_wf, DOMAIN_CIFAR


class DomainTag(Dataset):
    def __init__(self, base: Dataset, domain_id: int):
        self.base = base
        self.domain_id = int(domain_id)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x0, ywf = self.base[idx]
        return x0, ywf, self.domain_id


def build_dataloaders(data_root: str, image_size: int, batch_size: int, num_workers: int,
                      psf_path: Optional[str], background: float) -> Tuple[DataLoader, DataLoader]:
    # MNIST (train/test)
    mnist_train = HFMNIST("train", image_size)
    mnist_val = HFMNIST("test", image_size)

    # CIFAR-10 (train/test)
    cifar_train = HFCIFAR10("train", image_size)
    cifar_val = HFCIFAR10("test", image_size)

    # Synthetic pairs (use forward model if psf_path provided)
    forward_model = None
    if psf_path:
        psf = PSF(psf_path)
        forward_model = ForwardModel(psf=psf.to_torch(device="cpu"), background=background, device="cpu")

    synth_train = SynthesisDataset(
        source_dir=os.path.join(data_root, "images", "train"),
        forward_model=forward_model,
        image_size=image_size,
        mode="train",
    )
    synth_val = SynthesisDataset(
        source_dir=os.path.join(data_root, "images", "val"),
        forward_model=forward_model,
        image_size=image_size,
        mode="val",
    )

    # Real pairs (WF/2P)
    real_train = RealPairsDataset(data_dir=os.path.join(data_root, "real_microscopy"), split="train", image_size=image_size, mode="train")
    real_val = RealPairsDataset(data_dir=os.path.join(data_root, "real_microscopy"), split="val", image_size=image_size, mode="val")

    # Tag datasets with domain IDs (WF for measurements synthesized or real)
    synth_train = DomainTag(synth_train, DOMAIN_WF)
    synth_val = DomainTag(synth_val, DOMAIN_WF)
    real_train = DomainTag(real_train, DOMAIN_WF)
    real_val = DomainTag(real_val, DOMAIN_WF)

    # Combine all domains
    train_concat = ConcatDataset([mnist_train, cifar_train, synth_train, real_train])
    val_concat = ConcatDataset([mnist_val, cifar_val, synth_val, real_val])

    train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_concat, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="pkl-diffusion-multidomain")
    parser.add_argument("--name", type=str, default="multidomain-ddpm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--psf-path", type=str, default="")
    parser.add_argument("--background", type=float, default=0.0)
    args = parser.parse_args()

    # Enforce 128x128 regardless of user input
    args.image_size = 128
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    if args.wandb:
        wandb.init(project=args.project, name=args.name, config=vars(args))

    print(f"[startup] name={args.name} size={args.image_size} batch={args.batch_size} workers={args.num_workers} device={device}", flush=True)

    # Data
    try:
        train_loader, val_loader = build_dataloaders(
            args.data_root, args.image_size, args.batch_size, args.num_workers,
            args.psf_path if args.psf_path else None, args.background
        )
        print(f"[data] train_len={len(train_loader.dataset)} val_len={len(val_loader.dataset)}", flush=True)
    except Exception as e:
        print(f"[error] dataloaders failed: {e}", flush=True)
        raise

    # Transform to model domain [-1,1] using a broad intensity window for microscopy
    to_model = IntensityToModel(minIntensity=0.0, maxIntensity=255.0)

    # Model: enable conditioning (x_t + wf)
    # x_t + conditioner (wf) + 5 domain-one-hot channels
    model_cfg = {
        "sample_size": args.image_size,
        "in_channels": 2 + 5,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": [64, 128, 256, 512],
        "down_block_types": ["DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"],
    }
    unet = DenoisingUNet(model_cfg)

    trainer_cfg = {
        "learning_rate": args.lr,
        "weight_decay": 1e-6,
        "use_scheduler": False,
        "num_timesteps": args.timesteps,
        "beta_schedule": "cosine",
        "use_ema": args.use_ema,
        "use_conditioning": True,
        "supervised_x0_weight": 0.1,
        "max_epochs": args.max_epochs,
    }
    ddpm = DDPMTrainer(model=unet, config=trainer_cfg, transform=to_model)
    ddpm.to(device)
    try:
        total_params = sum(p.numel() for p in ddpm.parameters())
        print(f"[model] params={total_params}", flush=True)
    except Exception:
        pass

    optim_or_pair: Union[torch.optim.Optimizer, Tuple] = ddpm.configure_optimizers()
    if isinstance(optim_or_pair, tuple) or isinstance(optim_or_pair, list):
        optimizer = optim_or_pair[0][0] if isinstance(optim_or_pair[0], list) else optim_or_pair[0]
        # Adaptive schedule: ReduceLROnPlateau on val loss
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)
    else:
        optimizer = optim_or_pair  # type: ignore
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    samples_dir = os.path.join("outputs", "samples", args.name)
    os.makedirs(samples_dir, exist_ok=True)

    scaler = GradScaler(enabled=(device == "cuda"))
    global_step = 0
    best_val = float('inf')
    patience = 7
    epochs_no_improve = 0

    print("[loop] entering training loop", flush=True)
    for epoch in range(args.max_epochs):
        print(f"[epoch] {epoch+1}/{args.max_epochs}", flush=True)
        ddpm.train()
        progress = tqdm(train_loader, desc=f"Multi Epoch {epoch+1}/{args.max_epochs} [train]", leave=False)
        for batch_idx, batch in enumerate(progress):
            # Unpack (x0, wf, domain_id)
            x_0, y_wf, dom_id = batch
            x_0 = x_0.to(device, non_blocking=True)
            y_wf = y_wf.to(device, non_blocking=True)
            dom_id = dom_id.to(device)
            x_0 = to_model(x_0)
            y_wf = to_model(y_wf)
            # Build 5-channel domain one-hot map
            b, _, h, w = x_0.shape
            dom_onehot = torch.zeros((b, 5, h, w), device=device, dtype=x_0.dtype)
            for i in range(b):
                idx = int(dom_id[i].item())
                dom_onehot[i, idx, :, :] = 1.0
            # Concatenate conditioner channels: wf + domain one-hot
            cond = torch.cat([y_wf, dom_onehot], dim=1)

            with autocast(enabled=(device == "cuda")):
                # Integrate with trainer's training_step via conditioning
                loss = ddpm.training_step((x_0, y_wf), batch_idx)

            # NaN/Inf guard
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected")
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

        # Validation
        ddpm.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Multi Epoch {epoch+1}/{args.max_epochs} [val]", leave=False):
                x_0, y_wf, dom_id = batch
                x_0 = to_model(x_0.to(device))
                y_wf = to_model(y_wf.to(device))
                b, _, h, w = x_0.shape
                dom_onehot = torch.zeros((b, 5, h, w), device=device, dtype=x_0.dtype)
                for i in range(b):
                    idx = int(dom_id[i].item())
                    dom_onehot[i, idx, :, :] = 1.0
                cond = torch.cat([y_wf, dom_onehot], dim=1)
                loss = ddpm.validation_step((x_0, cond), 0)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1
        avg_val = val_loss_accum / max(1, val_batches)
        writer.add_scalar("val/loss", avg_val, epoch + 1)
        if args.wandb:
            wandb.log({"val/loss": avg_val, "epoch": epoch + 1})

        if scheduler is not None:
            try:
                scheduler.step(avg_val)
            except Exception:
                pass

        # Early stopping and checkpoints
        ckpt_prefix = os.path.join("checkpoints", args.name)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(ddpm.state_dict(), f"{ckpt_prefix}_epoch_{epoch+1:03d}.pt")
        try:
            torch.save(ddpm.model.state_dict(), f"{ckpt_prefix}_model_epoch_{epoch+1:03d}.pt")
            if hasattr(ddpm, "ema_model") and ddpm.ema_model is not None:
                torch.save(ddpm.ema_model.state_dict(), f"{ckpt_prefix}_ema_epoch_{epoch+1:03d}.pt")
        except Exception:
            pass

        if avg_val < best_val - 1e-6:
            best_val = avg_val
            epochs_no_improve = 0
            torch.save(ddpm.state_dict(), f"{ckpt_prefix}_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        # Save per-epoch samples
        try:
            with torch.no_grad():
                num_rows, num_cols = 2, 8
                num_images = num_rows * num_cols
                H = args.image_size
                W = args.image_size
                samples = ddpm.ddpm_sample(num_images=num_images, image_shape=(1, H, W), use_ema=True)
                samples = (samples.clamp(-1, 1) + 1) / 2.0
                samples = samples.cpu().numpy()
                grid_h = num_rows * H
                grid_w = num_cols * W
                grid = np.zeros((grid_h, grid_w), dtype=np.float32)
                for i in range(num_images):
                    r = i // num_cols
                    c = i % num_cols
                    img = samples[i, 0]
                    grid[r*H:(r+1)*H, c*W:(c+1)*W] = img
                grid_img = (grid * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(grid_img).save(os.path.join(samples_dir, f"epoch_{epoch+1:03d}.png"))
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


