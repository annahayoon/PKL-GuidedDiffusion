import os
import argparse
from omegaconf import OmegaConf
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.real_pairs import RealPairsDataset
from pkl_dg.data.transforms import IntensityToModel, AnscombeToModel


def build_transform(noise_model: str, min_intensity: float, max_intensity: float):
    if noise_model.lower() == "poisson":
        return AnscombeToModel(maxIntensity=float(max_intensity))
    return IntensityToModel(minIntensity=float(min_intensity), maxIntensity=float(max_intensity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/real_microscopy")
    parser.add_argument("--checkpoints", type=str, default="checkpoints/real")
    parser.add_argument("--logs", type=str, default="logs/real")
    parser.add_argument("--name", type=str, default="real_run")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=600)  # 500-800 target
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--conditioning", action="store_true")
    parser.add_argument("--noise_model", type=str, default="poisson", choices=["poisson", "gaussian"]) 
    parser.add_argument("--min_intensity", type=float, default=0.0)
    parser.add_argument("--max_intensity", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--use_diffusers_scheduler", action="store_true")
    parser.add_argument("--scheduler_type", type=str, default="ddpm", choices=["ddpm", "ddim", "dpm_solver"]) 
    args = parser.parse_args()

    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.logs, args.name))

    # Dataset & loaders
    transform = build_transform(args.noise_model, args.min_intensity, args.max_intensity)
    train_ds = RealPairsDataset(args.data_root, split="train", transform=transform)
    val_ds = RealPairsDataset(args.data_root, split="val", transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Model & trainer
    model_cfg = {
        "sample_size": args.image_size,
        "in_channels": 2 if args.conditioning else 1,
        "out_channels": 1,
        "gradient_checkpointing": True,
    }
    unet = DenoisingUNet(model_cfg)

    training_cfg = {
        "num_timesteps": 1000,
        "use_ema": bool(args.ema),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "mixed_precision": bool(args.mixed_precision),
        "use_diffusers_scheduler": bool(args.use_diffusers_scheduler),
        "scheduler_type": args.scheduler_type,
        "use_conditioning": bool(args.conditioning),
        "beta_schedule": "cosine",
        "max_epochs": args.max_epochs,
    }
    ddpm = DDPMTrainer(unet, training_cfg, transform=transform)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddpm = ddpm.to(device)

    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=bool(args.mixed_precision) and device == "cuda")

    # Early stopping
    best_val = float("inf")
    best_epoch = -1
    patience = args.early_stop_patience

    # Training loop
    for epoch in range(1, args.max_epochs + 1):
        ddpm.train()
        train_loss_accum = 0.0
        train_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.max_epochs} [train]", leave=False)
        for x_0, y_wf in pbar:
            x_0 = x_0.to(device, non_blocking=True)
            y_wf = y_wf.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.mixed_precision and device == "cuda":
                with autocast(dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    loss = ddpm.training_step((x_0, y_wf), 0)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = ddpm.training_step((x_0, y_wf), 0)
                loss.backward()
                optimizer.step()

            train_loss_accum += float(loss.detach().item())
            train_batches += 1
            writer.add_scalar("train/loss", float(loss.detach().item()), (epoch * 100000) + train_batches)

        avg_train = train_loss_accum / max(1, train_batches)

        # Validation
        ddpm.eval()
        val_loss_accum = 0.0
        val_batches = 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.max_epochs} [val]", leave=False)
            for x_0, y_wf in vbar:
                x_0 = x_0.to(device, non_blocking=True)
                y_wf = y_wf.to(device, non_blocking=True)
                loss = ddpm.validation_step((x_0, y_wf), 0)
                val_loss_accum += float(loss.detach().item())
                val_batches += 1

        avg_val = val_loss_accum / max(1, val_batches)
        writer.add_scalar("epoch/train_loss", avg_train, epoch)
        writer.add_scalar("epoch/val_loss", avg_val, epoch)

        # Checkpointing
        if epoch % max(1, args.save_every) == 0:
            ckpt_prefix = os.path.join(args.checkpoints, f"epoch_{epoch:04d}")
            torch.save(ddpm.state_dict(), f"{ckpt_prefix}_trainer.pt")
            torch.save(ddpm.model.state_dict(), f"{ckpt_prefix}_model.pt")
            if getattr(ddpm, "ema_model", None) is not None:
                torch.save(ddpm.ema_model.state_dict(), f"{ckpt_prefix}_ema_model.pt")

        # Early stopping on validation loss
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            best_epoch = epoch
            torch.save(ddpm.state_dict(), os.path.join(args.checkpoints, "best_trainer.pt"))
            torch.save(ddpm.model.state_dict(), os.path.join(args.checkpoints, "best_model.pt"))
            if getattr(ddpm, "ema_model", None) is not None:
                torch.save(ddpm.ema_model.state_dict(), os.path.join(args.checkpoints, "best_ema_model.pt"))
        elif epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (best={best_epoch}, val={best_val:.4f})")
            break

    writer.flush(); writer.close()


if __name__ == "__main__":
    main()


