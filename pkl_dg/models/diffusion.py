from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pytorch_lightning as pl  # type: ignore

    LightningModuleBase = pl.LightningModule  # type: ignore
except Exception:  # pragma: no cover - fallback when lightning is unavailable
    class LightningModuleBase(nn.Module):
        def __init__(self):
            super().__init__()
            self._global_step = 0

        def log(self, *args, **kwargs):
            # no-op in fallback
            return None

        @property
        def global_step(self) -> int:
            return getattr(self, "_global_step", 0)


class DDPMTrainer(LightningModuleBase):
    """DDPM training with PyTorch Lightning."""

    def __init__(
        self, model: nn.Module, config: Dict[str, Any], transform: Optional[Any] = None
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.transform = transform
        self.num_timesteps = config.get("num_timesteps", 1000)
        self.beta_schedule = config.get("beta_schedule", "cosine")
        self._setup_noise_schedule()
        self.use_ema = config.get("use_ema", True)
        if self.use_ema:
            self.ema_model = self._create_ema_model()

    def _log_if_trainer(self, *args, **kwargs) -> None:
        """Log only if a Lightning Trainer is attached to avoid warnings in tests."""
        trainer_ref = getattr(self, "trainer", None)
        if trainer_ref is not None:
            # type: ignore[attr-defined]
            self.log(*args, **kwargs)  # pragma: no cover - passthrough when trainer present

    def _setup_noise_schedule(self):
        if self.beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            steps = self.num_timesteps + 1
            x = torch.linspace(0, self.num_timesteps, steps)
            alphas_cumprod = (
                torch.cos(((x / self.num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            )
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def _create_ema_model(self):
        from copy import deepcopy

        ema_model = deepcopy(self.model)
        ema_model.requires_grad_(False)
        return ema_model

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1
        )
        return sqrt_alpha_t * x_0 + sqrt_one_minus_alpha_t * noise

    def training_step(self, batch, batch_idx):
        x_0, _ = batch
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        self._log_if_trainer("train/loss", loss, prog_bar=True)
        if self.use_ema and self.global_step % 10 == 0:
            self._update_ema()
        # advance step counter in fallback environments
        if not hasattr(self, "_global_step"):
            self._global_step = 0
        self._global_step += 1
        return loss

    def _update_ema(self, decay: float = 0.999):
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def validation_step(self, batch, batch_idx):
        x_0, _ = batch
        device = x_0.device
        losses = []
        for t_val in [100, 500, 900]:
            t = torch.full((x_0.shape[0],), t_val, device=device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            noise_pred = self.model(x_t, t)
            losses.append(F.mse_loss(noise_pred, noise))
        avg_loss = torch.stack(losses).mean()
        self._log_if_trainer("val/loss", avg_loss, prog_bar=True)
        return avg_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 1e-4),
            betas=(0.9, 0.999),
            weight_decay=self.config.get("weight_decay", 1e-6),
        )
        if self.config.get("use_scheduler", True):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.get("max_epochs", 100), eta_min=1e-6
            )
            return [optimizer], [scheduler]
        return optimizer


