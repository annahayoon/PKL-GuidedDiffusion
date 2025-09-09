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
        """Log only if a Lightning Trainer is attached to avoid warnings in tests.

        Accessing `self.trainer` on a LightningModule that is not attached raises.
        Wrap in try/except to keep standalone loops working.
        """
        try:
            trainer_ref = getattr(self, "trainer")  # Lightning property may raise
        except Exception:
            trainer_ref = None
        if trainer_ref is not None:
            try:
                # type: ignore[attr-defined]
                self.log(*args, **kwargs)  # pragma: no cover - passthrough when trainer present
            except Exception:
                pass

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
        # Precompute posterior coefficients as in ddpm.py for DDPM reverse sampling
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", posterior_log_variance_clipped)
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def _create_ema_model(self):
        from copy import deepcopy

        ema_model = deepcopy(self.model)
        ema_model.requires_grad_(False)
        return ema_model

    # ===== Helper conversions between model and intensity domains =====
    @staticmethod
    def _model_to_intensity(x_model: torch.Tensor, transform: Optional[Any]) -> torch.Tensor:
        """Map model domain [-1,1] to non-negative intensity domain via transform.inverse.

        If no transform is provided, assume inputs are already intensities and clamp to >= 0.
        """
        if transform is None:
            return torch.clamp(x_model, min=0)
        # Transform modules in this repo expose .inverse() for model->intensity
        return torch.clamp(transform.inverse(x_model), min=0)

    @staticmethod
    def _intensity_to_model(x_intensity: torch.Tensor, transform: Optional[Any]) -> torch.Tensor:
        """Map non-negative intensity domain to model domain [-1,1].

        If no transform is provided, pass through (assumes model trained in intensity domain).
        """
        if transform is None:
            return x_intensity
        return torch.clamp(transform(x_intensity), -1.0, 1.0)

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

    # ===== DDPM reverse-process utilities (mirroring ddpm.py) =====
    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Gather a[t] then reshape to broadcast over x."""
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        return (
            self._extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        model_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return model_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise: torch.Tensor, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True):
        # Recover x0 then compute posterior q(x_{t-1} | x_t, x0)
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, pred_noise: torch.Tensor, x: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(
            pred_noise, x=x, t=t, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

    # ===== Guided DDPM sampling per ICLR_2025.tex (PKL, L2, Anscombe) =====
    @torch.no_grad()
    def ddpm_guided_sample(
        self,
        y: torch.Tensor,
        forward_model: Any,
        guidance_strategy: Any,
        schedule: Optional[Any] = None,
        transform: Optional[Any] = None,
        num_steps: Optional[int] = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """Guided ancestral sampling using a physics-informed gradient.

        Args:
            y: Observed measurement tensor in intensity counts, shape [B, C, H, W]
            forward_model: Instance providing apply_psf(.), apply_psf_adjoint(.), and .background
            guidance_strategy: Instance with compute_gradient(x0_hat, y, forward_model, t)
            schedule: Instance with get_lambda_t(gradient, t) implementing the adaptive schedule
            transform: Transform object mapping intensity<->model domains (e.g., IntensityToModel)
            num_steps: Optional number of diffusion steps; defaults to configured self.num_timesteps
            use_ema: Whether to use EMA weights for prediction

        Returns:
            Samples in model domain [-1, 1] with shape [B, C, H, W]
        """
        device = next(self.model.parameters()).device
        B, C, H, W = y.shape
        steps = int(num_steps) if num_steps is not None else int(self.num_timesteps)

        # Initialize x_T ~ N(0, I) in model domain
        x_t = torch.randn((B, C, H, W), device=device)
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model

        for t_scalar in reversed(range(0, steps)):
            tt = torch.full((B,), t_scalar, device=device, dtype=torch.long)
            # Predict noise epsilon_theta(x_t, t)
            pred_noise = net(x_t, tt)

            # Recover current estimate x0_hat in model domain
            x0_hat_model = self.predict_start_from_noise(x_t, t=tt, noise=pred_noise)
            x0_hat_model = torch.clamp(x0_hat_model, -1.0, 1.0)

            # Convert to intensity domain for physics guidance
            x0_hat_int = self._model_to_intensity(x0_hat_model, transform)

            # Compute guidance gradient (in intensity domain) if components are provided
            if (forward_model is not None) and (guidance_strategy is not None) and (schedule is not None):
                grad = guidance_strategy.compute_gradient(x0_hat_int, y, forward_model, t_scalar)
                lambda_t = schedule.get_lambda_t(grad, t_scalar)
                x0_hat_int_guided = torch.clamp(x0_hat_int - float(lambda_t) * grad, min=0)
                x0_hat_model_guided = self._intensity_to_model(x0_hat_int_guided, transform)
            else:
                # No guidance; use uncorrected estimate
                x0_hat_model_guided = x0_hat_model

            # Use guided x0_hat in posterior q(x_{t-1} | x_t, x0)
            model_mean, _, model_log_variance = self.q_posterior(x_start=x0_hat_model_guided, x_t=x_t, t=tt)

            # DDPM noise injection (no noise when t == 0)
            noise = torch.randn_like(x_t)
            nonzero_mask = (tt != 0).float().view(B, 1, 1, 1)
            x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

        return x_t

    @torch.no_grad()
    def ddpm_sample(self, num_images: int, image_shape: tuple, use_ema: bool = True) -> torch.Tensor:
        """Generate samples by full DDPM ancestral sampling (slow).

        Args:
            num_images: Batch size
            image_shape: (C, H, W)
            use_ema: Whether to use EMA weights for prediction
        Returns:
            Samples in model domain [-1, 1]
        """
        device = next(self.model.parameters()).device
        samples = torch.randn((num_images, *image_shape), device=device)
        net = self.ema_model if (use_ema and getattr(self, 'use_ema', False)) else self.model
        for t in reversed(range(0, self.num_timesteps)):
            tt = torch.full((num_images,), t, device=device, dtype=torch.long)
            pred_noise = net(samples, tt)
            samples = self.p_sample(pred_noise, samples, tt, clip_denoised=True)
        return samples

    def training_step(self, batch, batch_idx):
        # Expect (target_2p, conditioner_wf)
        x_0, c_wf = batch
        b = x_0.shape[0]
        device = x_0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        # Pass conditioner to the model if enabled
        use_conditioning = bool(self.config.get("use_conditioning", True))
        if use_conditioning:
            try:
                noise_pred = self.model(x_t, t, cond=c_wf)
            except TypeError:
                noise_pred = self.model(x_t, t)
        else:
            noise_pred = self.model(x_t, t)
        loss = F.mse_loss(noise_pred, noise)
        # Optional supervised x0 loss to encourage paired mapping
        if self.config.get("supervised_x0_weight", 0.0) > 0:
            # Reconstruct x0 from epsilon prediction (epsilon parameterization)
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t + 1e-8)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t + 1e-8)
            x0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            loss_x0 = F.l1_loss(x0_hat, x_0)
            loss = loss + float(self.config.get("supervised_x0_weight", 0.0)) * loss_x0
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
        x_0, c_wf = batch
        device = x_0.device
        losses = []
        for t_val in [100, 500, 900]:
            t = torch.full((x_0.shape[0],), t_val, device=device)
            noise = torch.randn_like(x_0)
            x_t = self.q_sample(x_0, t, noise)
            use_conditioning = bool(self.config.get("use_conditioning", True))
            if use_conditioning:
                try:
                    noise_pred = self.model(x_t, t, cond=c_wf)
                except TypeError:
                    noise_pred = self.model(x_t, t)
            else:
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


