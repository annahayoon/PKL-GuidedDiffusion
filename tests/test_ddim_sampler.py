import torch
import torch.nn as nn

from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.guidance import PKLGuidance, AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.physics.forward_model import ForwardModel


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DummyTrainer(nn.Module):
    def __init__(self, T: int = 1000, use_ema: bool = False):
        super().__init__()
        self.model = DummyNet()
        self.use_ema = use_ema
        if use_ema:
            self.ema_model = DummyNet()
        # simple linear alphas_cumprod from 1->0
        alphas = torch.linspace(1.0, 1e-3, T)
        self.register_buffer("alphas_cumprod", alphas)


def _make_forward_model(device: str = "cpu") -> ForwardModel:
    psf = torch.ones(9, 9) / 81.0
    return ForwardModel(psf=psf, background=0.0, device=device)


def test_ddim_sampler_runs_and_returns_intensity():
    device = "cpu"
    trainer = DummyTrainer(T=100, use_ema=False).to(device)
    fm = _make_forward_model(device)
    guidance = PKLGuidance()
    schedule = AdaptiveSchedule(lambda_base=0.05, T_threshold=80, T_total=100)
    transform = IntensityToModel(min_intensity=0, max_intensity=1000)
    sampler = DDIMSampler(
        model=trainer,
        forward_model=fm,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=100,
        ddim_steps=10,
        eta=0.0,
    )

    B, C, H, W = 2, 1, 16, 16
    y = torch.rand(B, C, H, W, device=device) * 10.0 + 0.1
    x0 = sampler.sample(y=y, shape=(B, C, H, W), device=device, verbose=False)
    assert x0.shape == (B, C, H, W)
    assert torch.isfinite(x0).all()
    assert torch.all(x0 >= 0)


def test_ddim_sampler_uses_ema_if_available():
    device = "cpu"
    trainer = DummyTrainer(T=50, use_ema=True).to(device)
    fm = _make_forward_model(device)
    guidance = PKLGuidance()
    schedule = AdaptiveSchedule(lambda_base=0.05, T_threshold=40, T_total=50)
    transform = IntensityToModel(min_intensity=0, max_intensity=1000)
    sampler = DDIMSampler(
        model=trainer,
        forward_model=fm,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=50,
        ddim_steps=5,
        eta=0.0,
    )
    y = torch.rand(1, 1, 8, 8, device=device) + 0.2
    out = sampler.sample(y=y, shape=(1, 1, 8, 8), device=device, verbose=False)
    assert out.shape == (1, 1, 8, 8)


