from typing import Any, Dict

import torch
import torch.nn as nn
from types import SimpleNamespace

try:
    from diffusers import UNet2DModel  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without diffusers deps
    UNet2DModel = None  # type: ignore

    class _TinyUNet(nn.Module):
        """
        Minimal UNet-like fallback to avoid heavy diffusers dependency during tests.
        Matches the forward API returning an object with .sample tensor.
        """

        def __init__(self, in_channels: int, out_channels: int, **_: Any):
            super().__init__()
            hidden = 32
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1),
            )

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> SimpleNamespace:
            out = self.net(x)
            return SimpleNamespace(sample=out)


class DenoisingUNet(nn.Module):
    """UNet wrapper for diffusion denoising."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize UNet from config.

        Args:
            config: Model configuration dict passed directly to UNet2DModel
        """
        super().__init__()

        # Create UNet directly from config dict
        # Example fields: sample_size, in_channels, out_channels, block_out_channels, down_block_types, up_block_types
        if UNet2DModel is None:
            # Map config to fallback constructor
            in_ch = int(config.get("in_channels", 1))
            out_ch = int(config.get("out_channels", in_ch))
            self.unet = _TinyUNet(in_channels=in_ch, out_channels=out_ch)
        else:
            self.unet = UNet2DModel(**config)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t).sample


