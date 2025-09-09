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

# Import our custom UNet
from .custom_unet import CustomUNet


class DenoisingUNet(nn.Module):
    """UNet wrapper for diffusion denoising.

    Supports optional conditioning via channel concatenation: if a conditioner
    tensor is provided, it will be concatenated along the channel dimension
    with the noised input before passing through the UNet. Ensure that the
    underlying UNet `in_channels` matches the total channels (e.g., 2 for
    WF→2P with x_t and WF as conditioner).
    """

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
            # Use our custom UNet implementation
            self.unet = CustomUNet(config)
        else:
            self.unet = UNet2DModel(**config)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        # Concatenate conditioner if provided and model expects it (in_channels >= x+cond)
        if cond is not None:
            if cond.dim() == 3:
                cond = cond.unsqueeze(0)
            if cond.shape[0] != x.shape[0]:
                cond = cond.expand(x.shape[0], *cond.shape[1:])
            # Only concatenate if UNet was configured for extra channel(s)
            try:
                expected_in = getattr(self.unet, 'in_channels', None)
            except Exception:
                expected_in = None
            if expected_in is None or (x.shape[1] + cond.shape[1]) == expected_in:
                x = torch.cat([x, cond], dim=1)

        if UNet2DModel is None:
            # Custom UNet returns tensor directly
            return self.unet(x, t)
        else:
            # Diffusers UNet returns object with .sample
            return self.unet(x, t).sample


