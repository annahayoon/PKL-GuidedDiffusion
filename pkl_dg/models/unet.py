from typing import Any, Dict, Optional
import warnings

import torch
import torch.nn as nn
from types import SimpleNamespace

try:
    from diffusers import UNet2DModel  # type: ignore
    DIFFUSERS_AVAILABLE = True
except (ImportError, RuntimeError, ModuleNotFoundError) as e:  # Handle various import failures
    DIFFUSERS_AVAILABLE = False
    UNet2DModel = None  # type: ignore
    import warnings
    warnings.warn(f"Diffusers not available ({e.__class__.__name__}). Using custom UNet implementation.")

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

# Import our custom UNet as fallback
from .custom_unet import CustomUNet


def create_optimized_unet_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create optimized UNet configuration for microscopy applications.
    
    This function implements the enhanced UNet configuration from enhancement.md
    with memory-efficient settings and attention optimizations.
    """
    # Default optimized configuration for microscopy (compatible with diffusers 0.24+)
    optimized_config = {
        "sample_size": config.get("sample_size", 256),
        "in_channels": config.get("in_channels", 2),  # x_t + conditioner
        "out_channels": config.get("out_channels", 1),
        "layers_per_block": 2,
        "block_out_channels": (128, 256, 512, 512),  # Memory-efficient progression
        "down_block_types": (
            "DownBlock2D",
            "AttnDownBlock2D", 
            "AttnDownBlock2D",
            "DownBlock2D"  # No attention on highest resolution
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D", 
            "UpBlock2D"
        ),
        "attention_head_dim": 8,  # Reduces memory usage vs default 64
        "norm_num_groups": 8,     # Efficient group normalization
        "act_fn": "silu",
        "norm_eps": 1e-5,
    }
    
    # Override with user-provided config, but filter out unsupported keys
    supported_keys = set(optimized_config.keys())
    for key, value in config.items():
        if key in supported_keys:
            optimized_config[key] = value
    
    return optimized_config


class DenoisingUNet(nn.Module):
    """Enhanced UNet wrapper for diffusion denoising with optimizations.

    Prioritizes diffusers.UNet2DModel for better performance and memory efficiency.
    Supports optional conditioning via channel concatenation and gradient checkpointing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize optimized UNet from config.

        Args:
            config: Model configuration dict. Will be enhanced with optimizations.
        """
        super().__init__()
        
        # Store original config and create optimized version
        self.config = config
        self.use_diffusers = config.get("use_diffusers", True) and DIFFUSERS_AVAILABLE
        self.gradient_checkpointing = config.get("gradient_checkpointing", False)
        
        if self.use_diffusers:
            # Use optimized diffusers UNet (Task 1.1 implementation)
            optimized_config = create_optimized_unet_config(config)
            
            try:
                self.unet = UNet2DModel(**optimized_config)
                self._using_diffusers = True
                
                # Enable gradient checkpointing if requested
                if self.gradient_checkpointing:
                    self.unet.enable_gradient_checkpointing()
                    
                print(f"✅ Using optimized diffusers UNet2DModel with {self.unet.config.in_channels} input channels")
                
            except Exception as e:
                warnings.warn(f"Failed to create diffusers UNet: {e}. Falling back to custom implementation.")
                self.unet = CustomUNet(config)
                self._using_diffusers = False
        else:
            # Fallback to custom implementation
            if not DIFFUSERS_AVAILABLE:
                warnings.warn("Diffusers not available. Using custom UNet implementation.")
            self.unet = CustomUNet(config)
            self._using_diffusers = False
            
        # Store input channels for conditioning logic
        if self._using_diffusers:
            self.in_channels = self.unet.config.in_channels
        else:
            self.in_channels = getattr(self.unet, 'in_channels', config.get('in_channels', 1))

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        if self._using_diffusers and hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
        elif hasattr(self.unet, 'enable_gradient_checkpointing'):
            self.unet.enable_gradient_checkpointing()
            
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        if self._using_diffusers and hasattr(self.unet, 'disable_gradient_checkpointing'):
            self.unet.disable_gradient_checkpointing()
        elif hasattr(self.unet, 'disable_gradient_checkpointing'):
            self.unet.disable_gradient_checkpointing()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional conditioning.
        
        Args:
            x: Noisy input tensor [B, C, H, W]
            t: Timestep tensor [B]
            cond: Optional conditioning tensor [B, C_cond, H, W]
            
        Returns:
            Predicted noise tensor [B, C, H, W]
        """
        # Handle conditioning via channel concatenation
        if cond is not None:
            # Ensure conditioning tensor has correct batch dimension
            if cond.dim() == 3:
                cond = cond.unsqueeze(0)
            if cond.shape[0] != x.shape[0]:
                cond = cond.expand(x.shape[0], *cond.shape[1:])
                
            # Concatenate if model expects it
            expected_channels = self.in_channels
            current_channels = x.shape[1] + cond.shape[1]
            
            if current_channels == expected_channels:
                x = torch.cat([x, cond], dim=1)
            elif x.shape[1] == expected_channels:
                # Model doesn't expect conditioning, ignore it
                pass
            else:
                warnings.warn(f"Channel mismatch: model expects {expected_channels}, got {current_channels}")
        else:
            # No conditioning provided - pad with zeros if model expects more channels
            expected_channels = self.in_channels
            current_channels = x.shape[1]
            
            if current_channels < expected_channels:
                padding_channels = expected_channels - current_channels
                padding = torch.zeros(x.shape[0], padding_channels, x.shape[2], x.shape[3], 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)

        # Forward pass through UNet
        if self._using_diffusers:
            # Diffusers UNet returns object with .sample
            return self.unet(x, t).sample
        else:
            # Custom UNet returns tensor directly
            return self.unet(x, t)
            
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        return {"allocated_gb": 0, "reserved_gb": 0, "max_allocated_gb": 0}


