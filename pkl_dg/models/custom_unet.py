import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, Any
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
            
        # Gradient checkpointing support
        self.gradient_checkpointing = False
    
    def _forward_impl(self, x, time_emb):
        """Internal forward implementation for checkpointing."""
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.dropout(h)
        
        # Residual connection
        return h + self.residual_conv(x)
    
    def forward(self, x, time_emb):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, time_emb, use_reentrant=False)
        else:
            return self._forward_impl(x, time_emb)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        self.gradient_checkpointing = False
    
    def _forward_impl(self, x, time_emb):
        """Internal forward implementation for checkpointing."""
        for layer in self.layers:
            x = layer(x, time_emb)
        return self.downsample(x)
    
    def forward(self, x, time_emb):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, time_emb, use_reentrant=False)
        else:
            return self._forward_impl(x, time_emb)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_layers=2):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        self.layers = nn.ModuleList([
            ResidualBlock(in_channels if i == 0 else out_channels, out_channels, time_emb_dim)
            for i in range(num_layers)
        ])
        self.gradient_checkpointing = False
    
    def _forward_impl(self, x, time_emb):
        """Internal forward implementation for checkpointing."""
        x = self.upsample(x)
        for layer in self.layers:
            x = layer(x, time_emb)
        return x
    
    def forward(self, x, time_emb):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, time_emb, use_reentrant=False)
        else:
            return self._forward_impl(x, time_emb)

class CustomUNet(nn.Module):
    """Custom UNet implementation for diffusion denoising.

    Expects input with `in_channels` that may already include conditioning
    channels (e.g., x_t and WF conditioner concatenated).
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Parse config
        self.in_channels = config.get("in_channels", 1)
        self.out_channels = config.get("out_channels", 1)
        self.sample_size = config.get("sample_size", 256)
        self.block_out_channels = config.get("block_out_channels", [64, 128, 256, 512])
        self.layers_per_block = config.get("layers_per_block", 2)
        
        # Time embedding
        time_emb_dim = self.block_out_channels[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.block_out_channels[0]),
            nn.Linear(self.block_out_channels[0], time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Input projection
        self.conv_in = nn.Conv2d(self.in_channels, self.block_out_channels[0], 3, padding=1)
        
        # Down blocks
        self.down_blocks = nn.ModuleList()
        in_ch = self.block_out_channels[0]
        for out_ch in self.block_out_channels[1:]:
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, time_emb_dim, self.layers_per_block)
            )
            in_ch = out_ch
        
        # Middle block
        self.middle_block = ResidualBlock(
            self.block_out_channels[-1], 
            self.block_out_channels[-1], 
            time_emb_dim
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList()
        in_ch = self.block_out_channels[-1]
        for out_ch in reversed(self.block_out_channels[:-1]):
            self.up_blocks.append(
                UpBlock(in_ch, out_ch, time_emb_dim, self.layers_per_block)
            )
            in_ch = out_ch
        
        # Output projection
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, self.block_out_channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.block_out_channels[0], self.out_channels, 3, padding=1),
        )
        
        # Gradient checkpointing support
        self.gradient_checkpointing = False
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = True
        
        # Enable for all ResidualBlocks in down blocks
        for down_block in self.down_blocks:
            down_block.gradient_checkpointing = True
            for layer in down_block.layers:
                layer.gradient_checkpointing = True
        
        # Enable for middle block
        self.middle_block.gradient_checkpointing = True
        
        # Enable for all ResidualBlocks in up blocks
        for up_block in self.up_blocks:
            up_block.gradient_checkpointing = True
            for layer in up_block.layers:
                layer.gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for all blocks."""
        self.gradient_checkpointing = False
        
        # Disable for all ResidualBlocks in down blocks
        for down_block in self.down_blocks:
            down_block.gradient_checkpointing = False
            for layer in down_block.layers:
                layer.gradient_checkpointing = False
        
        # Disable for middle block
        self.middle_block.gradient_checkpointing = False
        
        # Disable for all ResidualBlocks in up blocks
        for up_block in self.up_blocks:
            up_block.gradient_checkpointing = False
            for layer in up_block.layers:
                layer.gradient_checkpointing = False
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_mlp(t)
        
        # Input
        h = self.conv_in(x)
        
        # Down blocks
        for down_block in self.down_blocks:
            h = down_block(h, time_emb)
        
        # Middle block
        h = self.middle_block(h, time_emb)
        
        # Up blocks
        for up_block in self.up_blocks:
            h = up_block(h, time_emb)
        
        # Output
        return self.conv_out(h)
