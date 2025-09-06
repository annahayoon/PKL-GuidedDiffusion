import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from pkl_dg.data import SynthesisDataset


class DummyForwardModel:
    def forward(self, x: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        # x: [B, 1, H, W]
        y = x.clone()
        if add_noise:
            y = y + 0.05 * torch.randn_like(y)
        return y


def _create_temp_images(tmp_dir: Path, num_images: int = 4, size: int = 64):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_images):
        arr = (np.random.rand(size, size) * 255).astype(np.uint8)
        Image.fromarray(arr).save(tmp_dir / f"img_{i}.png")


def test_synthesis_dataset_basic(tmp_path):
    source_dir = tmp_path / "images"
    _create_temp_images(source_dir)

    ds = SynthesisDataset(
        sourceDir=str(source_dir),
        forwardModel=DummyForwardModel(),
        imageSize=64,
        mode="train",
    )

    assert len(ds) == 4

    x_2p, y_wf = ds[0]
    assert isinstance(x_2p, torch.Tensor) and isinstance(y_wf, torch.Tensor)
    assert x_2p.ndim == 3 and y_wf.ndim == 3  # [C, H, W]
    assert x_2p.shape[0] == 1 and y_wf.shape[0] == 1
    assert x_2p.shape[1] == 64 and x_2p.shape[2] == 64

    # 2P is positive after scaling
    assert torch.all(x_2p >= 0)
    # y_wf should be finite and same shape
    assert torch.isfinite(y_wf).all()
    assert y_wf.shape == x_2p.shape


def test_synthesis_dataset_eval_mode(tmp_path):
    source_dir = tmp_path / "images"
    _create_temp_images(source_dir)

    ds = SynthesisDataset(
        sourceDir=str(source_dir),
        forwardModel=DummyForwardModel(),
        imageSize=32,
        mode="val",
    )

    x_2p, y_wf = ds[1]
    assert x_2p.shape[-1] == 32 and y_wf.shape[-1] == 32
    # No strong assertion on noise, but should be finite
    assert torch.isfinite(y_wf).all()


