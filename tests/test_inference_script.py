from pathlib import Path

import numpy as np
import pytest
import torch
import tifffile
from omegaconf import OmegaConf

from scripts.inference import run_inference
from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer


def _save_dummy_checkpoint(checkpoint_path: Path, model_cfg: dict, training_cfg: dict) -> None:
    unet = DenoisingUNet(model_cfg)
    ddpm = DDPMTrainer(model=unet, config=training_cfg)
    torch.save(ddpm.state_dict(), str(checkpoint_path))


def _write_input_tiffs(input_dir: Path, num_images: int = 2, size: int = 32) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)
    for i in range(num_images):
        arr = (rng.rand(size, size).astype(np.float32) + 0.1).astype(np.float32)
        tifffile.imwrite(str(input_dir / f"meas_{i}.tif"), arr)


@pytest.mark.cpu
@pytest.mark.parametrize("guidance_type", ["pkl", "l2", "anscombe"])
def test_run_inference_writes_outputs(tmp_path: Path, guidance_type: str):
    # Directories
    data_dir = tmp_path / "data"
    inputs_dir = data_dir / "val"
    outputs_dir = tmp_path / "outs"
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "final_model.pt"

    # Configs
    model_cfg = {
        "sample_size": 32,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 1,
        "block_out_channels": [8, 16],
        "down_block_types": ["DownBlock2D", "DownBlock2D"],
        "up_block_types": ["UpBlock2D", "UpBlock2D"],
    }
    training_cfg = {
        "num_timesteps": 10,
        "beta_schedule": "cosine",
        "use_ema": False,
        "learning_rate": 1e-4,
        "use_scheduler": False,
        "max_epochs": 1,
    }

    # Save a minimal checkpoint compatible with the constructed trainer
    _save_dummy_checkpoint(checkpoint_path, model_cfg, training_cfg)

    # Write input images
    _write_input_tiffs(inputs_dir, num_images=2, size=32)

    # Build OmegaConf config matching Hydra structure expected by run_inference
    cfg = OmegaConf.create(
        {
            "experiment": {"device": "cpu", "seed": 0, "mixed_precision": False},
            "paths": {"data": str(data_dir), "outputs": str(tmp_path / "outputs"), "checkpoints": str(ckpt_dir)},
            "physics": {"psf_path": None, "background": 0.0},
            "data": {"image_size": 32, "min_intensity": 0.0, "max_intensity": 1000.0},
            "training": training_cfg,
            "model": model_cfg,
            "guidance": {
                "type": guidance_type,
                "epsilon": 1.0e-6,
                "lambda_base": 0.05,
                "schedule": {"T_threshold": 8, "epsilon_lambda": 1.0e-3},
            },
            "inference": {
                "checkpoint_path": str(checkpoint_path),
                "input_dir": str(inputs_dir),
                "output_dir": str(outputs_dir),
                "ddim_steps": 5,
                "eta": 0.0,
            },
        }
    )

    saved_paths = run_inference(cfg)

    # Verify two outputs were saved
    assert len(saved_paths) == 2
    for out_path in saved_paths:
        assert out_path.exists()
        assert out_path.suffix == ".tif"
        arr = tifffile.imread(str(out_path))
        assert arr.dtype == np.float32
        assert arr.shape == (32, 32)


