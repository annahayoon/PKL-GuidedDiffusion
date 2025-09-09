import os
from pathlib import Path
import re
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import tifffile
from tqdm import tqdm
from PIL import Image

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.psf_estimator import build_psf_bank
 
from pkl_dg.guidance.pkl import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel


def run_inference(cfg: DictConfig) -> List[Path]:
    """Run guided diffusion inference and return saved file paths."""

    # Device
    device = str(cfg.experiment.device)

    # Load model and wrap in trainer (to get buffers/noise schedule)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    use_conditioning = bool(getattr(cfg.training, "use_conditioning", False))
    conditioning_type = str(getattr(cfg.training, "conditioning_type", "wf")).lower()
    if use_conditioning and conditioning_type == "wf" and int(model_cfg.get("in_channels", 1)) == 1:
        model_cfg["in_channels"] = 2
    unet = DenoisingUNet(model_cfg)
    ddpm = DDPMTrainer(
        model=unet,
        config=OmegaConf.to_container(cfg.training, resolve=True),
    )

    checkpoint_path = Path(str(cfg.inference.checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Load state dict (includes EMA weights if present)
    ddpm.load_state_dict(state_dict, strict=False)
    ddpm.eval()
    ddpm.to(device)

    # Allow toggling EMA usage from config (default true per ICLR spec)
    try:
        ddpm.use_ema = bool(getattr(cfg.inference, "use_ema", True))
    except Exception:
        ddpm.use_ema = True

    # Forward model per paper (PSF + background)
    forward_model = None
    try:
        phys_cfg = cfg.physics
        use_psf = bool(getattr(phys_cfg, "use_psf", False))
        background = float(getattr(phys_cfg, "background", 0.0))
        if use_psf:
            psf_path = getattr(phys_cfg, "psf_path", None)
            use_bead = bool(getattr(phys_cfg, "use_bead_psf", False))
            if use_bead:
                beads_dir = str(getattr(phys_cfg, "beads_dir", ""))
                bank = build_psf_bank(beads_dir)
                mode = getattr(phys_cfg, "bead_mode", None)
                if mode is None:
                    psf_t = bank.get("with_AO", bank.get("no_AO"))
                else:
                    psf_t = bank.get(str(mode), next(iter(bank.values())))
                psf = psf_t.to(device=device, dtype=torch.float32)
                if psf.ndim == 2:
                    psf = psf.unsqueeze(0).unsqueeze(0)
            elif psf_path is not None:
                psf = PSF(psf_path=str(psf_path)).to_torch(device=device)
            else:
                psf = PSF().to_torch(device=device)
            forward_model = ForwardModel(psf=psf, background=background, device=device)
    except Exception:
        forward_model = None

    # Guidance
    guidance_type = str(getattr(cfg.guidance, "type", "pkl"))
    epsilon = float(getattr(cfg.guidance, "epsilon", 1e-6))
    if guidance_type == "pkl":
        guidance = PKLGuidance(epsilon=epsilon)
    elif guidance_type == "l2":
        guidance = L2Guidance()
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")

    # Schedule
    lambda_base = float(getattr(cfg.guidance, "lambda_base", 0.1))
    schedule_cfg = getattr(cfg.guidance, "schedule", {})
    T_threshold = int(getattr(schedule_cfg, "T_threshold", 800))
    epsilon_lambda = float(getattr(schedule_cfg, "epsilon_lambda", 1e-3))
    schedule = AdaptiveSchedule(
        lambda_base=lambda_base,
        T_threshold=T_threshold,
        epsilon_lambda=epsilon_lambda,
        T_total=int(cfg.training.num_timesteps),
    )

    # Transform
    transform = IntensityToModel(
        min_intensity=float(cfg.data.min_intensity),
        max_intensity=float(cfg.data.max_intensity),
    )

    # Sampler (DDIM by default; can be extended to DDPM guided sampling/averaging)
    sampler = DDIMSampler(
        model=ddpm,
        forward_model=forward_model,
        guidance_strategy=guidance,
        schedule=schedule,
        transform=transform,
        num_timesteps=int(cfg.training.num_timesteps),
        ddim_steps=int(cfg.inference.ddim_steps),
        eta=float(cfg.inference.eta),
        use_autocast=bool(getattr(cfg.inference, "use_autocast", True)),
    )

    # IO
    input_dir = Path(str(cfg.inference.input_dir))
    output_dir = Path(str(cfg.inference.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Support tif/tiff/png inputs
    image_paths = []
    for ext in ("*.tif", "*.tiff", "*.png"):
        image_paths += list(input_dir.glob(ext))

    saved_paths: List[Path] = []
    max_intensity = float(cfg.data.max_intensity)

    for img_path in tqdm(image_paths, desc="Inference"):
        # Load measurement (PNG/TIF)
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            img_np = np.array(Image.open(img_path))
        else:
            img_np = tifffile.imread(str(img_path))

        y = torch.from_numpy(img_np).float().to(device)
        if y.ndim == 2:
            y = y.unsqueeze(0).unsqueeze(0)
        elif y.ndim == 3 and y.shape[-1] in (3, 4):
            # If RGB/RGBA, convert to grayscale by averaging
            y = y.mean(dim=-1, keepdim=False).unsqueeze(0).unsqueeze(0)
        # Optional per-image scaling: match dynamic range to configured intensity
        if bool(getattr(cfg.inference, "auto_scale_measurement", True)):
            y_min = float(y.amin().item())
            y_max = float(y.amax().item())
            if y_max > y_min:
                y = (y - y_min) / max(1e-8, (y_max - y_min))
                y = y * float(cfg.data.max_intensity)
        shape = tuple(y.shape)

        use_conditioning = bool(getattr(cfg.training, "use_conditioning", True))
        conditioner = transform(y) if (use_conditioning and conditioning_type == "wf") else None
        # DDIM: deterministic fast inference; DDPM averaging if configured
        num_samples = int(getattr(cfg.inference, "num_samples_avg", 1))
        if num_samples <= 1:
            reconstruction = sampler.sample(y, shape, device=device, verbose=False, conditioner=conditioner)
        else:
            # Average multiple stochastic DDPM reconstructions via ddpm_guided_sample
            preds = []
            for _ in range(num_samples):
                pred = ddpm.ddpm_guided_sample(
                    y=y,
                    forward_model=forward_model,
                    guidance_strategy=guidance,
                    schedule=schedule,
                    transform=transform,
                    num_steps=int(cfg.training.num_timesteps),
                    use_ema=bool(getattr(cfg.inference, "use_ema", True)),
                )
                preds.append(pred)
            reconstruction = torch.stack(preds, dim=0).mean(dim=0)
        out = reconstruction.squeeze().detach().cpu().numpy().astype(np.float32)

        # Save TIF
        tif_path = output_dir / f"{img_path.stem}_reconstructed.tif"
        tifffile.imwrite(str(tif_path), out)
        saved_paths.append(tif_path)

        # Save PNG (scaled to 0-255 based on intensity range)
        out_png = np.clip(out / max_intensity * 255.0, 0, 255).astype(np.uint8)
        png_path = output_dir / f"{img_path.stem}_reconstructed.png"
        Image.fromarray(out_png).save(str(png_path))

        # Also save composite WF | Prediction | GT 2P if GT exists
        try:
            # WF input (grayscale 0-255)
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                wf_np = np.array(Image.open(img_path))
            else:
                wf_np = tifffile.imread(str(img_path))
            if wf_np.ndim == 3:
                wf_np = wf_np.mean(axis=-1)
            wf_u8 = np.clip(wf_np, 0, 255).astype(np.uint8)

            # Prediction already prepared as out_png (uint8)
            pred_u8 = out_png

            # Ground truth path (if using real_pairs structure)
            gt_path = img_path.parent.parent / "2p" / img_path.name
            if gt_path.exists():
                gt_np = np.array(Image.open(gt_path))
                if gt_np.ndim == 3:
                    gt_np = gt_np.mean(axis=-1)
                gt_u8 = np.clip(gt_np, 0, 255).astype(np.uint8)
            else:
                # If not available, create a blank tile
                gt_u8 = np.zeros_like(pred_u8, dtype=np.uint8)

            # Ensure shapes match
            h = min(wf_u8.shape[0], pred_u8.shape[0], gt_u8.shape[0])
            w = min(wf_u8.shape[1], pred_u8.shape[1], gt_u8.shape[1])
            wf_u8 = wf_u8[:h, :w]
            pred_u8 = pred_u8[:h, :w]
            gt_u8 = gt_u8[:h, :w]

            composite = np.concatenate([wf_u8, pred_u8, gt_u8], axis=1)
            comp_path = output_dir / f"{img_path.stem}_wf_pred_gt.png"
            Image.fromarray(composite).save(str(comp_path))
        except Exception:
            # Best effort composite; continue on failure
            pass

    # --- Optional: stitch full-FOV composites when inputs are patch tiles ---
    try:
        # Detect tiled naming pattern
        patt = re.compile(r"^frame_(\d{3})_patch_(\d{3})")
        pred_tiles_by_frame = {}
        for p in output_dir.iterdir():
            if not p.is_file():
                continue
            m = patt.match(p.stem)
            if m and p.stem.endswith("_reconstructed"):
                frame_id = m.group(1)
                tile_id = int(m.group(2))
                pred_tiles_by_frame.setdefault(frame_id, {})[tile_id] = p

        if len(pred_tiles_by_frame) > 0:
            # Helper functions
            def _read_img(path: Path) -> np.ndarray:
                ext = path.suffix.lower()
                if ext in (".png", ".jpg", ".jpeg"):
                    return np.array(Image.open(path))
                return tifffile.imread(str(path))

            def _to_uint8(a: np.ndarray) -> np.ndarray:
                a = a.astype(np.float32)
                lo, hi = np.percentile(a, (1, 99))
                if hi <= lo:
                    lo, hi = float(a.min()), float(a.max())
                if hi > lo:
                    a = (a - lo) / (hi - lo)
                a = np.clip(a, 0.0, 1.0)
                return (a * 255).astype(np.uint8)

            def _infer_grid_side(tile_ids: list[int]) -> int:
                if not tile_ids:
                    return 0
                max_idx = max(tile_ids)
                side = int(round(np.sqrt(max_idx + 1)))
                if side * side == (max_idx + 1):
                    return side
                return int(np.ceil(np.sqrt(len(tile_ids))))

            def _stitch(tiles: dict[int, Path], grid_side: int, patch_size: int | None = None) -> np.ndarray:
                if len(tiles) == 0:
                    raise ValueError("No tiles to stitch")
                # Determine patch size
                if patch_size is None:
                    sample = _read_img(next(iter(tiles.values())))
                    if sample.ndim > 2:
                        sample = sample[..., 0]
                    ph, pw = sample.shape[:2]
                else:
                    ph = pw = int(patch_size)
                # Heuristic stride: assume 50% overlap for dense grids
                stride = ph // 2 if grid_side > 8 else ph
                # Canvas size
                H = W = 0
                for idx in tiles.keys():
                    r = idx // grid_side
                    c = idx % grid_side
                    y0 = r * stride
                    x0 = c * stride
                    H = max(H, y0 + ph)
                    W = max(W, x0 + pw)
                accum = np.zeros((H, W), dtype=np.float32)
                weight = np.zeros((H, W), dtype=np.float32)
                for idx, path in tiles.items():
                    r = idx // grid_side
                    c = idx % grid_side
                    y0 = r * stride
                    x0 = c * stride
                    tile = _read_img(path)
                    if tile.ndim > 2:
                        tile = tile[..., 0]
                    tile = tile.astype(np.float32)
                    h, w = tile.shape
                    accum[y0:y0+h, x0:x0+w] += tile
                    weight[y0:y0+h, x0:x0+w] += 1.0
                out = np.zeros_like(accum)
                m = weight > 0
                out[m] = accum[m] / weight[m]
                return out.astype(np.float32)

            # WF and GT directories (if available)
            wf_dir = input_dir
            gt_dir = input_dir.parent / "2p"

            # Save under run-specific directory
            run_name = str(getattr(cfg.experiment, "name", "run"))
            fov_out_dir = Path(str(cfg.paths.outputs)) / "tmp_eval_png" / run_name
            fov_out_dir.mkdir(parents=True, exist_ok=True)

            for frame_id, pred_tiles in sorted(pred_tiles_by_frame.items()):
                tile_ids_sorted = sorted(pred_tiles.keys())
                grid_side = _infer_grid_side(tile_ids_sorted)

                pred_fov = _stitch({k: pred_tiles[k] for k in tile_ids_sorted}, grid_side)
                pred_fov_u8 = _to_uint8(pred_fov)

                # WF stitched from same indices
                wf_tiles = {}
                for k in tile_ids_sorted:
                    wf_path = wf_dir / f"frame_{frame_id}_patch_{k:03d}.png"
                    if wf_path.exists():
                        wf_tiles[k] = wf_path
                wf_fov_u8 = None
                if len(wf_tiles) > 0:
                    wf_fov = _stitch(wf_tiles, grid_side)
                    wf_fov_u8 = _to_uint8(wf_fov)

                # GT 2P stitched from same indices if present
                gt_tiles = {}
                if gt_dir.exists():
                    for k in tile_ids_sorted:
                        gt_path = gt_dir / f"frame_{frame_id}_patch_{k:03d}.png"
                        if gt_path.exists():
                            gt_tiles[k] = gt_path
                gt_fov_u8 = None
                if len(gt_tiles) > 0:
                    gt_fov = _stitch(gt_tiles, grid_side)
                    gt_fov_u8 = _to_uint8(gt_fov)

                # Compose [WF | Pred | GT]
                panels = []
                if wf_fov_u8 is not None:
                    panels.append(wf_fov_u8)
                panels.append(pred_fov_u8)
                if gt_fov_u8 is not None:
                    panels.append(gt_fov_u8)
                strip = np.concatenate(panels, axis=1)
                Image.fromarray(strip).save(str(fov_out_dir / f"frame_{frame_id}_wf_pred_gt_fov.png"))
    except Exception:
        pass

    return saved_paths


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def inference(cfg: DictConfig):
    run_inference(cfg)


if __name__ == "__main__":
    inference()


