import os
from pathlib import Path
from typing import Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import tifffile
from tqdm import tqdm

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl import PKLGuidance
from pkl_dg.guidance.l2 import L2Guidance
from pkl_dg.guidance.anscombe import AnscombeGuidance
from pkl_dg.guidance.schedules import AdaptiveSchedule
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.evaluation import Metrics, RobustnessTests, HallucinationTests
from pkl_dg.evaluation.tasks import DownstreamTasks
from pkl_dg.baselines import richardson_lucy_restore

try:
    from pkl_dg.baselines import RCANWrapper  # optional
    HAS_RCAN = True
except Exception:
    HAS_RCAN = False


def _load_model_and_sampler(cfg: DictConfig, guidance_type: str):
    device = str(cfg.experiment.device)
    unet = DenoisingUNet(OmegaConf.to_container(cfg.model, resolve=True))
    ddpm = DDPMTrainer(model=unet, config=OmegaConf.to_container(cfg.training, resolve=True))
    checkpoint_path = Path(str(cfg.inference.checkpoint_path))
    state_dict = torch.load(checkpoint_path, map_location=device)
    ddpm.load_state_dict(state_dict)
    ddpm.eval().to(device)
    psf = PSF(getattr(cfg.physics, "psf_path", None))
    forward_model = ForwardModel(psf=psf.to_torch(device=device), background=float(cfg.physics.background), device=device)
    if guidance_type == "pkl":
        guidance = PKLGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    elif guidance_type == "l2":
        guidance = L2Guidance()
    elif guidance_type == "anscombe":
        guidance = AnscombeGuidance(epsilon=float(getattr(cfg.guidance, "epsilon", 1e-6)))
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
    schedule_cfg = getattr(cfg.guidance, "schedule", {})
    schedule = AdaptiveSchedule(
        lambda_base=float(getattr(cfg.guidance, "lambda_base", 0.1)),
        T_threshold=int(getattr(schedule_cfg, "T_threshold", 800)),
        epsilon_lambda=float(getattr(schedule_cfg, "epsilon_lambda", 1e-3)),
        T_total=int(cfg.training.num_timesteps),
    )
    transform = IntensityToModel(min_intensity=float(cfg.data.min_intensity), max_intensity=float(cfg.data.max_intensity))
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
    return sampler


def _compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pr = pred.astype(np.float32)
    gt = target.astype(np.float32)
    data_range = float(gt.max() - gt.min()) if gt.size > 0 else 1.0
    return {
        "psnr": Metrics.psnr(pr, gt, data_range=data_range),
        "ssim": Metrics.ssim(pr, gt, data_range=data_range),
        "frc": Metrics.frc(pr, gt, threshold=0.143),
    }


def _compute_downstream_metrics(
    pred: np.ndarray, gt_masks: np.ndarray
) -> Dict[str, float]:
    """Compute metrics for downstream tasks like segmentation."""
    try:
        # Cellpose F1 score
        f1 = DownstreamTasks.cellpose_f1(pred, gt_masks)
        
        # Hausdorff distance requires predicted masks, so we run cellpose again
        # This is slightly inefficient but decouples the metrics
        from cellpose import models
        model = models.Cellpose(model_type='cyto')
        pred_masks, _, _, _ = model.eval([pred], diameter=None, channels=[0, 0])

        hd = DownstreamTasks.hausdorff_distance(pred_masks[0], gt_masks)
        
        return {"cellpose_f1": f1, "hausdorff_distance": hd}
    except Exception:
        return {"cellpose_f1": 0.0, "hausdorff_distance": np.inf}


def _load_pairs(input_dir: Path) -> List[Path]:
    return list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))


def _read_tif(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def evaluate(cfg: DictConfig) -> Dict[str, Dict[str, float]]:
    device = str(cfg.experiment.device)
    input_dir = Path(str(cfg.inference.input_dir))
    gt_dir = Path(str(cfg.inference.gt_dir))
    mask_dir = Path(str(getattr(cfg.inference, "mask_dir", "")))
    image_paths = _load_pairs(input_dir)

    # Prepare samplers per guidance
    samplers = {
        "l2": _load_model_and_sampler(cfg, "l2"),
        "anscombe": _load_model_and_sampler(cfg, "anscombe"),
        "pkl": _load_model_and_sampler(cfg, "pkl"),
    }

    # Optional RCAN
    rcan = None
    if HAS_RCAN and getattr(cfg.baselines, "rcan_checkpoint", None) is not None:
        try:
            rcan = RCANWrapper(checkpoint_path=str(cfg.baselines.rcan_checkpoint), device=device)
        except Exception:
            rcan = None

    # Results accumulators
    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    def _acc(name: str, metrics: Dict[str, float]):
        if not metrics:
            return
        if name not in sums:
            sums[name] = {k: 0.0 for k in metrics}
            counts[name] = 0
        for k, v in metrics.items():
            if np.isfinite(v):
                sums[name][k] += float(v)
        counts[name] += 1

    for img_path in tqdm(image_paths, desc="Evaluate"):
        y = _read_tif(img_path)
        x_gt = _read_tif(gt_dir / img_path.name)
        
        # Load masks for downstream tasks if available
        gt_masks = None
        if mask_dir.is_dir():
            mask_path = mask_dir / img_path.name
            if mask_path.exists():
                gt_masks = _read_tif(mask_path)

        # WF input baseline
        _acc("wf", _compute_metrics(y, x_gt))
        if gt_masks is not None:
            _acc("wf", _compute_downstream_metrics(y, gt_masks))
            
        # RL baseline
        try:
            psf = PSF(getattr(cfg.physics, "psf_path", None)).psf
            rl = richardson_lucy_restore(y, psf, num_iter=int(getattr(cfg.baselines, "rl_iters", 30)))
            _acc("rl", _compute_metrics(rl, x_gt))
            if gt_masks is not None:
                _acc("rl", _compute_downstream_metrics(rl, gt_masks))
        except Exception:
            pass
            
        # Diffusion baselines
        for name, sampler in samplers.items():
            ten_y = torch.from_numpy(y).float().to(device)
            if ten_y.ndim == 2:
                ten_y = ten_y.unsqueeze(0).unsqueeze(0)
            pred = sampler.sample(ten_y, tuple(ten_y.shape), device=device, verbose=False)
            out = pred.squeeze().detach().cpu().numpy().astype(np.float32)
            _acc(name, _compute_metrics(out, x_gt))
            if gt_masks is not None:
                _acc(name, _compute_downstream_metrics(out, gt_masks))

        # RCAN if available
        if rcan is not None:
            try:
                rcan_out = rcan.infer(y)
                _acc("rcan", _compute_metrics(rcan_out, x_gt))
                if gt_masks is not None:
                    _acc("rcan", _compute_downstream_metrics(rcan_out, gt_masks))
            except Exception:
                pass

        # --- Adversarial Evaluations ---
        psf_true = PSF(getattr(cfg.physics, "psf_path", None))
        
        # Robustness: PSF mismatch
        try:
            x_mismatch = RobustnessTests.psf_mismatch_test(samplers["pkl"], ten_y, psf_true, mismatch_factor=1.1)
            out = x_mismatch.squeeze().detach().cpu().numpy().astype(np.float32)
            _acc("pkl_psf_mismatch", _compute_metrics(out, x_gt))
        except Exception:
            pass

        # Robustness: Alignment Error
        try:
            x_shifted = RobustnessTests.alignment_error_test(samplers["pkl"], ten_y, shift_pixels=0.5)
            out = x_shifted.squeeze().detach().cpu().numpy().astype(np.float32)
            _acc("pkl_alignment_error", _compute_metrics(out, x_gt))
        except Exception:
            pass

        # Hallucination: Commission Error (SAR)
        try:
            art_img, art_mask = HallucinationTests.add_out_of_focus_artifact(y, center=(y.shape[0]//2, y.shape[1]//2))
            ten_art = torch.from_numpy(art_img).float().to(device).unsqueeze(0).unsqueeze(0)
            out = samplers["pkl"].sample(ten_art, ten_art.shape, device=device, verbose=False)
            out_np = out.squeeze().detach().cpu().numpy().astype(np.float32)
            sar = HallucinationTests.commission_sar(out_np, art_mask)
            _acc("pkl_commission_sar", {"sar": sar})
        except Exception:
            pass
            
        # Hallucination: Omission Error (Fidelity of faint structure)
        try:
            faint_gt, faint_mask = HallucinationTests.insert_faint_structure(x_gt, start=(8, 8), end=(x_gt.shape[0]-8, x_gt.shape[1]-8))
            fm = samplers["pkl"].forward_model
            ten_faint = torch.from_numpy(faint_gt).float().to(device).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                wf_like = fm.forward(ten_faint, add_noise=True).squeeze().cpu().numpy()
            
            ten_wf = torch.from_numpy(wf_like).float().to(device).unsqueeze(0).unsqueeze(0)
            out = samplers["pkl"].sample(ten_wf, ten_wf.shape, device=device, verbose=False)
            out_np = out.squeeze().detach().cpu().numpy()
            psnr_mask = HallucinationTests.structure_fidelity_psnr(out_np, faint_gt, faint_mask)
            _acc("pkl_omission_psnr", {"psnr_faint": psnr_mask})
        except Exception:
            pass


    # Aggregate means
    results: Dict[str, Dict[str, float]] = {}
    for name, agg in sums.items():
        denom = max(counts.get(name, 1), 1)
        results[name] = {k: v / denom for k, v in agg.items()}
    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    res = evaluate(cfg)
    # Print terse summary
    for name, metrics in res.items():
        print(name, {k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()


