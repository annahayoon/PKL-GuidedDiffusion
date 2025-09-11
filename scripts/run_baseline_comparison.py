#!/usr/bin/env python3
"""
Hydra-based Full FOV Baseline Comparison Runner

This script provides a Hydra-based interface for running comprehensive
baseline comparisons between Richardson-Lucy and RCAN methods.

Usage:
    # Basic usage with default config
    python scripts/run_baseline_comparison.py

    # Override specific parameters
    python scripts/run_baseline_comparison.py \
        data.input_dir=data/test/wf \
        data.gt_dir=data/test/2p \
        processing.max_images=10

    # Use different config file
    python scripts/run_baseline_comparison.py \
        --config-name=baseline_comparison_custom

    # Run with specific RCAN checkpoint
    python scripts/run_baseline_comparison.py \
        model.rcan_checkpoint=checkpoints/my_rcan_model.pt
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PKL-DG imports
from pkl_dg.baselines import richardson_lucy_restore, RCANWrapper
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.evaluation import Metrics, RobustnessTests, HallucinationTests
from pkl_dg.evaluation.tasks import DownstreamTasks


class HydraBaselineComparison:
    """Hydra-based baseline comparison system."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize with Hydra configuration."""
        self.cfg = cfg
        self.device = str(cfg.processing.device)
        
        # Extract configuration parameters
        self.patch_size = int(cfg.processing.patch_size)
        self.stride = int(cfg.processing.stride)
        self.max_images = cfg.processing.get("max_images", None)
        
        # Richardson-Lucy parameters
        self.rl_iterations = int(cfg.richardson_lucy.iterations)
        self.rl_clip = bool(cfg.richardson_lucy.clip)
        
        # Physics parameters
        self.background_level = float(cfg.physics.background_level)
        self.read_noise_sigma = float(cfg.physics.read_noise_sigma)
        
        # Load PSF
        psf_path = str(cfg.data.psf_path)
        print(f"Loading PSF from {psf_path}")
        self.psf = self._load_psf(psf_path)
        
        # Initialize RCAN if enabled and checkpoint provided
        self.rcan_model = None
        rcan_checkpoint = cfg.model.get("rcan_checkpoint", None)
        if cfg.rcan.enabled and rcan_checkpoint and os.path.exists(rcan_checkpoint):
            try:
                print(f"Loading RCAN model from {rcan_checkpoint}")
                self.rcan_model = RCANWrapper(checkpoint_path=rcan_checkpoint, device=self.device)
                print("RCAN model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load RCAN model: {e}")
                if cfg.rcan.checkpoint_required:
                    raise RuntimeError(f"RCAN checkpoint required but failed to load: {e}")
                print("RCAN comparisons will be skipped")
        elif cfg.rcan.enabled and not rcan_checkpoint:
            print("RCAN enabled but no checkpoint provided - RCAN comparisons will be skipped")
        else:
            print("RCAN disabled - RCAN comparisons will be skipped")
    
    def _load_psf(self, psf_path: str) -> np.ndarray:
        """Load PSF from file."""
        psf_path = Path(psf_path)
        
        if psf_path.suffix.lower() in ['.tif', '.tiff']:
            psf = tifffile.imread(str(psf_path))
        elif psf_path.suffix.lower() == '.npy':
            psf = np.load(psf_path)
        else:
            raise ValueError(f"Unsupported PSF format: {psf_path.suffix}")
        
        # Ensure PSF is 2D
        if psf.ndim == 3 and psf.shape[0] == 1:
            psf = psf[0]
        elif psf.ndim == 3:
            raise ValueError("3D PSF not supported - please provide 2D PSF")
        
        # Normalize PSF
        psf = psf.astype(np.float32)
        psf = psf / (psf.sum() + 1e-12)
        
        print(f"PSF loaded: shape={psf.shape}, sum={psf.sum():.6f}")
        return psf
    
    def _extract_patches(self, image: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract overlapping patches from full FOV image."""
        patches = {}
        patch_id = 0
        
        h, w = image.shape
        patches_y = (h - self.patch_size) // self.stride + 1
        patches_x = (w - self.patch_size) // self.stride + 1
        
        print(f"Extracting patches: {patches_y} rows × {patches_x} cols = {patches_y * patches_x} patches")
        
        for row in range(patches_y):
            for col in range(patches_x):
                y_start = row * self.stride
                x_start = col * self.stride
                y_end = y_start + self.patch_size
                x_end = x_start + self.patch_size
                
                patch = image[y_start:y_end, x_start:x_end]
                patches[patch_id] = patch
                patch_id += 1
        
        return patches
    
    def _reconstruct_from_patches(
        self, 
        patches: Dict[int, np.ndarray], 
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Reconstruct full FOV image from patches with seamless blending."""
        h, w = original_shape
        patches_y = (h - self.patch_size) // self.stride + 1
        patches_x = (w - self.patch_size) // self.stride + 1
        
        # Initialize canvas
        canvas = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)
        
        # Get feather size from config or use auto
        feather_size = self.cfg.visualization.get("feather_size", self.stride // 2)
        
        # Process all patches
        for patch_id, patch_data in patches.items():
            row = patch_id // patches_x
            col = patch_id % patches_x
            
            # Calculate position in the original image
            y_start = row * self.stride
            x_start = col * self.stride
            y_end = y_start + self.patch_size
            x_end = x_start + self.patch_size
            
            # Create feathering weights for seamless blending
            patch_weight = np.ones((self.patch_size, self.patch_size), dtype=np.float32)
            
            # Feather the edges to create smooth blending
            # Top edge feathering
            if row > 0:
                patch_weight[:feather_size, :] *= np.linspace(0, 1, feather_size)[:, np.newaxis]
            
            # Bottom edge feathering
            if row < patches_y - 1:
                patch_weight[-feather_size:, :] *= np.linspace(1, 0, feather_size)[:, np.newaxis]
            
            # Left edge feathering
            if col > 0:
                patch_weight[:, :feather_size] *= np.linspace(0, 1, feather_size)[np.newaxis, :]
            
            # Right edge feathering
            if col < patches_x - 1:
                patch_weight[:, -feather_size:] *= np.linspace(1, 0, feather_size)[np.newaxis, :]
            
            # Add patch to canvas
            canvas[y_start:y_end, x_start:x_end] += patch_data * patch_weight
            weight_map[y_start:y_end, x_start:x_end] += patch_weight
        
        # Normalize by weights
        mask = weight_map > 0
        canvas[mask] = canvas[mask] / weight_map[mask]
        
        # Apply smoothing
        smoothing_sigma = float(self.cfg.visualization.smoothing_sigma)
        canvas = gaussian_filter(canvas, sigma=smoothing_sigma)
        
        return canvas
    
    def _richardson_lucy_patch(self, patch: np.ndarray) -> np.ndarray:
        """Apply Richardson-Lucy deconvolution to a single patch."""
        return richardson_lucy_restore(
            image=patch,
            psf=self.psf,
            num_iter=self.rl_iterations,
            clip=self.rl_clip
        )
    
    def _rcan_patch(self, patch: np.ndarray) -> np.ndarray:
        """Apply RCAN super-resolution to a single patch."""
        if self.rcan_model is None:
            raise RuntimeError("RCAN model not loaded")
        
        # Preprocess patch for RCAN
        # RCAN typically expects normalized input
        patch_norm = (patch - patch.min()) / (patch.max() - patch.min() + 1e-8)
        
        # Apply RCAN
        result = self.rcan_model.infer(patch_norm)
        
        # Postprocess to match original intensity range
        result = result * (patch.max() - patch.min()) + patch.min()
        
        return result
    
    def process_image(
        self, 
        wf_image: np.ndarray, 
        method: str = "richardson_lucy"
    ) -> np.ndarray:
        """Process full FOV image using specified method."""
        print(f"Processing full FOV image using {method}")
        
        # Extract patches
        patches = self._extract_patches(wf_image)
        
        # Process each patch
        processed_patches = {}
        
        if method == "richardson_lucy":
            for patch_id, patch in tqdm(patches.items(), desc="Richardson-Lucy patches"):
                processed_patches[patch_id] = self._richardson_lucy_patch(patch)
        elif method == "rcan":
            if self.rcan_model is None:
                raise RuntimeError("RCAN model not loaded")
            for patch_id, patch in tqdm(patches.items(), desc="RCAN patches"):
                processed_patches[patch_id] = self._rcan_patch(patch)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reconstruct full FOV
        reconstructed = self._reconstruct_from_patches(processed_patches, wf_image.shape)
        
        return reconstructed
    
    def compute_metrics(
        self, 
        pred: np.ndarray, 
        target: np.ndarray,
        wf_input: np.ndarray
    ) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic image quality metrics
        data_range = float(target.max() - target.min()) if target.size > 0 else 1.0
        metrics["psnr"] = Metrics.psnr(pred, target, data_range=data_range)
        metrics["ssim"] = Metrics.ssim(pred, target, data_range=data_range)
        metrics["frc"] = Metrics.frc(pred, target, threshold=0.143)
        
        # Signal-to-Artifact Ratio (SAR)
        try:
            metrics["sar"] = Metrics.sar(pred, target, wf_input)
        except Exception:
            metrics["sar"] = 0.0
        
        # Robustness metrics
        try:
            robustness = RobustnessTests()
            metrics["psf_mismatch"] = robustness.psf_mismatch_robustness(pred, target, self.psf)
            metrics["alignment_error"] = robustness.alignment_error_robustness(pred, target)
        except Exception:
            metrics["psf_mismatch"] = 0.0
            metrics["alignment_error"] = 0.0
        
        # Hallucination detection
        try:
            hallucination = HallucinationTests()
            metrics["hallucination_score"] = hallucination.detect_hallucinations(pred, target)
        except Exception:
            metrics["hallucination_score"] = 0.0
        
        return metrics
    
    def compute_downstream_metrics(
        self, 
        pred: np.ndarray, 
        gt_masks: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute downstream task metrics."""
        if not self.cfg.evaluation.downstream_tasks.enabled or gt_masks is None:
            return {"cellpose_f1": 0.0, "hausdorff_distance": np.inf}
        
        try:
            # Cellpose F1 score
            if self.cfg.evaluation.downstream_tasks.cellpose_f1:
                f1 = DownstreamTasks.cellpose_f1(pred, gt_masks)
            else:
                f1 = 0.0
            
            # Hausdorff distance
            if self.cfg.evaluation.downstream_tasks.hausdorff_distance:
                from cellpose import models
                model = models.Cellpose(model_type='cyto')
                pred_masks, _, _, _ = model.eval([pred], diameter=None, channels=[0, 0])
                hd = DownstreamTasks.hausdorff_distance(pred_masks[0], gt_masks)
            else:
                hd = np.inf
            
            return {"cellpose_f1": f1, "hausdorff_distance": hd}
        except Exception:
            return {"cellpose_f1": 0.0, "hausdorff_distance": np.inf}
    
    def normalize_for_visualization(self, image: np.ndarray) -> np.ndarray:
        """Normalize image for visualization."""
        percentile_clip = float(self.cfg.visualization.percentile_clip)
        image_norm = np.clip(image, 0, np.percentile(image, percentile_clip))
        if image_norm.max() > image_norm.min():
            image_norm = (image_norm - image_norm.min()) / (image_norm.max() - image_norm.min())
        return (image_norm * 255).astype(np.uint8)
    
    def create_comparison_visualization(
        self,
        wf_input: np.ndarray,
        rl_result: np.ndarray,
        rcan_result: Optional[np.ndarray],
        gt_target: np.ndarray,
        output_path: str
    ) -> None:
        """Create comprehensive comparison visualization."""
        # Normalize all images
        wf_norm = self.normalize_for_visualization(wf_input)
        rl_norm = self.normalize_for_visualization(rl_result)
        gt_norm = self.normalize_for_visualization(gt_target)
        
        # Create comparison grid
        if rcan_result is not None:
            rcan_norm = self.normalize_for_visualization(rcan_result)
            # WF | RL | RCAN | GT
            comparison = np.concatenate([wf_norm, rl_norm, rcan_norm, gt_norm], axis=1)
        else:
            # WF | RL | GT
            comparison = np.concatenate([wf_norm, rl_norm, gt_norm], axis=1)
        
        # Save comparison
        Image.fromarray(comparison).save(output_path)
        print(f"Comparison saved to {output_path}")
    
    def create_summary_plots(self, all_results: Dict[str, List[Dict]], output_dir: Path) -> None:
        """Create summary plots and statistical analysis."""
        if not all_results:
            return
        
        # Convert to DataFrame for easier analysis
        data_for_plot = []
        for method, results in all_results.items():
            for result in results:
                row = {"method": method}
                row.update(result)
                data_for_plot.append(row)
        
        if not data_for_plot:
            return
        
        df = pd.DataFrame(data_for_plot)
        
        # Create output directory for plots
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comparison plots for each metric
        metrics_to_plot = ["psnr", "ssim", "frc", "sar"]
        
        for metric in metrics_to_plot:
            if metric not in df.columns:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Box plot
            methods = df['method'].unique()
            data_for_box = [df[df['method'] == method][metric].values for method in methods]
            
            ax1.boxplot(data_for_box, labels=methods)
            ax1.set_ylabel(metric.upper())
            ax1.set_title(f'{metric.upper()} Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Violin plot
            sns.violinplot(data=df, x='method', y=metric, ax=ax2)
            ax2.set_title(f'{metric.upper()} Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'{metric}_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        # Create correlation heatmap
        if len(methods) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Compute correlation matrix for each method
            correlation_data = []
            for method in methods:
                method_data = df[df['method'] == method][metrics_to_plot].corr()
                correlation_data.append(method_data)
            
            # Create subplot for each method
            fig, axes = plt.subplots(1, len(methods), figsize=(5*len(methods), 4))
            if len(methods) == 1:
                axes = [axes]
            
            for i, method in enumerate(methods):
                sns.heatmap(correlation_data[i], annot=True, cmap='coolwarm', center=0,
                           ax=axes[i], cbar_kws={'shrink': 0.8})
                axes[i].set_title(f'{method.upper()} Correlations')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Summary plots saved to {plots_dir}")


@hydra.main(version_base=None, config_path="../configs", config_name="baseline_comparison")
def main(cfg: DictConfig) -> None:
    """Main function for Hydra-based baseline comparison."""
    
    # Create output directory
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # Initialize comparison system
    comparison = HydraBaselineComparison(cfg)
    
    # Load image pairs
    input_dir = Path(cfg.data.input_dir)
    gt_dir = Path(cfg.data.gt_dir)
    
    # Find matching image pairs
    input_files = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.png"))
    if cfg.processing.max_images:
        input_files = input_files[:cfg.processing.max_images]
    
    print(f"Found {len(input_files)} input images")
    
    # Results storage
    all_results = {
        "richardson_lucy": [],
        "rcan": []
    }
    
    # Process each image pair
    for input_path in tqdm(input_files, desc="Processing images"):
        # Find corresponding GT file
        gt_path = gt_dir / input_path.name
        if not gt_path.exists():
            print(f"Warning: No GT file found for {input_path.name}")
            continue
        
        # Load images
        wf_image = tifffile.imread(str(input_path)).astype(np.float32)
        gt_image = tifffile.imread(str(gt_path)).astype(np.float32)
        
        # Ensure single channel
        if wf_image.ndim == 3 and wf_image.shape[0] == 1:
            wf_image = wf_image[0]
        if gt_image.ndim == 3 and gt_image.shape[0] == 1:
            gt_image = gt_image[0]
        
        print(f"Processing {input_path.name}: {wf_image.shape} -> {gt_image.shape}")
        
        # Process with Richardson-Lucy
        try:
            rl_result = comparison.process_image(wf_image, method="richardson_lucy")
            rl_metrics = comparison.compute_metrics(rl_result, gt_image, wf_image)
            all_results["richardson_lucy"].append(rl_metrics)
            print(f"RL metrics: PSNR={rl_metrics['psnr']:.2f}, SSIM={rl_metrics['ssim']:.3f}")
        except Exception as e:
            print(f"Error processing RL for {input_path.name}: {e}")
            continue
        
        # Process with RCAN if available
        rcan_result = None
        if comparison.rcan_model is not None:
            try:
                rcan_result = comparison.process_image(wf_image, method="rcan")
                rcan_metrics = comparison.compute_metrics(rcan_result, gt_image, wf_image)
                all_results["rcan"].append(rcan_metrics)
                print(f"RCAN metrics: PSNR={rcan_metrics['psnr']:.2f}, SSIM={rcan_metrics['ssim']:.3f}")
            except Exception as e:
                print(f"Error processing RCAN for {input_path.name}: {e}")
        
        # Create comparison visualization if enabled
        if cfg.output.create_visualizations:
            comparison_path = output_dir / f"{input_path.stem}_comparison.png"
            comparison.create_comparison_visualization(
                wf_input=wf_image,
                rl_result=rl_result,
                rcan_result=rcan_result,
                gt_target=gt_image,
                output_path=str(comparison_path)
            )
    
    # Compute summary statistics
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    
    summary_stats = {}
    for method, results in all_results.items():
        if not results:
            continue
        
        print(f"\n{method.upper()} RESULTS:")
        print(f"Number of images: {len(results)}")
        
        # Compute mean and std for each metric
        metrics_summary = {}
        for metric_name in results[0].keys():
            values = [r[metric_name] for r in results]
            metrics_summary[metric_name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        
        summary_stats[method] = metrics_summary
        
        # Print summary
        for metric_name, stats in metrics_summary.items():
            print(f"  {metric_name.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f} - {stats['max']:.3f})")
    
    # Save detailed results
    if cfg.output.save_individual_results:
        results_path = output_dir / "detailed_results.npz"
        np.savez(results_path, **all_results)
        print(f"\nDetailed results saved to {results_path}")
    
    # Save summary statistics
    if cfg.output.save_summary:
        summary_path = output_dir / "summary_statistics.npz"
        np.savez(summary_path, **summary_stats)
        print(f"Summary statistics saved to {summary_path}")
    
    # Create summary plots
    comparison.create_summary_plots(all_results, output_dir)
    
    print("\nBaseline comparison completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
