#!/usr/bin/env python3
"""
Example: Full FOV Baseline Comparison

This example demonstrates how to run comprehensive baseline comparisons
between Richardson-Lucy deconvolution and RCAN super-resolution for
full field-of-view microscopy image reconstruction.

The example shows:
1. Basic usage with default parameters
2. Custom configuration overrides
3. Batch processing of multiple images
4. Results analysis and visualization
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import tifffile
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.run_baseline_comparison import HydraBaselineComparison
from omegaconf import DictConfig, OmegaConf


def create_example_data():
    """Create example synthetic data for demonstration."""
    print("Creating example synthetic data...")
    
    # Create output directory
    data_dir = Path("examples/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic PSF (Gaussian)
    psf_size = 32
    sigma = 2.0
    y, x = np.ogrid[-psf_size//2:psf_size//2, -psf_size//2:psf_size//2]
    psf = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    psf = psf / psf.sum()
    
    # Save PSF
    tifffile.imwrite(data_dir / "psf.tif", psf.astype(np.float32))
    
    # Create synthetic images
    image_size = 512
    num_images = 3
    
    for i in range(num_images):
        # Create synthetic ground truth (random structures)
        np.random.seed(42 + i)
        gt = np.random.poisson(100, (image_size, image_size)).astype(np.float32)
        
        # Add some structured features
        gt[100:200, 100:200] += 200  # Bright square
        gt[300:400, 300:400] += 150  # Medium square
        gt[50:100, 400:450] += 100   # Small square
        
        # Apply Gaussian smoothing to make it more realistic
        from scipy.ndimage import gaussian_filter
        gt = gaussian_filter(gt, sigma=1.0)
        
        # Create wide-field image by convolving with PSF and adding noise
        from scipy.signal import fftconvolve
        wf = fftconvolve(gt, psf, mode='same')
        
        # Add Poisson noise
        wf = np.random.poisson(wf).astype(np.float32)
        
        # Add background
        wf += 10
        
        # Save images
        tifffile.imwrite(data_dir / f"wf_{i:03d}.tif", wf)
        tifffile.imwrite(data_dir / f"gt_{i:03d}.tif", gt)
    
    print(f"Example data created in {data_dir}")
    return data_dir


def run_basic_comparison():
    """Run basic baseline comparison with default settings."""
    print("\n" + "="*60)
    print("RUNNING BASIC BASELINE COMPARISON")
    print("="*60)
    
    # Create example data
    data_dir = create_example_data()
    
    # Create basic configuration
    cfg = OmegaConf.create({
        "data": {
            "input_dir": str(data_dir),
            "gt_dir": str(data_dir),
            "psf_path": str(data_dir / "psf.tif"),
            "mask_dir": ""
        },
        "model": {
            "rcan_checkpoint": None  # No RCAN for basic example
        },
        "processing": {
            "device": "cpu",  # Use CPU for example
            "patch_size": 128,
            "stride": 64,
            "max_images": 2
        },
        "richardson_lucy": {
            "iterations": 20,
            "clip": True
        },
        "rcan": {
            "enabled": False
        },
        "physics": {
            "background_level": 0.0,
            "read_noise_sigma": 0.0
        },
        "output": {
            "directory": "examples/outputs/basic_comparison",
            "save_individual_results": True,
            "save_summary": True,
            "create_visualizations": True
        },
        "evaluation": {
            "metrics": ["psnr", "ssim", "frc"],
            "downstream_tasks": {
                "enabled": False
            }
        },
        "visualization": {
            "percentile_clip": 99.5,
            "feather_size": None,
            "smoothing_sigma": 0.5
        }
    })
    
    # Create output directory
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comparison system
    comparison = HydraBaselineComparison(cfg)
    
    # Process images
    input_dir = Path(cfg.data.input_dir)
    gt_dir = Path(cfg.data.gt_dir)
    
    input_files = sorted(input_dir.glob("wf_*.tif"))
    if cfg.processing.max_images:
        input_files = input_files[:cfg.processing.max_images]
    
    print(f"Processing {len(input_files)} images...")
    
    results = []
    
    for input_path in input_files:
        # Find corresponding GT file
        gt_name = input_path.name.replace("wf_", "gt_")
        gt_path = gt_dir / gt_name
        
        if not gt_path.exists():
            print(f"Warning: No GT file found for {input_path.name}")
            continue
        
        # Load images
        wf_image = tifffile.imread(str(input_path)).astype(np.float32)
        gt_image = tifffile.imread(str(gt_path)).astype(np.float32)
        
        print(f"Processing {input_path.name}: {wf_image.shape}")
        
        # Process with Richardson-Lucy
        try:
            rl_result = comparison.process_image(wf_image, method="richardson_lucy")
            rl_metrics = comparison.compute_metrics(rl_result, gt_image, wf_image)
            results.append({
                "image": input_path.name,
                "method": "richardson_lucy",
                **rl_metrics
            })
            
            print(f"RL metrics: PSNR={rl_metrics['psnr']:.2f}, SSIM={rl_metrics['ssim']:.3f}")
            
            # Create visualization
            comparison_path = output_dir / f"{input_path.stem}_comparison.png"
            comparison.create_comparison_visualization(
                wf_input=wf_image,
                rl_result=rl_result,
                rcan_result=None,
                gt_target=gt_image,
                output_path=str(comparison_path)
            )
            
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*40)
    print("SUMMARY RESULTS")
    print("="*40)
    
    if results:
        df = pd.DataFrame(results)
        summary = df.groupby('method')[['psnr', 'ssim', 'frc']].agg(['mean', 'std'])
        print(summary)
        
        # Save results
        results_path = output_dir / "results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")
    
    print(f"\nBasic comparison completed! Results in {output_dir}")
    return output_dir


def run_advanced_comparison():
    """Run advanced comparison with custom settings."""
    print("\n" + "="*60)
    print("RUNNING ADVANCED BASELINE COMPARISON")
    print("="*60)
    
    # Create example data
    data_dir = create_example_data()
    
    # Create advanced configuration
    cfg = OmegaConf.create({
        "data": {
            "input_dir": str(data_dir),
            "gt_dir": str(data_dir),
            "psf_path": str(data_dir / "psf.tif"),
            "mask_dir": ""
        },
        "model": {
            "rcan_checkpoint": None  # Still no RCAN for example
        },
        "processing": {
            "device": "cpu",
            "patch_size": 256,
            "stride": 128,
            "max_images": None  # Process all images
        },
        "richardson_lucy": {
            "iterations": 50,  # More iterations
            "clip": True
        },
        "rcan": {
            "enabled": False
        },
        "physics": {
            "background_level": 5.0,  # Add background
            "read_noise_sigma": 1.0   # Add read noise
        },
        "output": {
            "directory": "examples/outputs/advanced_comparison",
            "save_individual_results": True,
            "save_summary": True,
            "create_visualizations": True
        },
        "evaluation": {
            "metrics": ["psnr", "ssim", "frc", "sar", "psf_mismatch", "alignment_error"],
            "downstream_tasks": {
                "enabled": False
            }
        },
        "visualization": {
            "percentile_clip": 99.0,
            "feather_size": 32,
            "smoothing_sigma": 1.0
        }
    })
    
    # Create output directory
    output_dir = Path(cfg.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize comparison system
    comparison = HydraBaselineComparison(cfg)
    
    # Process images
    input_dir = Path(cfg.data.input_dir)
    gt_dir = Path(cfg.data.gt_dir)
    
    input_files = sorted(input_dir.glob("wf_*.tif"))
    
    print(f"Processing {len(input_files)} images with advanced settings...")
    
    results = []
    
    for input_path in input_files:
        # Find corresponding GT file
        gt_name = input_path.name.replace("wf_", "gt_")
        gt_path = gt_dir / gt_name
        
        if not gt_path.exists():
            continue
        
        # Load images
        wf_image = tifffile.imread(str(input_path)).astype(np.float32)
        gt_image = tifffile.imread(str(gt_path)).astype(np.float32)
        
        print(f"Processing {input_path.name}: {wf_image.shape}")
        
        # Process with Richardson-Lucy
        try:
            rl_result = comparison.process_image(wf_image, method="richardson_lucy")
            rl_metrics = comparison.compute_metrics(rl_result, gt_image, wf_image)
            results.append({
                "image": input_path.name,
                "method": "richardson_lucy",
                **rl_metrics
            })
            
            print(f"RL metrics: PSNR={rl_metrics['psnr']:.2f}, SSIM={rl_metrics['ssim']:.3f}, FRC={rl_metrics['frc']:.3f}")
            
            # Create visualization
            comparison_path = output_dir / f"{input_path.stem}_comparison.png"
            comparison.create_comparison_visualization(
                wf_input=wf_image,
                rl_result=rl_result,
                rcan_result=None,
                gt_target=gt_image,
                output_path=str(comparison_path)
            )
            
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            continue
    
    # Create summary plots
    if results:
        import pandas as pd
        
        df = pd.DataFrame(results)
        
        # Create plots directory
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics = ['psnr', 'ssim', 'frc', 'sar']
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                ax = axes[i//2, i%2]
                ax.hist(df[metric], bins=10, alpha=0.7, edgecolor='black')
                ax.set_xlabel(metric.upper())
                ax.set_ylabel('Frequency')
                ax.set_title(f'{metric.upper()} Distribution')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n" + "="*40)
        print("ADVANCED COMPARISON SUMMARY")
        print("="*40)
        
        summary = df.groupby('method')[metrics].agg(['mean', 'std'])
        print(summary)
        
        # Save results
        results_path = output_dir / "advanced_results.csv"
        df.to_csv(results_path, index=False)
        print(f"\nAdvanced results saved to {results_path}")
    
    print(f"\nAdvanced comparison completed! Results in {output_dir}")
    return output_dir


def demonstrate_configuration_options():
    """Demonstrate various configuration options."""
    print("\n" + "="*60)
    print("CONFIGURATION OPTIONS DEMONSTRATION")
    print("="*60)
    
    # Show different configuration examples
    configs = {
        "High Quality": {
            "processing": {"patch_size": 512, "stride": 256},
            "richardson_lucy": {"iterations": 100},
            "visualization": {"smoothing_sigma": 0.1}
        },
        "Fast Processing": {
            "processing": {"patch_size": 128, "stride": 64},
            "richardson_lucy": {"iterations": 10},
            "visualization": {"smoothing_sigma": 1.0}
        },
        "Robust Evaluation": {
            "evaluation": {
                "metrics": ["psnr", "ssim", "frc", "sar", "psf_mismatch", "alignment_error", "hallucination_score"]
            },
            "physics": {"background_level": 10.0, "read_noise_sigma": 2.0}
        }
    }
    
    for config_name, config_options in configs.items():
        print(f"\n{config_name} Configuration:")
        for section, options in config_options.items():
            print(f"  {section}:")
            for key, value in options.items():
                print(f"    {key}: {value}")
    
    print("\nTo use these configurations, you can:")
    print("1. Modify the config file directly")
    print("2. Override via command line:")
    print("   python scripts/run_baseline_comparison.py processing.patch_size=512")
    print("3. Create custom config files")
    print("4. Use Hydra's multirun for parameter sweeps")


def main():
    """Run all examples."""
    print("Full FOV Baseline Comparison Examples")
    print("="*60)
    
    try:
        # Run basic comparison
        basic_output = run_basic_comparison()
        
        # Run advanced comparison
        advanced_output = run_advanced_comparison()
        
        # Demonstrate configuration options
        demonstrate_configuration_options()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Basic results: {basic_output}")
        print(f"Advanced results: {advanced_output}")
        print("\nNext steps:")
        print("1. Examine the generated comparison images")
        print("2. Review the quantitative results")
        print("3. Modify configurations for your specific use case")
        print("4. Run with your own data and PSF")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
