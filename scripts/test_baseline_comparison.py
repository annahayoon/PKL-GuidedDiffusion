#!/usr/bin/env python3
"""
Test script for baseline comparison implementation.

This script validates the baseline comparison system with synthetic data
to ensure all components work correctly.
"""

import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tifffile
from omegaconf import OmegaConf

from scripts.run_baseline_comparison import HydraBaselineComparison


def create_test_data(temp_dir: Path) -> dict:
    """Create synthetic test data."""
    print("Creating synthetic test data...")
    
    # Create PSF (Gaussian)
    psf_size = 16
    sigma = 1.5
    y, x = np.ogrid[-psf_size//2:psf_size//2, -psf_size//2:psf_size//2]
    psf = np.exp(-(x*x + y*y) / (2*sigma*sigma))
    psf = psf / psf.sum()
    
    # Save PSF
    psf_path = temp_dir / "psf.tif"
    tifffile.imwrite(psf_path, psf.astype(np.float32))
    
    # Create test images
    image_size = 128  # Small for testing
    num_images = 2
    
    input_dir = temp_dir / "wf"
    gt_dir = temp_dir / "2p"
    input_dir.mkdir()
    gt_dir.mkdir()
    
    for i in range(num_images):
        # Create synthetic ground truth
        np.random.seed(42 + i)
        gt = np.random.poisson(50, (image_size, image_size)).astype(np.float32)
        
        # Add structured features
        gt[20:40, 20:40] += 100  # Bright square
        gt[60:80, 60:80] += 75   # Medium square
        
        # Apply smoothing
        from scipy.ndimage import gaussian_filter
        gt = gaussian_filter(gt, sigma=0.5)
        
        # Create wide-field image
        from scipy.signal import fftconvolve
        wf = fftconvolve(gt, psf, mode='same')
        wf = np.random.poisson(wf).astype(np.float32)
        wf += 5  # Background
        
        # Save images
        tifffile.imwrite(input_dir / f"test_{i:03d}.tif", wf)
        tifffile.imwrite(gt_dir / f"test_{i:03d}.tif", gt)
    
    return {
        "psf_path": str(psf_path),
        "input_dir": str(input_dir),
        "gt_dir": str(gt_dir)
    }


def test_psf_loading():
    """Test PSF loading functionality."""
    print("\nTesting PSF loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        # Test PSF loading
        cfg = OmegaConf.create({
            "data": {"psf_path": test_data["psf_path"]},
            "processing": {"device": "cpu"},
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            assert comparison.psf is not None
            assert comparison.psf.shape == (16, 16)
            assert abs(comparison.psf.sum() - 1.0) < 1e-6
            print("✓ PSF loading test passed")
        except Exception as e:
            print(f"✗ PSF loading test failed: {e}")
            raise


def test_patch_extraction():
    """Test patch extraction functionality."""
    print("\nTesting patch extraction...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        cfg = OmegaConf.create({
            "data": {"psf_path": test_data["psf_path"]},
            "processing": {"device": "cpu", "patch_size": 32, "stride": 16},
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            
            # Load test image
            input_files = list(Path(test_data["input_dir"]).glob("*.tif"))
            wf_image = tifffile.imread(str(input_files[0])).astype(np.float32)
            
            # Extract patches
            patches = comparison._extract_patches(wf_image)
            
            # Check patch count
            expected_patches = ((128 - 32) // 16 + 1) ** 2
            assert len(patches) == expected_patches
            assert all(patch.shape == (32, 32) for patch in patches.values())
            
            print("✓ Patch extraction test passed")
        except Exception as e:
            print(f"✗ Patch extraction test failed: {e}")
            raise


def test_richardson_lucy():
    """Test Richardson-Lucy processing."""
    print("\nTesting Richardson-Lucy processing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        cfg = OmegaConf.create({
            "data": {"psf_path": test_data["psf_path"]},
            "processing": {"device": "cpu", "patch_size": 32, "stride": 16},
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            
            # Load test image
            input_files = list(Path(test_data["input_dir"]).glob("*.tif"))
            wf_image = tifffile.imread(str(input_files[0])).astype(np.float32)
            
            # Process with Richardson-Lucy
            result = comparison.process_image(wf_image, method="richardson_lucy")
            
            # Check result
            assert result.shape == wf_image.shape
            assert np.isfinite(result).all()
            assert result.min() >= 0  # Should be non-negative
            
            print("✓ Richardson-Lucy test passed")
        except Exception as e:
            print(f"✗ Richardson-Lucy test failed: {e}")
            raise


def test_metrics_computation():
    """Test metrics computation."""
    print("\nTesting metrics computation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        cfg = OmegaConf.create({
            "data": {"psf_path": test_data["psf_path"]},
            "processing": {"device": "cpu", "patch_size": 32, "stride": 16},
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            
            # Load test images
            input_files = list(Path(test_data["input_dir"]).glob("*.tif"))
            gt_files = list(Path(test_data["gt_dir"]).glob("*.tif"))
            
            wf_image = tifffile.imread(str(input_files[0])).astype(np.float32)
            gt_image = tifffile.imread(str(gt_files[0])).astype(np.float32)
            
            # Process with Richardson-Lucy
            result = comparison.process_image(wf_image, method="richardson_lucy")
            
            # Compute metrics
            metrics = comparison.compute_metrics(result, gt_image, wf_image)
            
            # Check metrics
            assert "psnr" in metrics
            assert "ssim" in metrics
            assert "frc" in metrics
            assert isinstance(metrics["psnr"], float)
            assert isinstance(metrics["ssim"], float)
            assert isinstance(metrics["frc"], float)
            assert 0 <= metrics["ssim"] <= 1
            
            print("✓ Metrics computation test passed")
        except Exception as e:
            print(f"✗ Metrics computation test failed: {e}")
            raise


def test_visualization():
    """Test visualization functionality."""
    print("\nTesting visualization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        cfg = OmegaConf.create({
            "data": {"psf_path": test_data["psf_path"]},
            "processing": {"device": "cpu", "patch_size": 32, "stride": 16},
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False},
            "visualization": {"percentile_clip": 99.5}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            
            # Load test images
            input_files = list(Path(test_data["input_dir"]).glob("*.tif"))
            gt_files = list(Path(test_data["gt_dir"]).glob("*.tif"))
            
            wf_image = tifffile.imread(str(input_files[0])).astype(np.float32)
            gt_image = tifffile.imread(str(gt_files[0])).astype(np.float32)
            
            # Process with Richardson-Lucy
            result = comparison.process_image(wf_image, method="richardson_lucy")
            
            # Create visualization
            output_path = temp_path / "test_comparison.png"
            comparison.create_comparison_visualization(
                wf_input=wf_image,
                rl_result=result,
                rcan_result=None,
                gt_target=gt_image,
                output_path=str(output_path)
            )
            
            # Check output file
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            print("✓ Visualization test passed")
        except Exception as e:
            print(f"✗ Visualization test failed: {e}")
            raise


def test_full_pipeline():
    """Test complete pipeline."""
    print("\nTesting full pipeline...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_data = create_test_data(temp_path)
        
        cfg = OmegaConf.create({
            "data": {
                "input_dir": test_data["input_dir"],
                "gt_dir": test_data["gt_dir"],
                "psf_path": test_data["psf_path"]
            },
            "processing": {
                "device": "cpu",
                "patch_size": 32,
                "stride": 16,
                "max_images": 2
            },
            "richardson_lucy": {"iterations": 5},
            "rcan": {"enabled": False},
            "output": {
                "directory": str(temp_path / "output"),
                "save_individual_results": True,
                "save_summary": True,
                "create_visualizations": True
            },
            "evaluation": {
                "metrics": ["psnr", "ssim", "frc"],
                "downstream_tasks": {"enabled": False}
            },
            "visualization": {"percentile_clip": 99.5}
        })
        
        try:
            comparison = HydraBaselineComparison(cfg)
            
            # Run full pipeline
            input_dir = Path(cfg.data.input_dir)
            gt_dir = Path(cfg.data.gt_dir)
            output_dir = Path(cfg.output.directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            input_files = sorted(input_dir.glob("*.tif"))
            
            results = []
            for input_path in input_files:
                gt_path = gt_dir / input_path.name
                if not gt_path.exists():
                    continue
                
                wf_image = tifffile.imread(str(input_path)).astype(np.float32)
                gt_image = tifffile.imread(str(gt_path)).astype(np.float32)
                
                # Process with Richardson-Lucy
                result = comparison.process_image(wf_image, method="richardson_lucy")
                metrics = comparison.compute_metrics(result, gt_image, wf_image)
                results.append(metrics)
                
                # Create visualization
                comparison_path = output_dir / f"{input_path.stem}_comparison.png"
                comparison.create_comparison_visualization(
                    wf_input=wf_image,
                    rl_result=result,
                    rcan_result=None,
                    gt_target=gt_image,
                    output_path=str(comparison_path)
                )
            
            # Check results
            assert len(results) > 0
            assert all("psnr" in r for r in results)
            assert all("ssim" in r for r in results)
            
            # Check output files
            comparison_files = list(output_dir.glob("*_comparison.png"))
            assert len(comparison_files) > 0
            
            print("✓ Full pipeline test passed")
        except Exception as e:
            print(f"✗ Full pipeline test failed: {e}")
            raise


def main():
    """Run all tests."""
    print("Baseline Comparison Test Suite")
    print("=" * 50)
    
    try:
        test_psf_loading()
        test_patch_extraction()
        test_richardson_lucy()
        test_metrics_computation()
        test_visualization()
        test_full_pipeline()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED! ✓")
        print("=" * 50)
        print("The baseline comparison system is working correctly.")
        print("You can now use it with your own data.")
        
    except Exception as e:
        print(f"\n" + "=" * 50)
        print("TESTS FAILED! ✗")
        print("=" * 50)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
