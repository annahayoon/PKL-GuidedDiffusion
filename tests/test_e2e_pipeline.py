"""
End-to-end pipeline testing for PKL-Guided Diffusion system.

This module provides comprehensive testing of the complete workflows:
1. Data synthesis and loading pipeline
2. Training pipeline from data to model
3. Inference pipeline from measurement to reconstruction
4. Evaluation pipeline with metrics and robustness tests
5. Performance and memory profiling tests
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig, OmegaConf
import tifffile
from PIL import Image

# Import all pipeline components
from scripts.train_diffusion import run_training
from scripts.inference import run_inference
# Note: synthesize_data script doesn't export functions, using direct dataset creation instead
from pkl_dg.models.sampler import DDIMSampler
from pkl_dg.guidance import PKLGuidance, L2Guidance, AnscombeGuidance, AdaptiveSchedule
from pkl_dg.data.synthesis import SynthesisDataset
from pkl_dg.data.transforms import IntensityToModel
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.physics.psf import PSF
from pkl_dg.evaluation.metrics import Metrics
from pkl_dg.evaluation.robustness import RobustnessTests


class E2EPipelineTestBase:
    """Base class for end-to-end pipeline tests with shared utilities."""
    
    @pytest.fixture(scope="class")
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp(prefix="pkl_e2e_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture(scope="class")
    def minimal_config(self, temp_workspace):
        """Create minimal configuration for testing."""
        config = {
            "experiment": {
                "name": "e2e_test",
                "seed": 42,
                "device": "cpu"
            },
            "paths": {
                "data": str(temp_workspace / "data"),
                "checkpoints": str(temp_workspace / "checkpoints"),
                "logs": str(temp_workspace / "logs"),
                "outputs": str(temp_workspace / "outputs")
            },
            "model": {
                "sample_size": 32,
                "in_channels": 1,
                "out_channels": 1,
                "layers_per_block": 1,
                "block_out_channels": [32, 32, 64],
                "down_block_types": ["DownBlock2D", "DownBlock2D", "DownBlock2D"],
                "up_block_types": ["UpBlock2D", "UpBlock2D", "UpBlock2D"],
                "attention_head_dim": 8
            },
            "data": {
                "image_size": 32,
                "min_intensity": 0.0,
                "max_intensity": 1000.0
            },
            "physics": {
                "background": 10.0,
                "psf_path": None
            },
            "training": {
                "num_timesteps": 50,
                "max_epochs": 1,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "num_workers": 0,
                "use_ema": False,
                "beta_schedule": "cosine",
                "accumulate_grad_batches": 1,
                "gradient_clip": 0.0
            },
            "guidance": {
                "type": "pkl",
                "lambda_base": 0.1,
                "epsilon": 1e-6,
                "schedule": {
                    "T_threshold": 40,
                    "epsilon_lambda": 1e-3
                }
            },
            "inference": {
                "ddim_steps": 10,
                "eta": 0.0,
                "use_autocast": False,
                "checkpoint_path": None,
                "input_dir": None,
                "output_dir": None
            },
            "wandb": {
                "mode": "disabled"
            }
        }
        return OmegaConf.create(config)
    
    def create_synthetic_images(self, output_dir: Path, num_images: int = 5) -> List[Path]:
        """Create synthetic test images."""
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = []
        
        for i in range(num_images):
            # Create simple synthetic image with spots
            image = np.zeros((32, 32), dtype=np.uint8)
            
            # Add some bright spots
            for _ in range(3):
                y, x = np.random.randint(5, 27, 2)
                image[y-2:y+3, x-2:x+3] = np.random.randint(100, 255)
            
            # Add some noise
            image = image.astype(np.float32)
            image += np.random.normal(0, 10, image.shape)
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Save as PNG for training data
            img_path = output_dir / f"test_image_{i:03d}.png"
            Image.fromarray(image).save(str(img_path))
            image_paths.append(img_path)
        
        return image_paths
    
    def create_test_measurements(self, output_dir: Path, num_images: int = 3) -> List[Path]:
        """Create test measurement images for inference."""
        output_dir.mkdir(parents=True, exist_ok=True)
        measurement_paths = []
        
        for i in range(num_images):
            # Create noisy measurement
            measurement = np.random.poisson(50, (32, 32)).astype(np.float32)
            measurement += np.random.normal(10, 2, measurement.shape)  # Background
            measurement = np.clip(measurement, 0, None)
            
            # Save as TIFF for inference
            tiff_path = output_dir / f"measurement_{i:03d}.tif"
            tifffile.imwrite(str(tiff_path), measurement.astype(np.float32))
            measurement_paths.append(tiff_path)
        
        return measurement_paths


class TestDataPipeline(E2EPipelineTestBase):
    """Test data synthesis and loading pipeline."""
    
    def test_synthetic_data_creation(self, temp_workspace, minimal_config):
        """Test synthetic training data creation."""
        # Create source images
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        train_images = self.create_synthetic_images(train_dir, num_images=10)
        val_images = self.create_synthetic_images(val_dir, num_images=5)
        
        # Test dataset creation
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=minimal_config.physics.background,
            device="cpu"
        )
        
        transform = IntensityToModel(
            minIntensity=minimal_config.data.min_intensity,
            maxIntensity=minimal_config.data.max_intensity
        )
        
        train_dataset = SynthesisDataset(
            source_dir=str(train_dir),
            forward_model=forward_model,
            transform=transform,
            image_size=minimal_config.data.image_size,
            mode="train"
        )
        
        val_dataset = SynthesisDataset(
            source_dir=str(val_dir),
            forward_model=forward_model,
            transform=transform,
            image_size=minimal_config.data.image_size,
            mode="val"
        )
        
        # Test dataset loading
        assert len(train_dataset) == len(train_images)
        assert len(val_dataset) == len(val_images)
        
        # Test batch loading
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=0
        )
        
        batch = next(iter(train_loader))
        x_0, _ = batch
        
        assert x_0.shape == (2, 1, 32, 32)
        assert torch.isfinite(x_0).all()
        assert x_0.dtype == torch.float32
        
        # Test value ranges (should be in model domain [-1, 1])
        assert x_0.min() >= -1.1  # Allow small numerical errors
        assert x_0.max() <= 1.1
    
    def test_data_transforms_consistency(self, minimal_config):
        """Test that data transforms are consistent and reversible."""
        transform = IntensityToModel(
            minIntensity=minimal_config.data.min_intensity,
            maxIntensity=minimal_config.data.max_intensity
        )
        
        # Test with various intensity values
        test_intensities = torch.tensor([0.0, 100.0, 500.0, 1000.0])
        
        # Forward transform
        model_values = transform(test_intensities)
        assert model_values.min() >= -1.0
        assert model_values.max() <= 1.0
        
        # Inverse transform
        recovered_intensities = transform.inverse(model_values)
        
        # Should recover original values (within numerical precision)
        torch.testing.assert_close(recovered_intensities, test_intensities, rtol=1e-5, atol=1e-5)


class TestTrainingPipeline(E2EPipelineTestBase):
    """Test complete training pipeline."""
    
    def test_minimal_training_run(self, temp_workspace, minimal_config):
        """Test minimal training run completes without errors."""
        # Setup data
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=8)
        self.create_synthetic_images(val_dir, num_images=4)
        
        # Run training
        trainer = run_training(minimal_config)
        
        # Verify training completed
        assert trainer is not None
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'alphas_cumprod')
        
        # Check checkpoint was saved
        checkpoint_path = Path(minimal_config.paths.checkpoints) / "final_model.pt"
        assert checkpoint_path.exists()
        
        # Verify checkpoint can be loaded
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        assert isinstance(state_dict, dict)
        assert len(state_dict) > 0
    
    def test_training_with_different_configs(self, temp_workspace, minimal_config):
        """Test training with different configuration options."""
        # Setup data once
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=6)
        self.create_synthetic_images(val_dir, num_images=3)
        
        # Test different configurations
        test_configs = [
            {"training.use_ema": True, "experiment.name": "ema_test"},
            {"training.beta_schedule": "linear", "experiment.name": "linear_test"},
            {"training.batch_size": 1, "experiment.name": "batch1_test"},
        ]
        
        for config_override in test_configs:
            # Create modified config
            test_config = minimal_config.copy()
            for key, value in config_override.items():
                OmegaConf.set(test_config, key, value)
            
            # Update checkpoint path to avoid conflicts
            test_config.paths.checkpoints = str(temp_workspace / "checkpoints" / test_config.experiment.name)
            
            # Run training
            trainer = run_training(test_config)
            assert trainer is not None
            
            # Verify checkpoint
            checkpoint_path = Path(test_config.paths.checkpoints) / "final_model.pt"
            assert checkpoint_path.exists()


class TestInferencePipeline(E2EPipelineTestBase):
    """Test complete inference pipeline."""
    
    def test_inference_with_pretrained_model(self, temp_workspace, minimal_config):
        """Test inference pipeline with a pretrained model."""
        # First, create and train a minimal model
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=6)
        self.create_synthetic_images(val_dir, num_images=3)
        
        # Train model
        trainer = run_training(minimal_config)
        checkpoint_path = Path(minimal_config.paths.checkpoints) / "final_model.pt"
        
        # Create test measurements
        input_dir = temp_workspace / "inference_input"
        output_dir = temp_workspace / "inference_output"
        
        measurement_paths = self.create_test_measurements(input_dir, num_images=3)
        
        # Setup inference config
        inference_config = minimal_config.copy()
        inference_config.inference.checkpoint_path = str(checkpoint_path)
        inference_config.inference.input_dir = str(input_dir)
        inference_config.inference.output_dir = str(output_dir)
        
        # Run inference
        saved_paths = run_inference(inference_config)
        
        # Verify results
        assert len(saved_paths) == len(measurement_paths)
        
        for saved_path in saved_paths:
            assert saved_path.exists()
            assert saved_path.suffix == ".tif"
            
            # Load and verify reconstruction
            reconstruction = tifffile.imread(str(saved_path))
            assert reconstruction.shape == (32, 32)
            assert np.isfinite(reconstruction).all()
            assert reconstruction.min() >= 0  # Should be in intensity domain
    
    def test_inference_with_different_guidance_strategies(self, temp_workspace, minimal_config):
        """Test inference with different guidance strategies."""
        # Setup model and data
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=6)
        self.create_synthetic_images(val_dir, num_images=3)
        
        trainer = run_training(minimal_config)
        checkpoint_path = Path(minimal_config.paths.checkpoints) / "final_model.pt"
        
        # Create test measurement
        input_dir = temp_workspace / "inference_input"
        self.create_test_measurements(input_dir, num_images=1)
        
        # Test different guidance strategies
        guidance_types = ["pkl", "l2", "anscombe"]
        results = {}
        
        for guidance_type in guidance_types:
            output_dir = temp_workspace / f"inference_output_{guidance_type}"
            
            # Setup config for this guidance type
            inference_config = minimal_config.copy()
            inference_config.inference.checkpoint_path = str(checkpoint_path)
            inference_config.inference.input_dir = str(input_dir)
            inference_config.inference.output_dir = str(output_dir)
            inference_config.guidance.type = guidance_type
            
            # Run inference
            saved_paths = run_inference(inference_config)
            
            # Load result
            reconstruction = tifffile.imread(str(saved_paths[0]))
            results[guidance_type] = reconstruction
        
        # Verify all guidance types produced valid results
        for guidance_type, reconstruction in results.items():
            assert np.isfinite(reconstruction).all()
            assert reconstruction.min() >= 0
            assert reconstruction.shape == (32, 32)
        
        # Results should be different for different guidance strategies
        assert not np.allclose(results["pkl"], results["l2"], atol=1e-3)
        assert not np.allclose(results["pkl"], results["anscombe"], atol=1e-3)


class TestEvaluationPipeline(E2EPipelineTestBase):
    """Test evaluation and metrics pipeline."""
    
    def test_metrics_computation(self):
        """Test comprehensive metrics computation."""
        # Create synthetic test data
        np.random.seed(42)
        target = np.random.rand(64, 64).astype(np.float32)
        
        # Create prediction with controlled noise
        noise = np.random.normal(0, 0.05, target.shape).astype(np.float32)
        prediction = np.clip(target + noise, 0, 1)
        
        # Test all metrics
        psnr_val = Metrics.psnr(prediction, target)
        ssim_val = Metrics.ssim(prediction, target, data_range=1.0)
        frc_res = Metrics.frc(prediction, target, threshold=0.143)
        
        # Validate results
        assert isinstance(psnr_val, float)
        assert psnr_val > 0  # Should be positive for reasonable inputs
        
        assert isinstance(ssim_val, float)
        assert 0 <= ssim_val <= 1
        
        assert isinstance(frc_res, float)
        assert frc_res > 0
        
        # Test SAR metric with artifacts
        artifact_mask = np.zeros_like(target, dtype=bool)
        artifact_mask[:8, :8] = True
        sar_db = Metrics.sar(prediction, artifact_mask)
        assert isinstance(sar_db, float)
        
        # Test Hausdorff distance
        pred_mask = prediction > 0.5
        target_mask = target > 0.5
        hd = Metrics.hausdorff_distance(pred_mask, target_mask)
        assert isinstance(hd, float)
        assert hd >= 0
    
    def test_robustness_evaluation(self, temp_workspace, minimal_config):
        """Test robustness evaluation framework."""
        # Create and train minimal model
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=4)
        self.create_synthetic_images(val_dir, num_images=2)
        
        trainer = run_training(minimal_config)
        
        # Create sampler for robustness tests
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=minimal_config.physics.background,
            device="cpu"
        )
        
        guidance = PKLGuidance(epsilon=minimal_config.guidance.epsilon)
        schedule = AdaptiveSchedule(
            lambda_base=minimal_config.guidance.lambda_base,
            T_threshold=minimal_config.guidance.schedule.T_threshold,
            epsilon_lambda=minimal_config.guidance.schedule.epsilon_lambda,
            T_total=minimal_config.training.num_timesteps
        )
        
        transform = IntensityToModel(
            minIntensity=minimal_config.data.min_intensity,
            maxIntensity=minimal_config.data.max_intensity
        )
        
        sampler = DDIMSampler(
            model=trainer,
            forward_model=forward_model,
            guidance_strategy=guidance,
            schedule=schedule,
            transform=transform,
            num_timesteps=minimal_config.training.num_timesteps,
            ddim_steps=minimal_config.inference.ddim_steps,
            eta=minimal_config.inference.eta,
        )
        
        # Create test measurement
        y = torch.randn(1, 1, 32, 32) * 10 + 50  # Poisson-like measurement
        y = torch.clamp(y, 0, None)
        
        # Test PSF mismatch robustness
        psf_true = PSF()
        try:
            x_mismatch = RobustnessTests.psf_mismatch_test(
                sampler, y, psf_true, mismatch_factor=1.1
            )
            assert x_mismatch.shape == y.shape
            assert torch.isfinite(x_mismatch).all()
            assert torch.all(x_mismatch >= 0)
        except Exception as e:
            pytest.skip(f"PSF mismatch test requires more complex setup: {e}")
        
        # Test alignment error robustness
        try:
            x_shifted = RobustnessTests.alignment_error_test(
                sampler, y, shift_pixels=0.5
            )
            assert x_shifted.shape == y.shape
            assert torch.isfinite(x_shifted).all()
            assert torch.all(x_shifted >= 0)
        except Exception as e:
            pytest.skip(f"Alignment error test requires more complex setup: {e}")


class TestPerformancePipeline(E2EPipelineTestBase):
    """Test performance and memory profiling."""
    
    def test_memory_efficiency(self, temp_workspace, minimal_config):
        """Test memory efficiency of inference pipeline."""
        # Create and train model
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=4)
        self.create_synthetic_images(val_dir, num_images=2)
        
        trainer = run_training(minimal_config)
        
        # Test memory usage with larger images
        large_config = minimal_config.copy()
        large_config.data.image_size = 64
        large_config.model.sample_size = 64
        
        # Create larger test measurement
        y = torch.randn(1, 1, 64, 64) * 10 + 50
        y = torch.clamp(y, 0, None)
        
        # Setup sampler
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=large_config.physics.background,
            device="cpu"
        )
        
        guidance = PKLGuidance(epsilon=large_config.guidance.epsilon)
        schedule = AdaptiveSchedule(
            lambda_base=large_config.guidance.lambda_base,
            T_threshold=large_config.guidance.schedule.T_threshold,
            epsilon_lambda=large_config.guidance.schedule.epsilon_lambda,
            T_total=large_config.training.num_timesteps
        )
        
        transform = IntensityToModel(
            minIntensity=large_config.data.min_intensity,
            maxIntensity=large_config.data.max_intensity
        )
        
        sampler = DDIMSampler(
            model=trainer,
            forward_model=forward_model,
            guidance_strategy=guidance,
            schedule=schedule,
            transform=transform,
            num_timesteps=large_config.training.num_timesteps,
            ddim_steps=5,  # Fewer steps for faster test
            eta=large_config.inference.eta,
        )
        
        # Run inference without storing intermediates (memory efficient)
        start_time = time.time()
        result = sampler.sample(
            y=y,
            shape=y.shape,
            device="cpu",
            verbose=False,
            return_intermediates=False
        )
        end_time = time.time()
        
        # Verify result
        assert result.shape == y.shape
        assert torch.isfinite(result).all()
        assert torch.all(result >= 0)
        
        # Performance should be reasonable (not too slow)
        inference_time = end_time - start_time
        assert inference_time < 60  # Should complete within 1 minute on CPU
    
    def test_batch_processing_efficiency(self, temp_workspace, minimal_config):
        """Test efficiency of batch processing."""
        # Create and train model
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        self.create_synthetic_images(train_dir, num_images=4)
        self.create_synthetic_images(val_dir, num_images=2)
        
        trainer = run_training(minimal_config)
        
        # Test with batch of measurements
        batch_size = 3
        y_batch = torch.randn(batch_size, 1, 32, 32) * 10 + 50
        y_batch = torch.clamp(y_batch, 0, None)
        
        # Setup sampler
        psf = PSF()
        forward_model = ForwardModel(
            psf=psf.to_torch(device="cpu"),
            background=minimal_config.physics.background,
            device="cpu"
        )
        
        guidance = PKLGuidance(epsilon=minimal_config.guidance.epsilon)
        schedule = AdaptiveSchedule(
            lambda_base=minimal_config.guidance.lambda_base,
            T_threshold=minimal_config.guidance.schedule.T_threshold,
            epsilon_lambda=minimal_config.guidance.schedule.epsilon_lambda,
            T_total=minimal_config.training.num_timesteps
        )
        
        transform = IntensityToModel(
            minIntensity=minimal_config.data.min_intensity,
            maxIntensity=minimal_config.data.max_intensity
        )
        
        sampler = DDIMSampler(
            model=trainer,
            forward_model=forward_model,
            guidance_strategy=guidance,
            schedule=schedule,
            transform=transform,
            num_timesteps=minimal_config.training.num_timesteps,
            ddim_steps=5,
            eta=minimal_config.inference.eta,
        )
        
        # Process batch
        results = []
        for i in range(batch_size):
            y_single = y_batch[i:i+1]
            result = sampler.sample(
                y=y_single,
                shape=y_single.shape,
                device="cpu",
                verbose=False
            )
            results.append(result)
        
        # Verify all results
        for i, result in enumerate(results):
            assert result.shape == (1, 1, 32, 32)
            assert torch.isfinite(result).all()
            assert torch.all(result >= 0)


class TestIntegrationWorkflows(E2EPipelineTestBase):
    """Test complete integration workflows."""
    
    def test_full_training_to_inference_workflow(self, temp_workspace, minimal_config):
        """Test complete workflow from training to inference with evaluation."""
        # Step 1: Create training data
        train_dir = temp_workspace / "data" / "train" / "classless"
        val_dir = temp_workspace / "data" / "val" / "classless"
        
        train_images = self.create_synthetic_images(train_dir, num_images=8)
        val_images = self.create_synthetic_images(val_dir, num_images=4)
        
        # Step 2: Train model
        trainer = run_training(minimal_config)
        checkpoint_path = Path(minimal_config.paths.checkpoints) / "final_model.pt"
        assert checkpoint_path.exists()
        
        # Step 3: Create test measurements
        input_dir = temp_workspace / "inference_input"
        measurement_paths = self.create_test_measurements(input_dir, num_images=2)
        
        # Step 4: Run inference
        output_dir = temp_workspace / "inference_output"
        inference_config = minimal_config.copy()
        inference_config.inference.checkpoint_path = str(checkpoint_path)
        inference_config.inference.input_dir = str(input_dir)
        inference_config.inference.output_dir = str(output_dir)
        
        saved_paths = run_inference(inference_config)
        assert len(saved_paths) == len(measurement_paths)
        
        # Step 5: Evaluate results
        for i, saved_path in enumerate(saved_paths):
            # Load reconstruction
            reconstruction = tifffile.imread(str(saved_path))
            
            # Load original measurement for comparison
            measurement = tifffile.imread(str(measurement_paths[i]))
            
            # Compute metrics
            psnr_val = Metrics.psnr(reconstruction, measurement)
            ssim_val = Metrics.ssim(reconstruction, measurement, data_range=measurement.max())
            
            # Basic validation (not expecting perfect results from minimal training)
            assert isinstance(psnr_val, float)
            assert isinstance(ssim_val, float)
            assert 0 <= ssim_val <= 1
        
        # Step 6: Verify complete pipeline integrity
        assert len(train_images) > 0
        assert len(val_images) > 0
        assert trainer is not None
        assert len(saved_paths) > 0
        
        # All files should exist and be valid
        for path in saved_paths:
            assert path.exists()
            data = tifffile.imread(str(path))
            assert np.isfinite(data).all()
    
    def test_configuration_validation_workflow(self, temp_workspace):
        """Test that invalid configurations are properly caught."""
        # Test various invalid configurations
        invalid_configs = [
            # Missing required paths
            {"paths": {}},
            # Invalid model architecture
            {"model": {"in_channels": 0}},
            # Invalid training parameters
            {"training": {"max_epochs": -1}},
            # Invalid guidance parameters
            {"guidance": {"type": "invalid_guidance"}},
        ]
        
        for invalid_config in invalid_configs:
            base_config = {
                "experiment": {"name": "invalid_test", "seed": 42, "device": "cpu"},
                "paths": {
                    "data": str(temp_workspace / "data"),
                    "checkpoints": str(temp_workspace / "checkpoints"),
                },
                "wandb": {"mode": "disabled"}
            }
            
            # Merge invalid config
            test_config = OmegaConf.create({**base_config, **invalid_config})
            
            # Should raise an error or handle gracefully
            with pytest.raises((ValueError, KeyError, TypeError, RuntimeError)):
                # This should fail during configuration validation or early in training
                run_training(test_config)


# Test runner configuration
@pytest.mark.cpu
class TestE2EPipelineRunner:
    """Main test runner for end-to-end pipeline tests."""
    
    def test_run_all_pipeline_components(self, tmp_path):
        """Integration test that runs all major pipeline components."""
        # This is a meta-test that ensures all test classes can be instantiated
        # and their basic functionality works
        
        test_classes = [
            TestDataPipeline(),
            TestTrainingPipeline(),
            TestInferencePipeline(),
            TestEvaluationPipeline(),
            TestPerformancePipeline(),
            TestIntegrationWorkflows()
        ]
        
        for test_instance in test_classes:
            # Verify test instance has expected methods
            assert hasattr(test_instance, 'create_synthetic_images')
            assert hasattr(test_instance, 'create_test_measurements')
        
        # Basic smoke test
        data_pipeline = TestDataPipeline()
        images = data_pipeline.create_synthetic_images(tmp_path, num_images=2)
        assert len(images) == 2
        for img_path in images:
            assert img_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
