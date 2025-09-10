#!/usr/bin/env python3
"""
Setup script for Google Colab environment with A100 GPU optimizations.
This script configures the environment for optimal training performance.
"""

import os
import sys
import subprocess
import torch
import gc
from pathlib import Path


def setup_colab_environment():
    """Setup Colab environment with A100 optimizations."""
    print("🚀 Setting up Google Colab environment for A100 GPU training...")
    
    # Detect environment
    is_colab = 'COLAB_GPU' in os.environ or 'google.colab' in str(os.getcwd())
    if not is_colab:
        print("⚠️ Warning: Not running in Google Colab environment")
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    
    # Memory optimization settings
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '0'  # Enable memory caching
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    
    print("✅ Environment variables set")


def optimize_pytorch_for_a100():
    """Optimize PyTorch settings for A100 GPU."""
    print("🎯 Optimizing PyTorch for A100 GPU...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ GPU: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
    
    # Check if it's A100
    is_a100 = 'A100' in gpu_name
    if is_a100:
        print("✅ A100 GPU detected - applying optimizations")
        
        # Enable TensorFloat-32 for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✅ TensorFloat-32 enabled")
        
        # Optimize cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ cuDNN optimizations enabled")
        
        # Set optimal memory allocation strategy
        torch.cuda.empty_cache()
        gc.collect()
        print("✅ Memory cache cleared")
        
    else:
        print(f"⚠️ Non-A100 GPU detected: {gpu_name}")
        print("   Some optimizations may not be optimal")
    
    return True


def setup_memory_management():
    """Setup memory management for long training runs."""
    print("🧠 Setting up memory management...")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Set memory fraction (use 90% of available memory)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
        print("✅ GPU memory fraction set to 90%")
    
    print("✅ Memory management configured")


def create_optimized_config():
    """Create an optimized configuration file for A100."""
    print("⚙️ Creating optimized configuration...")
    
    config_content = """
# A100-optimized configuration
defaults:
  - model: unet
  - data: synthesis
  - physics: microscopy
  - guidance: pkl
  - inference: default
  - training: ddpm_colab
  - override hydra/launcher: basic
  - _self_

experiment:
  name: a100_${now:%Y-%m-%d_%H-%M-%S}
  seed: 42
  device: cuda
  mixed_precision: true
  enable_memory_profiling: false

paths:
  root: ${oc.env:PROJECT_ROOT,/content/PKL-DiffusionDenoising}
  data: ${paths.root}/data
  checkpoints: ${paths.root}/checkpoints
  outputs: ${paths.root}/outputs
  logs: ${paths.root}/logs

wandb:
  project: pkl-diffusion-a100
  entity: null
  mode: online

# A100-specific optimizations
a100_optimizations:
  # Memory optimizations
  gradient_checkpointing: true
  empty_cache_frequency: 50
  memory_efficient_attention: true
  
  # Training optimizations
  use_diffusers_scheduler: true
  scheduler_type: dpm_solver
  use_karras_sigmas: true
  
  # Checkpointing
  checkpoint_every_n_epochs: 3
  save_samples_every_n_epochs: 5
  
  # Early stopping
  early_stopping_monitor: val/loss
  early_stopping_min_delta: 0.001
"""
    
    config_path = Path("configs/config_a100.yaml")
    config_path.write_text(config_content)
    print(f"✅ A100 config created: {config_path}")


def setup_data_directories():
    """Setup data directories for training."""
    print("📁 Setting up data directories...")
    
    directories = [
        "data/train",
        "data/val", 
        "checkpoints",
        "outputs/samples",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("✅ Data directories ready")


def install_requirements():
    """Install required packages for Colab."""
    print("📦 Installing required packages...")
    
    packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
        "diffusers>=0.21.0",
        "accelerate>=0.21.0",
        "pytorch-lightning>=2.0.0",
        "tensorboard>=2.13.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], 
                         check=True, capture_output=True)
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")


def run_memory_test():
    """Run a memory test to verify A100 optimization."""
    print("🧪 Running memory test...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available for memory test")
        return False
    
    try:
        # Test tensor operations
        device = torch.device('cuda')
        
        # Create large tensors to test memory
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Test mixed precision
        with torch.cuda.amp.autocast():
            z = torch.matmul(x, y)
        
        # Test memory allocation
        memory_allocated = torch.cuda.memory_allocated(device) / (1024**3)
        memory_reserved = torch.cuda.memory_reserved(device) / (1024**3)
        
        print(f"✅ Memory test passed")
        print(f"   Allocated: {memory_allocated:.2f} GB")
        print(f"   Reserved: {memory_reserved:.2f} GB")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("=" * 60)
    print("🚀 PKL Diffusion Denoising - Colab A100 Setup")
    print("=" * 60)
    
    # Run setup steps
    setup_colab_environment()
    optimize_pytorch_for_a100()
    setup_memory_management()
    create_optimized_config()
    setup_data_directories()
    
    # Test setup
    if run_memory_test():
        print("\n✅ Setup completed successfully!")
        print("🎯 Ready for A100-optimized training")
    else:
        print("\n⚠️ Setup completed with warnings")
        print("   Some optimizations may not be available")
    
    print("\n📋 Next steps:")
    print("1. Run: python scripts/synthesize_data.py --config-name=config_a100")
    print("2. Run: python scripts/train_diffusion_colab.py --config-name=config_a100")
    print("3. Monitor training progress in the logs")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
