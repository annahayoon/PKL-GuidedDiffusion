# Distributed Training and Inference Plan for PKL-DiffusionDenoising

## Overview
This document provides a detailed plan to make the PKL-DiffusionDenoising code run efficiently on both multi-GPU systems (DGX-1 with 8x V100) and single GPU systems (H100). The implementation will automatically detect the available hardware and configure accordingly.

## Table of Contents
1. [Hardware Detection and Configuration](#hardware-detection)
2. [Code Modifications Required](#code-modifications)
3. [Training Script Updates](#training-script-updates)
4. [Inference Script Updates](#inference-script-updates)
5. [Launch Commands](#launch-commands)
6. [Testing and Validation](#testing-validation)
7. [Performance Expectations](#performance-expectations)

## 1. Hardware Detection and Configuration

### 1.1 Create Hardware Detection Utility
**File**: `pkl_dg/utils/hardware.py`

```python
import torch
import os
from typing import Dict, Any, Optional

class HardwareDetector:
    """Detects available hardware and provides optimal configuration."""
    
    def __init__(self):
        self.gpu_count = torch.cuda.device_count()
        self.gpu_memory = self._get_gpu_memory()
        self.gpu_arch = self._get_gpu_architecture()
        
    def _get_gpu_memory(self) -> Dict[int, float]:
        """Get memory per GPU in GB."""
        memory = {}
        for i in range(self.gpu_count):
            memory[i] = torch.cuda.get_device_properties(i).total_memory / 1e9
        return memory
    
    def _get_gpu_architecture(self) -> str:
        """Detect GPU architecture."""
        if self.gpu_count == 0:
            return "cpu"
        
        # Get compute capability
        major, minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return "ampere"  # A100, H100
        elif major >= 7:
            return "turing"  # RTX 20xx, V100
        else:
            return "pascal"  # GTX 10xx
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration for current hardware."""
        config = {
            "gpu_count": self.gpu_count,
            "gpu_memory": self.gpu_memory,
            "architecture": self.gpu_arch,
            "use_ddp": self.gpu_count > 1,
            "batch_size_per_gpu": self._get_optimal_batch_size(),
            "mixed_precision": self._should_use_mixed_precision(),
            "gradient_checkpointing": self._should_use_gradient_checkpointing(),
        }
        return config
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size per GPU based on memory."""
        if self.gpu_count == 0:
            return 1
        
        memory_gb = self.gpu_memory[0]
        if memory_gb >= 40:  # H100, A100
            return 16
        elif memory_gb >= 30:  # A40
            return 12
        elif memory_gb >= 16:  # V100
            return 8
        else:
            return 4
    
    def _should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision should be used."""
        return self.gpu_arch in ["ampere", "turing"]
    
    def _should_use_gradient_checkpointing(self) -> bool:
        """Determine if gradient checkpointing should be used."""
        return self.gpu_arch in ["pascal", "turing"] or self.gpu_memory[0] < 20

# Global instance
hardware_detector = HardwareDetector()
```

### 1.2 Update Requirements
**File**: `requirements.txt`
```
# Add distributed training dependencies
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
# ... existing requirements ...
```

## 2. Code Modifications Required

### 2.1 Update DDPMTrainer for DDP Support
**File**: `pkl_dg/models/diffusion.py`

Add these methods to the `DDPMTrainer` class:

```python
def setup_ddp(self, device_ids: Optional[List[int]] = None):
    """Setup DistributedDataParallel wrapper."""
    if not torch.distributed.is_initialized():
        return self
    
    from torch.nn.parallel import DistributedDataParallel as DDP
    self.model = DDP(self.model, device_ids=device_ids)
    return self

def get_model_for_inference(self):
    """Get model for inference (unwrap DDP if needed)."""
    if hasattr(self.model, 'module'):
        return self.model.module
    return self.model

def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
    """Save checkpoint with DDP awareness."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return  # Only save on rank 0
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.get_model_for_inference().state_dict(),
        'optimizer_state_dict': getattr(self, 'optimizer', None),
        'scheduler_state_dict': getattr(self, 'scheduler', None),
        'is_best': is_best,
    }
    
    if hasattr(self, 'ema_model') and self.ema_model is not None:
        checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
    
    torch.save(checkpoint, path)
```

### 2.2 Create Distributed Data Loading Utility
**File**: `pkl_dg/utils/distributed.py`

```python
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Callable, Any

def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs
) -> DataLoader:
    """Create DataLoader with distributed sampling if needed."""
    
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

def setup_distributed(backend: str = 'nccl'):
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
```

## 3. Training Script Updates

### 3.1 Update Training Script for DDP
**File**: `scripts/train_real.py`

Add these imports at the top:
```python
import os
from pkl_dg.utils.hardware import hardware_detector
from pkl_dg.utils.distributed import setup_distributed, cleanup_distributed, create_distributed_dataloader
```

Update the `parse_args` function:
```python
def parse_args():
    parser = argparse.ArgumentParser(description="Train PKL-DiffusionDenoising model on real microscopy data.")
    
    # Hardware detection
    parser.add_argument("--auto_config", action="store_true", default=True,
                       help="Automatically detect hardware and configure settings")
    
    # Distributed training
    parser.add_argument("--distributed", action="store_true",
                       help="Enable distributed training")
    parser.add_argument("--backend", type=str, default="nccl",
                       help="Distributed backend")
    
    # ... existing arguments ...
    
    return parser.parse_args()
```

Update the `run_training` function:
```python
def run_training(args):
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(args.backend)
    
    # Hardware detection and auto-configuration
    if args.auto_config:
        hw_config = hardware_detector.get_optimal_config()
        print(f"Hardware detected: {hw_config['gpu_count']} GPUs, {hw_config['architecture']} architecture")
        
        # Override args with optimal settings
        if not hasattr(args, 'batch_size') or args.batch_size == 4:
            args.batch_size = hw_config['batch_size_per_gpu']
        if not hasattr(args, 'mixed_precision'):
            args.mixed_precision = hw_config['mixed_precision']
        if not hasattr(args, 'gradient_checkpointing'):
            args.gradient_checkpointing = hw_config['gradient_checkpointing']
    
    # Set device
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # Only print on rank 0
    if rank == 0:
        print(f"Training on {world_size} GPUs, using device {device}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    
    # ... rest of existing setup ...
    
    # Create distributed data loaders
    train_loader = create_distributed_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = create_distributed_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    
    # Setup model with DDP
    if args.distributed:
        ddpm_trainer.setup_ddp(device_ids=[local_rank])
    
    # ... rest of training loop ...
    
    # Cleanup
    cleanup_distributed()
```

### 3.2 Add Training Loop Updates
Update the training loop to handle distributed training:

```python
# In the training loop
for epoch in range(args.max_epochs):
    # Set epoch for distributed sampler
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
    
    # ... training step ...
    
    # Only save checkpoints on rank 0
    if rank == 0:
        # ... checkpointing code ...
    
    # Synchronize all processes
    if args.distributed:
        torch.distributed.barrier()
```

## 4. Inference Script Updates

### 4.1 Create Distributed Inference Script
**File**: `scripts/infer_distributed.py`

```python
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.data.transforms import AnscombeToModel, IntensityToModel
from pkl_dg.utils.hardware import hardware_detector
from pkl_dg.utils.distributed import setup_distributed, cleanup_distributed

def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference for PKL-DiffusionDenoising")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model if available")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed inference")
    return parser.parse_args()

def run_inference(args):
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model_cfg = checkpoint.get('model_config', {})
    training_cfg = checkpoint.get('training_config', {})
    
    unet = DenoisingUNet(model_cfg)
    unet.load_state_dict(checkpoint['model_state_dict'])
    
    ddpm = DDPMTrainer(unet, training_cfg)
    ddpm.to(device)
    
    # Load EMA model if available
    if args.use_ema and 'ema_model_state_dict' in checkpoint:
        ddpm.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
    
    # Setup DDP if distributed
    if args.distributed:
        ddpm.setup_ddp(device_ids=[local_rank])
    
    # Get input files for this rank
    input_files = list(Path(args.input_dir).glob("*.png"))
    files_per_rank = len(input_files) // world_size
    start_idx = rank * files_per_rank
    end_idx = start_idx + files_per_rank if rank < world_size - 1 else len(input_files)
    rank_files = input_files[start_idx:end_idx]
    
    # Process files
    ddpm.eval()
    with torch.no_grad():
        for file_path in rank_files:
            # Load and process image
            img = Image.open(file_path).convert('L')
            img_tensor = torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            
            # Apply transform
            transform = AnscombeToModel(maxIntensity=1.0)
            img_model = transform(img_tensor)
            
            # Generate
            output = ddpm.fast_sample(
                shape=img_model.shape,
                num_inference_steps=args.num_inference_steps,
                device=device,
                use_ema=args.use_ema,
                conditioner=img_model
            )
            
            # Save result
            output_path = Path(args.output_dir) / f"denoised_{file_path.stem}.png"
            output_img = transform.inverse(output.clamp(-1, 1)).squeeze().cpu().numpy()
            output_img = (np.clip(output_img, 0, 1) * 255).astype('uint8')
            Image.fromarray(output_img).save(output_path)
    
    cleanup_distributed()

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
```

## 5. Launch Commands

### 5.1 Single GPU (H100)
```bash
# H100 with auto-configuration
python scripts/train_real.py \
  --data_root data/real_microscopy \
  --checkpoints checkpoints/h100_run \
  --logs logs/h100_run \
  --name h100_run \
  --auto_config \
  --mixed_precision \
  --conditioning \
  --noise_model poisson \
  --use_diffusers_scheduler \
  --scheduler_type dpm_solver
```

### 5.2 Multi-GPU (DGX-1)
```bash
# 8x V100 with distributed training
torchrun --nproc_per_node=8 scripts/train_real.py \
  --data_root data/real_microscopy \
  --checkpoints checkpoints/dgx1_run \
  --logs logs/dgx1_run \
  --name dgx1_run \
  --distributed \
  --auto_config \
  --mixed_precision \
  --conditioning \
  --noise_model poisson \
  --use_diffusers_scheduler \
  --scheduler_type dpm_solver
```

### 5.3 Multi-Node (Multiple DGX-1)
```bash
# 2 nodes, 8 GPUs each
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
  --master_addr="192.168.1.1" --master_port=12345 \
  scripts/train_real.py \
  --data_root data/real_microscopy \
  --checkpoints checkpoints/multi_node_run \
  --logs logs/multi_node_run \
  --name multi_node_run \
  --distributed \
  --auto_config \
  --mixed_precision \
  --conditioning \
  --noise_model poisson \
  --use_diffusers_scheduler \
  --scheduler_type dpm_solver
```

## 6. Testing and Validation

### 6.1 Create Test Script
**File**: `scripts/test_distributed.py`

```python
import torch
from pkl_dg.utils.hardware import hardware_detector
from pkl_dg.utils.distributed import setup_distributed, cleanup_distributed

def test_hardware_detection():
    """Test hardware detection."""
    hw_config = hardware_detector.get_optimal_config()
    print(f"GPU Count: {hw_config['gpu_count']}")
    print(f"GPU Memory: {hw_config['gpu_memory']}")
    print(f"Architecture: {hw_config['architecture']}")
    print(f"Use DDP: {hw_config['use_ddp']}")
    print(f"Batch Size per GPU: {hw_config['batch_size_per_gpu']}")
    print(f"Mixed Precision: {hw_config['mixed_precision']}")
    print(f"Gradient Checkpointing: {hw_config['gradient_checkpointing']}")

def test_distributed_setup():
    """Test distributed setup."""
    try:
        rank, world_size, local_rank = setup_distributed()
        print(f"Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")
        cleanup_distributed()
        print("Distributed setup test passed")
    except Exception as e:
        print(f"Distributed setup test failed: {e}")

if __name__ == "__main__":
    test_hardware_detection()
    test_distributed_setup()
```

### 6.2 Run Tests
```bash
# Test hardware detection
python scripts/test_distributed.py

# Test distributed setup
torchrun --nproc_per_node=2 scripts/test_distributed.py
```

## 7. Performance Expectations

### 7.1 Single GPU Performance
| GPU | Memory | Batch Size | Expected Time |
|-----|--------|------------|---------------|
| H100 | 80GB | 16 | 24-40 hours |
| A100 | 40GB | 12 | 30-50 hours |
| A40 | 48GB | 12 | 35-55 hours |
| V100 | 16GB | 8 | 47-78 hours |

### 7.2 Multi-GPU Performance
| Configuration | GPUs | Expected Speedup | Expected Time |
|---------------|------|------------------|---------------|
| DGX-1 | 8x V100 | 6-7x | 6-10 hours |
| 2x H100 | 2x H100 | 1.7-1.8x | 12-20 hours |
| 4x A100 | 4x A100 | 3.2-3.5x | 8-15 hours |

### 7.3 Memory Usage
- **Single V100**: ~12-14GB (batch_size=8)
- **Single H100**: ~20-25GB (batch_size=16)
- **8x V100**: ~12-14GB per GPU (batch_size=8)

## 8. Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Create `pkl_dg/utils/hardware.py`
- [ ] Create `pkl_dg/utils/distributed.py`
- [ ] Update `pkl_dg/models/diffusion.py` with DDP support
- [ ] Update `scripts/train_real.py` for distributed training

### Phase 2: Testing and Validation
- [ ] Create `scripts/test_distributed.py`
- [ ] Test hardware detection
- [ ] Test distributed setup
- [ ] Validate training on single GPU
- [ ] Validate training on multi-GPU

### Phase 3: Inference and Optimization
- [ ] Create `scripts/infer_distributed.py`
- [ ] Test inference on single GPU
- [ ] Test inference on multi-GPU
- [ ] Optimize batch sizes for different hardware
- [ ] Add memory profiling for distributed training

### Phase 4: Documentation and Deployment
- [ ] Update README with distributed training instructions
- [ ] Create example launch scripts
- [ ] Document performance benchmarks
- [ ] Create troubleshooting guide

## 9. Troubleshooting

### Common Issues
1. **NCCL errors**: Ensure all GPUs are visible and accessible
2. **Memory errors**: Reduce batch size or enable gradient checkpointing
3. **Communication errors**: Check network connectivity between nodes
4. **Checkpoint loading**: Ensure checkpoints are saved/loaded correctly with DDP

### Debug Commands
```bash
# Check GPU visibility
nvidia-smi

# Check distributed setup
python -c "import torch; print(torch.distributed.is_available())"

# Test NCCL
python -c "import torch; print(torch.distributed.is_nccl_available())"
```

This plan provides a comprehensive approach to making the code run efficiently on both single GPU and multi-GPU systems, with automatic hardware detection and configuration.
