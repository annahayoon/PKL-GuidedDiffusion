#!/usr/bin/env python3
"""
Test suite for Enhancement Task 2.3: Memory Profiling Utilities
"""

import torch
import time
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.diffusion import DDPMTrainer
from pkl_dg.utils.memory import (
    MemoryProfiler,
    profile_memory,
    get_memory_summary,
    cleanup_memory,
)


def test_memory_profiling():
    """Test basic memory profiling functionality."""
    print("🧪 Testing Memory Profiling...")
    
    try:
        profiler = MemoryProfiler(verbose=True)
        
        # Test snapshot
        snapshot = profiler.take_snapshot(step=0, phase="test")
        assert snapshot is not None
        print("✅ Memory snapshot works")
        
        # Test continuous monitoring
        profiler.start_monitoring()
        
        # Allocate memory
        if torch.cuda.is_available():
            tensor = torch.randn(1000, 1000, device="cuda")
        else:
            tensor = torch.randn(1000, 1000)
        
        time.sleep(0.2)
        profile = profiler.stop_monitoring()
        
        assert len(profile.snapshots) >= 2
        print(f"✅ Monitoring captured {len(profile.snapshots)} snapshots")
        
        # Test context manager
        with profile_memory("test_op", verbose=False) as prof:
            result = torch.sum(tensor ** 2)
        
        assert len(prof.profile.snapshots) >= 2
        print("✅ Context manager profiling works")
        
        # Test memory summary
        summary = get_memory_summary()
        assert "gpu_allocated" in summary
        print("✅ Memory summary works")
        
        # Test cleanup
        del tensor
        cleanup_memory(verbose=False)
        print("✅ Memory cleanup works")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory profiling test failed: {e}")
        return False


def test_trainer_integration():
    """Test integration with trainer."""
    print("\n🧪 Testing Trainer Integration...")
    
    try:
        # Create trainer
        config = {"num_timesteps": 50, "use_ema": False}
        unet_config = {"in_channels": 2, "out_channels": 1, "sample_size": 64}
        
        unet = DenoisingUNet(unet_config)
        trainer = DDPMTrainer(unet, config)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = trainer.to(device)
        
        # Profile training step
        profiler = MemoryProfiler(verbose=False)
        
        with profiler.profile_context("training"):
            x_0 = torch.randn(2, 1, 64, 64, device=device)
            c_wf = torch.randn(2, 1, 64, 64, device=device)
            
            trainer.train()
            loss = trainer.training_step((x_0, c_wf), 0)
        
        assert len(profiler.profile.snapshots) >= 2
        assert torch.isfinite(loss)
        
        print(f"✅ Training step profiled: loss = {loss:.4f}")
        print(f"✅ Peak GPU: {profiler.profile.peak_gpu_allocated:.3f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer integration test failed: {e}")
        return False


def main():
    """Run tests for Task 2.3."""
    print("🚀 Testing Enhancement Task 2.3: Memory Profiling Utilities")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    if test_memory_profiling():
        success_count += 1
    
    if test_trainer_integration():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"🎉 Task 2.3 Testing Complete: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ Task 2.3 implementation is working correctly!")
        return True
    else:
        print("❌ Task 2.3 implementation needs more work")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)