#!/usr/bin/env python3
"""
Tests for:
- Optimized FFT convolution caching in ForwardModel
- Batch-aware PKL guidance computation
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl import PKLGuidance


def test_fft_cache_speed_and_hits():
    print("🧪 Testing FFT cache behavior...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psf = torch.randn(64, 64)
    fm = ForwardModel(psf, device=device)

    x = torch.randn(4, 1, 256, 256, device=device)

    # First call - warmup and measure
    start = time.time()
    y1 = fm.apply_psf(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t1 = time.time() - start

    # Second call - should be faster due to cache
    start = time.time()
    y2 = fm.apply_psf(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t2 = time.time() - start

    stats = fm.get_cache_stats()
    print(f"   Cache stats: {stats}")
    print(f"   First: {t1:.4f}s, Second: {t2:.4f}s")

    assert stats["base_entries"] > 0, "Base cache should be populated"
    assert stats["device_entries"] > 0, "Device cache should be populated"
    assert t2 <= t1 * 0.85 or t2 < 0.005, "Second call should be faster (or very fast)"

    print("✅ FFT caching works")


def test_batch_aware_guidance():
    print("🧪 Testing batch-aware PKL guidance...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    psf = torch.rand(21, 21)
    fm = ForwardModel(psf, background=0.1, device=device, read_noise_sigma=0.0)

    # Single sample
    x_single = torch.rand(1, 1, 128, 128, device=device)
    y_single = fm.forward(x_single)

    # Batch
    x_batch = torch.rand(4, 1, 128, 128, device=device)
    y_batch = fm.forward(x_batch)

    g = PKLGuidance(epsilon=1e-6)

    grad_single = g.compute_gradient(x_single, y_single, fm, t=100)
    grad_batch = g.compute_gradient(x_batch, y_batch, fm, t=100)

    assert grad_single.shape == x_single.shape
    assert grad_batch.shape == x_batch.shape

    # Check vectorization by comparing per-sample to batch slice
    xb = x_batch[0:1].clone()
    yb = y_batch[0:1].clone()
    grad_b0 = g.compute_gradient(xb, yb, fm, t=100)

    grad_batch_recomputed = g.compute_gradient(x_batch.clone(), y_batch.clone(), fm, t=100)
    diff = (grad_b0 - grad_batch_recomputed[0:1]).abs().max().item()
    print(f"   Max diff batch vs single: {diff:.6f}")

    assert diff < 1e-5, "Batch and single-sample gradients should match for same inputs"
    print("✅ Batch-aware guidance works")


def main():
    ok = True
    try:
        test_fft_cache_speed_and_hits()
    except AssertionError as e:
        print(f"❌ FFT cache test failed: {e}")
        ok = False
    try:
        test_batch_aware_guidance()
    except AssertionError as e:
        print(f"❌ Batch guidance test failed: {e}")
        ok = False

    print("\n🎉 Enhancement Task: FFT caching + batch guidance:", "PASS" if ok else "NEEDS WORK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())