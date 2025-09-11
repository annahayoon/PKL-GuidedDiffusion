# Performance Enhancement TODO List

## Overview
This document outlines performance improvements for the PKL-DiffusionDenoising project. Each item includes background, current implementation location, and step-by-step enhancement instructions.

**Note**: We avoid `torch.compile` due to debugging difficulties. All optimizations focus on algorithmic improvements and better use of existing PyTorch features.

---

## 🔴 High Priority (Quick Wins)

### 1. Optimize FFT Convolution Caching
**Why needed**: Currently, FFT operations in `pkl_dg/physics/forward_model.py` recompute transforms frequently. FFTs are expensive operations that can be cached and reused.

**Current code location**: 
- File: `pkl_dg/physics/forward_model.py`
- Method: `_get_psf_fft()` (lines 97-122)

**Problem**: 
- FFT of PSF is recomputed when switching between different tensor dtypes
- Cache invalidation is too aggressive

**Enhancement steps**:
```python
# Step 1: In forward_model.py, modify the caching strategy
# Current (line 99-102):
key = (height, width, dtype)
cached = self._psf_fft_cache.get(key)

# Enhanced: Add persistent cache with better key management
def _get_psf_fft(self, height: int, width: int, dtype: torch.dtype, device: torch.device):
    # Use device-agnostic caching
    key = f"{height}_{width}_{str(dtype)}"
    
    # Step 2: Pre-compute common sizes at initialization
    # Add to __init__ method:
    self.common_sizes = [(256, 256), (512, 512), (128, 128)]
    for size in self.common_sizes:
        for dtype in [torch.float32, torch.float16]:
            self._precompute_fft(*size, dtype)
```

**Expected improvement**: 15-20% faster inference by avoiding redundant FFT computations

---

### 2. Batch-Aware Guidance Computation
**Why needed**: Current guidance computes gradients sample-by-sample. Batch processing leverages GPU parallelism better.

**Current code location**:
- Files: `pkl_dg/guidance/pkl.py`, `l2.py`, `anscombe.py`
- Method: `compute_gradient()` (processes one sample at a time)

**Problem**: 
- Not utilizing GPU's parallel processing capabilities
- Redundant memory transfers between CPU and GPU

**Enhancement steps**:
```python
# Step 1: In pkl.py, modify compute_gradient to handle batches
# Current (line 9-29):
def compute_gradient(self, x0_hat: torch.Tensor, y: torch.Tensor, ...):
    Ax = forward_model.apply_psf(x0_hat)  # Single sample
    
# Enhanced:
def compute_gradient(self, x0_hat: torch.Tensor, y: torch.Tensor, ...):
    # Check if batch dimension exists
    if x0_hat.dim() == 4:  # Batch processing
        # Process entire batch at once
        Ax = forward_model.apply_psf(x0_hat)  # Works on full batch
        # All operations are already vectorized in PyTorch
    else:
        # Fallback to single sample
        Ax = forward_model.apply_psf(x0_hat.unsqueeze(0)).squeeze(0)
```

**Testing**: Ensure both single sample and batch inputs work correctly

**Expected improvement**: 25-30% faster when processing batches

---

### 3. Memory-Efficient Attention in UNet
**Why needed**: Attention blocks use O(n²) memory. Memory-efficient attention reduces this without losing quality.

**Current code location**:
- File: `pkl_dg/models/unet.py`
- Lines: 47-69 (UNet configuration)

**Problem**:
- Standard attention is memory-intensive
- Limits batch size on smaller GPUs

**Enhancement steps**:
```python
# Step 1: Check if using diffusers >= 0.24.0 (already in requirements.txt)
# Step 2: In unet.py, modify create_optimized_unet_config():

def create_optimized_unet_config(config: Dict[str, Any]) -> Dict[str, Any]:
    optimized_config = {
        # ... existing config ...
        
        # Add memory efficient attention
        "use_memory_efficient_attention": True,  # Add this line
        
        # Optional: Use scaled dot-product attention (SDPA)
        "attention_type": "sdpa",  # PyTorch 2.0+ native attention
    }
    
# Step 3: For custom UNet (if not using diffusers):
# In custom_unet.py, replace attention implementation:
class EfficientAttention(nn.Module):
    def forward(self, q, k, v):
        # Use PyTorch's native scaled_dot_product_attention
        return F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=0.0 if not self.training else self.dropout,
            is_causal=False
        )
```

**Expected improvement**: 40% memory reduction, allowing larger batch sizes

---

## 🟡 Medium Priority

### 4. Optimize DDIM Sampler with Vectorized Operations
**Why needed**: DDIM sampler has redundant operations and unnecessary tensor copies.

**Current code location**:
- File: `pkl_dg/models/sampler.py`
- Method: `_ddim_step()` (lines 238-271)

**Problem**:
- Multiple tensor reshaping operations
- Unnecessary clamps and checks in inner loop

**Enhancement steps**:
```python
# Step 1: Pre-allocate tensors outside the loop
# In sample() method (line 88):
def sample(self, y, shape, device=None, verbose=True):
    # Pre-allocate reusable tensors
    self.noise_buffer = torch.empty(shape, device=device)
    self.alpha_buffer = torch.empty((shape[0], 1, 1, 1), device=device)
    
# Step 2: Vectorize alpha computations
# In _ddim_step (line 245-250):
# Current:
alpha_cur = self.model.alphas_cumprod[t_cur]
alpha_next = self.model.alphas_cumprod[t_next] if t_next > 0 else torch.tensor(1.0)

# Enhanced: Pre-compute all alphas
def _precompute_ddim_coefficients(self):
    # At initialization, compute all coefficients
    self.ddim_alphas = {}
    self.ddim_sigmas = {}
    for i, t in enumerate(self.ddim_timesteps):
        t_cur = t.item()
        t_next = self.ddim_timesteps[i + 1].item() if i < len(self.ddim_timesteps) - 1 else 0
        # Store precomputed values
        self.ddim_alphas[(t_cur, t_next)] = (alpha_cur, alpha_next)
        self.ddim_sigmas[(t_cur, t_next)] = sigma_t
```

**Expected improvement**: 10-15% faster sampling

---

### 5. Implement Gradient Checkpointing for Large Models
**Why needed**: Reduces memory usage during training by recomputing activations instead of storing them.

**Current code location**:
- File: `pkl_dg/models/unet.py`
- Property: `gradient_checkpointing` (line 99)

**Problem**:
- Currently defined but not implemented
- Large models run out of memory on smaller GPUs

**Enhancement steps**:
```python
# Step 1: In unet.py DenoisingUNet class:
def __init__(self, config):
    # ... existing code ...
    self.gradient_checkpointing = config.get("gradient_checkpointing", False)
    
    # Step 2: Apply checkpointing to the model
    if self.gradient_checkpointing and self.use_diffusers:
        self.model.enable_gradient_checkpointing()
    
# Step 3: For custom UNet implementation:
# In custom_unet.py:
from torch.utils.checkpoint import checkpoint

class CheckpointedBlock(nn.Module):
    def forward(self, x, t):
        if self.training and self.use_checkpointing:
            # Recompute activations during backward pass
            return checkpoint(self._forward_impl, x, t, use_reentrant=False)
        return self._forward_impl(x, t)
    
    def _forward_impl(self, x, t):
        # Actual forward logic here
        pass
```

**Expected improvement**: 30-40% memory reduction during training

---

### 6. Use PyTorch Native Mixed Precision (AMP) Properly
**Why needed**: Current mixed precision implementation is manual. PyTorch's native AMP is more robust.

**Current code location**:
- File: `pkl_dg/models/diffusion.py`
- Lines: 63-70 (mixed precision setup)

**Problem**:
- Manual mixed precision handling is error-prone
- Not using GradScaler optimally

**Enhancement steps**:
```python
# Step 1: Simplify mixed precision in diffusion.py
# Replace manual handling with context managers:

from torch.cuda.amp import autocast, GradScaler

class DDPMTrainer(LightningModuleBase):
    def __init__(self, model, config):
        # ... existing code ...
        self.automatic_optimization = False  # Take control of optimization
        self.scaler = GradScaler(enabled=self.mixed_precision)
    
    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        
        # Clear gradients
        opt.zero_grad()
        
        # Forward pass with autocast
        with autocast(enabled=self.mixed_precision, dtype=self.autocast_dtype):
            # Your forward pass
            loss = self.compute_loss(batch)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step with unscaling
        self.scaler.step(opt)
        self.scaler.update()
        
        return loss
```

**Expected improvement**: More stable training, 20% faster on compatible GPUs

---

## 🟢 Low Priority (Nice to Have)

### 7. Implement Dynamic Batching Based on GPU Memory
**Why needed**: Automatically adjust batch size to maximize GPU utilization without OOM errors.

**Current code location**:
- File: `pkl_dg/utils/adaptive_batch.py`
- Already partially implemented!

**Enhancement steps**:
```python
# In your training script, use the existing AdaptiveBatchSizer:
from pkl_dg.utils.adaptive_batch import AdaptiveBatchSizer

# Before training:
batch_sizer = AdaptiveBatchSizer(safety_factor=0.9)
optimal_batch_size = batch_sizer.find_optimal_batch_size(
    model=model,
    input_shape=(2, 256, 256),  # Your input shape
    device="cuda"
)
print(f"Using batch size: {optimal_batch_size}")
```

---

### 8. Add Profiling Tools
**Why needed**: Identify bottlenecks scientifically rather than guessing.

**Enhancement steps**:
```python
# Create a new file: pkl_dg/utils/profiler.py
import torch.profiler as profiler
import time

class PerformanceProfiler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = {}
    
    def profile_forward_pass(self, model, input_batch):
        if not self.enabled:
            return model(input_batch)
        
        with profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True
        ) as prof:
            output = model(input_batch)
        
        # Save profiling results
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("profile_trace.json")
        
        return output

# Usage in training:
profiler = PerformanceProfiler(enabled=True)
output = profiler.profile_forward_pass(model, batch)
```

---

## Testing Each Enhancement

After implementing each enhancement:

1. **Run unit tests**: 
   ```bash
   pytest tests/ -v
   ```

2. **Benchmark performance**:
   ```python
   # Create benchmark.py
   import time
   import torch
   from pkl_dg.models import DenoisingUNet
   
   # Time before enhancement
   start = time.time()
   # Run inference 100 times
   for _ in range(100):
       output = model(input)
   baseline_time = time.time() - start
   
   # Time after enhancement
   # ... same code with enhanced model ...
   
   print(f"Speedup: {baseline_time / enhanced_time:.2f}x")
   ```

3. **Check memory usage**:
   ```python
   print(f"Memory before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   # Run model
   print(f"Memory after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
   ```

---

## Implementation Order

1. Start with **High Priority** items (1-3) - easiest to implement, biggest impact
2. Test thoroughly after each change
3. Move to **Medium Priority** items (4-6) - require more careful implementation
4. Consider **Low Priority** items only after others are complete

## Questions to Ask Before Each Enhancement

1. Will this break existing functionality?
2. Have I created a backup/git branch?
3. Do I have benchmarks to measure improvement?
4. Have I tested with both single samples and batches?
5. Does this work with both float32 and float16?

---

## Tracking Progress

- [ ] FFT Convolution Caching
- [ ] Batch-Aware Guidance
- [ ] Memory-Efficient Attention
- [ ] DDIM Sampler Optimization
- [ ] Gradient Checkpointing
- [ ] Native AMP Integration
- [ ] Dynamic Batching
- [ ] Profiling Tools

Mark items as complete after implementation and testing.
