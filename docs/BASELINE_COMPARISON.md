# Full FOV Baseline Comparison: Richardson-Lucy vs RCAN

This document describes the comprehensive baseline comparison system for full field-of-view reconstructions between Richardson-Lucy deconvolution and RCAN super-resolution methods.

## Overview

The baseline comparison system provides:

- **Full FOV Reconstruction**: Processes entire microscopy images using patch-based approach with seamless stitching
- **Multiple Baseline Methods**: Richardson-Lucy deconvolution and RCAN super-resolution
- **Comprehensive Evaluation**: Multiple metrics including PSNR, SSIM, FRC, SAR, robustness tests
- **Visual Comparisons**: Side-by-side comparison images
- **Statistical Analysis**: Summary statistics and significance testing
- **Flexible Configuration**: Hydra-based configuration system

## Quick Start

### Basic Usage

```bash
# Run with default configuration
python scripts/run_baseline_comparison.py

# Override specific parameters
python scripts/run_baseline_comparison.py \
    data.input_dir=data/test/wf \
    data.gt_dir=data/test/2p \
    processing.max_images=10
```

### Command Line Interface

```bash
# Direct script usage (without Hydra)
python scripts/baseline_comparison_full_fov.py \
    --input-dir data/test/wf \
    --gt-dir data/test/2p \
    --psf-path data/psf/psf.tif \
    --output-dir outputs/baseline_comparison \
    --rcan-checkpoint checkpoints/rcan_model.pt
```

## Configuration

### Main Configuration File

The system uses Hydra configuration files (`configs/baseline_comparison.yaml`):

```yaml
# Data paths
data:
  input_dir: "data/test/wf"           # Wide-field input images
  gt_dir: "data/test/2p"              # Ground truth 2P images
  psf_path: "data/psf/psf.tif"        # PSF file
  mask_dir: ""                        # Optional mask directory

# Model paths
model:
  rcan_checkpoint: "checkpoints/rcan_model.pt"  # RCAN checkpoint

# Processing parameters
processing:
  device: "cuda"                      # Computation device
  patch_size: 256                     # Patch size for processing
  stride: 128                         # Stride between patches
  max_images: null                    # Maximum images to process

# Richardson-Lucy parameters
richardson_lucy:
  iterations: 30                       # Number of RL iterations
  clip: true                          # Clip negative values

# RCAN parameters
rcan:
  enabled: true                       # Enable RCAN comparison
  checkpoint_required: false          # Whether RCAN checkpoint is required
```

### Key Parameters

#### Processing Parameters
- `patch_size`: Size of patches for processing (default: 256)
- `stride`: Stride between patches (default: 128)
- `max_images`: Maximum number of images to process (null = all)

#### Richardson-Lucy Parameters
- `iterations`: Number of Richardson-Lucy iterations (default: 30)
- `clip`: Whether to clip negative values (default: true)

#### RCAN Parameters
- `enabled`: Enable RCAN comparison (default: true)
- `checkpoint_required`: Whether RCAN checkpoint is required (default: false)

## Methods

### Richardson-Lucy Deconvolution

Classical iterative deconvolution algorithm based on Poisson noise model:

```python
def richardson_lucy_restore(image, psf, num_iter=30, clip=True):
    """
    Richardson-Lucy deconvolution.
    
    Args:
        image: Input wide-field image
        psf: Point spread function
        num_iter: Number of iterations
        clip: Clip negative values
    """
```

**Advantages:**
- Well-established method with theoretical foundation
- Handles Poisson noise naturally
- No training data required
- Computationally efficient

**Limitations:**
- Requires accurate PSF knowledge
- Can produce ringing artifacts
- Limited resolution improvement

### RCAN Super-Resolution

Residual Channel Attention Network for image super-resolution:

```python
class RCANWrapper:
    """Wrapper for RCAN super-resolution model."""
    
    def __init__(self, checkpoint_path, device="cpu"):
        # Load pre-trained RCAN model
    
    def infer(self, image):
        # Apply RCAN super-resolution
```

**Advantages:**
- State-of-the-art super-resolution performance
- Learns complex image priors
- Can handle various degradation types
- Good at preserving fine details

**Limitations:**
- Requires training data
- Computationally expensive
- May hallucinate details
- Model-dependent performance

## Evaluation Metrics

### Image Quality Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures reconstruction fidelity
   - Higher values indicate better quality
   - Range: 0 to ∞ dB

2. **SSIM (Structural Similarity Index)**
   - Measures structural similarity
   - Accounts for luminance, contrast, and structure
   - Range: 0 to 1

3. **FRC (Fourier Ring Correlation)**
   - Measures effective resolution
   - Based on frequency domain analysis
   - Lower values indicate better resolution

4. **SAR (Signal-to-Artifact Ratio)**
   - Measures artifact suppression
   - Compares reconstruction to input
   - Higher values indicate fewer artifacts

### Robustness Metrics

1. **PSF Mismatch Robustness**
   - Tests sensitivity to PSF errors
   - Simulates PSF estimation errors
   - Higher values indicate better robustness

2. **Alignment Error Robustness**
   - Tests sensitivity to registration errors
   - Simulates misalignment between WF and 2P
   - Higher values indicate better robustness

3. **Hallucination Score**
   - Detects artificial details
   - Uses adversarial testing
   - Lower values indicate fewer hallucinations

### Downstream Task Metrics

1. **Cellpose F1 Score**
   - Segmentation performance
   - Uses Cellpose for cell segmentation
   - Range: 0 to 1

2. **Hausdorff Distance**
   - Shape preservation metric
   - Measures boundary accuracy
   - Lower values indicate better preservation

## Output Structure

```
outputs/baseline_comparison_full_fov/
├── config.yaml                    # Configuration used
├── detailed_results.npz           # Detailed results
├── summary_statistics.npz        # Summary statistics
├── image001_comparison.png       # Individual comparisons
├── image002_comparison.png
├── ...
└── plots/                        # Summary plots
    ├── psnr_comparison.png
    ├── ssim_comparison.png
    ├── frc_comparison.png
    ├── sar_comparison.png
    └── correlation_heatmap.png
```

## Usage Examples

### Example 1: Basic Comparison

```python
from scripts.run_baseline_comparison import HydraBaselineComparison
from omegaconf import OmegaConf

# Create configuration
cfg = OmegaConf.create({
    "data": {
        "input_dir": "data/test/wf",
        "gt_dir": "data/test/2p",
        "psf_path": "data/psf/psf.tif"
    },
    "processing": {
        "device": "cuda",
        "patch_size": 256,
        "stride": 128
    },
    "richardson_lucy": {
        "iterations": 30
    },
    "rcan": {
        "enabled": False  # Skip RCAN for basic example
    }
})

# Initialize and run comparison
comparison = HydraBaselineComparison(cfg)
```

### Example 2: Custom Parameters

```bash
# High-quality processing
python scripts/run_baseline_comparison.py \
    processing.patch_size=512 \
    processing.stride=256 \
    richardson_lucy.iterations=100

# Fast processing
python scripts/run_baseline_comparison.py \
    processing.patch_size=128 \
    processing.stride=64 \
    richardson_lucy.iterations=10

# Robust evaluation
python scripts/run_baseline_comparison.py \
    evaluation.metrics="psnr,ssim,frc,sar,psf_mismatch,alignment_error" \
    physics.background_level=10.0
```

### Example 3: Batch Processing

```python
# Process multiple datasets
datasets = [
    {"name": "dataset1", "input_dir": "data/dataset1/wf", "gt_dir": "data/dataset1/2p"},
    {"name": "dataset2", "input_dir": "data/dataset2/wf", "gt_dir": "data/dataset2/2p"},
]

for dataset in datasets:
    # Run comparison for each dataset
    python scripts/run_baseline_comparison.py \
        data.input_dir={dataset["input_dir"]} \
        data.gt_dir={dataset["gt_dir"]} \
        output.directory=outputs/{dataset["name"]}_comparison
```

## Performance Considerations

### Computational Requirements

- **Memory**: ~4-8 GB GPU memory for 512×512 images
- **Time**: ~2-5 minutes per image (depending on parameters)
- **Storage**: ~10-50 MB per comparison (depending on image size)

### Optimization Tips

1. **Patch Size**: Larger patches reduce stitching artifacts but increase memory usage
2. **Stride**: Smaller stride improves quality but increases computation time
3. **RL Iterations**: More iterations improve quality but increase computation time
4. **Device**: Use GPU for faster processing when available

### Scaling to Large Datasets

```bash
# Process in batches
python scripts/run_baseline_comparison.py \
    processing.max_images=100 \
    output.directory=outputs/batch1

python scripts/run_baseline_comparison.py \
    processing.max_images=100 \
    data.input_dir=data/batch2/wf \
    data.gt_dir=data/batch2/2p \
    output.directory=outputs/batch2
```

## Troubleshooting

### Common Issues

1. **RCAN Model Loading Error**
   ```
   Error: RCAN dependencies not available
   ```
   **Solution**: Install RCAN implementation or disable RCAN comparison

2. **PSF Format Error**
   ```
   Error: Unsupported PSF format
   ```
   **Solution**: Use .tif or .npy format for PSF files

3. **Memory Error**
   ```
   Error: CUDA out of memory
   ```
   **Solution**: Reduce patch_size or use CPU processing

4. **No GT Files Found**
   ```
   Warning: No GT file found for image.tif
   ```
   **Solution**: Ensure GT files have matching names in GT directory

### Performance Issues

1. **Slow Processing**
   - Use GPU if available
   - Reduce patch_size and stride
   - Reduce RL iterations

2. **Poor Quality Results**
   - Increase RL iterations
   - Use larger patch_size
   - Check PSF quality

3. **Stitching Artifacts**
   - Reduce stride
   - Increase feather_size
   - Adjust smoothing_sigma

## Advanced Usage

### Custom Metrics

```python
def custom_metric(pred, target):
    """Define custom evaluation metric."""
    # Your metric implementation
    return metric_value

# Add to evaluation pipeline
comparison.compute_metrics = lambda pred, target, wf: {
    **comparison.compute_metrics(pred, target, wf),
    "custom_metric": custom_metric(pred, target)
}
```

### Custom Visualization

```python
def custom_visualization(wf, rl, rcan, gt):
    """Create custom comparison visualization."""
    # Your visualization code
    pass

# Use in comparison
comparison.create_comparison_visualization = custom_visualization
```

### Integration with Other Tools

```python
# Integration with Weights & Biases
import wandb

def log_results(results):
    """Log results to W&B."""
    wandb.log(results)

# Integration with other evaluation tools
from pkl_dg.evaluation import Metrics
metrics = Metrics.compute_all(pred, target)
```

## References

1. Richardson, W. H. "Bayesian-based iterative method of image restoration." JOSA 62.1 (1972): 55-59.
2. Lucy, L. B. "An iterative technique for the rectification of observed distributions." AJ 79.6 (1974): 745-754.
3. Zhang, Yulun, et al. "Image super-resolution using very deep residual channel attention networks." ECCV 2018.
4. PKL-DG Project: Microscopy Denoising Diffusion with Poisson-aware Physical Guidance.

## Support

For questions and issues:
1. Check the troubleshooting section above
2. Review the example scripts in `examples/`
3. Examine the configuration files in `configs/`
4. Refer to the main project documentation
