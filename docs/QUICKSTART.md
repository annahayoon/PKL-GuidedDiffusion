# Quick Start Guide for Junior Engineers

## Overview
This guide will help you implement the PKL-Guided Diffusion system for microscopy image restoration. We'll build a system that transforms blurry widefield (WF) microscopy images into sharp two-photon (2P) quality images using a novel physics-aware guidance mechanism.

## What You're Building

### The Problem
- **Input**: Blurry, noisy widefield microscopy images (like looking through frosted glass)
- **Output**: Clear, high-resolution images (like having perfect vision)
- **Challenge**: The noise in microscopy follows Poisson statistics (not Gaussian), so standard methods fail

### The Solution
We use a diffusion model (like DALL-E but for microscopy) with special "physics guidance" that understands how photons behave in microscopes.

## Prerequisites

### Required Knowledge
- Python programming (intermediate level)
- Basic understanding of neural networks (CNNs)
- Familiarity with PyTorch basics
- Command line usage

### Required Software
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU (slower)
- 16GB+ RAM
- 100GB free disk space

## Step-by-Step Implementation

### Day 1: Environment Setup

#### 1. Create Project Directory
```bash
# Create main project folder
mkdir PKL-DiffusionDenoising
cd PKL-DiffusionDenoising

# Create folder structure
mkdir -p {pkl_dg,scripts,configs,tests,assets,docs,notebooks}
mkdir -p pkl_dg/{data,physics,models,guidance,baselines,evaluation,utils}
mkdir -p assets/{psf,examples}
mkdir -p data/{raw,synthesized,train,val,test}
```

#### 2. Setup Python Environment
```bash
# Create virtual environment (isolates dependencies)
python -m venv venv

# Activate it (you'll do this every time you work on the project)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### 3. Install Dependencies
Create `requirements.txt`:
```txt
# Core ML
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0

# Diffusion models
diffusers>=0.21.0
transformers>=4.30.0

# Configuration
hydra-core>=1.3.0
omegaconf>=2.3.0

# Experiment tracking
wandb>=0.15.0

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
scikit-image>=0.21.0
kornia>=0.7.0

# Image I/O
tifffile>=2023.0.0
Pillow>=9.5.0

# Evaluation
cellpose>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
ipywidgets>=8.0.0

# Utilities
tqdm>=4.65.0
einops>=0.6.0

# Development
pytest>=7.3.0
black>=23.0.0
isort>=5.12.0
```

Install everything:
```bash
pip install -r requirements.txt
```

### Day 2: Understanding the Physics

#### Key Concepts Explained

**1. Point Spread Function (PSF)**
- Think of it as the "blur kernel" of the microscope
- Like how a camera lens can blur an image
- We need to model this blur mathematically

**2. Poisson Noise**
- In microscopy, we count individual photons
- Few photons = more noise (like a grainy photo in low light)
- The noise depends on the signal strength (unlike regular Gaussian noise)

**3. Forward Model**
- How we simulate what the microscope sees
- Clean image → Blur with PSF → Add Poisson noise → Noisy measurement

#### Implement PSF Handler
Create `pkl_dg/physics/psf.py`:
```python
import numpy as np
import torch
from pathlib import Path
import tifffile  # For reading microscopy images

class PSF:
    """Handles the Point Spread Function (blur kernel) of the microscope."""
    
    def __init__(self, psf_path=None):
        """
        Initialize PSF from file or create default.
        
        Args:
            psf_path: Path to measured PSF file (optional)
        """
        if psf_path and Path(psf_path).exists():
            # Load real PSF from microscope calibration
            self.psf = tifffile.imread(psf_path)
            print(f"Loaded PSF from {psf_path}")
        else:
            # Create synthetic Gaussian PSF for testing
            self.psf = self._create_gaussian_psf()
            print("Created synthetic Gaussian PSF")
        
        # Normalize so it sums to 1 (preserves image brightness)
        self.psf = self.psf.astype(np.float32)
        self.psf = self.psf / self.psf.sum()
    
    def _create_gaussian_psf(self, size=15, sigma=2.0):
        """
        Create a Gaussian blur kernel.
        
        This simulates the blur of a microscope.
        Larger sigma = more blur.
        """
        # Create coordinate grid
        center = size // 2
        x = np.arange(size) - center
        y = np.arange(size) - center
        xx, yy = np.meshgrid(x, y)
        
        # Gaussian formula: e^(-(x²+y²)/(2σ²))
        psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        return psf
    
    def to_torch(self, device='cpu'):
        """Convert PSF to PyTorch tensor for GPU operations."""
        return torch.from_numpy(self.psf).float().to(device)
    
    def visualize(self):
        """Show what the PSF looks like."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.psf, cmap='hot')
        plt.colorbar(label='Intensity')
        plt.title('Point Spread Function (PSF)')
        plt.xlabel('Pixels')
        plt.ylabel('Pixels')
        plt.show()

# Test it
if __name__ == "__main__":
    psf = PSF()
    psf.visualize()
    print(f"PSF shape: {psf.psf.shape}")
    print(f"PSF sum: {psf.psf.sum():.6f} (should be 1.0)")
```

### Day 3: Forward Model Implementation

#### Understanding Convolution
Convolution is how we apply blur:
- Think of sliding the PSF over each pixel of the image
- At each position, multiply and sum
- FFT makes this fast (O(n log n) instead of O(n²))

Create `pkl_dg/physics/forward_model.py`:
```python
import torch
import torch.nn.functional as F

class ForwardModel:
    """
    Simulates how the microscope captures images.
    
    Clean image → Blur → Add noise → Measurement
    """
    
    def __init__(self, psf_tensor, background=3.0, device='cuda'):
        """
        Args:
            psf_tensor: The PSF as a torch tensor
            background: Dark counts from detector (constant noise floor)
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        self.device = device
        self.background = background
        
        # Prepare PSF for convolution
        # Shape needs to be [1, 1, H, W] for batch processing
        if psf_tensor.ndim == 2:
            psf_tensor = psf_tensor.unsqueeze(0).unsqueeze(0)
        
        self.psf = psf_tensor.to(device)
        
        # Pre-compute FFT of PSF for efficiency
        self.psf_fft = torch.fft.rfft2(self.psf)
    
    def apply_blur(self, image):
        """
        Apply PSF blur using FFT convolution.
        
        Why FFT? It's much faster for large kernels.
        Regular convolution: O(n²k²) where k is kernel size
        FFT convolution: O(n² log n)
        """
        # Pad image to avoid edge artifacts
        pad_h = self.psf.shape[-2] // 2
        pad_w = self.psf.shape[-1] // 2
        image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        # FFT convolution: multiply in frequency domain
        image_fft = torch.fft.rfft2(image_padded)
        blurred_fft = image_fft * self.psf_fft
        blurred = torch.fft.irfft2(blurred_fft, image_padded.shape[-2:])
        
        # Remove padding
        blurred = blurred[..., pad_h:-pad_h, pad_w:-pad_w]
        
        return blurred
    
    def add_poisson_noise(self, image):
        """
        Add Poisson noise (photon shot noise).
        
        Key insight: In Poisson noise, variance = mean
        Bright areas have more noise (but better SNR)
        Dark areas have less noise (but worse SNR)
        """
        # Ensure non-negative (can't have negative photons!)
        image = torch.clamp(image, min=0)
        
        # Sample from Poisson distribution
        noisy = torch.poisson(image)
        
        return noisy
    
    def forward(self, clean_image, add_noise=True):
        """
        Full forward model: simulate microscope measurement.
        
        Args:
            clean_image: The true image we want to recover
            add_noise: Whether to add Poisson noise
        
        Returns:
            Simulated microscope measurement
        """
        # Step 1: Apply PSF blur
        blurred = self.apply_blur(clean_image)
        
        # Step 2: Add background (detector dark counts)
        with_background = blurred + self.background
        
        # Step 3: Add Poisson noise (optional, for training)
        if add_noise:
            noisy = self.add_poisson_noise(with_background)
            return noisy
        else:
            return with_background
    
    def apply_adjoint(self, image):
        """
        Apply adjoint operator (transpose of blur).
        
        This is used in the guidance gradient computation.
        For convolution, adjoint = correlation (flip kernel and convolve).
        """
        # Flip PSF for correlation
        psf_flipped = torch.flip(self.psf, dims=[-2, -1])
        
        # Apply correlation (same as convolution with flipped kernel)
        # We reuse the FFT approach for efficiency
        pad_h = psf_flipped.shape[-2] // 2
        pad_w = psf_flipped.shape[-1] // 2
        image_padded = F.pad(image, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        
        psf_flipped_fft = torch.fft.rfft2(psf_flipped)
        image_fft = torch.fft.rfft2(image_padded)
        result_fft = image_fft * psf_flipped_fft
        result = torch.fft.irfft2(result_fft, image_padded.shape[-2:])
        
        result = result[..., pad_h:-pad_h, pad_w:-pad_w]
        
        return result

# Test the forward model
if __name__ == "__main__":
    from pkl_dg.physics.psf import PSF
    import matplotlib.pyplot as plt
    
    # Create a simple test image
    test_image = torch.zeros(1, 1, 256, 256)
    test_image[0, 0, 100:150, 100:150] = 100  # Bright square
    test_image[0, 0, 50:60, 50:60] = 50       # Dimmer square
    
    # Setup forward model
    psf = PSF()
    forward_model = ForwardModel(psf.to_torch(), device='cpu')
    
    # Simulate measurement
    measurement = forward_model.forward(test_image, add_noise=True)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(test_image[0, 0], cmap='gray')
    axes[0].set_title('Clean Image')
    
    axes[1].imshow(measurement[0, 0].cpu(), cmap='gray')
    axes[1].set_title('Simulated Measurement')
    
    diff = measurement - test_image
    axes[2].imshow(diff[0, 0].cpu(), cmap='RdBu_r')
    axes[2].set_title('Difference (Blur + Noise)')
    
    plt.tight_layout()
    plt.show()
```

### Day 4: Data Synthesis Pipeline

#### Why Synthetic Data?
- Real microscopy paired data is expensive and rare
- We can create training data from regular images
- This teaches the model what "sharp" images look like

Create `pkl_dg/data/synthesis.py`:
```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms

class NormalizationTransform:
    """
    Handles conversion between intensity and model domains.
    - Intensity: [0, max_photons] (e.g., [0, 1000])
    - Model: [-1, 1] (what the neural network expects)
    """
    def __init__(self, max_intensity=1000):
        self.max_intensity = max_intensity

    def to_model_scale(self, x):
        """Intensity -> Model"""
        # Normalize to [0, 1]
        x = x / self.max_intensity
        # Scale to [-1, 1]
        return 2 * x - 1

    def to_intensity_scale(self, x):
        """Model -> Intensity"""
        # Scale from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Scale to intensity range and ensure non-negative
        return torch.clamp(x * self.max_intensity, min=0)


class TrainingDataSynthesizer(Dataset):
    """
    Creates training pairs from regular images.
    
    Takes ImageNet/BioTISR images and creates:
    - Clean "2P-like" targets
    - Degraded "WF-like" inputs
    """
    
    def __init__(self, image_dir, forward_model, image_size=256, transform=None):
        """
        Args:
            image_dir: Folder with source images
            forward_model: Physics model for degradation
            image_size: Size to resize images to
            transform: The normalization transform to apply
        """
        self.image_dir = Path(image_dir)
        self.forward_model = forward_model
        self.image_size = image_size
        self.normalization = transform
        
        # Find all images
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif']:
            self.image_paths.extend(self.image_dir.glob(f"**/{ext}"))
        
        print(f"Found {len(self.image_paths)} images")
        
        # Basic image transforms
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),  # Convert to grayscale
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get one training pair.
        
        Returns:
            (clean_image, degraded_image)
        """
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # Convert to tensor and resize
        image_intensity = self.base_transform(image)
        
        # Scale to realistic photon counts (10-1000 photons)
        # This matches real microscopy intensity ranges
        image_intensity = image_intensity * 500 + 10
        
        # Create degraded version (WF-like)
        with torch.no_grad():
            # Add batch dimension
            image_batch = image_intensity.unsqueeze(0)
            
            # Apply forward model (blur + noise)
            degraded_intensity = self.forward_model.forward(image_batch, add_noise=True)
            
            # Remove batch dimension
            degraded_intensity = degraded_intensity.squeeze(0)
        
        # The clean image is our target, degraded is our input.
        # Both must be normalized for the model.
        clean_model = self.normalization.to_model_scale(image_intensity)
        degraded_model = self.normalization.to_model_scale(degraded_intensity)
        
        # Return the normalized clean image for training, and the
        # un-normalized degraded image for guidance reference.
        return clean_model, degraded_intensity
    
    @staticmethod
    def create_data_loaders(train_dir, val_dir, forward_model, batch_size=32, transform=None):
        """
        Helper to create train and validation loaders.
        """
        train_dataset = TrainingDataSynthesizer(train_dir, forward_model, transform=transform)
        val_dataset = TrainingDataSynthesizer(val_dir, forward_model, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True  # Faster GPU transfer
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader

# Test data synthesis
if __name__ == "__main__":
    from pkl_dg.physics.psf import PSF
    from pkl_dg.physics.forward_model import ForwardModel
    import matplotlib.pyplot as plt
    
    # Setup
    psf = PSF()
    forward_model = ForwardModel(psf.to_torch(), device='cpu')
    transform = NormalizationTransform()
    
    # Create dataset (you'll need some images in data/raw/)
    dataset = TrainingDataSynthesizer('data/raw', forward_model, transform=transform)
    
    if len(dataset) > 0:
        # Get one sample
        clean_model, degraded_intensity = dataset[0]
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # We convert back to intensity for visualization
        axes[0].imshow(transform.to_intensity_scale(clean_model)[0], cmap='gray')
        axes[0].set_title('Clean Target (Intensity)')
        
        axes[1].imshow(degraded_intensity[0], cmap='gray')
        axes[1].set_title('Degraded Input (Intensity)')
        
        plt.show()
```

### Day 5: Diffusion Model Setup

#### Understanding Diffusion Models
Think of it like this:
1. **Forward process**: Gradually add noise until image becomes pure noise
2. **Reverse process**: Learn to remove noise step by step
3. **Key insight**: If we can remove noise, we can generate clean images!

Create `pkl_dg/models/unet.py`:
```python
from diffusers import UNet2DModel
import torch.nn as nn

class DenoisingUNet(nn.Module):
    """
    U-Net architecture for noise prediction.
    
    We use Hugging Face's pre-built UNet to save time.
    U-Net is perfect for image tasks because it has:
    - Encoder: Compresses image to features
    - Decoder: Reconstructs from features
    - Skip connections: Preserves fine details
    """
    
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Use Hugging Face's UNet implementation
        self.unet = UNet2DModel(
            sample_size=256,  # Image size
            in_channels=in_channels,  # 1 for grayscale
            out_channels=out_channels,  # Predict noise (same size as input)
            layers_per_block=2,
            block_out_channels=(128, 256, 512, 512),  # Channel progression
            down_block_types=(
                "DownBlock2D",  # Regular downsampling
                "DownBlock2D",
                "AttnDownBlock2D",  # With attention (captures long-range)
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",  # Attention in decoder too
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=8,
            norm_num_groups=32,
            dropout=0.0,
        )
    
    def forward(self, noisy_image, timestep):
        """
        Predict the noise in the image.
        
        Args:
            noisy_image: Image with noise at timestep t
            timestep: How much noise was added (0 = no noise, 999 = pure noise)
        
        Returns:
            Predicted noise to remove
        """
        return self.unet(noisy_image, timestep).sample
```

### Day 6: The Key Innovation - PKL Guidance

#### Why PKL Guidance?
Standard methods assume Gaussian noise (same everywhere). But in microscopy:
- Bright areas have more noise (absolute) but better signal-to-noise ratio
- Dark areas have less noise (absolute) but worse signal-to-noise ratio
- PKL guidance adapts to this automatically!

Create `pkl_dg/guidance/pkl.py`:
```python
import torch

class PKLGuidance:
    """
    Poisson-Kullback-Leibler guidance.
    
    This is our key innovation! It respects Poisson statistics.
    """
    
    def __init__(self, epsilon=1e-6):
        """
        Args:
            epsilon: Small number to avoid division by zero
        """
        self.epsilon = epsilon
    
    def compute_gradient(self, prediction, measurement, forward_model):
        """
        Compute the PKL gradient for guidance.
        
        The magic formula: ∇ = A^T(1 - y/(A(x) + B + ε))
        
        This automatically:
        - Applies strong corrections in dark areas (low photon count)
        - Applies weak corrections in bright areas (high photon count)
        
        Args:
            prediction: Current prediction of clean image
            measurement: What we actually measured
            forward_model: Physics model
        
        Returns:
            Gradient to improve the prediction
        """
        # Apply forward model to prediction
        predicted_measurement = forward_model.apply_blur(prediction)
        predicted_measurement = predicted_measurement + forward_model.background
        
        # Compute the ratio y/(A(x) + B)
        # This tells us how well our prediction matches the measurement
        ratio = measurement / (predicted_measurement + self.epsilon)
        
        # Compute residual: 1 - ratio
        # If ratio = 1, our prediction is perfect (residual = 0)
        # If ratio < 1, we overestimated (residual > 0)
        # If ratio > 1, we underestimated (residual < 0)
        residual = 1.0 - ratio
        
        # Apply adjoint to get gradient in image space
        gradient = forward_model.apply_adjoint(residual)
        
        return gradient
    
    def apply_correction(self, prediction, gradient, step_size):
        """
        Apply the gradient to improve prediction.
        
        Args:
            prediction: Current prediction
            gradient: PKL gradient
            step_size: How much to trust the gradient
        
        Returns:
            Corrected prediction
        """
        # Gradient descent step
        corrected = prediction - step_size * gradient
        
        # Ensure non-negative (no negative photons!)
        corrected = torch.clamp(corrected, min=0)
        
        return corrected

# For comparison: Standard L2 guidance
class L2Guidance:
    """
    Standard L2 (Mean Squared Error) guidance.
    
    This is what most people use, but it's wrong for Poisson noise!
    """
    
    def compute_gradient(self, prediction, measurement, forward_model):
        """
        L2 gradient: ∇ = A^T(A(x) - y)
        
        This treats all pixels equally, ignoring Poisson statistics.
        """
        # Apply forward model
        predicted_measurement = forward_model.apply_blur(prediction)
        predicted_measurement = predicted_measurement + forward_model.background
        
        # Simple difference
        residual = predicted_measurement - measurement
        
        # Apply adjoint
        gradient = forward_model.apply_adjoint(residual)
        
        return gradient
    
    def apply_correction(self, prediction, gradient, step_size):
        """Same as PKL."""
        corrected = prediction - step_size * gradient
        corrected = torch.clamp(corrected, min=0)
        return corrected
```

### Day 7: DDIM Sampler Implementation

#### What is DDIM?
- DDIM = Denoising Diffusion Implicit Models
- Like DDPM but deterministic and faster
- We inject our physics guidance at each denoising step

Create `pkl_dg/models/sampler.py`:
```python
import torch
from tqdm import tqdm

class GuidedDDIMSampler:
    """
    DDIM sampler with physics-based guidance.
    
    This is where everything comes together!
    """
    
    def __init__(self, model, forward_model, guidance, transform, num_steps=100):
        """
        Args:
            model: Trained diffusion model
            forward_model: Physics model
            guidance: Guidance strategy (PKL or L2)
            transform: Normalization transform object
            num_steps: Number of denoising steps
        """
        self.model = model
        self.forward_model = forward_model
        self.guidance = guidance
        self.transform = transform
        self.num_steps = num_steps
        
        # Setup timestep schedule (which steps to use)
        # We skip steps for faster inference
        total_timesteps = 1000
        step_size = total_timesteps // num_steps
        self.timesteps = torch.arange(0, total_timesteps, step_size).flip(0)
    
    @torch.no_grad()
    def sample(self, measurement, device='cuda'):
        """
        Generate clean image from noisy measurement.
        
        Args:
            measurement: The WF microscopy image
            device: Where to run computation
        
        Returns:
            Reconstructed 2P-quality image
        """
        # Move everything to device
        measurement = measurement.to(device)
        self.model = self.model.to(device)
        
        # Start from pure noise
        shape = measurement.shape
        x_t = torch.randn(shape, device=device)
        
        # Denoising loop
        for i, t in enumerate(tqdm(self.timesteps, desc="Denoising")):
            # Step 1: Predict clean image from noisy version
            x_0_pred = self.predict_clean(x_t, t)
            
            # Step 2: Apply physics guidance to improve prediction
            if t > 0:  # Don't guide at the last step
                x_0_pred = self.apply_guidance(x_0_pred, measurement, t)
            
            # Step 3: Compute next noisy image (less noise than current)
            if i < len(self.timesteps) - 1:
                t_next = self.timesteps[i + 1]
                x_t = self.ddim_step(x_t, x_0_pred, t, t_next)
            else:
                # Last step: return clean prediction
                return x_0_pred
        
        return x_0_pred
    
    def predict_clean(self, x_t, t):
        """
        Predict clean image from noisy image.
        
        Uses the formula:
        x_0 = (x_t - √(1-α_t) * ε_θ(x_t, t)) / √α_t
        """
        # Get noise schedule parameters
        alpha_t = self.get_alpha(t)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict noise
        t_tensor = torch.tensor([t], device=x_t.device)
        noise_pred = self.model(x_t, t_tensor)
        
        # Compute clean image
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        return x_0_pred
    
    def apply_guidance(self, x_0_pred, measurement, t):
        """
        Apply physics guidance to improve prediction.
        
        This is where PKL vs L2 makes a difference!
        """
        # Convert from model scale [-1, 1] to intensity [0, max]
        x_0_intensity = self.transform.to_intensity_scale(x_0_pred)
        
        # Compute guidance gradient
        gradient = self.guidance.compute_gradient(
            x_0_intensity, measurement, self.forward_model
        )
        
        # Adaptive step size (smaller gradients → larger steps)
        step_size = self.get_step_size(gradient, t)
        
        # Apply correction
        x_0_corrected = self.guidance.apply_correction(
            x_0_intensity, gradient, step_size
        )
        
        # Convert back to model scale
        x_0_corrected_model = self.transform.to_model_scale(x_0_corrected)
        
        return x_0_corrected_model
    
    def get_step_size(self, gradient, t):
        """
        Adaptive step size schedule.
        
        Key ideas:
        - Normalize by gradient magnitude (stability)
        - Start weak, get stronger (warm-up)
        """
        base_step = 0.1
        
        # Normalize by gradient magnitude
        grad_norm = torch.norm(gradient) + 1e-6
        normalized_step = base_step / grad_norm
        
        # Warm-up: weak guidance early, strong guidance later
        warmup = min(t / 100, 1.0)
        
        return normalized_step * warmup
    
    def ddim_step(self, x_t, x_0_pred, t, t_next):
        """
        DDIM denoising step.
        
        Computes x_{t-1} from x_t and predicted x_0.
        """
        # Get noise schedule parameters
        alpha_t = self.get_alpha(t)
        alpha_next = self.get_alpha(t_next)
        
        # DDIM formula (deterministic)
        sqrt_alpha_next = torch.sqrt(alpha_next)
        pred_noise = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)
        x_next = sqrt_alpha_next * x_0_pred + torch.sqrt(1 - alpha_next) * pred_noise
        
        return x_next
    
    def get_alpha(self, t):
        """Get noise schedule parameter."""
        # Cosine schedule (better than linear)
        # More noise preserved early, faster denoising later
        s = 0.008
        cos_val = torch.cos(((t / 1000) + s) / (1 + s) * torch.pi * 0.5)
        return cos_val ** 2
    
```

### Day 8: Training Script

Create `scripts/train.py`:
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb  # For experiment tracking

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.data.synthesis import TrainingDataSynthesizer
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel

class DiffusionTrainer:
    """
    Trains the diffusion model.
    
    Key idea: Learn to predict noise at various levels.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Initialize Weights & Biases for tracking
        wandb.init(project="pkl-diffusion", name="training")
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (clean_images, _) in enumerate(tqdm(dataloader, desc="Training")):
            # clean_images are already normalized to [-1, 1] by the dataset
            clean_images = clean_images.to(self.device)
            batch_size = clean_images.shape[0]
            
            # Sample random timesteps for each image
            timesteps = torch.randint(0, 1000, (batch_size,), device=self.device)
            
            # Add noise according to timestep
            noise = torch.randn_like(clean_images)
            noisy_images = self.add_noise(clean_images, noise, timesteps)
            
            # Predict noise
            predicted_noise = self.model(noisy_images, timesteps)
            
            # Compute loss
            loss = self.criterion(predicted_noise, noise)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log to W&B
            if batch_idx % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
        
        return total_loss / len(dataloader)
    
    def add_noise(self, clean, noise, timesteps):
        """
        Add noise according to diffusion schedule.
        
        x_t = √(α_t) * x_0 + √(1 - α_t) * ε
        """
        # Get noise schedule parameters for each image
        alpha_t = self.get_alpha_batch(timesteps)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Add noise
        noisy = sqrt_alpha_t * clean + sqrt_one_minus_alpha_t * noise
        
        return noisy
    
    def get_alpha_batch(self, timesteps):
        """Get alpha for batch of timesteps."""
        # Cosine schedule
        s = 0.008
        cos_vals = torch.cos(((timesteps / 1000) + s) / (1 + s) * torch.pi * 0.5)
        alphas = cos_vals ** 2
        
        # Reshape for broadcasting
        return alphas.view(-1, 1, 1, 1)
    
    def save_checkpoint(self, epoch, path):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

def main():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = DenoisingUNet()
    
    # Create data
    psf = PSF()
    forward_model = ForwardModel(psf.to_torch(), device='cpu')
    transform = NormalizationTransform()
    
    train_loader, val_loader = TrainingDataSynthesizer.create_data_loaders(
        train_dir='data/train',
        val_dir='data/val',
        forward_model=forward_model,
        batch_size=16,
        transform=transform
    )
    
    # Create trainer
    trainer = DiffusionTrainer(model, device)
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Training loss: {train_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(epoch + 1, f"checkpoints/model_epoch_{epoch+1}.pt")
    
    # Save final model
    trainer.save_checkpoint(num_epochs, "checkpoints/final_model.pt")
    wandb.finish()

if __name__ == "__main__":
    main()
```

### Day 9: Inference Script

Create `scripts/inference.py`:
```python
import torch
from pathlib import Path
import tifffile
import numpy as np
from tqdm import tqdm

from pkl_dg.models.unet import DenoisingUNet
from pkl_dg.models.sampler import GuidedDDIMSampler
from pkl_dg.physics.psf import PSF
from pkl_dg.physics.forward_model import ForwardModel
from pkl_dg.guidance.pkl import PKLGuidance, L2Guidance

def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    model = DenoisingUNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device)

def process_image(image_path, sampler, device='cuda'):
    """Process a single microscopy image."""
    # Load image
    image = tifffile.imread(image_path)
    
    # Convert to torch tensor
    image = torch.from_numpy(image).float()
    
    # Add batch and channel dimensions if needed
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    # Move to device
    image = image.to(device)
    
    # Run reconstruction
    reconstruction = sampler.sample(image, device)
    
    # Convert back to numpy
    reconstruction = reconstruction.squeeze().cpu().numpy()
    
    return reconstruction

def main():
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = 'checkpoints/final_model.pt'
    input_dir = Path('data/test/widefield')
    output_dir = Path('outputs/reconstructions')
    guidance_type = 'pkl'  # 'pkl' or 'l2'
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    
    # Setup physics
    psf = PSF('assets/psf/measured_psf.tif')  # Use measured PSF if available
    forward_model = ForwardModel(psf.to_torch(device), device=device)
    
    # Choose guidance
    if guidance_type == 'pkl':
        print("Using PKL guidance (recommended)")
        guidance = PKLGuidance()
    else:
        print("Using L2 guidance (baseline)")
        guidance = L2Guidance()
    
    # Create sampler
    transform = NormalizationTransform()
    sampler = GuidedDDIMSampler(
        model=model,
        forward_model=forward_model,
        guidance=guidance,
        transform=transform,
        num_steps=100
    )
    
    # Process all images
    image_paths = list(input_dir.glob('*.tif'))
    print(f"Processing {len(image_paths)} images...")
    
    for img_path in tqdm(image_paths):
        # Process image
        reconstruction = process_image(img_path, sampler, device)
        
        # Save result
        output_path = output_dir / f"{img_path.stem}_reconstructed.tif"
        tifffile.imwrite(output_path, reconstruction.astype(np.float32))
    
    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
```

### Day 10: Evaluation and Visualization

Create `scripts/evaluate.py`:
```python
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_reconstruction(pred_path, target_path):
    """
    Evaluate reconstruction quality.
    
    Returns:
        Dictionary of metrics
    """
    # Load images
    pred = tifffile.imread(pred_path)
    target = tifffile.imread(target_path)
    
    # Ensure same shape
    assert pred.shape == target.shape, "Images must have same shape"
    
    # Normalize to [0, 1] for metrics
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    target = (target - target.min()) / (target.max() - target.min())
    
    # Compute metrics
    metrics = {
        'PSNR': peak_signal_noise_ratio(target, pred, data_range=1.0),
        'SSIM': structural_similarity(target, pred, data_range=1.0)
    }
    
    return metrics

def compare_methods():
    """Compare PKL vs L2 guidance."""
    
    # Paths
    pkl_dir = Path('outputs/pkl')
    l2_dir = Path('outputs/l2')
    target_dir = Path('data/test/twophoton')
    
    # Collect results
    results = []
    
    for target_path in target_dir.glob('*.tif'):
        name = target_path.stem
        
        # PKL reconstruction
        pkl_path = pkl_dir / f"{name}_reconstructed.tif"
        if pkl_path.exists():
            pkl_metrics = evaluate_reconstruction(pkl_path, target_path)
            results.append({
                'Image': name,
                'Method': 'PKL',
                'PSNR': pkl_metrics['PSNR'],
                'SSIM': pkl_metrics['SSIM']
            })
        
        # L2 reconstruction
        l2_path = l2_dir / f"{name}_reconstructed.tif"
        if l2_path.exists():
            l2_metrics = evaluate_reconstruction(l2_path, target_path)
            results.append({
                'Image': name,
                'Method': 'L2',
                'PSNR': l2_metrics['PSNR'],
                'SSIM': l2_metrics['SSIM']
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Compute average metrics
    avg_metrics = df.groupby('Method')[['PSNR', 'SSIM']].mean()
    print("\nAverage Metrics:")
    print(avg_metrics)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # PSNR comparison
    axes[0].boxplot([
        df[df['Method'] == 'PKL']['PSNR'],
        df[df['Method'] == 'L2']['PSNR']
    ], labels=['PKL', 'L2'])
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('PSNR Comparison')
    axes[0].grid(True, alpha=0.3)
    
    # SSIM comparison
    axes[1].boxplot([
        df[df['Method'] == 'PKL']['SSIM'],
        df[df['Method'] == 'L2']['SSIM']
    ], labels=['PKL', 'L2'])
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('SSIM Comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/metrics_comparison.png', dpi=150)
    plt.show()
    
    # Save detailed results
    df.to_csv('outputs/detailed_metrics.csv', index=False)
    print(f"\nDetailed results saved to outputs/detailed_metrics.csv")

if __name__ == "__main__":
    compare_methods()
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)
**Problem**: GPU runs out of memory during training/inference.
**Solutions**:
- Reduce batch size in training
- Use gradient accumulation
- Enable mixed precision training (FP16)
- Use CPU if GPU unavailable (slower)

### Issue 2: Poor Reconstruction Quality
**Problem**: Reconstructions are blurry or have artifacts.
**Solutions**:
- Train for more epochs
- Check if PSF is correctly measured/aligned
- Tune guidance strength (lambda_base)
- Ensure proper normalization

### Issue 3: Training Instability
**Problem**: Loss explodes or oscillates.
**Solutions**:
- Reduce learning rate
- Use gradient clipping
- Check data normalization
- Use EMA (Exponential Moving Average) of weights

### Issue 4: Slow Inference
**Problem**: Reconstruction takes too long.
**Solutions**:
- Reduce DDIM steps (try 50 instead of 100)
- Use GPU instead of CPU
- Batch multiple images together
- Use compiled model (PyTorch 2.0)

## Testing Your Implementation

### Unit Test for PSF
```python
def test_psf():
    psf = PSF()
    assert psf.psf.sum() == 1.0, "PSF should sum to 1"
    assert psf.psf.min() >= 0, "PSF should be non-negative"
    print("✓ PSF test passed")

test_psf()
```

### Unit Test for Forward Model
```python
def test_forward_model():
    psf = PSF()
    fm = ForwardModel(psf.to_torch(), device='cpu')
    
    # Test shape preservation
    x = torch.ones(1, 1, 64, 64)
    y = fm.forward(x, add_noise=False)
    assert y.shape == x.shape, "Shape should be preserved"
    
    # Test adjoint
    x = torch.randn(1, 1, 32, 32)
    y = torch.randn(1, 1, 32, 32)
    
    Ax = fm.apply_blur(x)
    ATy = fm.apply_adjoint(y)
    
    inner1 = (Ax * y).sum()
    inner2 = (x * ATy).sum()
    
    assert torch.abs(inner1 - inner2) < 1e-3, "Adjoint test failed"
    print("✓ Forward model test passed")

test_forward_model()
```

## Next Steps

Once you have the basic system working:

1. **Optimize Performance**
   - Implement mixed precision training
   - Use torch.compile() for faster inference
   - Optimize data loading pipeline

2. **Improve Quality**
   - Fine-tune on real microscopy data
   - Implement perceptual losses
   - Add regularization terms

3. **Extend Functionality**
   - Support 3D volumes
   - Multi-channel fluorescence
   - Time-lapse sequences

4. **Production Deployment**
   - Create REST API for inference
   - Docker containerization
   - Cloud deployment (AWS/GCP)

## Resources for Learning More

### Papers to Read
1. Original DDPM paper (Ho et al., 2020)
2. DDIM paper (Song et al., 2021)
3. Diffusion for Inverse Problems (Chung et al., 2022)

### Online Courses
1. Deep Learning Specialization (Coursera)
2. PyTorch Tutorials (Official)
3. Hugging Face Diffusion Course

### Communities
1. PyTorch Forums
2. Hugging Face Discord
3. Papers with Code

## Conclusion

Congratulations! You've built a state-of-the-art microscopy denoising system. The key insights:

1. **Physics Matters**: PKL guidance respects Poisson statistics
2. **Diffusion is Powerful**: Can learn complex image priors
3. **Guidance is Flexible**: Can inject domain knowledge

Remember: This is research code. Expect to iterate and debug. The joy is in understanding why it works!

Good luck with your implementation! 🚀
