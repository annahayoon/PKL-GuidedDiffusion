import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

try:
    import kornia.filters as K_filters
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False


class ForwardModel:
    """WF to 2P forward model with PSF convolution and noise."""

    def __init__(self, psf: torch.Tensor, background: float = 0.0, device: str = "cuda", 
                 common_sizes: Optional[list] = None):
        """
        Initialize forward model.

        Args:
            psf: Point spread function tensor
            background: Background intensity level
            device: Computation device
            common_sizes: List of (height, width) tuples for pre-computing FFTs
        """
        self.device = device
        self.background = background

        # Ensure PSF is shape [1, 1, H, W]
        self.psf = psf.to(device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        # Cache dict for PSF FFTs keyed by (H, W, dtype)
        self._psf_fft_cache = {}
        
        # Pre-compute FFTs for common image sizes
        if common_sizes is None:
            # Default common sizes for microscopy (powers of 2 and common patch sizes)
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        
        self._precompute_common_ffts(common_sizes)

    def _precompute_common_ffts(self, common_sizes: list) -> None:
        """Pre-compute PSF FFTs for common image sizes to improve runtime performance."""
        common_dtypes = [torch.float32, torch.float16]  # Most common dtypes in inference
        
        for height, width in tqdm(common_sizes, desc="Pre-computing PSF FFTs", leave=False):
            for dtype in common_dtypes:
                try:
                    # Pre-compute and cache FFT for this size/dtype combination
                    self._get_psf_fft(height, width, dtype, self.psf.device)
                except Exception:
                    # Skip if there are memory or compatibility issues
                    continue

    def set_psf(self, psf: torch.Tensor, common_sizes: Optional[list] = None) -> None:
        """Update PSF and clear cached FFTs to maintain correctness.

        Args:
            psf: New PSF tensor, will be moved to model device and shaped to [1,1,H,W]
            common_sizes: Optional list of sizes to pre-compute FFTs for
        """
        self.psf = psf.to(self.device)
        if self.psf.ndim == 2:
            self.psf = self.psf.unsqueeze(0).unsqueeze(0)
        # Invalidate cache since kernel changed
        self._psf_fft_cache = {}
        
        # Re-precompute common FFTs for new PSF
        if common_sizes is None:
            common_sizes = [(256, 256), (512, 512), (128, 128), (64, 64), (1024, 1024)]
        self._precompute_common_ffts(common_sizes)

    def _get_psf_fft(self, height: int, width: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Return cached PSF FFT for given size/dtype or compute and store it."""
        key = (height, width, dtype)
        cached = self._psf_fft_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        psf_h, psf_w = self.psf.shape[-2:]
        psf_padded = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
        psf_src = self.psf.to(device=device, dtype=dtype)
        # If image smaller than PSF, center-crop PSF; otherwise place PSF then roll to center
        if height < psf_h or width < psf_w:
            start_h = max((psf_h - height) // 2, 0)
            start_w = max((psf_w - width) // 2, 0)
            end_h = start_h + min(psf_h, height)
            end_w = start_w + min(psf_w, width)
            cropped = psf_src[..., start_h:end_h, start_w:end_w]
            ph, pw = cropped.shape[-2:]
            psf_padded[..., :ph, :pw] = cropped
            psf_padded = torch.roll(psf_padded, shifts=(-(ph // 2), -(pw // 2)), dims=(-2, -1))
        else:
            psf_padded[..., :psf_h, :psf_w] = psf_src
            psf_padded = torch.roll(psf_padded, shifts=(-(psf_h // 2), -(psf_w // 2)), dims=(-2, -1))
        psf_fft = torch.fft.rfft2(psf_padded)
        self._psf_fft_cache[key] = psf_fft
        return psf_fft

    def _fft_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """
        Circular convolution via FFT with kernel centered at origin.
        Ensures discrete adjointness with correlation for apply_psf_adjoint.
        """
        _, _, img_h, img_w = x.shape
        x_fft = torch.fft.rfft2(x)
        psf_fft = self._get_psf_fft(img_h, img_w, x.dtype, x.device)
        y_fft = x_fft * psf_fft
        y = torch.fft.irfft2(y_fft, s=(img_h, img_w))
        return y

    def apply_psf(self, x: torch.Tensor, use_kornia: bool = False) -> torch.Tensor:
        """Apply PSF convolution using FFT with padding or Kornia filters."""
        if use_kornia and KORNIA_AVAILABLE:
            return self._kornia_convolve(x, self.psf)
        return self._fft_convolve(x, self.psf)

    def _kornia_convolve(self, x: torch.Tensor, psf: torch.Tensor) -> torch.Tensor:
        """Alternative convolution using Kornia for differentiable operations."""
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia not available. Use FFT convolution instead.")
        
        # Kornia expects kernel in shape [1, 1, H, W] 
        kernel = psf if psf.ndim == 4 else psf.unsqueeze(0).unsqueeze(0)
        
        # Use kornia's filter2d which handles padding automatically
        return K_filters.filter2d(x, kernel, border_type='reflect')

    def apply_psf_adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Apply adjoint (correlation) using conjugate multiplication in Fourier domain."""
        _, _, img_h, img_w = y.shape
        y_fft = torch.fft.rfft2(y)
        psf_fft = self._get_psf_fft(img_h, img_w, y.dtype, y.device)
        at_fft = y_fft * torch.conj(psf_fft)
        at = torch.fft.irfft2(at_fft, s=(img_h, img_w))
        return at

    def forward(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """
        Full forward model: A(x) + B [+ Poisson noise].
        """
        y = self.apply_psf(x)
        y = y + self.background
        if add_noise:
            y = torch.poisson(torch.clamp(y, min=0))
        return y


