import numpy as np
import torch
from pathlib import Path
import tifffile


class PSF:
    """Point Spread Function handler for microscopy."""

    def __init__(self, psf_path: str = None, psf_array: np.ndarray = None):
        """
        Initialize PSF from file or array.

        Args:
            psf_path: Path to PSF TIFF file
            psf_array: Direct PSF array
        """
        if psf_path is not None:
            self.psf = self._load_psf(psf_path)
        elif psf_array is not None:
            arr = psf_array
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            self.psf = arr.astype(np.float32)
        else:
            # Default Gaussian PSF
            self.psf = self._create_gaussian_psf()

        # Normalize PSF to sum to 1
        s = float(self.psf.sum())
        if s == 0.0:
            # Fallback to default Gaussian if provided PSF is degenerate
            self.psf = self._create_gaussian_psf()
        else:
            self.psf = self.psf / s

    def _load_psf(self, path: str) -> np.ndarray:
        """Load PSF from TIFF file."""
        psf = tifffile.imread(path)
        if psf.ndim == 3:  # If 3D, take central slice
            psf = psf[psf.shape[0] // 2]
        return psf.astype(np.float32)

    def _create_gaussian_psf(self, size: int = 15, sigma: float = 2.0) -> np.ndarray:
        """Create default Gaussian PSF."""
        x = np.arange(size) - size // 2
        y = np.arange(size) - size // 2
        xx, yy = np.meshgrid(x, y)
        psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        return psf.astype(np.float32)

    def to_torch(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        """Convert PSF to torch tensor."""
        return torch.from_numpy(self.psf).to(device).to(dtype)

    def broaden(self, factor: float = 1.1):
        """Broaden PSF for robustness testing."""
        from scipy.ndimage import gaussian_filter

        sigma = (factor - 1.0) * 2.0  # Heuristic scaling
        broadened = gaussian_filter(self.psf, sigma)
        return PSF(psf_array=broadened)


