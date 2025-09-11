import torch
import numpy as np
from typing import Optional, Tuple


class Normalize:
    """Normalize images to [-1, 1] for diffusion model."""

    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class IntensityToModel:
    """Convert intensity domain to model domain [-1, 1]."""

    def __init__(self, minIntensity: float = 0, maxIntensity: float = 1000, **kwargs):
        # Accept snake_case aliases used in docs/tests
        if "min_intensity" in kwargs:
            minIntensity = kwargs["min_intensity"]
        if "max_intensity" in kwargs:
            maxIntensity = kwargs["max_intensity"]
        self.minIntensity = float(minIntensity)
        self.maxIntensity = float(maxIntensity)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Clip to valid range
        x = torch.clamp(x, self.minIntensity, self.maxIntensity)
        # Scale to [0, 1]
        x = (x - self.minIntensity) / (self.maxIntensity - self.minIntensity)
        # Scale to [-1, 1]
        return 2 * x - 1

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # From [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # From [0, 1] to intensity
        x = x * (self.maxIntensity - self.minIntensity) + self.minIntensity
        return torch.clamp(x, min=0)  # Ensure non-negative


class RandomCrop:
    """Random crop for data augmentation."""

    def __init__(self, size: int):
        self.size = size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        if h > self.size and w > self.size:
            top = torch.randint(0, h - self.size + 1, (1,)).item()
            left = torch.randint(0, w - self.size + 1, (1,)).item()
            x = x[:, top : top + self.size, left : left + self.size]
        return x


class AnscombeToModel:
    """Variance-stabilizing transform for Poisson data to model domain [-1, 1].

    Applies forward Anscombe transform z = 2 * sqrt(x + 3/8), then scales to [-1, 1]
    using the expected maximum intensity to compute a scale. The inverse first
    rescales to z, then applies the inverse Anscombe approximation and clamps to >= 0.
    """

    def __init__(self, maxIntensity: float = 1000.0):
        # We assume inputs are intensities in [0, maxIntensity]
        self.maxIntensity = float(maxIntensity)
        # Maximum z approximately for maxIntensity
        self._z_max = 2.0 * np.sqrt(self.maxIntensity + 3.0 / 8.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure non-negative intensities
        x = torch.clamp(x, min=0)
        # Anscombe forward
        z = 2.0 * torch.sqrt(x + 3.0 / 8.0)
        # Normalize to [-1, 1]
        z01 = torch.clamp(z / max(self._z_max, 1e-8), 0.0, 1.0)
        return 2.0 * z01 - 1.0

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        # From [-1, 1] to [0, 1]
        z01 = (x + 1.0) / 2.0
        # Scale back to z domain
        z = z01 * self._z_max
        # Inverse Anscombe (approximate unbiased) per Makitalo & Foi
        z2 = z * 0.5
        # Use polynomial approximation for inverse A^{-1}(z)
        # lambda ~ (z/2)^2 - 1/8 + (1/4)*(z/2)^{-2} - (11/8)*(z/2)^{-4} + (5/8)*(z/2)^{-6}
        eps = 1e-8
        u2 = torch.clamp(z2, min=eps) ** 2
        inv = u2 - 1.0 / 8.0 + 1.0 / (4.0 * u2) - 11.0 / (8.0 * u2 * u2) + 5.0 / (8.0 * u2 * u2 * u2)
        return torch.clamp(inv, min=0)


def compute_quantile_mapping(source_values: np.ndarray, reference_values: np.ndarray, num_bins: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a piecewise-linear mapping from source to reference via quantile matching.

    Returns arrays (src_points, ref_points) such that np.interp(x, src_points, ref_points)
    maps source quantiles to reference.
    """
    # Flatten and remove NaNs
    s = np.asarray(source_values).reshape(-1)
    r = np.asarray(reference_values).reshape(-1)
    s = s[np.isfinite(s)]
    r = r[np.isfinite(r)]
    if s.size == 0 or r.size == 0:
        # Degenerate; identity
        xs = np.linspace(0.0, 1.0, num_bins)
        return xs, xs
    qs = np.linspace(0.0, 1.0, num_bins)
    s_q = np.quantile(s, qs)
    r_q = np.quantile(r, qs)
    # Ensure monotonic (handle potential ties)
    s_q = np.maximum.accumulate(s_q)
    r_q = np.maximum.accumulate(r_q)
    return s_q, r_q


class PerModalityQuantileMatcher:
    """Per-modality quantile matching to a shared reference histogram.

    Usage:
        - Initialize with precomputed reference quantiles in intensity domain, or
          call build_reference(modality_to_samples) to compute.
        - Call transform(y, modality) to map intensities to the shared reference.
    """

    def __init__(self, reference_src_points: Optional[np.ndarray] = None, reference_ref_points: Optional[np.ndarray] = None):
        self.reference_src_points = reference_src_points
        self.reference_ref_points = reference_ref_points

    @staticmethod
    def build_reference(modality_to_samples: dict, num_bins: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
        """Build a shared reference by averaging modality quantiles in intensity domain."""
        qs = np.linspace(0.0, 1.0, num_bins)
        q_arrays = []
        for modality, arr in modality_to_samples.items():
            arr = np.asarray(arr).reshape(-1)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            q_arrays.append(np.quantile(arr, qs))
        if not q_arrays:
            # identity
            xs = np.linspace(0.0, 1.0, num_bins)
            return xs, xs
        ref_q = np.mean(np.stack(q_arrays, axis=0), axis=0)
        ref_q = np.maximum.accumulate(ref_q)
        # For mapping we set src_points=ref_q and ref_points=ref_q (reference identity);
        # per-modality mapping will compute modality->ref via compute_quantile_mapping(modality, ref_q)
        return ref_q, ref_q

    def transform(self, y: torch.Tensor, modality: str, modality_samples_quantiles: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> torch.Tensor:
        """Map intensities y (tensor) of a modality to the shared reference using interpolation."""
        y_np = y.detach().cpu().numpy()
        if modality_samples_quantiles is None:
            # Identity if no mapping provided
            return y
        src_points, ref_points = modality_samples_quantiles
        y_mapped = np.interp(y_np, src_points, ref_points)
        return torch.from_numpy(y_mapped).type_as(y)


class GeneralizedAnscombeToModel:
    """Generalized Anscombe VST for Poisson–Gaussian noise to model domain [-1, 1].

    Assumes observations y = alpha * Poisson(lambda) + mu + n, with n ~ N(0, sigma^2).

    Forward (Makitalo & Foi):
        z = 2 * sqrt( x + 3/8 + (sigma/alpha)^2 ), where x = (y - mu) / alpha
    Then scale z to [-1, 1] using z_max computed from maxIntensity in intensity units.

    Inverse uses the unbiased inverse of (pure) Anscombe applied to z, then removes
    the Gaussian term and re-applies linear scaling back to intensity domain.
    """

    def __init__(
        self,
        maxIntensity: float = 1000.0,
        alpha: float = 1.0,
        mu: float = 0.0,
        sigma: float = 0.0,
    ):
        self.maxIntensity = float(maxIntensity)
        self.alpha = float(alpha)
        self.mu = float(mu)
        self.sigma = float(sigma)
        # Compute maximum z for normalization based on maxIntensity
        # Using x_max = (I_max - mu)/alpha, clamped to >= 0
        x_max = max((self.maxIntensity - self.mu) / max(self.alpha, 1e-8), 0.0)
        z_max = 2.0 * np.sqrt(x_max + 3.0 / 8.0 + (self.sigma / max(self.alpha, 1e-8)) ** 2)
        self._z_max = max(z_max, 1e-6)

    @staticmethod
    def _inverse_anscombe_unbiased(z: torch.Tensor) -> torch.Tensor:
        """Approximate unbiased inverse for pure Anscombe A(lambda)=2*sqrt(lambda+3/8).

        Returns an estimate of lambda given z.
        """
        z2 = z * 0.5
        eps = 1e-8
        u2 = torch.clamp(z2, min=eps) ** 2
        # Polynomial from Makitalo & Foi (2011/2012)
        lam = u2 - 1.0 / 8.0 + 1.0 / (4.0 * u2) - 11.0 / (8.0 * u2 * u2) + 5.0 / (8.0 * u2 * u2 * u2)
        return lam

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        # Ensure non-negative intensities in input domain
        y = torch.clamp(y, min=0)
        # Normalize to Poisson domain x
        alpha = max(self.alpha, 1e-8)
        x = (y - self.mu) / alpha
        x = torch.clamp(x, min=0)
        # Generalized Anscombe forward
        z = 2.0 * torch.sqrt(x + 3.0 / 8.0 + (self.sigma / alpha) ** 2)
        # Normalize to [-1, 1]
        z01 = torch.clamp(z / self._z_max, 0.0, 1.0)
        return 2.0 * z01 - 1.0

    def inverse(self, x_model: torch.Tensor) -> torch.Tensor:
        # From [-1, 1] back to z
        z = (torch.clamp(x_model, -1.0, 1.0) + 1.0) * 0.5 * self._z_max
        # Unbiased inverse of pure Anscombe to estimate x_plus = x + (sigma/alpha)^2
        x_plus = self._inverse_anscombe_unbiased(z)
        # Remove Gaussian term and map back to intensity domain
        alpha = max(self.alpha, 1e-8)
        x_hat = x_plus - (self.sigma / alpha) ** 2
        y_hat = alpha * x_hat + self.mu
        return torch.clamp(y_hat, min=0)

