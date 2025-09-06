from typing import Optional

import numpy as np
from tqdm import tqdm


def richardson_lucy_restore(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 30,
    clip: bool = True,
) -> np.ndarray:
    """Richardson–Lucy deconvolution baseline using scikit-image if available.

    Falls back to a minimal NumPy implementation if scikit-image is not installed.
    Expects single-channel 2D arrays.
    """
    try:
        from skimage.restoration import richardson_lucy  # type: ignore

        return richardson_lucy(image, psf, iterations=num_iter, clip=clip)
    except Exception:
        # Simple fallback RL without acceleration
        from scipy.signal import fftconvolve  # type: ignore

        img = image.astype(np.float32)
        kernel = psf.astype(np.float32)
        kernel = kernel / (kernel.sum() + 1e-12)
        estimate = np.maximum(img, 1e-12)
        psf_mirror = kernel[::-1, ::-1]
        for _ in tqdm(range(num_iter), desc="Richardson-Lucy iterations"):
            conv = fftconvolve(estimate, kernel, mode="same")
            relative_blur = img / (conv + 1e-12)
            estimate *= fftconvolve(relative_blur, psf_mirror, mode="same")
            if clip:
                estimate = np.clip(estimate, 0, None)
        return estimate


