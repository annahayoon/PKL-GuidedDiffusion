import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image


def _load_grayscale_images(dir_path: Path) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    if not dir_path.exists():
        return images
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for p in dir_path.glob(ext):
            arr = np.array(Image.open(p))
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            images.append(arr.astype(np.float32))
    return images


def _center_crop(img: np.ndarray, size: int = 33) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = np.unravel_index(np.argmax(img), img.shape)
    half = size // 2
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = min(y0 + size, h)
    x1 = min(x0 + size, w)
    crop = img[y0:y1, x0:x1]
    # pad if near boundaries
    if crop.shape[0] != size or crop.shape[1] != size:
        pad_y = size - crop.shape[0]
        pad_x = size - crop.shape[1]
        crop = np.pad(crop, ((0, pad_y), (0, pad_x)), mode="constant")
    return crop


def _normalize_unit_sum(psf: np.ndarray) -> np.ndarray:
    s = float(psf.sum())
    if s <= 0:
        return psf
    return psf / s


def estimate_psf_from_beads(bead_images: List[np.ndarray], crop_size: int = 33) -> np.ndarray:
    """Estimate 2D PSF by centering, cropping, and averaging bead images."""
    if not bead_images:
        raise ValueError("No bead images provided for PSF estimation")
    crops = []
    for img in bead_images:
        img = img.astype(np.float32)
        # background subtract using median
        img = img - float(np.median(img))
        img = np.clip(img, 0, None)
        crop = _center_crop(img, size=crop_size)
        crop = crop / (crop.max() + 1e-8)
        crops.append(crop)
    psf = np.mean(np.stack(crops, axis=0), axis=0)
    psf = np.clip(psf, 0, None)
    psf = _normalize_unit_sum(psf)
    return psf.astype(np.float32)


def fit_second_moments_sigma(psf: np.ndarray) -> Tuple[float, float]:
    """Estimate Gaussian sigma_x, sigma_y from second moments of PSF."""
    psf = np.clip(psf, 0, None)
    psf = _normalize_unit_sum(psf)
    h, w = psf.shape
    yy, xx = np.mgrid[0:h, 0:w]
    y0 = (psf * yy).sum()
    x0 = (psf * xx).sum()
    var_y = (psf * (yy - y0) ** 2).sum()
    var_x = (psf * (xx - x0) ** 2).sum()
    sigma_y = float(np.sqrt(max(var_y, 1e-8)))
    sigma_x = float(np.sqrt(max(var_x, 1e-8)))
    return sigma_x, sigma_y


def build_psf_bank(bead_root: str | Path) -> Dict[str, torch.Tensor]:
    """Build a PSF bank from bead data. Expects subdirs like 'with_AO' and 'no_AO'."""
    root = Path(bead_root)
    bank: Dict[str, torch.Tensor] = {}
    modes = {"with_AO": ["with_AO", "with-ao", "withao"], "no_AO": ["no_AO", "no-ao", "noao"]}
    for key, aliases in modes.items():
        for alias in aliases:
            dir_path = root / alias
            imgs = _load_grayscale_images(dir_path)
            if imgs:
                psf_np = estimate_psf_from_beads(imgs)
                bank[key] = torch.from_numpy(psf_np)
                break
    # Fallback: if only one found, clone to the other key
    if "with_AO" in bank and "no_AO" not in bank:
        bank["no_AO"] = bank["with_AO"].clone()
    if "no_AO" in bank and "with_AO" not in bank:
        bank["with_AO"] = bank["no_AO"].clone()
    if not bank:
        raise ValueError(f"No bead images found under {root}")
    return bank


def psf_params_from_tensor(psf: torch.Tensor) -> Tuple[float, float]:
    """Compute (sigma_x, sigma_y) from PSF tensor via second moments."""
    arr = psf.detach().cpu().numpy().astype(np.float32)
    sx, sy = fit_second_moments_sigma(arr)
    return float(sx), float(sy)


