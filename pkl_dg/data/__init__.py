
"""Data pipeline modules for PKL Diffusion Denoising."""

from .synthesis import SynthesisDataset
from .transforms import Normalize, IntensityToModel, RandomCrop, AnscombeToModel, GeneralizedAnscombeToModel

# Optional dependency: zarr
try:
    from .zarr_io import ZarrPatchesDataset  # type: ignore
    _HAS_ZARR = True
except Exception:
    ZarrPatchesDataset = None  # type: ignore
    _HAS_ZARR = False

__all__ = [
    "SynthesisDataset",
    "Normalize",
    "IntensityToModel",
    "RandomCrop",
    "AnscombeToModel",
    "GeneralizedAnscombeToModel",
]

if _HAS_ZARR:
    __all__.append("ZarrPatchesDataset")


