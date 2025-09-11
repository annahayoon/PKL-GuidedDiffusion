import os
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset
from PIL import Image


def _load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("F")  # 32-bit float
    arr = torch.from_numpy(np.array(img, dtype=np.float32))
    return arr.unsqueeze(0)  # [1, H, W]


class RealPairsDataset(Dataset):
    """Dataset for real microscopy WF/2P paired images.

    Expects directory structure:
      root/
        splits/
          train/{wf,2p}
          val/{wf,2p}
          test/{wf,2p}
    and matching filenames between wf/ and 2p/ subfolders.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        self.dir_wf = os.path.join(root, "splits", split, "wf")
        self.dir_2p = os.path.join(root, "splits", split, "2p")

        wf_files = sorted([f for f in os.listdir(self.dir_wf) if not f.startswith('.')])
        two_p_files = sorted([f for f in os.listdir(self.dir_2p) if not f.startswith('.')])

        # Intersect by filenames
        wf_set = set(wf_files)
        two_p_set = set(two_p_files)
        common = sorted(list(wf_set.intersection(two_p_set)))
        self.files = common

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.files[idx]
        wf_path = os.path.join(self.dir_wf, fname)
        two_p_path = os.path.join(self.dir_2p, fname)

        wf_img = Image.open(wf_path).convert("F")
        two_p_img = Image.open(two_p_path).convert("F")

        wf = torch.from_numpy(np.array(wf_img, dtype=np.float32)).unsqueeze(0)
        two_p = torch.from_numpy(np.array(two_p_img, dtype=np.float32)).unsqueeze(0)

        # Apply optional transform mapping intensities to model space
        if self.transform is not None:
            wf = self.transform(wf)
            two_p = self.transform(two_p)

        # Return (target_2p, conditioner_wf)
        return two_p, wf

"""
Real microscopy data dataset for WF/2P paired training.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np
from PIL import Image


class RealPairsDataset(Dataset):
    """Dataset for real WF/2P microscopy pairs."""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_size: Optional[int] = None,
        mode: str = "train",
        align_pairs: bool = True,
        max_shift: int = 4,
        normalize_per_image: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        **kwargs,
    ):
        """
        Initialize real pairs dataset.

        Args:
            data_dir: Root directory containing splits/
            split: 'train', 'val', or 'test'
            transform: Transform to apply to images
            image_size: Target image size (for compatibility, images are already 256x256)
            mode: Training mode (for compatibility)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size or 256
        self.mode = mode
        self.align_pairs = align_pairs
        self.max_shift = int(max_shift)
        self.normalize_per_image = normalize_per_image
        self.percentile_low = float(percentile_low)
        self.percentile_high = float(percentile_high)

        # Paths to WF and 2P data
        self.wf_dir = self.data_dir / "splits" / split / "wf"
        self.tp_dir = self.data_dir / "splits" / split / "2p"

        if not self.wf_dir.exists() or not self.tp_dir.exists():
            raise ValueError(f"Data directories not found: {self.wf_dir}, {self.tp_dir}")

        # Get matching pairs
        wf_files = sorted(list(self.wf_dir.glob("*.png")))
        tp_files = sorted(list(self.tp_dir.glob("*.png")))

        # Verify matching pairs
        wf_names = {f.name for f in wf_files}
        tp_names = {f.name for f in tp_files}
        
        if wf_names != tp_names:
            print(f"Warning: Mismatch in WF/2P files")
            print(f"WF files: {len(wf_names)}, 2P files: {len(tp_names)}")
            # Use intersection
            common_names = wf_names & tp_names
            wf_files = [f for f in wf_files if f.name in common_names]
            tp_files = [f for f in tp_files if f.name in common_names]

        self.wf_files = sorted(wf_files)
        self.tp_files = sorted(tp_files)

        print(f"Loaded {len(self.wf_files)} WF/2P pairs for {split} split")

    def __len__(self) -> int:
        return len(self.wf_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a WF/2P pair.

        Returns:
            Tuple of (2P clean image, WF noisy image) - note the order for compatibility
        """
        # Load images
        wf_img = Image.open(self.wf_files[idx])
        tp_img = Image.open(self.tp_files[idx])

        # Convert to tensors in raw intensity domain
        wf_np = np.array(wf_img).astype(np.float32)
        tp_np = np.array(tp_img).astype(np.float32)

        # Optional per-image robust normalization to [0, 255]
        if self.normalize_per_image:
            wf_np = self._robust_rescale_01(wf_np, self.percentile_low, self.percentile_high) * 255.0
            tp_np = self._robust_rescale_01(tp_np, self.percentile_low, self.percentile_high) * 255.0

        # Optional alignment (align WF to 2P)
        if self.align_pairs:
            dy, dx = self._estimate_shift_phase_corr(tp_np, wf_np, max_shift=self.max_shift)
            wf_np = np.roll(wf_np, shift=int(dy), axis=0)
            wf_np = np.roll(wf_np, shift=int(dx), axis=1)

        wf_tensor = torch.from_numpy(wf_np).float()
        tp_tensor = torch.from_numpy(tp_np).float()

        # Add channel dimension
        wf_tensor = wf_tensor.unsqueeze(0)  # [1, H, W]
        tp_tensor = tp_tensor.unsqueeze(0)  # [1, H, W]

        # Resize if needed
        if self.image_size != 256:
            import torch.nn.functional as F
            wf_tensor = F.interpolate(wf_tensor.unsqueeze(0), 
                                    size=(self.image_size, self.image_size), 
                                    mode='bilinear', align_corners=False).squeeze(0)
            tp_tensor = F.interpolate(tp_tensor.unsqueeze(0), 
                                    size=(self.image_size, self.image_size), 
                                    mode='bilinear', align_corners=False).squeeze(0)

        # Apply transforms if any
        if self.transform:
            wf_tensor = self.transform(wf_tensor)
            tp_tensor = self.transform(tp_tensor)

        # Return (clean_target, noisy_input) for diffusion training
        # 2P is cleaner, WF is noisier
        return tp_tensor, wf_tensor

    @staticmethod
    def _robust_rescale_01(arr: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        """Rescale array to [0,1] using robust percentiles to reduce outlier influence."""
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        a = (arr - lo) / (hi - lo)
        return np.clip(a, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _estimate_shift_phase_corr(ref: np.ndarray, mov: np.ndarray, max_shift: int = 4) -> Tuple[int, int]:
        """Estimate integer pixel shift (dy, dx) aligning mov to ref via phase correlation.

        Args:
            ref: Reference image (e.g., 2P)
            mov: Moving image to align (e.g., WF)
            max_shift: Limit search radius to avoid wraparound artifacts
        Returns:
            dy, dx integer shifts such that roll(mov, (dy, dx)) aligns to ref
        """
        # Ensure float32
        ref = ref.astype(np.float32)
        mov = mov.astype(np.float32)
        # Windowing to reduce edge effects
        h, w = ref.shape[:2]
        wy = np.hanning(h)[:, None]
        wx = np.hanning(w)[None, :]
        win = (wy * wx).astype(np.float32)
        ref_w = ref * win
        mov_w = mov * win
        # FFTs
        F_ref = np.fft.rfft2(ref_w)
        F_mov = np.fft.rfft2(mov_w)
        cross_power = F_ref * np.conj(F_mov)
        denom = np.abs(cross_power) + 1e-8
        R = cross_power / denom
        r = np.fft.irfft2(R, s=ref_w.shape)
        # Zero out unrealistic large shifts by masking to a central window
        if max_shift is not None and max_shift > 0:
            yy, xx = np.ogrid[:h, :w]
            cy, cx = h // 2, w // 2
            mask = (np.abs(yy - cy) <= max_shift) & (np.abs(xx - cx) <= max_shift)
            masked = np.full_like(r, -np.inf)
            masked[mask] = r[mask]
            r = masked
        peak = np.unravel_index(np.argmax(r), r.shape)
        dy = int(peak[0]) - (h // 2)
        dx = int(peak[1]) - (w // 2)
        return dy, dx

    def get_metadata(self) -> dict:
        """Get dataset metadata."""
        return {
            "dataset_type": "real_microscopy_pairs",
            "split": self.split,
            "num_pairs": len(self),
            "image_size": f"{self.image_size}x{self.image_size}",
            "data_dir": str(self.data_dir),
            "modalities": ["2P", "WF"]
        }


class BeadDataset(Dataset):
    """Dataset for bead calibration data."""

    def __init__(
        self,
        data_dir: str,
        bead_type: str = "both",  # "no_AO", "with_AO", or "both"
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Initialize bead dataset.

        Args:
            data_dir: Root directory containing beads/
            bead_type: Which bead data to use
            transform: Transform to apply to images
        """
        self.data_dir = Path(data_dir)
        self.bead_type = bead_type
        self.transform = transform

        beads_dir = self.data_dir / "beads"
        if not beads_dir.exists():
            raise ValueError(f"Beads directory not found: {beads_dir}")

        # Collect bead images
        self.bead_files = []
        
        if bead_type in ["no_AO", "both"]:
            no_ao_files = list(beads_dir.glob("bead_no_AO_*.png"))
            no_ao_slices = list((beads_dir / "bead_no_AO_slices").glob("*.png"))
            self.bead_files.extend(no_ao_files + no_ao_slices)

        if bead_type in ["with_AO", "both"]:
            with_ao_files = list(beads_dir.glob("bead_with_AO_*.png"))
            with_ao_slices = list((beads_dir / "bead_with_AO_slices").glob("*.png"))
            self.bead_files.extend(with_ao_files + with_ao_slices)

        self.bead_files = sorted(self.bead_files)
        print(f"Loaded {len(self.bead_files)} bead images")

    def __len__(self) -> int:
        return len(self.bead_files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a bead image."""
        # Load image
        bead_img = Image.open(self.bead_files[idx])
        
        # Convert to tensor [0, 1]
        bead_tensor = torch.from_numpy(np.array(bead_img)).float() / 255.0
        
        # Add channel dimension
        bead_tensor = bead_tensor.unsqueeze(0)  # [1, H, W]
        
        # Apply transforms if any
        if self.transform:
            bead_tensor = self.transform(bead_tensor)
        
        return bead_tensor
