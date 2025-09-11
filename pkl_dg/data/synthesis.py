import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Callable
import numpy as np
from PIL import Image
from tqdm import tqdm


class SynthesisDataset(Dataset):
    """Dataset for synthesizing training pairs from ImageNet/BioTISR."""

    def __init__(
        self,
        sourceDir: Optional[str] = None,
        forwardModel: Optional["ForwardModel"] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        imageSize: int = 256,
        mode: str = "train",
        *,
        source_dir: Optional[str] = None,
        forward_model: Optional["ForwardModel"] = None,
        image_size: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize synthesis dataset.

        Args:
            sourceDir/source_dir: Directory with source images
            forwardModel/forward_model: Forward model for WF simulation
            transform: Additional transforms
            imageSize/image_size: Target image size
            mode: 'train', 'val', or 'test'
        """
        # Backward-compatible arg aliases
        if sourceDir is None:
            sourceDir = source_dir or kwargs.get("sourceDir")
        if forwardModel is None:
            forwardModel = forward_model or kwargs.get("forwardModel")
        if imageSize is None:
            if image_size is not None:
                imageSize = int(image_size)
            elif "imageSize" in kwargs:
                imageSize = int(kwargs["imageSize"])  # override if provided

        if sourceDir is None:
            raise TypeError("SynthesisDataset requires sourceDir/source_dir")

        # Keep camelCase attributes internally per codebase style
        self.sourceDir = Path(sourceDir)
        self.forwardModel = forwardModel
        self.imageSize = int(imageSize)
        self.mode = mode

        # Collect image paths (support common lowercase/uppercase extensions)
        patterns = [
            "**/*.png", "**/*.PNG",
            "**/*.jpg", "**/*.JPG",
            "**/*.jpeg", "**/*.JPEG",
            "**/*.tif", "**/*.TIF",
            "**/*.tiff", "**/*.TIFF",
        ]
        image_paths = []
        for pat in tqdm(patterns, desc="Scanning for images", leave=False):
            image_paths += list(self.sourceDir.glob(pat))
        self.image_paths = image_paths

        # Basic transforms implemented without torchvision
        self.base_transform = None  # Placeholder for clarity

        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get synthesized training pair.

        Returns:
            Tuple of (2P-like clean image, WF-like noisy image)
        """
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Apply base transforms: resize -> grayscale -> to tensor in [0,1]
        img = self._resize_and_grayscale(img, self.imageSize)
        img = self._pil_to_tensor(img)

        # Create "2P-like" clean image (mild processing)
        x2p = self._create_2p_like(img)

        # Create "WF-like" measurement
        if self.forwardModel is not None:
            # Use provided forward model (PSF) path
            fmDevice = getattr(self.forwardModel, "device", "cpu")
            x2pDev = x2p.to(fmDevice)
            with torch.no_grad():
                yWf = self.forwardModel.forward(x2pDev.unsqueeze(0), add_noise=True).squeeze(0)
            yWf = yWf.to("cpu")
            x2p = x2p.to("cpu")
        else:
            # Fallback: simple degraded WF-like by blurring and adding noise
            yWf = self._simple_wf_like(x2p)

        # Add Gaussian background on CPU for train mode
        if self.mode == "train":
            backgroundNoise = torch.randn_like(yWf) * 0.1
            yWf = yWf + torch.abs(backgroundNoise)

        # Apply additional transforms if any
        if self.transform:
            x2p = self.transform(x2p)
            yWf = self.transform(yWf)

        return x2p, yWf

    def _create_2p_like(self, img: torch.Tensor) -> torch.Tensor:
        """
        Process image to be more 2P-like.

        Args:
            img: Input image

        Returns:
            Processed image resembling 2P microscopy
        """
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Enhance contrast (gamma correction)
        img = torch.pow(img, 0.8)

        # Scale to realistic photon counts (e.g., 10-1000 photons)
        img = img * 500 + 10

        return img

    def _resize_and_grayscale(self, img: Image.Image, image_size: int) -> Image.Image:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        img = img.convert("L")  # single channel
        return img

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # [1, H, W]
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)  # [C, H, W]
        else:
            raise ValueError("Unsupported image tensor shape from PIL conversion")
        return tensor

    def _simple_wf_like(self, x: torch.Tensor) -> torch.Tensor:
        """Create a WF-like measurement without PSF using cheap ops."""
        # light gaussian blur via separable conv with tiny kernel
        k = torch.tensor([0.27901, 0.44198, 0.27901], dtype=x.dtype)
        k2d = torch.outer(k, k).to(x.device)
        k2d = k2d / k2d.sum()
        kernel = k2d.view(1, 1, 3, 3)
        pad = (1, 1, 1, 1)
        x_pad = torch.nn.functional.pad(x.unsqueeze(0), pad, mode="reflect").squeeze(0)
        y = torch.nn.functional.conv2d(x_pad.unsqueeze(0), kernel, padding=0).squeeze(0)
        # add poisson-like noise approximation
        noise = torch.randn_like(y) * 0.05 * (y + 0.1).sqrt()
        y = torch.clamp(y + noise, min=0)
        return y


