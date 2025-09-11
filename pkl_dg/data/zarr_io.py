import os
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
import zarr
from numcodecs import Blosc


def compute_image_stats(img: np.ndarray) -> Dict[str, float]:
    arr = img.astype(np.float32)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p1": float(np.quantile(arr, 0.01)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def tile_image(img: np.ndarray, tile_size: int = 256, overlap: int = 32) -> List[Tuple[slice, slice]]:
    h, w = img.shape[-2], img.shape[-1]
    step = max(1, tile_size - overlap)
    tiles: List[Tuple[slice, slice]] = []
    for top in range(0, max(1, h - tile_size + 1), step):
        for left in range(0, max(1, w - tile_size + 1), step):
            tiles.append((slice(top, top + tile_size), slice(left, left + tile_size)))
    # Ensure bottom/right coverage
    if h >= tile_size and w >= tile_size:
        if not tiles or tiles[-1][0].stop < h:
            tiles.append((slice(h - tile_size, h), slice(0, tile_size)))
        last_row_tops = sorted(set([sl[0].start for sl in tiles if sl[0].stop == h]))
        if last_row_tops:
            top_last = last_row_tops[-1]
            if not any(sl[1].stop == w and sl[0].start == top_last for sl in tiles):
                tiles.append((slice(top_last, top_last + tile_size), slice(w - tile_size, w)))
    return tiles


def write_zarr_patches(
    x_images: List[np.ndarray],
    y_images: List[np.ndarray],
    out_path: str,
    tile_size: int = 256,
    overlap: int = 32,
    compressor: Optional[Blosc] = None,
) -> None:
    """Write paired images as tiled patches into a Zarr store.

    x_images and y_images are lists of 2D arrays (H, W) in intensity domain.
    """
    assert len(x_images) == len(y_images), "Mismatched pairs"
    store = zarr.DirectoryStore(out_path)
    root = zarr.group(store=store, overwrite=True)
    comp = compressor or Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    # Precompute counts
    patch_records: List[Tuple[int, slice, slice]] = []
    for idx, (x, y) in enumerate(zip(x_images, y_images)):
        tiles = tile_image(x, tile_size, overlap)
        for sl_h, sl_w in tiles:
            patch_records.append((idx, sl_h, sl_w))

    num_patches = len(patch_records)
    patches_x = root.create_dataset(
        "patches/x",
        shape=(num_patches, 1, tile_size, tile_size),
        chunks=(64, 1, tile_size, tile_size),
        dtype="f4",
        compressor=comp,
    )
    patches_y = root.create_dataset(
        "patches/y",
        shape=(num_patches, 1, tile_size, tile_size),
        chunks=(64, 1, tile_size, tile_size),
        dtype="f4",
        compressor=comp,
    )
    meta = root.create_dataset(
        "patches/meta",
        shape=(num_patches, 6),
        chunks=(256, 6),
        dtype="f4",
        compressor=comp,
    )
    index = root.create_dataset(
        "index",
        shape=(num_patches, 3),
        chunks=(1024, 3),
        dtype="i4",
        compressor=comp,
    )

    # Write
    write_i = 0
    for img_idx, (x, y) in enumerate(zip(x_images, y_images)):
        tiles = tile_image(x, tile_size, overlap)
        for sl_h, sl_w in tiles:
            x_patch = x[sl_h, sl_w].astype(np.float32)
            y_patch = y[sl_h, sl_w].astype(np.float32)
            patches_x[write_i, 0] = x_patch
            patches_y[write_i, 0] = y_patch
            stats = compute_image_stats(y_patch)
            meta[write_i] = np.array([stats["mean"], stats["std"], stats["p1"], stats["p99"], stats["min"], stats["max"]], dtype=np.float32)
            index[write_i] = np.array([img_idx, sl_h.start, sl_w.start], dtype=np.int32)
            write_i += 1

    root.attrs.update({
        "tile_size": int(tile_size),
        "overlap": int(overlap),
        "num_images": int(len(x_images)),
        "num_patches": int(num_patches),
    })


class ZarrPatchesDataset(torch.utils.data.Dataset):
    """Dataset reading paired patches from a Zarr store, with optional mapping/transform.
    Returns (x_clean, y_noisy) tensors.
    """

    def __init__(self, zarr_path: str, transform=None):
        self.store = zarr.DirectoryStore(zarr_path)
        self.root = zarr.open(self.store, mode="r")
        self.patches_x = self.root["patches/x"]
        self.patches_y = self.root["patches/y"]
        self.meta = self.root["patches/meta"]
        self.index = self.root["index"]
        self.transform = transform

    def __len__(self) -> int:
        return int(self.patches_x.shape[0])

    def __getitem__(self, idx: int):
        x = self.patches_x[idx]
        y = self.patches_y[idx]
        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        if self.transform is not None:
            x_t = self.transform(x_t)
            y_t = self.transform(y_t)
        return x_t, y_t


