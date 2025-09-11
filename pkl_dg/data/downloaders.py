import os
import tarfile
import zipfile
from pathlib import Path
from typing import Optional, List

import requests
from tqdm import tqdm


class DownloadError(RuntimeError):
    pass


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stream_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        if r.status_code != 200:
            raise DownloadError(f"Failed to download {url}: HTTP {r.status_code}")
        total = int(r.headers.get("content-length") or 0)
        downloaded = 0
        with open(dest, "wb") as f:
            with tqdm(total=total, unit='B', unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pbar.update(len(chunk))


def _extract_archive(archive_path: Path, to_dir: Path) -> None:
    if archive_path.suffixes[-2:] == [".tar", ".gz"] or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(to_dir)
    elif archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(to_dir)
    else:
        raise DownloadError(f"Unsupported archive format: {archive_path}")


def download_imagenet_subset(raw_dir: str | Path, urls: Optional[List[str]] = None) -> Path:
    """
    Download a small, license-friendly subset of images serving as an ImageNet proxy.

    Notes:
    - Full ImageNet requires manual registration and cannot be programmatically fetched.
    - This utility retrieves a small open subset (e.g., from academic mirrors) or uses
      placeholder public domain images to bootstrap the pipeline.
    """
    dest_root = _ensure_dir(Path(raw_dir) / "imagenet_subset")

    # Prefer the open Imagenette subset hosted by fastai (train/val folders included)
    imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    archive_path = dest_root / "imagenette2-320.tgz"
    extract_dir = dest_root
    extracted_marker = dest_root / "imagenette2-320"

    if not extracted_marker.exists():
        if not archive_path.exists():
            _stream_download(imagenette_url, archive_path)
        _extract_archive(archive_path, extract_dir)

    return extracted_marker


def prepare_image_folders(raw_roots: List[Path], out_dir: str | Path, *, train_ratio: float = 0.9) -> Path:
    """
    Aggregate images from multiple raw roots into a unified folder structure:
    out_dir/{train,val}/classless/{*.png|*.jpg|*.tif}
    """
    from glob import glob
    import shutil
    import random
    import hashlib

    out = Path(out_dir)
    train_dir = _ensure_dir(out / "train" / "classless")
    val_dir = _ensure_dir(out / "val" / "classless")

    all_images: List[str] = []
    patterns = [
        "**/*.png",
        "**/*.PNG",
        "**/*.jpg",
        "**/*.JPG",
        "**/*.jpeg",
        "**/*.JPEG",
        "**/*.tif",
        "**/*.TIF",
        "**/*.tiff",
        "**/*.TIFF",
    ]
    for root in raw_roots:
        root = Path(root)
        for pat in patterns:
            all_images.extend(glob(str(root / pat), recursive=True))

    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    def _copy_list(img_list: List[str], dest: Path):
        for src in tqdm(img_list, desc=f"Copying to {dest.name}"):
            ext = Path(src).suffix.lower() or ".jpg"
            # Deterministic name based on absolute path hash ensures idempotency across runs
            try:
                abs_path = str(Path(src).resolve())
            except Exception:
                abs_path = str(Path(src))
            name_hash = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:16]
            dst = dest / f"{name_hash}{ext}"
            if dst.exists():
                continue
            try:
                shutil.copy2(src, dst)
            except Exception:
                continue

    _copy_list(train_images, train_dir)
    _copy_list(val_images, val_dir)

    return out


