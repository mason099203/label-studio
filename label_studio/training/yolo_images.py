"""Decode, validate, and normalize images for YOLO training datasets."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}


def register_extra_image_openers() -> None:
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


def is_probably_html(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html") or head.startswith(b"<?xml")


def canonicalize_image_to_jpeg(src: Path, dst: Path) -> bool:
    from PIL import Image

    register_extra_image_openers()
    try:
        with Image.open(src) as im:
            im.load()
            rgb = im.convert("RGB")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        rgb.save(tmp, format="JPEG", quality=95, optimize=True)
        tmp.replace(dst)
        return dst.is_file() and dst.stat().st_size > 0
    except Exception as exc:
        logger.warning("Cannot convert image %s to JPEG: %s", src, exc)
        try:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False


def decodable_image_path(path: Path) -> Path | None:
    """Return path if file exists, has image-like ext, and PIL can decode it."""
    if not path.is_file():
        return None
    try:
        if path.stat().st_size <= 0:
            return None
    except OSError:
        return None
    if path.suffix.lower() not in _IMAGE_EXTS:
        return None
    from PIL import Image

    register_extra_image_openers()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
        return path
    except Exception:
        return None


def canonicalize_yolo_images_dir(images_dir: Path) -> int:
    """
    Rewrite all images under images_dir as valid JPEG files (same stem).
    Removes files that cannot be decoded.
    """
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    converted = 0
    for path in list(images_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".tmp" or path.name.endswith(".jpg.tmp"):
            continue

        dst = images_dir / f"{path.stem}.jpg"
        if canonicalize_image_to_jpeg(path, dst):
            if path.resolve() != dst.resolve() and path.exists():
                path.unlink()
            converted += 1
        else:
            logger.warning("Removing undecodable training image: %s", path)
            path.unlink(missing_ok=True)

    if converted == 0:
        raise FileNotFoundError(
            f"No decodable images under {images_dir}. "
            "Ensure task images are valid JPEG/PNG (not HTML error pages or broken downloads)."
        )
    logger.info("Canonicalized %s YOLO training images under %s", converted, images_dir)
    return converted
