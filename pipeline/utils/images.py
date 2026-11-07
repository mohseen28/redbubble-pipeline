"""Pillow / NumPy / OpenCV helpers shared by the image stages."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

U8 = np.uint8


def allow_large_images(max_dim_px: int) -> None:
    """Raise Pillow's decompression-bomb limit to what export.max_dim_px allows (posters exceed 89 MP)."""
    limit = max_dim_px * max_dim_px
    if Image.MAX_IMAGE_PIXELS is not None and limit > Image.MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = limit


@lru_cache(maxsize=1)
def srgb_icc() -> bytes:
    """sRGB ICC profile bytes, embedded in every exported file."""
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return bytes(profile.tobytes())


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def to_srgb(img: Image.Image) -> Image.Image:
    """Convert an image with an embedded non-sRGB profile to sRGB (no-op otherwise)."""
    icc = img.info.get("icc_profile")
    if not icc:
        return img
    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        mode = img.mode if img.mode in ("RGB", "RGBA") else "RGBA"
        out = ImageCms.profileToProfile(img.convert(mode), src, dst, outputMode=mode)
        return out if out is not None else img
    except (ImageCms.PyCMSError, OSError):
        return img


def load_rgba(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return to_srgb(im).convert("RGBA")


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return to_srgb(im).convert("RGB")


def save_png(img: Image.Image, path: str | Path, dpi: int = 300) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", dpi=(dpi, dpi), icc_profile=srgb_icc(), compress_level=6)
    return path.stat().st_size


def save_jpg(img: Image.Image, path: str | Path, quality: int = 95, dpi: int = 300) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, format="JPEG", quality=quality, dpi=(dpi, dpi),
                            icc_profile=srgb_icc(), optimize=False, subsampling=0)
    return path.stat().st_size


# ----------------------------------------------------------- background / mask
def estimate_bg_color(rgb: np.ndarray, border: int = 4) -> np.ndarray:
    """Median colour of the outer border ring."""
    b = max(1, border)
    ring = np.concatenate([
        rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3),
    ])
    med: np.ndarray = np.median(ring, axis=0).astype(np.float32)
    return med


def subject_mask(rgb: np.ndarray, tolerance: float, bg: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask of pixels that differ from the (estimated) background colour."""
    bg = estimate_bg_color(rgb) if bg is None else bg
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
    mask: np.ndarray = dist > tolerance
    return mask


# ---------------------------------------------------------------- alpha tools
def clean_alpha(alpha: np.ndarray, erode_px: int = 1, feather_px: int = 1, min_speck_area: int = 64) -> np.ndarray:
    """Erode, drop isolated specks below ``min_speck_area`` px, feather, clear the outer ring."""
    a: np.ndarray = alpha.astype(U8).copy()
    if min_speck_area > 0:
        binary = (a > 16).astype(U8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        small = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] < min_speck_area]
        if small:
            a[np.isin(labels, small)] = 0
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        a = cv2.erode(a, k, iterations=1)
    if feather_px > 0:
        ksize = 2 * feather_px + 1
        a = cv2.GaussianBlur(a, (ksize, ksize), sigmaX=feather_px * 0.6)
    a[a < 3] = 0
    clear_border(a, 1)
    return a


def clear_border(alpha: np.ndarray, px: int = 1) -> np.ndarray:
    """Zero the outer ``px`` ring in place so corners are always fully transparent."""
    if px > 0:
        alpha[:px, :] = 0
        alpha[-px:, :] = 0
        alpha[:, :px] = 0
        alpha[:, -px:] = 0
    return alpha


def corners_transparent(img: Image.Image) -> bool:
    if img.mode != "RGBA":
        return False
    w, h = img.size
    a = img.getchannel("A")
    return all(a.getpixel(p) == 0 for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)))


def bleed_colors(rgba: np.ndarray) -> np.ndarray:
    """Fill transparent pixels with nearby edge colours (push-pull pyramid).

    Prevents dark halos when the RGB channels are resampled separately from alpha.
    Opaque pixels keep their colour; semi-transparent ones are blended with the fill.
    """
    rgb = rgba[..., :3].astype(np.float32)
    w = (rgba[..., 3].astype(np.float32) / 255.0)
    if not (w > 0).any():
        return rgba[..., :3].copy()
    premult = rgb * w[..., None]
    filled = _push_pull(premult, w)
    out = premult + (1.0 - w[..., None]) * filled
    result: np.ndarray = np.clip(out + 0.5, 0, 255).astype(U8)
    return result


def _push_pull(premult: np.ndarray, w: np.ndarray) -> np.ndarray:
    h, wd = w.shape
    if h <= 2 or wd <= 2:
        total = float(w.sum())
        mean = premult.reshape(-1, 3).sum(axis=0) / max(total, 1e-6)
        return np.broadcast_to(mean, (h, wd, 3)).astype(np.float32)
    size = (max(1, wd // 2), max(1, h // 2))
    p_small = cv2.resize(premult, size, interpolation=cv2.INTER_AREA)
    w_small = cv2.resize(w, size, interpolation=cv2.INTER_AREA)
    coarse = _push_pull(p_small, w_small)
    up = cv2.resize(coarse, (wd, h), interpolation=cv2.INTER_LINEAR)
    color = np.where(w[..., None] > 1e-4, premult / np.maximum(w[..., None], 1e-4), up)
    wc = np.clip(w * 4.0, 0, 1)[..., None]       # trust partially covered pixels a bit more
    blended: np.ndarray = (wc * color + (1 - wc) * up).astype(np.float32)
    return blended


def alpha_bbox(img: Image.Image, threshold: int = 8) -> tuple[int, int, int, int] | None:
    """Bounding box of pixels with alpha > threshold (None if fully transparent)."""
    a = img.getchannel("A")
    if threshold > 0:
        a = a.point(lambda v: 255 if v > threshold else 0)
    return a.getbbox()


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((xx // cell) + (yy // cell)) % 2).astype(U8)
    arr = np.where(board[..., None] == 1, 205, 245).astype(U8).repeat(3, axis=2)
    return Image.fromarray(arr, "RGB")


def on_checkerboard(img: Image.Image, max_side: int = 384) -> Image.Image:
    im = img.convert("RGBA")
    im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    bg = checkerboard(im.size).convert("RGBA")
    bg.alpha_composite(im)
    return bg.convert("RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
