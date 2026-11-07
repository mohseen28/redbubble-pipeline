"""Stage 6: upscale the approved cutout to the master (approved -> mastered).

RGB goes through Real-ESRGAN after edge colours are bled into transparent
pixels; alpha is upscaled separately with Lanczos; the two are recombined.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image

from pipeline.config import AppConfig
from pipeline.schemas import DesignState
from pipeline.utils.gpu import torch_device
from pipeline.utils.images import bleed_colors, clear_border, load_rgba, save_png
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "upscale"
INPUT_STATUS = "approved"
OUTPUT_STATUS = "mastered"
log = get_logger("upscale")

# Official Real-ESRGAN release weights (BSD-3-Clause).
WEIGHTS: dict[str, tuple[str, str]] = {
    "realesrgan_x4plus_anime_6b": (
        "RealESRGAN_x4plus_anime_6B.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "realesrgan_x4plus": (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
}
PHOTO_STYLE_WORDS = ("photo", "photograph", "realistic", "realism", "painterly", "oil painting",
                     "watercolor", "watercolour", "cinematic", "3d render")


class Upscaler(Protocol):
    name: str
    scale: int

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 -> (H*scale)x(W*scale)x3 uint8."""
        ...


class LanczosUpscaler:
    """Dry-run / fallback upscaler."""

    name = "lanczos"

    def __init__(self, scale: int = 4) -> None:
        self.scale = scale

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb, "RGB").resize((w * self.scale, h * self.scale), Image.Resampling.LANCZOS)
        return np.asarray(img)


def download(url: str, dest: Path) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as fh:  # fixed https URL
        shutil.copyfileobj(resp, fh)
    tmp.replace(dest)
    return dest


class RealESRGANUpscaler:
    """Real-ESRGAN weights loaded with spandrel (basicsr is broken on current torchvision)."""

    name = "realesrgan"

    def __init__(self, model: str, models_dir: Path, tile: int = 400, tile_pad: int = 10,
                 device: str = "auto") -> None:
        key = model.lower()
        if key not in WEIGHTS:
            raise ValueError(f"unknown upscale model {model!r}; known: {sorted(WEIGHTS)}")
        self.model_name = model
        self.models_dir = models_dir
        self.tile = tile
        self.tile_pad = tile_pad
        self.device = torch_device(device)
        if self.device == "cpu":
            log.warning("Real-ESRGAN running on CPU: this is slow (minutes per image)")
        self.scale = 4
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            import torch
            from spandrel import ImageModelDescriptor, ModelLoader

            fname, url = WEIGHTS[self.model_name.lower()]
            path = download(url, self.models_dir / "realesrgan" / fname)
            desc = ModelLoader().load_from_file(str(path))
            if not isinstance(desc, ImageModelDescriptor):
                raise TypeError(f"{fname} is not a single-image model")
            desc.to(torch.device(self.device)).eval()
            if self.device == "cuda" and desc.supports_half:
                desc.half()
            self.scale = int(desc.scale)
            self._model = desc
        return self._model

    def upscale(self, rgb: np.ndarray) -> np.ndarray:
        import torch

        model = self._load()
        dtype = torch.float16 if (self.device == "cuda" and model.supports_half) else torch.float32
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        x = x.to(self.device, dtype=dtype) / 255.0
        tile = self.tile
        while True:
            try:
                with torch.inference_mode():
                    y = self._tiled(model, x, tile)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if tile == 0:
                    tile = 512
                elif tile > 64:
                    tile //= 2
                else:
                    raise
                log.warning("CUDA OOM in Real-ESRGAN; retrying with tile=%d", tile)
        out = (y.clamp(0, 1) * 255.0).round().byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.asarray(out)

    def _tiled(self, model: Any, x: Any, tile: int) -> Any:
        if tile <= 0:
            return model(x)
        _, c, h, w = x.shape
        s = self.scale
        out = x.new_zeros((1, c, h * s, w * s))
        pad = self.tile_pad
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(y0 - pad, 0), max(x0 - pad, 0)
                py1, px1 = min(y1 + pad, h), min(x1 + pad, w)
                o = model(x[:, :, py0:py1, px0:px1])
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = o[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        return out


def pick_model(style: str | None, cfg: AppConfig) -> str:
    """Anime/illustration model for flat art, general model for photographic styles."""
    s = (style or "").lower()
    return cfg.s.upscale.general_model if any(w in s for w in PHOTO_STYLE_WORDS) else cfg.s.upscale.model


def upscale_rgba(rgba: np.ndarray, upscaler: Upscaler) -> np.ndarray:
    """Bleed edges, upscale RGB with the model and alpha with Lanczos, recombine."""
    rgb = bleed_colors(rgba)
    up = upscaler.upscale(rgb)
    H, W = up.shape[:2]
    alpha = np.array(Image.fromarray(rgba[..., 3], "L").resize((W, H), Image.Resampling.LANCZOS))
    alpha[alpha < 3] = 0
    clear_border(alpha, max(1, W // rgba.shape[1]))
    return np.dstack([up[..., :3], alpha]).astype(np.uint8)


def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    if not state.cutout_image:
        raise ValueError("no cutout image to upscale")
    rgba = np.array(load_rgba(state.cutout_image))
    model = pick_model(state.brief.style if state.brief else None, cfg)
    master = upscale_rgba(rgba, deps.upscaler(model))
    H, W = master.shape[:2]
    out = cfg.dir("master") / f"{state.slug}_master_{W}x{H}.png"
    save_png(Image.fromarray(master, "RGBA"), out, dpi=cfg.s.export.dpi)
    return state.patch(master_image=str(out), status=OUTPUT_STATUS)
