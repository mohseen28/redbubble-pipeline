"""Stage 4: background removal to a transparent PNG (qa_passed -> awaiting_review)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from PIL import Image

from pipeline.config import AppConfig
from pipeline.schemas import DesignState
from pipeline.utils.images import clean_alpha, estimate_bg_color, load_rgb, save_png

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "cutout"
INPUT_STATUS = "qa_passed"
OUTPUT_STATUS = "awaiting_review"


class Cutter(Protocol):
    name: str

    def remove(self, img: Image.Image) -> Image.Image:
        """Return an RGBA image the same size as ``img``."""
        ...


class RembgCutter:
    """rembg with a persistent session (u2net by default; BRIA weights are not used)."""

    name = "rembg"

    def __init__(self, model: str = "u2net", alpha_matting: bool = False) -> None:
        if model.startswith("bria"):
            raise ValueError("BRIA RMBG weights are not licensed for commercial use; pick another model")
        self.model = model
        self.alpha_matting = alpha_matting
        self._session: Any = None

    def remove(self, img: Image.Image) -> Image.Image:
        from rembg import new_session, remove

        if self._session is None:
            self._session = new_session(self.model)
        out = remove(img.convert("RGB"), session=self._session, alpha_matting=self.alpha_matting)
        if not isinstance(out, Image.Image):
            raise TypeError("rembg returned an unexpected type")
        return out.convert("RGBA")


class ColorKeyCutter:
    """Dry-run / fallback cutter: keys out the plain background colour estimated from the border."""

    name = "colorkey"

    def __init__(self, tolerance: float = 30.0) -> None:
        self.tolerance = float(tolerance)

    def remove(self, img: Image.Image) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        bg = estimate_bg_color(rgb)
        dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)
        lo, hi = self.tolerance * 0.5, self.tolerance * 1.5
        alpha = np.clip((dist - lo) / (hi - lo), 0, 1) * 255
        rgba = np.dstack([rgb, alpha.astype(np.uint8)])
        return Image.fromarray(rgba, "RGBA")


def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    if not state.raw_image:
        raise ValueError("no raw image to cut out")
    raw = load_rgb(state.raw_image)
    cut = deps.cutter.remove(raw)
    if cut.size != raw.size:
        cut = cut.resize(raw.size, Image.Resampling.LANCZOS)
    arr = np.array(cut.convert("RGBA"))
    c = cfg.s.cutout
    arr[..., 3] = clean_alpha(arr[..., 3], c.erode_px, c.feather_px, c.min_speck_area)
    if not (arr[..., 3] > 0).any():
        raise ValueError("cutout removed the whole image (empty alpha)")
    out = cfg.dir("cutout") / f"{state.design_id}.png"
    save_png(Image.fromarray(arr, "RGBA"), out, dpi=cfg.s.export.dpi)
    return state.patch(cutout_image=str(out), status=OUTPUT_STATUS)
