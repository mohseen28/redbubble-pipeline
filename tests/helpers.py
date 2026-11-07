"""Synthetic test images and state builders (no AI models)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.schemas import DesignBrief, DesignState


def brief(niche: str = "cozy mushroom cottage", bg: str = "#f5efe0") -> DesignBrief:
    return DesignBrief(brief_id="b1", niche=niche, concept="A cozy mushroom cottage.",
                       prompt="A cozy mushroom cottage, flat vector, on a plain solid cream background.",
                       style="flat vector sticker", palette=["#c0392b", "#27ae60", "#2c3e50"], bg_color=bg)


def rgba_disc(size: int = 256, frac: float = 0.35, color: tuple[int, int, int] = (200, 40, 40),
              offset: tuple[int, int] = (0, 0)) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    r = size * frac
    cx, cy = size / 2 + offset[0], size / 2 + offset[1]
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*color, 255))
    return img


def rgb_on_bg(size: int = 256, bg: tuple[int, int, int] = (245, 239, 224), frac: float = 0.3) -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    r = size * frac
    d.ellipse((size / 2 - r, size / 2 - r, size / 2 + r, size / 2 + r), fill=(30, 90, 160))
    return img


def save(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path)


def state(design_id: str = "d1", slug: str = "cozy-mushroom-cottage-01", **kw: object) -> DesignState:
    return DesignState(design_id=design_id, slug=slug, status=kw.pop("status", "new"),  # type: ignore[arg-type]
                       brief=kw.pop("brief", brief()), **kw)  # type: ignore[arg-type]


def alpha(img: Image.Image) -> np.ndarray:
    return np.asarray(img.getchannel("A"))
