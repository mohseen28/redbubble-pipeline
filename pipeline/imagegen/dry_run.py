"""CPU placeholder backend: a simple shape on a plain background (no AI, no network)."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

from pipeline.schemas import DesignBrief
from pipeline.utils.images import hex_to_rgb

_SHAPES = ("circle", "star", "hexagon", "triangle", "rounded_square", "diamond", "flower")
_FALLBACK_PALETTE = ("#e76f51", "#2a9d8f", "#264653", "#f4a261", "#e9c46a", "#8338ec")


class DryRunBackend:
    name = "dry_run"
    model_id = "dry-run-placeholder"

    def generate(self, prompt: str, seed: int, width: int, height: int,
                 brief: DesignBrief | None = None) -> Image.Image:
        rng = random.Random(seed)
        bg = hex_to_rgb(brief.bg_color) if brief else (250, 246, 238)
        palette = [hex_to_rgb(c) for c in (brief.palette if brief else _FALLBACK_PALETTE)]
        fills = [c for c in palette if sum(abs(a - b) for a, b in zip(c, bg, strict=True)) > 90] or [
            (40, 40, 40) if sum(bg) > 380 else (230, 230, 230)]
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)
        cx = width / 2 + rng.uniform(-0.06, 0.06) * width
        cy = height / 2 + rng.uniform(-0.06, 0.06) * height
        r = min(width, height) * rng.uniform(0.24, 0.33)
        main = rng.choice(fills)
        accent = rng.choice(fills)
        outline = (30, 30, 30) if sum(bg) > 380 else (245, 245, 245)
        shape = _SHAPES[rng.randrange(len(_SHAPES))]
        _draw_shape(draw, shape, cx, cy, r, main, outline, max(4, width // 120), rng)
        inner = _SHAPES[rng.randrange(len(_SHAPES))]
        _draw_shape(draw, inner, cx + rng.uniform(-0.15, 0.15) * r, cy + rng.uniform(-0.15, 0.15) * r,
                    r * rng.uniform(0.3, 0.5), accent, outline, max(2, width // 250), rng)
        return img

    def unload(self) -> None:
        return None


def _poly(cx: float, cy: float, r: float, n: int, rot: float) -> list[tuple[float, float]]:
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n), cy + r * math.sin(rot + 2 * math.pi * i / n))
            for i in range(n)]


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float, r: float,
                fill: tuple[int, int, int], outline: tuple[int, int, int], width: int,
                rng: random.Random) -> None:
    rot = rng.uniform(0, math.pi)
    if shape == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    elif shape == "star":
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.45
            a = rot + math.pi * i / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        draw.polygon(pts, fill=fill, outline=outline, width=width)
    elif shape == "hexagon":
        draw.polygon(_poly(cx, cy, r, 6, rot), fill=fill, outline=outline, width=width)
    elif shape == "triangle":
        draw.polygon(_poly(cx, cy, r, 3, -math.pi / 2), fill=fill, outline=outline, width=width)
    elif shape == "diamond":
        draw.polygon(_poly(cx, cy, r, 4, 0), fill=fill, outline=outline, width=width)
    elif shape == "flower":
        for i in range(6):
            a = rot + 2 * math.pi * i / 6
            px, py = cx + r * 0.55 * math.cos(a), cy + r * 0.55 * math.sin(a)
            draw.ellipse((px - r * 0.45, py - r * 0.45, px + r * 0.45, py + r * 0.45),
                         fill=fill, outline=outline, width=width)
    else:  # rounded_square
        draw.rounded_rectangle((cx - r, cy - r, cx + r, cy + r), radius=r * 0.3,
                               fill=fill, outline=outline, width=width)
