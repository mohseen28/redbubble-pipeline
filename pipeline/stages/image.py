"""Stage 2: generate the raw image (new -> generated)."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from PIL import Image

from pipeline.config import AppConfig
from pipeline.schemas import DesignState
from pipeline.utils.images import save_png

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "image"
INPUT_STATUS = "new"
OUTPUT_STATUS = "generated"


def new_seed() -> int:
    return secrets.randbelow(2**31 - 1)


def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    """Always draws a fresh random seed: first attempt, QA retry and review-regenerate alike."""
    if state.brief is None:
        raise ValueError("design has no brief")
    g = cfg.s.imagegen
    seed = new_seed()
    img = deps.imagegen.generate(state.brief.prompt, seed, g.width, g.height, state.brief)
    if img.size != (g.width, g.height):
        img = img.resize((g.width, g.height), Image.Resampling.LANCZOS)
    out_dir = cfg.dir("raw") / state.design_id
    out_dir.mkdir(parents=True, exist_ok=True)
    version = len(list(out_dir.glob("v*.png"))) + 1
    path = out_dir / f"v{version}.png"
    save_png(img.convert("RGB"), path, dpi=cfg.s.export.dpi)
    return state.patch(seed=seed, raw_image=str(path), cutout_image=None, master_image=None,
                       status=OUTPUT_STATUS)
