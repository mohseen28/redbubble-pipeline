"""Image generation backend interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from pipeline.schemas import DesignBrief


class ImageGenError(RuntimeError):
    """Generation failed (e.g. NaN / black output)."""


@runtime_checkable
class ImageBackend(Protocol):
    name: str
    model_id: str

    def generate(self, prompt: str, seed: int, width: int, height: int,
                 brief: DesignBrief | None = None) -> Image.Image:
        """Return an RGB image of exactly width x height."""
        ...

    def unload(self) -> None:
        """Release model memory (VRAM)."""
        ...
