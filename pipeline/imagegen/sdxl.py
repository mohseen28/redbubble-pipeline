"""SDXL 1.0 fallback backend (config switch only: imagegen.backend = sdxl)."""

from __future__ import annotations

import os
from typing import Any

from PIL import Image

from pipeline.config import ImageGenCfg
from pipeline.imagegen.flux_schnell import check_not_blank
from pipeline.schemas import DesignBrief
from pipeline.utils.gpu import free_vram


class SDXLBackend:
    name = "sdxl"

    def __init__(self, cfg: ImageGenCfg, cpu_offload: bool = False) -> None:
        self.cfg = cfg
        self.model_id = cfg.sdxl_repo
        self.cpu_offload = cpu_offload
        self._pipe: Any = None

    def load(self) -> Any:
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.cfg.sdxl_repo, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
                token=os.environ.get("HF_TOKEN") or None)
            if torch.cuda.is_available():
                pipe.enable_model_cpu_offload() if self.cpu_offload else pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
        return self._pipe

    def generate(self, prompt: str, seed: int, width: int, height: int,
                 brief: DesignBrief | None = None) -> Image.Image:
        import torch

        pipe = self.load()
        gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)
        with torch.inference_mode():
            out = pipe(prompt=prompt, width=width, height=height, num_inference_steps=self.cfg.sdxl_steps,
                       guidance_scale=self.cfg.sdxl_guidance, generator=gen,
                       negative_prompt="text, watermark, signature, logo, blurry, cropped")
        img: Image.Image = out.images[0].convert("RGB")
        check_not_blank(img)
        return img

    def unload(self) -> None:
        self._pipe = None
        free_vram()
