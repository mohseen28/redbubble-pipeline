"""FLUX.1 schnell (Apache 2.0) via Diffusers, transformer + T5 quantized to NF4 (bitsandbytes).

T4 GPUs have no bfloat16, so compute dtype defaults to float16.
FLUX.1 dev is NOT used (non-commercial licence).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import Image

from pipeline.config import ImageGenCfg
from pipeline.imagegen.base import ImageGenError
from pipeline.schemas import DesignBrief
from pipeline.utils.gpu import free_vram
from pipeline.utils.logging import get_logger

log = get_logger("flux")


class FluxSchnellBackend:
    name = "flux_schnell"

    def __init__(self, cfg: ImageGenCfg) -> None:
        if "dev" in cfg.flux_repo.lower():
            raise ValueError("FLUX.1 dev is non-commercial; only FLUX.1 schnell is allowed")
        self.cfg = cfg
        self.model_id = cfg.flux_repo
        self._pipe: Any = None

    def _dtype(self) -> Any:
        import torch

        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            self.cfg.compute_dtype]

    def load(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import FluxPipeline, FluxTransformer2DModel

        dtype = self._dtype()
        token = os.environ.get("HF_TOKEN") or None
        repo = self.cfg.flux_repo
        kwargs: dict[str, Any] = {}
        if self.cfg.quantization == "nf4":
            from diffusers import BitsAndBytesConfig as DiffusersBnB
            from transformers import BitsAndBytesConfig as TfBnB
            from transformers import T5EncoderModel

            log.info("loading FLUX transformer (NF4, compute %s)", self.cfg.compute_dtype)
            kwargs["transformer"] = FluxTransformer2DModel.from_pretrained(
                repo, subfolder="transformer", torch_dtype=dtype, token=token,
                quantization_config=DiffusersBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                 bnb_4bit_compute_dtype=dtype))
            log.info("loading T5 encoder (NF4)")
            kwargs["text_encoder_2"] = T5EncoderModel.from_pretrained(
                repo, subfolder="text_encoder_2", torch_dtype=dtype, token=token,
                quantization_config=TfBnB(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                          bnb_4bit_compute_dtype=dtype))
        pipe = FluxPipeline.from_pretrained(repo, torch_dtype=dtype, token=token, **kwargs)
        if not torch.cuda.is_available():
            log.warning("CUDA not available: FLUX on CPU is impractically slow")
        elif self.cfg.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def generate(self, prompt: str, seed: int, width: int, height: int,
                 brief: DesignBrief | None = None) -> Image.Image:
        import torch

        pipe = self.load()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gen = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            result = pipe(prompt=prompt, width=width, height=height,
                          num_inference_steps=self.cfg.steps, guidance_scale=self.cfg.guidance,
                          max_sequence_length=self.cfg.max_sequence_length, generator=gen)
        img: Image.Image = result.images[0].convert("RGB")
        check_not_blank(img)
        return img

    def unload(self) -> None:
        self._pipe = None
        free_vram()


def check_not_blank(img: Image.Image) -> None:
    """fp16 overflow shows up as an all-black (NaN) image; fail loudly so QA can retry."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    if arr.std() < 1.0:
        raise ImageGenError(f"generated image is blank (mean={arr.mean():.1f}); possible fp16 overflow")
