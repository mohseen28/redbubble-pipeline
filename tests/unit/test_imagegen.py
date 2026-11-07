"""FLUX / SDXL backends with a fake torch + fake pipeline (no GPU, no downloads)."""

from __future__ import annotations

import sys
import types
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from PIL import Image

from pipeline.config import ImageGenCfg
from pipeline.imagegen.base import ImageBackend, ImageGenError
from pipeline.imagegen.dry_run import DryRunBackend
from pipeline.imagegen.flux_schnell import FluxSchnellBackend, check_not_blank
from pipeline.imagegen.sdxl import SDXLBackend


@pytest.fixture
def fake_torch(monkeypatch):
    t = types.ModuleType("torch")
    t.cuda = MagicMock(is_available=lambda: False)
    t.Generator = MagicMock()
    t.inference_mode = nullcontext
    t.float16, t.bfloat16, t.float32 = "fp16", "bf16", "fp32"
    monkeypatch.setitem(sys.modules, "torch", t)
    return t


def _pipe(img):
    pipe = MagicMock()
    pipe.return_value.images = [img]
    return pipe


def test_flux_call_parameters(fake_torch):
    b = FluxSchnellBackend(ImageGenCfg())
    b._pipe = _pipe(Image.effect_noise((64, 64), 50).convert("RGB"))
    out = b.generate("a frog", 123, 1024, 1024)
    kw = b._pipe.call_args.kwargs
    assert kw["num_inference_steps"] == 4 and kw["guidance_scale"] == 0.0
    assert kw["max_sequence_length"] == 256 and (kw["width"], kw["height"]) == (1024, 1024)
    assert "negative_prompt" not in kw
    fake_torch.Generator.return_value.manual_seed.assert_called_with(123)
    assert out.mode == "RGB"
    assert isinstance(b, ImageBackend) and isinstance(DryRunBackend(), ImageBackend)


def test_flux_blank_output_raises(fake_torch):
    b = FluxSchnellBackend(ImageGenCfg())
    b._pipe = _pipe(Image.new("RGB", (64, 64)))
    with pytest.raises(ImageGenError, match="blank"):
        b.generate("x", 1, 1024, 1024)


def test_flux_dev_refused():
    with pytest.raises(ValueError, match="non-commercial"):
        FluxSchnellBackend(ImageGenCfg(flux_repo="black-forest-labs/FLUX.1-dev"))


def test_sdxl_call(fake_torch):
    b = SDXLBackend(ImageGenCfg())
    b._pipe = _pipe(Image.effect_noise((64, 64), 50).convert("RGB"))
    b.generate("x", 5, 1024, 1024)
    assert b._pipe.call_args.kwargs["num_inference_steps"] == 25
    b.unload()
    assert b._pipe is None


def test_check_not_blank_ok():
    check_not_blank(Image.effect_noise((32, 32), 40).convert("RGB"))
