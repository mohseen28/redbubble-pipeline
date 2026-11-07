from __future__ import annotations

import numpy as np
from PIL import Image

from pipeline.imagegen.dry_run import DryRunBackend
from pipeline.stages import cutout, upscale
from pipeline.stages.cutout import ColorKeyCutter
from pipeline.stages.upscale import LanczosUpscaler, pick_model, upscale_rgba
from pipeline.utils.images import bleed_colors, clean_alpha, corners_transparent, load_rgba
from tests.helpers import alpha, brief, rgb_on_bg, rgba_disc, save, state


def test_clean_alpha_removes_specks_and_clears_corners():
    a = np.zeros((100, 100), np.uint8)
    a[20:80, 20:80] = 255
    a[2:5, 90:93] = 255                     # 9-px speck
    a[0, 0] = 255
    out = clean_alpha(a, erode_px=1, feather_px=1, min_speck_area=64)
    assert out[3, 91] == 0 and out[0, 0] == 0
    assert out[50, 50] == 255
    assert 0 < out[20, 50] < 255 or out[21, 50] < 255   # edge eroded/feathered


def test_cutout_stage_alpha_and_corners(cfg, tmp_path):
    cfg.ensure_dirs()
    raw = save(rgb_on_bg(256), tmp_path / "raw.png")
    st = state(status="qa_passed", raw_image=raw)
    from pipeline.deps import Deps
    out = cutout.run(st, cfg, Deps(cfg, dry_run=True))
    assert out.status == "awaiting_review"
    img = Image.open(out.cutout_image)
    assert img.mode == "RGBA" and corners_transparent(img)
    assert alpha(img)[128, 128] == 255


def test_colorkey_on_dry_run_image():
    img = DryRunBackend().generate("x", 3, 256, 256, brief())
    cut = ColorKeyCutter(30).remove(img)
    a = alpha(cut)
    assert a[0, 0] == 0 and a.max() == 255


def test_bleed_has_no_dark_halo():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[16:48, 16:48] = (250, 220, 10, 255)
    rgb = bleed_colors(rgba)
    assert (rgb[0, 0] > 150).all() or rgb[0, 0][0] > 200   # transparent area takes edge colour, not black
    assert tuple(rgb[30, 30]) == (250, 220, 10)


def test_upscale_preserves_alpha_and_corners():
    src = np.array(rgba_disc(64, 0.4, color=(250, 220, 10)))
    out = upscale_rgba(src, LanczosUpscaler(4))
    assert out.shape == (256, 256, 4)
    img = Image.fromarray(out, "RGBA")
    assert corners_transparent(img)
    assert out[128, 128, 3] == 255
    edge = out[(out[..., 3] > 30) & (out[..., 3] < 220)]
    assert edge[:, 0].mean() > 200, "edge pixels darkened (halo)"


def test_upscale_stage(cfg, tmp_path):
    cfg.ensure_dirs()
    from pipeline.deps import Deps
    cut = save(rgba_disc(128, 0.4), tmp_path / "cut.png")
    st = state(status="approved", cutout_image=cut)
    out = upscale.run(st, cfg, Deps(cfg, dry_run=True))
    assert out.status == "mastered"
    assert out.master_image.endswith("cozy-mushroom-cottage-01_master_512x512.png")
    m = load_rgba(out.master_image)
    assert m.size == (512, 512) and corners_transparent(m)


def test_pick_model(cfg):
    assert pick_model("flat vector sticker", cfg) == "realesrgan_x4plus_anime_6B"
    assert pick_model("Photorealistic oil painting", cfg) == "realesrgan_x4plus"
