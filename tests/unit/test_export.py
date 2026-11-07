from __future__ import annotations

import csv
import re

import numpy as np
import pytest
from PIL import Image

from pipeline.config import ProductSpec
from pipeline.db import DB
from pipeline.deps import Deps
from pipeline.stages import export
from pipeline.stages.export import ExportError, file_name, placement, render
from tests.conftest import make_cfg
from tests.helpers import rgba_disc, save, state


def spec(**kw):
    base = dict(key="test_product", category="home", width=1000, height=500, transparent=False,
                format="jpg", margin_pct=5, anchor="center", scale=1.0)
    base.update(kw)
    return ProductSpec(**base)


def test_placement_contain_with_margins():
    pl = placement(400, 400, 1000, 500, 5, "center")
    # inner box = 900 x 450 -> limited by height
    assert (pl.width, pl.height) == (450, 450)
    assert pl.scale_factor == pytest.approx(450 / 400)
    assert (pl.x, pl.y) == (50 + (900 - 450) // 2, 25)


@pytest.mark.parametrize(("anchor", "xy"), [
    ("center", (275, 25)), ("left", (50, 25)), ("right", (500, 25)),
    ("top", (275, 25)), ("bottom_right", (500, 25)), ("top_left", (50, 25)),
])
def test_anchoring(anchor, xy):
    pl = placement(400, 400, 1000, 500, 5, anchor)
    assert (pl.x, pl.y) == xy


def test_anchoring_vertical():
    assert placement(400, 400, 500, 1000, 0, "top").y == 0
    assert placement(400, 400, 500, 1000, 0, "bottom").y == 500
    assert placement(400, 400, 500, 1000, 0, "center").y == 250


def test_render_exact_size_bg_fill_and_transparency():
    design = rgba_disc(200, 0.5)  # disc fills the whole 200x200 box
    img, pl = render(design, spec(), "#336699")
    assert img.size == (1000, 500) and img.mode == "RGB"
    assert img.getpixel((5, 5)) == (0x33, 0x66, 0x99)              # bg fill in the margin
    assert img.getpixel((500, 250))[0] > 150                         # subject in the centre
    timg, _ = render(design, spec(transparent=True, format="png"), "#336699")
    assert timg.mode == "RGBA"
    assert all(timg.getpixel(p)[3] == 0 for p in [(0, 0), (999, 0), (0, 499), (999, 499)])
    assert pl.scale_factor == pytest.approx(450 / 200)


def test_scale_factor_by_canvas_scale():
    s = spec(width=1000, height=1000, scale=2.0)
    assert (s.canvas_w, s.canvas_h) == (2000, 2000)


def test_file_name_pattern():
    s = spec(key="premium_tshirt", width=2875, height=3900, format="png", transparent=True)
    assert file_name("cozy-mushroom-cottage-01", s) == "cozy-mushroom-cottage-01_premium-tshirt_2875x3900.png"


def test_hard_limits(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(ExportError, match="max_dim_px"):
        export.check_hard_limits(spec(width=14000, height=100), cfg)
    with pytest.raises(ExportError, match="no room"):
        placement(10, 10, 10, 10, 49.99, "center")


def _export_setup(tmp_path, cfg, master_size=256, frac=0.45):
    db = DB(cfg.db_path)
    run_id = db.create_run("cozy", False, 1, True)
    name = f"cozy-mushroom-cottage-01_master_{master_size}x{master_size}.png"
    master = save(rgba_disc(master_size, frac), cfg.dir("master") / name)
    st = state(status="mastered", master_image=master, listing={"title": "T", "tags": ["a"], "description": "D"})
    db.insert(st, run_id)
    return db, run_id, st, Deps(cfg, dry_run=True, db=db)


def test_export_run_folder_manifest_and_low_res(small_cfg, tmp_path):
    cfg = small_cfg
    cfg.ensure_dirs()
    _, run_id, st, deps = _export_setup(tmp_path, cfg)
    out = export.run(st, cfg, deps)
    assert out.status == "exported"
    run_dir = cfg.output_dir / run_id
    ddir = run_dir / st.slug
    assert (ddir / "design.json").is_file() and (ddir / "listing.txt").is_file()
    assert (ddir / "master" / "cozy-mushroom-cottage-01_master_256x256.png").is_file()
    assert (ddir / "apparel" / "cozy-mushroom-cottage-01_premium-tshirt_575x780.png").is_file()
    assert (ddir / "home" / "cozy-mushroom-cottage-01_mug-standard_540x224.jpg").is_file()
    assert len(out.exports) == len(cfg.products)
    for rec in out.exports:
        with Image.open(run_dir / rec.path) as im:
            assert im.size == (rec.width_px, rec.height_px)
            assert im.mode == ("RGBA" if rec.transparent else "RGB")
            assert im.info.get("dpi") and round(im.info["dpi"][0]) == 300
            assert im.info.get("icc_profile")
        assert re.match(r"^[a-z0-9-]+_[a-z0-9-]+_\d+x\d+\.(png|jpg)$", rec.path.rsplit("/", 1)[-1])
    # trimmed disc is ~231 px; poster 574x820 at 5% margin -> ~2.2x, below 2.5
    # small master forces low_res on nothing here; check flag logic directly:
    assert all(r.low_res == (r.scale_factor > cfg.s.export.max_scale_factor) for r in out.exports)
    rows = list(csv.DictReader((run_dir / "manifest.csv").open()))
    assert rows[0]["product"] == "master"
    assert len(rows) == len(cfg.products) + 1
    assert list(rows[0].keys()) == export.MANIFEST_COLUMNS
    # resume: exporting again replaces rows, no duplicates
    export.run(out.patch(status="mastered"), cfg, deps)
    assert len(list(csv.DictReader((run_dir / "manifest.csv").open()))) == len(cfg.products) + 1


def test_low_res_flag_set_and_logged(small_cfg, tmp_path):
    cfg = small_cfg.with_overrides({"export.max_scale_factor": 1.0})
    cfg.ensure_dirs()
    db, _, st, deps = _export_setup(tmp_path, cfg, master_size=128)
    out = export.run(st, cfg, deps)
    low = [r for r in out.exports if r.low_res]
    assert low and all(r.scale_factor > 1.0 for r in low)
    assert db.events(st.design_id, action="warning")


def test_render_cache_renders_identical_specs_once(small_cfg, tmp_path, mocker):
    cfg = small_cfg
    cfg.ensure_dirs()
    _, run_id, st, deps = _export_setup(tmp_path, cfg)
    spy = mocker.spy(export, "render")
    out = export.run(st, cfg, deps)
    # products_small has sticker_small and sticker_medium with identical size/format/margin
    assert spy.call_count == len(cfg.products) - 1
    a, b = (cfg.output_dir / run_id / r.path for r in out.exports if r.product.startswith("sticker"))
    assert a.read_bytes() == b.read_bytes() and a.name != b.name


def test_file_size_cap(small_cfg, tmp_path):
    cfg = small_cfg.with_overrides({"export.max_file_mb": 0.0001})
    cfg.ensure_dirs()
    _, _, st, deps = _export_setup(tmp_path, cfg)
    with pytest.raises(ExportError, match="max_file_mb"):
        export.run(st, cfg, deps)


def test_transparent_master_is_rejected(small_cfg, tmp_path):
    cfg = small_cfg
    cfg.ensure_dirs()
    _, _, st, deps = _export_setup(tmp_path, cfg)
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(st.master_image)
    with pytest.raises(ExportError, match="fully transparent"):
        export.run(st, cfg, deps)


def test_trim_to_alpha_bbox(small_cfg, tmp_path):
    """An off-centre subject is trimmed, so it ends up centred on the canvas."""
    cfg = small_cfg
    cfg.ensure_dirs()
    _, run_id, st, deps = _export_setup(tmp_path, cfg)
    rgba_disc(256, 0.2, offset=(-70, -70)).save(st.master_image)
    out = export.run(st, cfg, deps)
    rec = next(r for r in out.exports if r.product == "coaster")
    arr = np.asarray(Image.open(cfg.output_dir / run_id / rec.path).convert("RGB")).astype(int)
    bg = np.array([0xf5, 0xef, 0xe0])
    ys, xs = np.where(np.abs(arr - bg).sum(axis=2) > 60)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    assert abs(cx - 186) <= 2 and abs(cy - 186) <= 2
