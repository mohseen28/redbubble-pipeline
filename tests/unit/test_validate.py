from __future__ import annotations

from PIL import Image

from pipeline.db import DB
from pipeline.deps import Deps
from pipeline.stages import export
from pipeline.validate import validate_output, validate_run
from tests.helpers import rgba_disc, save, state


def _exported(cfg):
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    run_id = db.create_run("cozy", False, 1, True)
    master = save(rgba_disc(256, 0.45), cfg.dir("master") / "cozy-mushroom-cottage-01_master_256x256.png")
    st = state(status="mastered", master_image=master)
    db.insert(st, run_id)
    export.run(st, cfg, Deps(cfg, dry_run=True, db=db))
    return cfg.output_dir / run_id


def test_validate_passes_then_detects_problems(small_cfg):
    run_dir = _exported(small_cfg)
    rep = validate_run(run_dir, small_cfg)
    assert rep.ok, rep.errors
    assert rep.designs == 1 and rep.files_checked == len(small_cfg.products)
    ddir = run_dir / "cozy-mushroom-cottage-01"
    # wrong size
    p = ddir / "home" / "cozy-mushroom-cottage-01_coaster_372x372.jpg"
    Image.new("RGB", (100, 100)).save(p, "JPEG")
    # missing product
    (ddir / "stickers" / "cozy-mushroom-cottage-01_sticker-medium_600x800.png").unlink()
    # wrong mode (opaque PNG for a transparent product)
    q = ddir / "apparel" / "cozy-mushroom-cottage-01_baseball-cap_480x228.png"
    Image.new("RGB", (480, 228)).save(q, "PNG")
    rep = validate_run(run_dir, small_cfg)
    text = "\n".join(rep.errors)
    assert not rep.ok
    assert "size (100, 100)" in text
    assert "missing product sticker_medium" in text
    assert "mode RGB != RGBA" in text


def test_validate_output_empty(small_cfg):
    reps = validate_output(small_cfg)
    assert not reps[0].ok
