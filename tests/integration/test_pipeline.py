"""End-to-end dry-run tests (stub LLM + placeholder images, CPU only, no network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from pipeline.db import DB
from pipeline.deps import Deps
from pipeline.imagegen.dry_run import DryRunBackend
from pipeline.runner import Runner, retry_errors
from pipeline.stages.prompt import PromptRequest
from pipeline.stages.review import apply_decision
from pipeline.validate import validate_run
from tests.conftest import FIXTURES, REPO, make_cfg


def _runner(cfg, **deps_kw):
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    return Runner(cfg, db, Deps(cfg, dry_run=True, db=db, **deps_kw)), db


def test_full_run_to_exported_matches_section_11(small_cfg):
    runner, db = _runner(small_cfg)
    rid = db.create_run("cozy mushroom cottage", False, 3, True)
    res = runner.run(PromptRequest(niche="cozy mushroom cottage", n=3), rid)
    assert res.stopped is None and res.errors == 0
    assert db.counts_by_status() == {"exported": 3}
    run_dir = small_cfg.output_dir / rid
    slugs = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    assert slugs == [f"cozy-mushroom-cottage-0{i}" for i in (1, 2, 3)]
    d = run_dir / slugs[0]
    assert (d / "design.json").is_file() and (d / "listing.txt").is_file()
    assert (d / "master" / f"{slugs[0]}_master_4096x4096.png").is_file()
    assert (d / "apparel" / f"{slugs[0]}_premium-tshirt_575x780.png").is_file()
    info = json.loads((d / "design.json").read_text())
    assert info["seed"] is not None and info["models"]["image_backend"] == "dry_run" and info["qa"]["passed"]
    rows = list(csv.DictReader((run_dir / "manifest.csv").open()))
    assert len(rows) == 3 * (len(small_cfg.products) + 1)
    rep = validate_run(run_dir, small_cfg)
    assert rep.ok, rep.errors


def test_resume_after_qa_only_remaining_stages_run(small_cfg, mocker):
    runner, db = _runner(small_cfg)
    rid = db.create_run("axolotl barista", False, 3, True)
    req = PromptRequest(niche="axolotl barista", n=3)
    runner.run(req, rid, until="qa")
    assert db.counts_by_status() == {"qa_passed": 3}
    seeds = {s.design_id: s.seed for s in db.all()}
    # "kill" and restart: new process objects, same DB
    runner2, db2 = _runner(small_cfg)
    llm_spy = mocker.spy(runner2.deps.llm, "complete_json")
    gen_spy = mocker.spy(DryRunBackend, "generate")
    res = runner2.run(req, rid)
    assert llm_spy.call_count == 0 and gen_spy.call_count == 0
    assert res.processed == {"cutout": 3, "review": 3, "upscale": 3, "export": 3}
    assert {s.design_id: s.seed for s in db2.all()} == seeds
    assert db2.counts_by_status() == {"exported": 3}


class _BlankBackend(DryRunBackend):
    def generate(self, prompt, seed, width, height, brief=None):
        from PIL import Image
        return Image.new("RGB", (width, height), (240, 240, 240))


def test_qa_failure_retries_then_stops(small_cfg):
    runner, db = _runner(small_cfg, imagegen=_BlankBackend())
    rid = db.create_run("pond frogs", False, 1, True)
    runner.run(PromptRequest(niche="pond frogs", n=1), rid)
    (s,) = db.all()
    assert s.status == "qa_failed" and s.qa_retries == small_cfg.s.run.max_qa_retries
    raws = list((small_cfg.dir("raw") / s.design_id).glob("v*.png"))
    assert len(raws) == small_cfg.s.run.max_qa_retries + 1


def test_manual_review_reject_and_regenerate(tmp_path):
    cfg = make_cfg(tmp_path, FIXTURES / "products_small.yaml", **{"review.mode": "manual"})
    runner, db = _runner(cfg)
    rid = db.create_run("cat astronaut picnic", False, 3, True)
    res = runner.run(PromptRequest(niche="cat astronaut picnic", n=3), rid)
    assert res.awaiting_review == 3 and db.counts_by_status() == {"awaiting_review": 3}
    a, b, c = db.all()
    rej = apply_decision(db, cfg, a.slug, "reject")
    assert rej.status == "rejected"
    assert (cfg.dir("rejected") / a.design_id / "raw" / "v1.png").is_file()
    assert not (cfg.dir("raw") / a.design_id).exists()
    assert rej.cutout_image and Path(rej.cutout_image).parent == cfg.dir("rejected") / a.design_id
    regen = apply_decision(db, cfg, b.design_id, "regenerate")
    assert regen.status == "new" and regen.regen_count == 1 and regen.seed is None
    apply_decision(db, cfg, c.design_id, "approve")
    with pytest.raises(ValueError):
        apply_decision(db, cfg, c.design_id, "approve")
    # decisions persist across a "kernel restart"
    db2 = DB(cfg.db_path)
    assert db2.get(a.design_id).status == "rejected" and db2.get(b.design_id).decision == "regenerate"
    runner2 = Runner(cfg, db2, Deps(cfg, dry_run=True, db=db2))
    runner2.run()                                   # regenerated design loops back through 2-4
    b2 = db2.get(b.design_id)
    assert b2.status == "awaiting_review" and b2.seed != b.seed and b2.raw_image.endswith("v2.png")
    assert db2.get(c.design_id).status == "exported"   # approved one continued to export
    apply_decision(db2, cfg, b.design_id, "approve")
    runner2.finish()
    assert db2.counts_by_status() == {"exported": 2, "rejected": 1}


def test_disk_cap_stops_cleanly(small_cfg):
    small_cfg = small_cfg.with_overrides({"run.min_free_disk_gb": 5})
    small_cfg.ensure_dirs()
    db = DB(small_cfg.db_path)
    runner = Runner(small_cfg, db, Deps(small_cfg, dry_run=True, db=db), disk_free_bytes=lambda: 2 * 2**30)
    rid = db.create_run("x", False, 2, True)
    res = runner.run(PromptRequest(niche="sleepy owls", n=2), rid)
    assert res.stopped and "min_free_disk_gb" in res.stopped
    assert db.counts_by_status() == {"new": 2}       # nothing lost, resumable
    assert db.events(action="stop")


def test_design_count_cap_stops_cleanly(small_cfg):
    runner, db = _runner(small_cfg)
    rid = db.create_run("x", False, 3, True)
    runner.create_designs(PromptRequest(niche="sleepy owls", n=3), rid)
    runner.cfg = small_cfg.with_overrides({"run.max_designs_per_run": 2})
    res = runner.run()
    assert res.stopped and "max_designs_per_run" in res.stopped
    counts = db.counts_by_status()
    assert counts.get("new") == 1 and counts.get("exported") is None   # stop happens in stage 2
    res2 = Runner(small_cfg, db, Deps(small_cfg, dry_run=True, db=db)).run()   # next session continues
    assert res2.stopped is None and db.counts_by_status() == {"exported": 3}


class _Boom(DryRunBackend):
    def __init__(self):
        self.n = 0

    def generate(self, *a, **k):
        self.n += 1
        if self.n == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return super().generate(*a, **k)


def test_error_in_one_design_does_not_stop_batch_and_retry(small_cfg):
    runner, db = _runner(small_cfg, imagegen=_Boom())
    rid = db.create_run("x", False, 2, True)
    res = runner.run(PromptRequest(niche="sleepy owls", n=2), rid)
    assert res.errors == 1 and db.counts_by_status() == {"error": 1, "exported": 1}
    err = db.by_status("error")[0]
    assert "out of memory" in err.error and db.events(err.design_id, action="error")[0].traceback
    (reset,) = retry_errors(db)
    assert reset.status == "new"
    Runner(small_cfg, db, Deps(small_cfg, dry_run=True, db=db)).run()
    assert db.counts_by_status() == {"exported": 2}


def test_cli_end_to_end(tmp_path):
    cli = CliRunner()
    base = ["--config", str(REPO / "config.yaml"), "--products", str(FIXTURES / "products_small.yaml"),
            "--work-dir", str(tmp_path / "w")]
    r = cli.invoke(app, [*base, "init"])
    assert r.exit_code == 0, r.output
    r = cli.invoke(app, [*base, "run", "--niche", "cozy mushroom cottage", "--n", "2", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "exported" not in r.output or "errors=0" in r.output
    r = cli.invoke(app, [*base, "validate"])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output
    r = cli.invoke(app, [*base, "status"])
    assert "exported=2" in r.output
    r = cli.invoke(app, [*base, "zip"])
    assert r.exit_code == 0 and ".zip" in r.output
    # rerunning the same command does not create new designs (nothing left to resume -> new run)
    r = cli.invoke(app, [*base, "run", "--niche", "pikachu", "--n", "1", "--dry-run"])
    assert r.exit_code == 2 and "safety" in r.output


def test_cli_manual_review_flow(tmp_path):
    cli = CliRunner()
    base = ["--config", str(REPO / "config.yaml"), "--products", str(FIXTURES / "products_small.yaml"),
            "--work-dir", str(tmp_path / "w")]
    r = cli.invoke(app, [*base, "run", "--niche", "sleepy owls", "--n", "2", "--dry-run", "--review", "manual"])
    assert "awaiting_review=2" in r.output, r.output
    r = cli.invoke(app, [*base, "review"])
    assert "2 design(s) awaiting review" in r.output
    r = cli.invoke(app, [*base, "review", "--approve", "sleepy-owls-01", "--reject", "sleepy-owls-02"])
    assert "approve -> approved" in r.output and "reject -> rejected" in r.output
    r = cli.invoke(app, [*base, "finish"])
    assert r.exit_code == 0 and "export=1" in r.output, r.output


def test_gallery_builds(small_cfg):
    pytest.importorskip("ipywidgets")
    from pipeline.stages.review import ReviewGallery
    cfg = small_cfg.with_overrides({"review.mode": "manual"})
    runner, db = _runner(cfg)
    rid = db.create_run("x", False, 2, True)
    runner.run(PromptRequest(niche="sleepy owls", n=2), rid)
    box = ReviewGallery(db, cfg).build()
    grid = box.children[2]
    assert len(grid.children) == 2
    card = grid.children[0]
    approve_btn = card.children[2].children[0]
    approve_btn.click()
    assert db.counts_by_status() == {"approved": 1, "awaiting_review": 1}
    box.children[1].click()                       # approve all remaining
    assert db.counts_by_status() == {"approved": 2}
