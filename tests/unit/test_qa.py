from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pipeline.db import DB
from pipeline.deps import Deps
from pipeline.schemas import VisionQAResult
from pipeline.stages import qa
from pipeline.stages.qa import ThumbnailEmbedder, technical_checks
from tests.helpers import rgb_on_bg, save, state


def test_technical_checks(cfg):
    assert technical_checks(rgb_on_bg(1024), cfg) == []
    empty = Image.new("RGB", (1024, 1024), (240, 240, 240))
    assert any("mostly empty" in r for r in technical_checks(empty, cfg))
    assert any("resolution" in r for r in technical_checks(rgb_on_bg(256), cfg))
    touching = Image.new("RGB", (1024, 1024), (240, 240, 240))
    ImageDraw.Draw(touching).rectangle((0, 300, 1023, 700), fill=(20, 20, 200))
    assert any("border" in r for r in technical_checks(touching, cfg))
    rng = np.random.default_rng(0)
    noisy = Image.fromarray(rng.integers(0, 255, (1024, 1024, 3), dtype=np.uint8))
    reasons = technical_checks(noisy, cfg)
    assert any("noise" in r for r in reasons)


def _qa_state(cfg, tmp_path, img=None, **kw):
    raw = save(img or rgb_on_bg(1024), tmp_path / "raw.png")
    return state(status="generated", raw_image=raw, **kw)


def test_pass_stores_embedding_and_dedupe_fails_copy(cfg, tmp_path):
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    rid = db.create_run("x", False, 2, True)
    deps = Deps(cfg, dry_run=True, db=db)
    s1 = _qa_state(cfg, tmp_path)
    db.insert(s1, rid)
    out = qa.run(s1, cfg, deps)
    assert out.status == "qa_passed" and out.qa.passed
    assert len(db.embeddings()) == 1
    s2 = state("d2", "cozy-mushroom-cottage-02", status="generated", raw_image=s1.raw_image)
    out2 = qa.run(s2, cfg, deps)
    assert out2.status == "new" and out2.qa_retries == 1 and out2.seed is None
    assert any("too similar" in r for r in out2.qa.reasons)
    assert out2.qa.similarity_matches[0]["score"] > 0.99


def test_reference_dir(tmp_path):
    from tests.conftest import make_cfg
    ref = tmp_path / "refs"
    save(rgb_on_bg(512), ref / "known.png")
    cfg = make_cfg(tmp_path, **{"qa.reference_dir": str(ref)})
    cfg.ensure_dirs()
    out = qa.run(_qa_state(cfg, tmp_path), cfg, Deps(cfg, dry_run=True))
    assert not out.qa.passed and "known.png" in out.qa.reasons[0]


def test_retries_exhausted_and_text_safety(cfg, tmp_path):
    from tests.helpers import brief
    b = brief().model_copy(update={"concept": "A Pikachu cottage"})
    s = _qa_state(cfg, tmp_path, brief=b, qa_retries=cfg.s.run.max_qa_retries)
    out = qa.run(s, cfg, Deps(cfg, dry_run=True))
    assert out.status == "qa_failed"
    assert any("pikachu" in r.lower() for r in out.qa.reasons)


class _Vision:
    def __init__(self, r):
        self.r = r

    def check(self, img):
        return self.r


def test_vision_qa(tmp_path):
    from tests.conftest import make_cfg
    cfg = make_cfg(tmp_path, **{"qa.vision_qa.enabled": True})
    cfg.ensure_dirs()
    bad = VisionQAResult(severe_artifacts=True, artifacts=["melted paw"], ocr_text="NIKE", ip_risk=0.8)
    out = qa.run(_qa_state(cfg, tmp_path), cfg, Deps(cfg, dry_run=True, vision=_Vision(bad)))
    text = " ".join(out.qa.reasons)
    assert "severe artifacts" in text and "ip_risk 0.80" in text and "nike" in text.lower()
    assert out.qa.ip_risk == 0.8
    good = VisionQAResult(severe_artifacts=False, ip_risk=0.1)
    out2 = qa.run(_qa_state(cfg, tmp_path), cfg, Deps(cfg, dry_run=True, vision=_Vision(good)))
    assert out2.qa.passed and out2.qa.ip_risk == 0.1


def test_thumbnail_embedder_normalised():
    v = ThumbnailEmbedder().embed(rgb_on_bg(128))
    assert abs(float(np.linalg.norm(v)) - 1) < 1e-5
