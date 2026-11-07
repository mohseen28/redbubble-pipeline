from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from pipeline.db import DB
from pipeline.schemas import DesignBrief, DesignState


def _state(i: int) -> DesignState:
    brief = DesignBrief(brief_id=f"b{i}", niche="cozy", concept="c", prompt="p", style="flat",
                        palette=["#112233", "abc"], bg_color="FFFFFF")
    return DesignState(design_id=f"d{i}", slug=f"cozy-{i:02d}", status="new", brief=brief)


def test_roundtrip_status_events_runs(tmp_path):
    db = DB(tmp_path / "p.db")
    run = db.create_run("cozy", False, 3, True, now=datetime(2026, 11, 6, tzinfo=UTC))
    assert run == "20261106_run01"
    assert db.create_run("x", False, 1, True, now=datetime(2026, 11, 6, tzinfo=UTC)) == "20261106_run02"
    for i in range(3):
        db.insert(_state(i), run)
    s = db.get("d1")
    assert s.brief is not None and s.brief.palette == ["#112233", "#aabbcc"] and s.brief.bg_color == "#ffffff"
    db.save(s.patch(status="generated", seed=42))
    assert [x.design_id for x in db.by_status("generated")] == ["d1"]
    assert db.counts_by_status(run) == {"new": 2, "generated": 1}
    assert db.find("cozy-02").design_id == "d2"
    assert db.run_id_of("d0") == run
    db.log_event("d1", "image", "start")
    db.log_event("d1", "image", "error", {"x": 1}, "tb")
    assert db.events("d1", action="error")[0].traceback == "tb"
    assert db.find_resumable_run("cozy", False, 3, True) is not None
    db.put_embedding("d1", "/p.png", np.ones(4))
    (did, _path, vec), = db.embeddings()
    assert did == "d1" and vec.shape == (4,)
    with pytest.raises(KeyError):
        db.save(_state(99))
    db.close()
    # persists across connections (kernel restart)
    db2 = DB(tmp_path / "p.db")
    assert db2.get("d1").seed == 42
