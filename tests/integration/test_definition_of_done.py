"""Spec section 12 definition of done: 3 designs x all 60 products, dry-run on CPU, validate passes.

Slow (~75 s on one CPU core): run with `pytest -m slow`; deselect with `-m "not slow"`.
"""

from __future__ import annotations

import warnings

import pytest
from typer.testing import CliRunner

from pipeline.cli import app
from tests.conftest import REPO


@pytest.mark.slow
def test_dod_60_products_three_designs(tmp_path):
    cli = CliRunner()
    base = ["--config", str(REPO / "config.yaml"), "--work-dir", str(tmp_path / "w")]
    r = cli.invoke(app, [*base, "run", "--niche", "cozy mushroom cottage", "--n", "3", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "export=3 errors=0" in r.output
    with warnings.catch_warnings():
        warnings.simplefilter("error")             # no DecompressionBombWarning on 98 MP posters
        r = cli.invoke(app, [*base, "validate"])
    assert r.exit_code == 0, r.output
    assert "PASS" in r.output and "3 designs, 180 files, 0 errors" in r.output
    run_dir = next((tmp_path / "w" / "output").iterdir())
    folders = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    assert folders == ["cozy-mushroom-cottage-01", "cozy-mushroom-cottage-02", "cozy-mushroom-cottage-03"]
