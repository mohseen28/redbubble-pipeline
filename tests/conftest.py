"""Shared fixtures. Every test runs on CPU without network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pipeline.config import AppConfig, load_config

REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def repo() -> Path:
    return REPO


def make_cfg(tmp_path: Path, products: str | Path | None = None, **overrides: Any) -> AppConfig:
    base: dict[str, Any] = {
        "runtime.work_dir": str(tmp_path / "work"),
        "runtime.gpu_layout": "single",
        "imagegen.backend": "dry_run",
        "review.mode": "auto",
        "run.min_free_disk_gb": 0,
    }
    base.update(overrides)
    return load_config(REPO / "config.yaml", products or REPO / "products.yaml", base)


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    """Full 60-product config with a temporary work dir."""
    return make_cfg(tmp_path)


@pytest.fixture
def small_cfg(tmp_path: Path) -> AppConfig:
    """A handful of small products across categories for fast end-to-end tests."""
    return make_cfg(tmp_path, FIXTURES / "products_small.yaml")
