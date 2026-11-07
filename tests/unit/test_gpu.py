from __future__ import annotations

import pytest

from pipeline.utils import gpu


@pytest.mark.parametrize(("configured", "n", "want"), [
    ("auto", 2, "dual"), ("auto", 1, "single"), ("auto", 0, "cpu"),
    ("dual", 1, "single"), ("single", 2, "single"), ("dual", 2, "dual"),
])
def test_resolve_layout(configured, n, want):
    assert gpu.resolve_layout(configured, n) == want


@pytest.mark.parametrize(("before", "after"), [(None, "1"), ("0,1", "1"), ("", "1"), ("1", "1"), ("0", "0")])
def test_dual_pins_process_to_gpu1(monkeypatch, before, after):
    monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
    if before is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", before)
    gpu.configure_process_devices("dual")
    assert gpu.os.environ["CUDA_VISIBLE_DEVICES"] == after


def test_single_layout_leaves_env(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    gpu.configure_process_devices("single")
    assert "CUDA_VISIBLE_DEVICES" not in gpu.os.environ
