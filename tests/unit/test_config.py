from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pipeline.config import ConfigError, load_config, parse_override, parse_products
from tests.conftest import REPO, make_cfg


def test_repo_config_loads_with_60_products(cfg):
    assert len(cfg.all_products) == 60
    assert len(cfg.products) == 60
    keys = {p.key for p in cfg.products}
    assert {"premium_tshirt", "poster_large", "sticker_xl", "cotton_tote"} <= keys
    poster = next(p for p in cfg.products if p.key == "poster_large")
    assert (poster.canvas_w, poster.canvas_h, poster.format, poster.transparent) == (8310, 11790, "jpg", False)
    tote = next(p for p in cfg.products if p.key == "cotton_tote")
    assert tote.transparent and tote.format == "png" and tote.margin_pct == 4 and tote.category == "bags"
    sticker = next(p for p in cfg.products if p.key == "sticker_small")
    assert sticker.margin_pct == 3 and sticker.file_key == "sticker-small"


def test_transparent_set_matches_decision_7(cfg):
    transparent = {p.category for p in cfg.products if p.transparent}
    assert transparent == {"apparel", "stickers", "bags"}
    assert [p.key for p in cfg.products if p.transparent and p.category == "bags"] == ["cotton_tote"]


def test_relative_output_dir_resolves_under_work_dir(cfg, tmp_path):
    assert cfg.output_dir == (tmp_path / "work").resolve() / "output"
    assert cfg.blocklist_dir == REPO / "assets/blocklist"


def _prod(tmp_path: Path, data: object) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"defaults": {}, "groups": {"g": {"a": [10]}}}, "size must be"),
        ({"defaults": {}, "groups": {"g": {"a": [10, -5]}}}, "size must be"),
        ({"defaults": {}, "groups": {"g": {"a": {"transparent": True}}}}, "needs 'size"),
        ({"defaults": {"format": "jpg"}, "groups": {"g": {"a": {"size": [5, 5], "transparent": True}}}},
         "transparent products must use format png"),
        ({"defaults": {"format": "gif"}, "groups": {"g": {"a": [5, 5]}}}, "format"),
        ({"defaults": {"colour": "red"}, "groups": {"g": {"a": [5, 5]}}}, "unknown keys"),
        ({"defaults": {"anchor": "middle"}, "groups": {"g": {"a": [5, 5]}}}, "anchor"),
        ({"defaults": {"margin_pct": 60}, "groups": {"g": {"a": [5, 5]}}}, "margin_pct"),
        ({"defaults": {}, "groups": {"g": {"a": [5, 5]}, "h": {"a": [6, 6]}}}, "duplicate"),
        ({"groups": {}}, "non-empty"),
        ([1, 2], "mapping"),
    ],
)
def test_bad_products_fail_fast(data, match):
    full = {"defaults": {"transparent": False, "format": "jpg", "margin_pct": 5, "anchor": "center", "scale": 1.0}}
    if isinstance(data, dict):
        merged = {**full, **data}
        merged["defaults"] = {**full["defaults"], **data.get("defaults", {})}
    else:
        merged = data
    with pytest.raises(ConfigError, match=match):
        parse_products(merged)


def test_enabled_false_on_product_and_group_and_only_groups(tmp_path):
    raw = yaml.safe_load((REPO / "products.yaml").read_text())
    raw["groups"]["stickers"]["_"]["enabled"] = False
    raw["groups"]["home"]["mug_tall"] = {"size": [2700, 1624], "enabled": False}
    path = _prod(tmp_path, raw)
    cfg = make_cfg(tmp_path, path)
    assert len(cfg.all_products) == 60
    assert len(cfg.products) == 60 - 4 - 1
    cfg2 = make_cfg(tmp_path, path, **{"export.only_groups": ["apparel", "home"]})
    assert {p.category for p in cfg2.products} == {"apparel", "home"}
    with pytest.raises(ConfigError, match="unknown groups"):
        make_cfg(tmp_path, path, **{"export.only_groups": ["nope"]})


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("run.max_qa_retries", -1, "max_qa_retries"),
        ("review.mode", "sometimes", "review.mode"),
        ("imagegen.backend", "midjourney", "imagegen.backend"),
        ("imagegen.height", 512, "square"),
        ("qa.similarity_threshold", 1.5, "similarity_threshold"),
        ("export.jpeg_quality", 0, "jpeg_quality"),
        ("llm.backend", "gemini", "llm.backend"),
        ("export.bogus_key", 1, "bogus_key"),
    ],
)
def test_bad_config_values_fail_fast(tmp_path, key, value, match):
    with pytest.raises(ConfigError, match=match):
        make_cfg(tmp_path, **{key: value})


def test_non_multiple_of_16(tmp_path):
    with pytest.raises(ConfigError, match="multiple of 16"):
        make_cfg(tmp_path, **{"imagegen.width": 1000, "imagegen.height": 1000})


def test_missing_and_invalid_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", REPO / "products.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad, REPO / "products.yaml")


def test_overrides_and_env(tmp_path, monkeypatch):
    assert parse_override("run.max_qa_retries=5") == ("run.max_qa_retries", 5)
    assert parse_override("export.only_groups=[apparel]") == ("export.only_groups", ["apparel"])
    with pytest.raises(ConfigError):
        parse_override("novalue")
    monkeypatch.setenv("PIPELINE_WORK_DIR", str(tmp_path / "envwork"))
    cfg = load_config(REPO / "config.yaml", REPO / "products.yaml")
    assert cfg.work_dir == (tmp_path / "envwork").resolve()
    cfg2 = cfg.with_overrides({"run.max_qa_retries": 7})
    assert cfg2.s.run.max_qa_retries == 7
