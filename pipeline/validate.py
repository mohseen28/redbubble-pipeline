"""Validate an output run folder against products.yaml (used by `pipeline validate`)."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from pipeline.config import AppConfig
from pipeline.stages.export import MANIFEST_COLUMNS, file_name, product_dir
from pipeline.utils.images import allow_large_images

NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*_[a-z0-9]+(?:-[a-z0-9]+)*_(\d+)x(\d+)\.(png|jpg)$")
SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{2,}$")
_FORMATS = {"png": "PNG", "jpg": "JPEG"}


@dataclass
class Report:
    run_dir: Path
    designs: int = 0
    files_checked: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.designs > 0

    def summary(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        return (f"{state} {self.run_dir.name}: {self.designs} designs, {self.files_checked} files, "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")


def validate_run(run_dir: str | Path, cfg: AppConfig) -> Report:
    run_dir = Path(run_dir)
    rep = Report(run_dir)
    if not run_dir.is_dir():
        rep.errors.append(f"run folder not found: {run_dir}")
        return rep
    manifest = run_dir / "manifest.csv"
    rows: dict[str, dict[str, str]] = {}
    if not manifest.is_file():
        rep.errors.append("manifest.csv missing")
    else:
        with manifest.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames != MANIFEST_COLUMNS:
                rep.errors.append(f"manifest columns {reader.fieldnames} != {MANIFEST_COLUMNS}")
            for r in reader:
                rows[r.get("path", "")] = r
        for rel in rows:
            if not (run_dir / rel).is_file():
                rep.errors.append(f"manifest lists missing file {rel}")

    e = cfg.s.export
    allow_large_images(e.max_dim_px)
    max_bytes = e.max_file_mb * 1024 * 1024
    design_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "design.json").is_file())
    rep.designs = len(design_dirs)
    if not design_dirs:
        rep.errors.append("no design folders (with design.json) found")
    for ddir in design_dirs:
        slug = ddir.name
        if not SLUG_RE.match(slug) or len(slug.rsplit("-", 1)[0]) > 40:
            rep.errors.append(f"{slug}: slug does not match kebab-case (max 40) + 2-digit index")
        masters = list((ddir / "master").glob(f"{slug}_master_*x*.png"))
        if len(masters) != 1:
            rep.errors.append(f"{slug}: expected exactly one master PNG, found {len(masters)}")
        else:
            with Image.open(masters[0]) as im:
                if im.mode != "RGBA":
                    rep.errors.append(f"{slug}: master is {im.mode}, expected RGBA")
        expected: set[Path] = {masters[0]} if len(masters) == 1 else set()
        for spec in cfg.products:
            path = product_dir(ddir, spec, e.layout) / file_name(slug, spec)
            expected.add(path)
            rel = path.relative_to(run_dir).as_posix()
            if not path.is_file():
                rep.errors.append(f"{slug}: missing product {spec.key} ({rel})")
                continue
            rep.files_checked += 1
            m = NAME_RE.match(path.name)
            if not m:
                rep.errors.append(f"{rel}: name does not match {{slug}}_{{product-key}}_{{W}}x{{H}}.{{png|jpg}}")
            size = path.stat().st_size
            if size > max_bytes:
                rep.errors.append(f"{rel}: {size / 2**20:.1f} MB exceeds {e.max_file_mb} MB")
            with Image.open(path) as im:
                if im.size != (spec.canvas_w, spec.canvas_h):
                    rep.errors.append(f"{rel}: size {im.size} != {(spec.canvas_w, spec.canvas_h)}")
                if max(im.size) > e.max_dim_px:
                    rep.errors.append(f"{rel}: side exceeds {e.max_dim_px}px")
                if im.format != _FORMATS[spec.format]:
                    rep.errors.append(f"{rel}: format {im.format} != {_FORMATS[spec.format]}")
                want_mode = "RGBA" if spec.transparent else "RGB"
                if im.mode != want_mode:
                    rep.errors.append(f"{rel}: mode {im.mode} != {want_mode}")
                if not im.info.get("icc_profile"):
                    rep.warnings.append(f"{rel}: no embedded ICC profile")
            row = rows.get(rel)
            if manifest.is_file():
                if row is None:
                    rep.errors.append(f"{rel}: not listed in manifest.csv")
                elif (row["product"] != spec.key or int(row["width_px"]) != spec.canvas_w
                      or int(row["height_px"]) != spec.canvas_h or row["format"] != spec.format
                      or int(row["bytes"]) != size):
                    rep.errors.append(f"{rel}: manifest row does not match the file")
                elif row["low_res"] == "True":
                    rep.warnings.append(f"{rel}: low_res (scale_factor {row['scale_factor']})")
        for extra in sorted(p for p in ddir.rglob("*") if p.suffix in (".png", ".jpg") and p not in expected):
            rep.warnings.append(f"{extra.relative_to(run_dir).as_posix()}: not an expected product file")
    return rep


def validate_output(cfg: AppConfig, run_id: str | None = None) -> list[Report]:
    out = cfg.output_dir
    if run_id:
        return [validate_run(out / run_id, cfg)]
    runs = sorted(p for p in out.iterdir() if p.is_dir()) if out.is_dir() else []
    return [validate_run(p, cfg) for p in runs] or [Report(out, errors=["no run folders in output_dir"])]
