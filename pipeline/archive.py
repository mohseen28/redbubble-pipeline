"""`pipeline zip`: one archive per run, PNG/JPEG stored (already compressed)."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from pipeline.config import AppConfig

_STORED = {".png", ".jpg", ".jpeg"}


def zip_run(cfg: AppConfig, run_id: str) -> Path:
    run_dir = cfg.output_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"no output folder for run {run_id}: {run_dir}")
    files = sorted(p for p in run_dir.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    free = shutil.disk_usage(cfg.output_dir).free
    if free < total * 1.05 + 50 * 2**20:
        raise OSError(f"not enough disk to zip {run_id}: need ~{total / 2**30:.1f} GB, "
                      f"have {free / 2**30:.1f} GB free")
    dest = cfg.output_dir / f"{run_id}.zip"
    tmp = dest.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as zf:
        for p in files:
            ctype = zipfile.ZIP_STORED if p.suffix.lower() in _STORED else zipfile.ZIP_DEFLATED
            zf.write(p, arcname=(Path(run_id) / p.relative_to(run_dir)).as_posix(), compress_type=ctype)
    tmp.replace(dest)
    return dest
