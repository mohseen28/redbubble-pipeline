"""Stage 7: export one correctly sized file per Redbubble product (mastered -> exported)."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from pipeline.config import AppConfig, ProductSpec
from pipeline.schemas import DesignState, ExportRecord, utcnow
from pipeline.utils.images import allow_large_images, alpha_bbox, hex_to_rgb, load_rgba, save_jpg, save_png
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "export"
INPUT_STATUS = "mastered"
OUTPUT_STATUS = "exported"
MANIFEST_COLUMNS = ["design_id", "slug", "category", "product", "width_px", "height_px", "format",
                    "transparent", "scale_factor", "low_res", "bytes", "path"]
log = get_logger("export")


class ExportError(RuntimeError):
    """A hard export limit was violated."""


@dataclass(frozen=True)
class Placement:
    width: int
    height: int
    x: int
    y: int
    scale_factor: float


def placement(trim_w: int, trim_h: int, canvas_w: int, canvas_h: int,
              margin_pct: float, anchor: str) -> Placement:
    """Contain-fit the trimmed design inside the canvas minus margin_pct on every side."""
    if trim_w <= 0 or trim_h <= 0:
        raise ValueError("empty design")
    mx = round(canvas_w * margin_pct / 100.0)
    my = round(canvas_h * margin_pct / 100.0)
    inner_w, inner_h = canvas_w - 2 * mx, canvas_h - 2 * my
    if inner_w <= 0 or inner_h <= 0:
        raise ExportError(f"margin {margin_pct}% leaves no room on a {canvas_w}x{canvas_h} canvas")
    f = min(inner_w / trim_w, inner_h / trim_h)
    w = max(1, min(inner_w, round(trim_w * f)))
    h = max(1, min(inner_h, round(trim_h * f)))
    parts = anchor.split("_")
    vert = parts[0] if parts[0] in ("top", "bottom") else "center"
    horiz = parts[-1] if parts[-1] in ("left", "right") else "center"
    x = {"left": mx, "right": canvas_w - mx - w}.get(horiz, mx + (inner_w - w) // 2)
    y = {"top": my, "bottom": canvas_h - my - h}.get(vert, my + (inner_h - h) // 2)
    return Placement(w, h, x, y, round(f, 4))


def render(trimmed: Image.Image, spec: ProductSpec, bg_color: str) -> tuple[Image.Image, Placement]:
    W, H = spec.canvas_w, spec.canvas_h
    pl = placement(trimmed.width, trimmed.height, W, H, spec.margin_pct, spec.anchor)
    resized = trimmed.resize((pl.width, pl.height), Image.Resampling.LANCZOS)
    if spec.transparent:
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        canvas.paste(resized, (pl.x, pl.y))
    else:
        canvas = Image.new("RGB", (W, H), hex_to_rgb(bg_color))
        canvas.paste(resized.convert("RGB"), (pl.x, pl.y), mask=resized.getchannel("A"))
    return canvas, pl


def file_name(slug: str, spec: ProductSpec) -> str:
    return f"{slug}_{spec.file_key}_{spec.canvas_w}x{spec.canvas_h}.{spec.format}"


def product_dir(design_dir: Path, spec: ProductSpec, layout: str) -> Path:
    return design_dir / spec.category if layout == "by_category" else design_dir


def check_hard_limits(spec: ProductSpec, cfg: AppConfig) -> None:
    e = cfg.s.export
    if spec.format not in ("png", "jpg"):
        raise ExportError(f"{spec.key}: format {spec.format!r} is not PNG or JPEG")
    if max(spec.canvas_w, spec.canvas_h) > e.max_dim_px:
        raise ExportError(f"{spec.key}: {spec.canvas_w}x{spec.canvas_h} exceeds max_dim_px={e.max_dim_px}")


def write_manifest(run_dir: Path, design_id: str, rows: list[dict[str, Any]]) -> Path:
    """Replace this design's rows in run_dir/manifest.csv (idempotent on resume)."""
    path = run_dir / "manifest.csv"
    kept: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as fh:
            kept = [r for r in csv.DictReader(fh) if r.get("design_id") != design_id]
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
        writer.writerows(rows)
    tmp.replace(path)
    return path


def listing_text(listing: dict[str, Any]) -> str:
    tags = ", ".join(listing.get("tags", []))
    return (f"Title: {listing.get('title', '')}\n"
            f"Tags: {tags}\n"
            f"Description:\n{listing.get('description', '')}\n")


def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    if not state.master_image or state.brief is None:
        raise ValueError("export needs a master image and a brief")
    e = cfg.s.export
    allow_large_images(e.max_dim_px)
    products = cfg.products
    for spec in products:
        check_hard_limits(spec, cfg)

    run_dir = deps.run_dir(state.design_id)
    design_dir = run_dir / state.slug
    if design_dir.exists():           # re-export after an interrupted session
        shutil.rmtree(design_dir)
    (design_dir / "master").mkdir(parents=True, exist_ok=True)

    master_src = Path(state.master_image)
    master_dst = design_dir / "master" / master_src.name
    shutil.copyfile(master_src, master_dst)

    master = load_rgba(master_src)
    master_w, master_h = master.size
    bbox = alpha_bbox(master, e.alpha_bbox_threshold)
    if bbox is None:
        raise ExportError("master image is fully transparent")
    trimmed = master.crop(bbox)
    del master

    max_bytes = int(e.max_file_mb * 1024 * 1024)
    cache: dict[tuple[Any, ...], Path] = {}
    records: list[ExportRecord] = []
    rows: list[dict[str, Any]] = [{
        "design_id": state.design_id, "slug": state.slug, "category": "master", "product": "master",
        "width_px": master_w, "height_px": master_h, "format": "png", "transparent": True,
        "scale_factor": 1.0, "low_res": False, "bytes": master_dst.stat().st_size,
        "path": master_dst.relative_to(run_dir).as_posix(),
    }]
    for spec in products:
        out_dir = product_dir(design_dir, spec, e.layout)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / file_name(state.slug, spec)
        key = (spec.canvas_w, spec.canvas_h, spec.transparent, spec.format, spec.margin_pct, spec.anchor)
        if key in cache:
            shutil.copyfile(cache[key], path)
            pl = placement(trimmed.width, trimmed.height, spec.canvas_w, spec.canvas_h,
                           spec.margin_pct, spec.anchor)
        else:
            img, pl = render(trimmed, spec, state.brief.bg_color)
            if spec.format == "png":
                save_png(img, path, dpi=e.dpi)
            else:
                save_jpg(img, path, quality=e.jpeg_quality, dpi=e.dpi)
            del img
            cache[key] = path
        size = path.stat().st_size
        if size > max_bytes:
            path.unlink()
            raise ExportError(f"{spec.key}: file is {size / 2**20:.1f} MB > max_file_mb={e.max_file_mb}")
        low_res = pl.scale_factor > e.max_scale_factor
        if low_res:
            msg = (f"{spec.key}: needs {pl.scale_factor:.2f}x enlargement of the master "
                   f"(> max_scale_factor {e.max_scale_factor}); exported with low_res=true")
            log.warning("%s %s", state.slug, msg)
            if deps.db is not None:
                deps.db.log_event(state.design_id, STAGE, "warning", msg)
        rec = ExportRecord(
            product=spec.key, category=spec.category, width_px=spec.canvas_w, height_px=spec.canvas_h,
            fmt=spec.format, transparent=spec.transparent, scale_factor=pl.scale_factor, low_res=low_res,
            path=path.relative_to(run_dir).as_posix(), bytes=size,
        )
        records.append(rec)
        rows.append({
            "design_id": state.design_id, "slug": state.slug, "category": rec.category, "product": rec.product,
            "width_px": rec.width_px, "height_px": rec.height_px, "format": rec.fmt,
            "transparent": rec.transparent, "scale_factor": rec.scale_factor, "low_res": rec.low_res,
            "bytes": rec.bytes, "path": rec.path,
        })

    info = {
        "design_id": state.design_id,
        "slug": state.slug,
        "run_id": run_dir.name,
        "brief": state.brief.model_dump(),
        "seed": state.seed,
        "models": deps.model_names(),
        "qa": state.qa.model_dump() if state.qa else None,
        "qa_retries": state.qa_retries,
        "regen_count": state.regen_count,
        "created_at": state.created_at.isoformat(),
        "exported_at": utcnow().isoformat(),
        "products_exported": len(records),
        "low_res_products": [r.product for r in records if r.low_res],
        "notice": "QA is advisory and is not a legal guarantee of originality. Check Redbubble policies "
                  "on AI content, trademarks and image requirements before any upload.",
    }
    (design_dir / "design.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    if cfg.s.metadata.enabled and state.listing:
        (design_dir / "listing.txt").write_text(listing_text(state.listing), encoding="utf-8")
    write_manifest(run_dir, state.design_id, rows)
    return state.patch(exports=[r.model_dump() for r in records], status=OUTPUT_STATUS)
