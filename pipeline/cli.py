"""Command line interface: pipeline init | run | review | finish | status | validate | zip | retry."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer

from pipeline.config import AppConfig, ConfigError, load_config, parse_override
from pipeline.db import DB
from pipeline.utils.logging import setup_logging
from pipeline.utils.text_safety import load_blocklist

STAGE_NAMES = ("prompt", "image", "qa", "cutout", "review", "upscale", "export")

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Redbubble AI design pipeline v1.1 (Kaggle edition).")


@dataclass
class Ctx:
    config: Path | None = None
    products: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def load(self, extra: dict[str, Any] | None = None) -> AppConfig:
        try:
            return load_config(self.config, self.products, {**self.overrides, **(extra or {})})
        except ConfigError as exc:
            typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from exc


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(help="Path to config.yaml")] = None,
    products: Annotated[Path | None, typer.Option(help="Path to products.yaml")] = None,
    work_dir: Annotated[str | None, typer.Option(help="Override runtime.work_dir")] = None,
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override a config key: key.path=value")] = None,
    log_level: Annotated[str, typer.Option(help="DEBUG, INFO, WARNING")] = "INFO",
) -> None:
    setup_logging(log_level.upper())
    overrides: dict[str, Any] = {}
    try:
        for item in set_ or []:
            k, v = parse_override(item)
            overrides[k] = v
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    if work_dir:
        overrides["runtime.work_dir"] = work_dir
    ctx.obj = Ctx(config=config, products=products, overrides=overrides)


@app.command()
def init(ctx: typer.Context) -> None:
    """Validate config + products, create work folders and the database."""
    cfg = ctx.obj.load()
    cfg.ensure_dirs()
    db = DB(cfg.db_path)
    db.close()
    bl = load_blocklist(cfg.blocklist_dir)
    cats: dict[str, int] = {}
    for p in cfg.products:
        cats[p.category] = cats.get(p.category, 0) + 1
    typer.echo(f"work_dir:   {cfg.work_dir}")
    typer.echo(f"database:   {cfg.db_path}")
    typer.echo(f"output_dir: {cfg.output_dir}")
    typer.echo(f"products:   {len(cfg.products)} enabled of {len(cfg.all_products)} "
               f"({', '.join(f'{k}={v}' for k, v in cats.items())})")
    typer.echo(f"blocklist:  {len(bl.entries)} entries")
    typer.secho("init OK", fg=typer.colors.GREEN)


@app.command()
def status(ctx: typer.Context,
           run_id: Annotated[str | None, typer.Option(help="Limit to one run")] = None,
           errors: Annotated[int, typer.Option(help="Show the last N error events")] = 5) -> None:
    """Show runs, design counts per status and recent errors."""
    cfg = ctx.obj.load()
    if not cfg.db_path.exists():
        typer.echo("No database yet. Run `pipeline init`.")
        raise typer.Exit(1)
    db = DB(cfg.db_path)
    try:
        for run in db.list_runs():
            if run_id and run.run_id != run_id:
                continue
            counts = db.counts_by_status(run.run_id)
            label = "auto-niche" if run.auto_niche else f"niche={run.niche!r}"
            typer.echo(f"{run.run_id}  {label} n={run.n}{' dry-run' if run.dry_run else ''}  "
                       + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        total = db.counts_by_status(run_id)
        typer.echo("total: " + ("  ".join(f"{k}={v}" for k, v in sorted(total.items())) or "no designs"))
        errs = db.events(action="error", limit=errors)
        if errs:
            typer.echo("recent errors:")
            for e in errs:
                typer.echo(f"  {e.ts} {e.design_id} [{e.stage}] {e.detail}")
    finally:
        db.close()


def _open(cfg: AppConfig) -> DB:
    cfg.ensure_dirs()
    return DB(cfg.db_path)


def _make_runner(cfg: AppConfig, db: DB, dry_run: bool) -> Any:
    from pipeline.deps import Deps
    from pipeline.runner import Runner
    from pipeline.utils.gpu import configure_process_devices, resolve_layout

    layout = "cpu" if dry_run else resolve_layout(cfg.s.runtime.gpu_layout)
    configure_process_devices(layout)          # before any torch import
    typer.echo(f"layout: {layout}{' (dry-run: no AI models, no network)' if dry_run else ''}")
    return Runner(cfg, db, Deps(cfg, dry_run=dry_run, db=db, layout=layout))


def _pending_all_dry(db: DB) -> bool:
    """True when every unfinished design belongs to a dry-run run (so finish needs no models)."""
    runs = {r.run_id: r for r in db.list_runs()}
    pending = db.by_status(["new", "generated", "qa_passed", "awaiting_review", "approved", "mastered"])
    return bool(pending) and all(runs[db.run_id_of(s.design_id)].dry_run for s in pending)


@app.command()
def run(
    ctx: typer.Context,
    niche: Annotated[str | None, typer.Option(help="Niche to design for")] = None,
    auto_niche: Annotated[bool, typer.Option("--auto-niche", help="Let the LLM propose niches")] = False,
    n: Annotated[int, typer.Option("--n", min=1, help="Number of designs")] = 10,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="CPU-only placeholder models, no network")] = False,
    until: Annotated[str, typer.Option(help="Stop after this stage")] = "export",
    review_mode: Annotated[str | None, typer.Option("--review", help="manual | auto (dry-run defaults to auto)")]
    = None,
    new_run: Annotated[bool, typer.Option("--new-run", help="Never resume; always start a new run")] = False,
) -> None:
    """Stages 1-4 (+ review gate). Without --niche/--auto-niche: continue existing designs only.

    In auto review mode the run continues through upscale and export.
    """
    if until not in STAGE_NAMES:
        raise typer.BadParameter(f"--until must be one of {STAGE_NAMES}")
    if niche and auto_niche:
        raise typer.BadParameter("use --niche or --auto-niche, not both")
    extra: dict[str, Any] = {}
    if review_mode:
        extra["review.mode"] = review_mode
    elif dry_run:
        extra["review.mode"] = "auto"
    cfg = ctx.obj.load(extra)
    db = _open(cfg)
    try:
        from pipeline.stages.prompt import PromptRequest

        req = None
        run_id = None
        if niche or auto_niche:
            req = PromptRequest(niche=niche, auto_niche=auto_niche, n=n)
            existing = None if new_run else db.find_resumable_run(niche, auto_niche, n, dry_run)
            if existing:
                run_id = existing.run_id
                typer.echo(f"resuming {run_id}")
            else:
                run_id = db.create_run(niche, auto_niche, n, dry_run)
                typer.echo(f"new run {run_id}")
        runner = _make_runner(cfg, db, dry_run)
        try:
            result = runner.run(req, run_id, until=until)
        except ValueError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(2) from exc
        typer.echo(result.summary())
        if result.awaiting_review and cfg.s.review.mode == "manual":
            typer.echo("Next: review in the notebook gallery (or `pipeline review`), then `pipeline finish`.")
    finally:
        db.close()


@app.command()
def review(
    ctx: typer.Context,
    approve: Annotated[list[str] | None, typer.Option(help="Design id or slug (repeatable)")] = None,
    reject: Annotated[list[str] | None, typer.Option(help="Design id or slug (repeatable)")] = None,
    regenerate: Annotated[list[str] | None, typer.Option(help="Design id or slug (repeatable)")] = None,
    approve_all: Annotated[bool, typer.Option("--approve-all")] = False,
) -> None:
    """List designs awaiting review, or record decisions. In a notebook use pipeline.notebook.review_gallery()."""
    from pipeline.stages.review import apply_decision

    cfg = ctx.obj.load()
    db = _open(cfg)
    try:
        decisions: list[tuple[str, Any]] = [(x, "approve") for x in approve or []]
        decisions += [(x, "reject") for x in reject or []]
        decisions += [(x, "regenerate") for x in regenerate or []]
        if approve_all:
            decisions += [(s.design_id, "approve") for s in db.by_status("awaiting_review")]
        for ident, decision in decisions:
            try:
                new = apply_decision(db, cfg, ident, decision)
                typer.echo(f"{new.slug}: {decision} -> {new.status}")
            except (KeyError, ValueError) as exc:
                typer.secho(f"{ident}: {exc}", fg=typer.colors.RED, err=True)
        if not decisions:
            waiting = db.by_status("awaiting_review")
            typer.echo(f"{len(waiting)} design(s) awaiting review:")
            for s in waiting:
                concept = s.brief.concept if s.brief else ""
                typer.echo(f"  {s.design_id}  {s.slug:<45} {concept[:70]}")
                typer.echo(f"      cutout: {s.cutout_image}")
    finally:
        db.close()


@app.command()
def finish(ctx: typer.Context,
           dry_run: Annotated[bool | None, typer.Option("--dry-run/--real",
                                                        help="Default: dry-run if all pending designs are")] = None,
           ) -> None:
    """Stages 6-7: upscale approved designs to masters and export every product size."""
    cfg = ctx.obj.load()
    db = _open(cfg)
    try:
        use_dry = _pending_all_dry(db) if dry_run is None else dry_run
        result = _make_runner(cfg, db, use_dry).finish()
        typer.echo(result.summary())
    finally:
        db.close()


@app.command()
def validate(ctx: typer.Context,
             run_id: Annotated[str | None, typer.Option(help="Validate one run (default: all)")] = None,
             max_errors: Annotated[int, typer.Option()] = 30) -> None:
    """Check every exported file against products.yaml (scripts/validate_exports.py)."""
    from pipeline.validate import validate_output

    cfg = ctx.obj.load()
    ok = True
    for rep in validate_output(cfg, run_id):
        typer.secho(rep.summary(), fg=typer.colors.GREEN if rep.ok else typer.colors.RED)
        for e in rep.errors[:max_errors]:
            typer.echo(f"  ERROR {e}")
        for w in rep.warnings[:10]:
            typer.echo(f"  warn  {w}")
        ok = ok and rep.ok
    if not ok:
        raise typer.Exit(1)


@app.command(name="zip")
def zip_cmd(ctx: typer.Context,
            run_id: Annotated[str | None, typer.Option(help="Run to archive (default: latest)")] = None,
            all_runs: Annotated[bool, typer.Option("--all")] = False) -> None:
    """Create one .zip per run in output_dir (download it from the Kaggle Output panel)."""
    from pipeline.archive import zip_run

    cfg = ctx.obj.load()
    db = _open(cfg)
    try:
        if all_runs:
            ids = [r.run_id for r in db.list_runs()]
        else:
            latest = db.latest_run()
            ids = [run_id] if run_id else ([latest.run_id] if latest else [])
    finally:
        db.close()
    if not ids:
        typer.echo("no runs to archive")
        raise typer.Exit(1)
    for rid in ids:
        try:
            path = zip_run(cfg, rid)
        except (FileNotFoundError, OSError) as exc:
            typer.secho(f"{rid}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from exc
        typer.echo(f"{path}  ({path.stat().st_size / 2**20:.1f} MB)")


@app.command()
def retry(ctx: typer.Context,
          design: Annotated[list[str] | None, typer.Option(help="Design id or slug (default: all errors)")] = None,
          ) -> None:
    """Reset designs in `error` to the stage that failed, so the next run/finish retries them."""
    from pipeline.runner import retry_errors

    cfg = ctx.obj.load()
    db = _open(cfg)
    try:
        for s in retry_errors(db, design):
            typer.echo(f"{s.slug}: -> {s.status}")
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    app()
