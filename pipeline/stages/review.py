"""Stage 5: review (awaiting_review -> approved | rejected | new via regenerate).

manual: decisions come from the notebook gallery (or `pipeline review --approve ...`)
        and are written to SQLite + events immediately, so they survive a kernel restart.
auto:   every design that passed QA is approved (dry-run and tests).
"""

from __future__ import annotations

import html
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.config import AppConfig
from pipeline.db import DB
from pipeline.schemas import Decision, DesignState, Status
from pipeline.utils.images import load_rgba, on_checkerboard, png_bytes

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "review"
INPUT_STATUS: Status = "awaiting_review"


def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    """Auto mode only: approve. (The runner skips this stage in manual mode.)"""
    return state.patch(status="approved", decision="approve")


def decide(state: DesignState, decision: Decision, cfg: AppConfig) -> DesignState:
    """Pure-ish transition for one decision (moves files on reject)."""
    if state.status != INPUT_STATUS:
        raise ValueError(f"{state.slug} is {state.status}, not awaiting_review")
    if decision == "approve":
        return state.patch(status="approved", decision="approve")
    if decision == "regenerate":
        return state.patch(status="new", decision="regenerate", regen_count=state.regen_count + 1,
                           qa_retries=0, seed=None, qa=None)
    if decision == "reject":
        dest = cfg.dir("rejected") / state.design_id
        dest.mkdir(parents=True, exist_ok=True)
        raw_dir = cfg.dir("raw") / state.design_id
        new_raw = state.raw_image
        if raw_dir.is_dir():
            shutil.move(str(raw_dir), str(dest / "raw"))
            if state.raw_image:
                new_raw = str(dest / "raw" / Path(state.raw_image).name)
        new_cut = state.cutout_image
        if state.cutout_image and Path(state.cutout_image).is_file():
            new_cut = str(shutil.move(state.cutout_image, str(dest / Path(state.cutout_image).name)))
        return state.patch(status="rejected", decision="reject", raw_image=new_raw, cutout_image=new_cut)
    raise ValueError(f"unknown decision {decision!r}")


def apply_decision(db: DB, cfg: AppConfig, id_or_slug: str, decision: Decision) -> DesignState:
    """Persist one decision immediately (DB row + events log)."""
    state = db.find(id_or_slug)
    new = decide(state, decision, cfg)
    db.save(new)
    if decision in ("reject", "regenerate"):
        db.delete_embedding(state.design_id)
    db.log_event(state.design_id, STAGE, decision, {"from": state.status, "to": new.status})
    return new


# ------------------------------------------------------------------ gallery
def _card_html(s: DesignState) -> str:
    b = s.brief
    qa = s.qa
    notes = "; ".join(qa.reasons) if qa and qa.reasons else "passed"
    sim = ""
    if qa and qa.similarity_matches:
        sim = f"<br><b>closest:</b> {float(qa.similarity_matches[0]['score']):.2f}"
    return (f"<div style='font-size:12px;line-height:1.35;max-width:380px'>"
            f"<b>{html.escape(s.slug)}</b><br>"
            f"<b>niche:</b> {html.escape(b.niche if b else '')}<br>"
            f"<b>concept:</b> {html.escape(b.concept if b else '')}<br>"
            f"<b>style:</b> {html.escape(b.style if b else '')} &nbsp; <b>seed:</b> {s.seed}<br>"
            f"<b>QA:</b> {html.escape(notes)} (retries {s.qa_retries}, regenerated {s.regen_count}){sim}</div>")


class ReviewGallery:
    """ipywidgets gallery: each cutout on a checkerboard with Approve / Reject / Regenerate."""

    def __init__(self, db: DB, cfg: AppConfig, thumb: int = 360, columns: int = 3) -> None:
        self.db, self.cfg, self.thumb, self.columns = db, cfg, thumb, columns

    def build(self) -> Any:
        import ipywidgets as w

        designs = self.db.by_status(INPUT_STATUS)
        header = w.HTML(f"<h3>{len(designs)} design(s) awaiting review</h3>"
                        "<p>Decisions are saved immediately. Then run <code>pipeline finish</code> "
                        "(and <code>pipeline run</code> again for regenerated designs).</p>")
        cards = [self._card(s, w) for s in designs]
        grid = w.GridBox(cards, layout=w.Layout(grid_template_columns=f"repeat({self.columns}, {self.thumb + 30}px)",
                                                grid_gap="12px"))
        bulk = w.Button(description="Approve all remaining", button_style="success")

        def approve_all(_: Any) -> None:
            for s in self.db.by_status(INPUT_STATUS):
                apply_decision(self.db, self.cfg, s.design_id, "approve")
            header.value = "<h3>All remaining designs approved.</h3>"

        bulk.on_click(approve_all)
        return w.VBox([header, bulk, grid])

    def _card(self, s: DesignState, w: Any) -> Any:
        img_bytes = b""
        if s.cutout_image and Path(s.cutout_image).is_file():
            img_bytes = png_bytes(on_checkerboard(load_rgba(s.cutout_image), self.thumb))
        image = w.Image(value=img_bytes, format="png", width=self.thumb)
        status = w.HTML("<i>pending</i>")
        buttons: list[Any] = []
        choices: tuple[tuple[str, Decision, str], ...] = (
            ("Approve", "approve", "success"), ("Reject", "reject", "danger"),
            ("Regenerate", "regenerate", "warning"))
        for label, decision, style in choices:
            btn = w.Button(description=label, button_style=style, layout=w.Layout(width="110px"))

            def on_click(_: Any, d: Decision = decision) -> None:
                try:
                    new = apply_decision(self.db, self.cfg, s.design_id, d)
                    status.value = f"<b>{new.status}</b>"
                    for b in buttons:
                        b.disabled = True
                except (ValueError, KeyError) as exc:
                    status.value = f"<span style='color:red'>{html.escape(str(exc))}</span>"

            btn.on_click(on_click)
            buttons.append(btn)
        return w.VBox([image, w.HTML(_card_html(s)), w.HBox(buttons), status],
                      layout=w.Layout(border="1px solid #ccc", padding="6px"))


def show_gallery(db: DB, cfg: AppConfig) -> Any:
    from IPython.display import display

    gallery = ReviewGallery(db, cfg).build()
    display(gallery)  # type: ignore[no-untyped-call]
    return gallery
