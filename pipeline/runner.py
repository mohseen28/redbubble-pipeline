"""Stage-batched runner with SQLite state (resumable after a Kaggle session restart).

Each stage selects designs by input status, processes them one by one and writes
the new state after every design. Re-running skips finished work. A failure in
one design sets it to `error` (traceback in events) and the batch continues.
"""

from __future__ import annotations

import shutil
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from types import ModuleType
from typing import Literal

from pipeline.config import AppConfig
from pipeline.db import DB
from pipeline.deps import Deps
from pipeline.schemas import DesignState, Status
from pipeline.stages import cutout, export, image, prompt, qa, review, upscale
from pipeline.stages.prompt import PromptRequest
from pipeline.utils.gpu import free_vram
from pipeline.utils.logging import get_logger

log = get_logger("runner")
StageName = Literal["prompt", "image", "qa", "cutout", "review", "upscale", "export"]
STAGES: tuple[StageName, ...] = ("prompt", "image", "qa", "cutout", "review", "upscale", "export")
STAGE_INPUT: dict[str, Status] = {
    "image": "new", "qa": "generated", "cutout": "qa_passed", "review": "awaiting_review",
    "upscale": "approved", "export": "mastered",
}
_MODULES: dict[str, ModuleType] = {
    "image": image, "qa": qa, "cutout": cutout, "review": review, "upscale": upscale, "export": export,
}
_DISK_STAGES = {"image", "cutout", "upscale", "export"}


class StopRun(Exception):
    """Clean stop (disk or design cap). Designs keep their status; rerun to continue."""


@dataclass
class RunResult:
    run_id: str | None = None
    created: int = 0
    processed: dict[str, int] = field(default_factory=dict)
    errors: int = 0
    stopped: str | None = None
    awaiting_review: int = 0

    def summary(self) -> str:
        parts = [f"{k}={v}" for k, v in self.processed.items() if v]
        msg = f"run {self.run_id or '-'}: created={self.created} " + " ".join(parts) + f" errors={self.errors}"
        if self.awaiting_review:
            msg += f" awaiting_review={self.awaiting_review}"
        if self.stopped:
            msg += f"  STOPPED: {self.stopped}"
        return msg


class Runner:
    def __init__(self, cfg: AppConfig, db: DB, deps: Deps,
                 disk_free_bytes: Callable[[], int] | None = None) -> None:
        self.cfg, self.db, self.deps = cfg, db, deps
        self._disk_free = disk_free_bytes or (lambda: shutil.disk_usage(self.cfg.work_dir).free)
        self.generated_ids: set[str] = set()
        self.result = RunResult()

    # ----------------------------------------------------------------- guards
    def check_disk(self) -> None:
        free_gb = self._disk_free() / 2**30
        if free_gb < self.cfg.s.run.min_free_disk_gb:
            raise StopRun(f"free disk {free_gb:.1f} GB < min_free_disk_gb={self.cfg.s.run.min_free_disk_gb}. "
                          "Run `pipeline zip`, download, delete old output, then rerun.")

    def _check_design_cap(self, state: DesignState) -> None:
        if state.design_id in self.generated_ids:
            return
        cap = self.cfg.s.run.max_designs_per_run
        if len(self.generated_ids) >= cap:
            raise StopRun(f"max_designs_per_run={cap} reached; rerun to continue with the rest")
        self.generated_ids.add(state.design_id)

    # ---------------------------------------------------------- stage driving
    def _process_one(self, stage: str, state: DesignState) -> DesignState:
        mod = _MODULES[stage]
        self.db.log_event(state.design_id, stage, "start")
        try:
            new: DesignState = mod.run(state, self.cfg, self.deps)
        except StopRun:
            raise
        except Exception as exc:
            tb = traceback.format_exc()
            log.error("%s failed for %s: %s", stage, state.slug, exc)
            new = state.patch(status="error", error=f"{stage}: {exc}")
            self.db.save(new)
            self.db.log_event(state.design_id, stage, "error",
                              {"input_status": state.status, "error": str(exc)}, tb)
            self.result.errors += 1
            return new
        self.db.save(new)
        self.db.log_event(state.design_id, stage, "end", {"status": new.status})
        self.result.processed[stage] = self.result.processed.get(stage, 0) + 1
        return new

    def run_stage(self, stage: str) -> int:
        designs = self.db.by_status(STAGE_INPUT[stage])
        if not designs:
            return 0
        if stage == "review" and self.cfg.s.review.mode == "manual":
            self.result.awaiting_review = len(designs)
            log.info("%d design(s) awaiting manual review (notebook gallery or `pipeline review`)", len(designs))
            return 0
        log.info("stage %s: %d design(s)", stage, len(designs))
        batch = self.cfg.s.run.designs_per_batch
        for i, state in enumerate(designs):
            if stage in _DISK_STAGES:
                self.check_disk()
            if stage == "image":
                self._check_design_cap(state)
            self._process_one(stage, state)
            if stage in ("image", "upscale") and (i + 1) % batch == 0:
                free_vram()
        return len(designs)

    def image_qa_loop(self) -> None:
        """Stages 2-3, looping while QA sends designs back for a new seed."""
        for _ in range(self.cfg.s.run.max_qa_retries + 2):
            if not self.db.by_status("new"):
                break
            self.run_stage("image")
            self.run_stage("qa")
        else:
            self.run_stage("qa")

    # ------------------------------------------------------------- commands
    def create_designs(self, req: PromptRequest, run_id: str) -> int:
        self.db.log_event(None, "prompt", "start", {"run_id": run_id, "niche": req.niche, "n": req.n})
        req.existing_slugs = self.db.slugs()
        states = prompt.generate(req, self.cfg, self.deps)
        for s in states:
            self.db.insert(s, run_id)
            self.db.log_event(s.design_id, "prompt", "end", {"slug": s.slug, "run_id": run_id})
        self.db.mark_prompt_done(run_id)
        self.result.created = len(states)
        self.result.processed["prompt"] = len(states)
        return len(states)

    def run(self, req: PromptRequest | None = None, run_id: str | None = None,
            until: StageName = "export", stages: tuple[StageName, ...] = STAGES) -> RunResult:
        self.result.run_id = run_id
        stop_at = STAGES.index(until)
        wanted = [s for s in stages if STAGES.index(s) <= stop_at]
        try:
            if "prompt" in wanted and req is not None and run_id is not None:
                run = self.db.get_run(run_id)
                if run is not None and run.prompt_done:
                    log.info("stage prompt already done for %s; resuming", run_id)
                else:
                    self.create_designs(req, run_id)
            if "image" in wanted and self.db.by_status("new"):
                if self.deps.layout == "single" and not self.deps.dry_run and "prompt" in wanted:
                    self.deps.release("llm")          # free the GPU for FLUX
                if "qa" in wanted:
                    self.image_qa_loop()
                else:
                    self.run_stage("image")
            elif "qa" in wanted:
                self.run_stage("qa")
            if "cutout" in wanted:
                self.run_stage("cutout")
            if "review" in wanted:
                self.run_stage("review")
            if "upscale" in wanted and self.db.by_status("approved"):
                self.deps.release("imagegen", "embedder")
                self.run_stage("upscale")
            if "export" in wanted:
                self.run_stage("export")
        except StopRun as exc:
            self.result.stopped = str(exc)
            log.warning("stopping cleanly: %s", exc)
            self.db.log_event(None, "runner", "stop", str(exc))
        self.result.awaiting_review = len(self.db.by_status("awaiting_review"))
        return self.result

    def finish(self) -> RunResult:
        """Stages 6-7 for approved designs."""
        return self.run(stages=("upscale", "export"))


def retry_errors(db: DB, design_ids: list[str] | None = None) -> list[DesignState]:
    """Reset `error` designs to the input status of the stage that failed."""
    out = []
    for s in db.by_status("error"):
        if design_ids and s.design_id not in design_ids and s.slug not in design_ids:
            continue
        evs = db.events(s.design_id, action="error", limit=1)
        prev: Status = "new"
        if evs and evs[0].detail:
            import json

            prev = json.loads(evs[0].detail).get("input_status", "new")
        new = s.patch(status=prev, error=None)
        db.save(new)
        db.log_event(s.design_id, "runner", "retry", {"to": prev})
        out.append(new)
    return out
