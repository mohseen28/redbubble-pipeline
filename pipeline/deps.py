"""Dependency container: builds model backends lazily, once per process.

Stage batching means each model is created on first use by its stage and can be
released afterwards. Tests inject fakes through the constructor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipeline.config import AppConfig
from pipeline.utils.gpu import Layout, free_vram
from pipeline.utils.logging import get_logger
from pipeline.utils.text_safety import Blocklist, load_blocklist

if TYPE_CHECKING:
    from pipeline.db import DB
    from pipeline.imagegen.base import ImageBackend
    from pipeline.llm.base import LLMClient
    from pipeline.stages.cutout import Cutter
    from pipeline.stages.qa import Embedder, VisionChecker
    from pipeline.stages.upscale import Upscaler

log = get_logger("deps")


class Deps:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        dry_run: bool = False,
        db: DB | None = None,
        layout: Layout = "cpu",
        llm: LLMClient | None = None,
        imagegen: ImageBackend | None = None,
        cutter: Cutter | None = None,
        upscaler: Upscaler | None = None,
        embedder: Embedder | None = None,
        vision: VisionChecker | None = None,
    ) -> None:
        self.cfg = cfg
        self.dry_run = dry_run
        self.db = db
        self.layout: Layout = layout
        self._llm = llm
        self._imagegen = imagegen
        self._cutter = cutter
        self._upscaler_override = upscaler
        self._upscalers: dict[str, Upscaler] = {}
        self._embedder = embedder
        self._vision = vision
        self._blocklist: Blocklist | None = None
        self.ref_embeddings: list[tuple[str, Any]] | None = None   # filled lazily by the QA stage

    # ---------------------------------------------------------------- text
    @property
    def blocklist(self) -> Blocklist:
        if self._blocklist is None:
            self._blocklist = load_blocklist(self.cfg.blocklist_dir)
        return self._blocklist

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            if self.dry_run:
                from pipeline.llm.dry_run import DryRunLLM

                self._llm = DryRunLLM()
            else:
                from pipeline.llm.ollama_client import OllamaClient

                keep_alive = "0" if self.layout == "single" else self.cfg.s.llm.keep_alive
                self._llm = OllamaClient.from_config(self.cfg.s.llm, keep_alive=keep_alive)
        return self._llm

    # --------------------------------------------------------------- images
    @property
    def imagegen(self) -> ImageBackend:
        if self._imagegen is None:
            backend = "dry_run" if self.dry_run else self.cfg.s.imagegen.backend
            if backend == "dry_run":
                from pipeline.imagegen.dry_run import DryRunBackend

                self._imagegen = DryRunBackend()
            elif backend == "sdxl":
                from pipeline.imagegen.sdxl import SDXLBackend

                self._imagegen = SDXLBackend(self.cfg.s.imagegen, cpu_offload=self.cfg.s.imagegen.cpu_offload)
            else:
                from pipeline.imagegen.flux_schnell import FluxSchnellBackend

                self._imagegen = FluxSchnellBackend(self.cfg.s.imagegen)
        return self._imagegen

    @property
    def cutter(self) -> Cutter:
        if self._cutter is None:
            from pipeline.stages.cutout import ColorKeyCutter, RembgCutter

            if self.dry_run:
                self._cutter = ColorKeyCutter(tolerance=self.cfg.s.qa.bg_tolerance)
            else:
                self._cutter = RembgCutter(self.cfg.s.cutout.model, self.cfg.s.cutout.alpha_matting)
        return self._cutter

    def upscaler(self, model: str) -> Upscaler:
        if self._upscaler_override is not None:
            return self._upscaler_override
        if model not in self._upscalers:
            from pipeline.stages.upscale import LanczosUpscaler, RealESRGANUpscaler

            if self.dry_run:
                self._upscalers[model] = LanczosUpscaler(scale=4)
            else:
                u = self.cfg.s.upscale
                self._upscalers[model] = RealESRGANUpscaler(
                    model, self.cfg.models_dir, tile=u.tile, tile_pad=u.tile_pad, device=u.device)
        return self._upscalers[model]

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            from pipeline.stages.qa import ClipEmbedder, ThumbnailEmbedder

            if self.dry_run:
                self._embedder = ThumbnailEmbedder()
            else:
                self._embedder = ClipEmbedder(self.cfg.s.qa.clip_model, self.cfg.s.qa.clip_pretrained)
        return self._embedder

    @property
    def vision(self) -> VisionChecker | None:
        if not self.cfg.s.qa.vision_qa.enabled:
            return None
        if self._vision is None:
            from pipeline.stages.qa import OllamaVisionChecker

            self._vision = OllamaVisionChecker(self.llm, self.cfg.s.qa.vision_qa.model)
        return self._vision

    # ------------------------------------------------------------- lifecycle
    def release(self, *names: str) -> None:
        """Unload backends to free (V)RAM between stage batches."""
        for name in names:
            if name == "llm" and self._llm is not None:
                try:
                    self._llm.unload()
                except Exception as exc:  # unloading is best-effort
                    log.warning("could not unload LLM: %s", exc)
            elif name == "imagegen" and self._imagegen is not None:
                self._imagegen.unload()
                self._imagegen = None
            elif name == "upscale":
                self._upscalers.clear()
            elif name == "cutout":
                self._cutter = None
            elif name == "embedder":
                self._embedder = None
                self.ref_embeddings = None
        free_vram()

    # ----------------------------------------------------------------- paths
    def run_dir(self, design_id: str) -> Path:
        if self.db is None:
            raise RuntimeError("Deps.db is required to resolve the run folder")
        return self.cfg.output_dir / self.db.run_id_of(design_id)

    def model_names(self) -> dict[str, Any]:
        """Model identifiers for design.json, derived from config (never loads a model)."""
        s = self.cfg.s
        backend = "dry_run" if self.dry_run else s.imagegen.backend
        image_model = {"dry_run": "dry-run-placeholder", "sdxl": s.imagegen.sdxl_repo}.get(
            backend, f"{s.imagegen.flux_repo} ({s.imagegen.quantization})")
        return {
            "text_model": "dry-run-stub" if self.dry_run else s.llm.text_model,
            "image_backend": backend,
            "image_model": image_model,
            "cutout": "colorkey-dry-run" if self.dry_run else s.cutout.model,
            "upscale": "lanczos-dry-run" if self.dry_run else [s.upscale.model, s.upscale.general_model],
            "clip": None if not s.qa.clip_dedupe else
            ("thumbnail-dry-run" if self.dry_run else f"{s.qa.clip_model}/{s.qa.clip_pretrained}"),
            "vision_qa": s.qa.vision_qa.model if s.qa.vision_qa.enabled else None,
        }
