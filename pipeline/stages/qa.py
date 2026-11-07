"""Stage 3: QA (generated -> qa_passed | new (retry with a new seed) | qa_failed).

QA is advisory. It is NOT a legal guarantee of originality.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
from PIL import Image

from pipeline.config import AppConfig
from pipeline.llm.base import LLMClient, load_prompt
from pipeline.schemas import DesignState, QAReport, VisionQAResult
from pipeline.utils.gpu import torch_device
from pipeline.utils.images import estimate_bg_color, load_rgb, png_bytes, subject_mask
from pipeline.utils.logging import get_logger

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "qa"
INPUT_STATUS = "generated"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
log = get_logger("qa")


# ------------------------------------------------------------- embedders
class Embedder(Protocol):
    name: str

    def embed(self, img: Image.Image) -> np.ndarray:
        """Return an L2-normalised 1-D float32 vector."""
        ...


class ThumbnailEmbedder:
    """Dry-run stand-in for CLIP: mean-centred 32x32 colour thumbnail."""

    name = "thumbnail"

    def embed(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR), np.float32).ravel()
        arr = arr - arr.mean()
        n = float(np.linalg.norm(arr))
        out: np.ndarray = (arr / n if n else arr).astype(np.float32)
        return out


class ClipEmbedder:
    """OpenCLIP ViT-B/32 image embeddings."""

    name = "openclip"

    def __init__(self, model: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.model_name, self.pretrained = model, pretrained
        self._model: Any = None
        self._preprocess: Any = None
        self.device = torch_device("auto")

    def embed(self, img: Image.Image) -> np.ndarray:
        import open_clip
        import torch

        if self._model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained)
            self._model = model.to(self.device).eval()
            self._preprocess = preprocess
        x = self._preprocess(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            v = self._model.encode_image(x).float()
            v = v / v.norm(dim=-1, keepdim=True)
        out: np.ndarray = v.squeeze(0).cpu().numpy().astype(np.float32)
        return out


# ---------------------------------------------------------- vision checker
class VisionChecker(Protocol):
    def check(self, img: Image.Image) -> VisionQAResult: ...


class OllamaVisionChecker:
    """Qwen2.5-VL through Ollama (OLLAMA_MAX_LOADED_MODELS=1 swaps it in when needed)."""

    def __init__(self, llm: LLMClient, model: str) -> None:
        self.llm, self.model = llm, model

    def check(self, img: Image.Image) -> VisionQAResult:
        small = img.convert("RGB")
        small.thumbnail((768, 768))
        return self.llm.complete_json(load_prompt("qa_vision_rubric"), "Review this design.", VisionQAResult,
                                      model=self.model, images=[png_bytes(small)], temperature=0.0)


# ---------------------------------------------------------- checks
def technical_checks(img: Image.Image, cfg: AppConfig) -> list[str]:
    q = cfg.s.qa
    reasons: list[str] = []
    w, h = img.size
    if min(w, h) < q.min_resolution:
        reasons.append(f"resolution {w}x{h} below minimum {q.min_resolution}")
    rgb = np.asarray(img.convert("RGB"))
    bg = estimate_bg_color(rgb)
    mask = subject_mask(rgb, q.bg_tolerance, bg)
    frac = float(mask.mean())
    if frac < q.min_subject_fraction:
        reasons.append(f"mostly empty image (subject covers {frac:.1%})")
    elif frac > q.max_subject_fraction:
        reasons.append(f"no plain background (subject covers {frac:.1%})")
    b = q.border_px
    ring = np.concatenate([mask[:b].ravel(), mask[-b:].ravel(), mask[:, :b].ravel(), mask[:, -b:].ravel()])
    if float(ring.mean()) > q.max_border_fraction:
        reasons.append(f"subject touches the border ({float(ring.mean()):.1%} of edge pixels)")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    noise = float(np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)).mean())
    if noise > q.max_noise:
        reasons.append(f"extreme noise (score {noise:.1f} > {q.max_noise})")
    return reasons


def reference_embeddings(cfg: AppConfig, deps: Deps) -> list[tuple[str, np.ndarray]]:
    if deps.ref_embeddings is None:
        refs: list[tuple[str, np.ndarray]] = []
        rdir = cfg.s.qa.reference_dir
        if rdir:
            d = Path(rdir).expanduser()
            if not d.is_dir():
                log.warning("qa.reference_dir %s not found; skipping", d)
            else:
                for p in sorted(d.rglob("*")):
                    if p.suffix.lower() in IMAGE_EXTS:
                        try:
                            refs.append((str(p), deps.embedder.embed(load_rgb(p))))
                        except OSError as exc:
                            log.warning("cannot read reference %s: %s", p, exc)
        deps.ref_embeddings = refs
    return deps.ref_embeddings


def similarity_matches(vec: np.ndarray, state: DesignState, cfg: AppConfig,
                       deps: Deps) -> list[dict[str, Any]]:
    pool: list[tuple[str, np.ndarray]] = list(reference_embeddings(cfg, deps))
    if deps.db is not None:
        pool += [(path, v) for did, path, v in deps.db.embeddings()
                 if did != state.design_id and v.shape == vec.shape]
    scored: list[tuple[float, str]] = sorted(((round(float(vec @ v), 4), p) for p, v in pool), reverse=True)
    return [{"path": p, "score": s} for s, p in scored[:3]]


# ---------------------------------------------------------- stage
def run(state: DesignState, cfg: AppConfig, deps: Deps) -> DesignState:
    if not state.raw_image or state.brief is None:
        raise ValueError("QA needs a raw image and a brief")
    q = cfg.s.qa
    img = load_rgb(state.raw_image)
    reasons: list[str] = []

    b = state.brief
    verdict = deps.blocklist.check(" | ".join(filter(None, [b.concept, b.prompt, b.text_on_design])))
    if not verdict.ok:
        reasons += [f"text safety: {h}" for h in verdict.hits]
    reasons += technical_checks(img, cfg)

    matches: list[dict[str, Any]] = []
    vec: np.ndarray | None = None
    if q.clip_dedupe:
        vec = deps.embedder.embed(img)
        matches = similarity_matches(vec, state, cfg, deps)
        top = matches[0] if matches else None
        if top is not None:
            score = float(top["score"])
            if score > q.similarity_threshold:
                reasons.append(f"too similar to {top['path']} ({score:.3f} > {q.similarity_threshold})")

    ip_risk: float | None = None
    checker = deps.vision
    if checker is not None:
        vr = checker.check(img)
        ip_risk = vr.ip_risk
        if vr.severe_artifacts:
            reasons.append("vision QA: severe artifacts " + ", ".join(vr.artifacts))
        if vr.ip_risk > q.vision_qa.ip_risk_threshold:
            reasons.append(f"vision QA: ip_risk {vr.ip_risk:.2f} > {q.vision_qa.ip_risk_threshold}")
        if vr.ocr_text.strip():
            ocr = deps.blocklist.check(vr.ocr_text)
            if not ocr.ok:
                reasons += [f"on-image text: {h}" for h in ocr.hits]
            if not cfg.s.llm.allow_text_on_design:
                reasons.append(f"unexpected text on design: {vr.ocr_text.strip()[:60]!r}")

    report = QAReport(passed=not reasons, reasons=reasons, ip_risk=ip_risk, similarity_matches=matches)
    if report.passed:
        if vec is not None and deps.db is not None:
            deps.db.put_embedding(state.design_id, state.raw_image, vec)
        return state.patch(qa=report.model_dump(), status="qa_passed")
    if state.qa_retries < cfg.s.run.max_qa_retries:
        return state.patch(qa=report.model_dump(), qa_retries=state.qa_retries + 1, seed=None, status="new")
    return state.patch(qa=report.model_dump(), status="qa_failed")
