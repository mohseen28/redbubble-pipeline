"""Stage 1: generate DesignBriefs with the LLM and create DesignStates (status new)."""

from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

from pipeline.config import AppConfig
from pipeline.llm.base import LLMError, load_prompt
from pipeline.schemas import BriefBatch, BriefDraft, DesignBrief, DesignState, Listing, NicheList
from pipeline.utils.logging import get_logger
from pipeline.utils.text_safety import Blocklist

if TYPE_CHECKING:
    from pipeline.deps import Deps

STAGE = "prompt"
OUTPUT_STATUS = "new"
CONCEPT_DUP_THRESHOLD = 0.90
SLUG_BASE_MAX = 40
log = get_logger("prompt")

_COLOR_NAMES: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255), "cream": (250, 243, 224), "light grey": (215, 215, 215),
    "grey": (140, 140, 140), "charcoal": (55, 55, 60), "black": (15, 15, 15),
    "pale yellow": (252, 244, 190), "yellow": (240, 200, 40), "orange": (240, 140, 50),
    "peach": (250, 210, 180), "pink": (245, 180, 200), "red": (200, 40, 45), "burgundy": (110, 25, 40),
    "lavender": (215, 200, 240), "purple": (110, 60, 160), "light blue": (190, 220, 245),
    "blue": (50, 100, 200), "navy": (20, 35, 80), "mint": (200, 240, 220), "green": (60, 150, 70),
    "dark green": (25, 70, 40), "teal": (30, 130, 130), "beige": (225, 210, 185), "brown": (120, 80, 50),
}


@dataclass
class PromptRequest:
    niche: str | None = None
    auto_niche: bool = False
    n: int = 10
    existing_slugs: set[str] = field(default_factory=set)


# ------------------------------------------------------------------ helpers
def kebab(text: str, max_len: int = SLUG_BASE_MAX) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    out = ""
    for w in words:
        cand = f"{out}-{w}" if out else w
        if len(cand) > max_len:
            break
        out = cand
    return out or (words[0][:max_len] if words else "design")


def make_slug(base_text: str, existing: set[str]) -> str:
    base = kebab(base_text)
    i = 1
    while f"{base}-{i:02d}" in existing:
        i += 1
    return f"{base}-{i:02d}"


def trigram_vector(text: str, dim: int = 2048) -> np.ndarray:
    """Hashed character-trigram embedding (cheap, model-free near-duplicate check)."""
    t = f"  {re.sub(r'[^a-z0-9 ]+', ' ', text.lower())}  "
    counts = Counter(t[i:i + 3] for i in range(len(t) - 2))
    v = np.zeros(dim, np.float32)
    for gram, c in counts.items():
        v[hash_stable(gram) % dim] += c
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def hash_stable(s: str) -> int:
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def color_name(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    rgb = (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
    return min(_COLOR_NAMES, key=lambda n: sum((a - b) ** 2 for a, b in zip(_COLOR_NAMES[n], rgb, strict=True)))


def ensure_plain_background(prompt: str, bg_hex: str) -> str:
    low = prompt.lower()
    if "plain" in low and "background" in low:
        return prompt
    return f"{prompt.rstrip().rstrip('.')}. Isolated on a plain solid {color_name(bg_hex)} background."


def _log(deps: Deps, action: str, detail: Any) -> None:
    if deps.db is not None:
        deps.db.log_event(None, STAGE, action, detail)


# ------------------------------------------------------------------ LLM calls
def concept_system(cfg: AppConfig) -> str:
    if cfg.s.llm.allow_text_on_design:
        text_rule = "Text on the design is allowed but optional: at most 4 original words, no slogans."
        field_rule = "null, or at most 4 original words that appear in the design."
    else:
        text_rule = "No text, letters, numbers or logos anywhere in the design."
        field_rule = "always null."
    return load_prompt("concept_system", text_rule=text_rule, text_field_rule=field_rule)


def propose_niches(cfg: AppConfig, deps: Deps, count: int) -> list[str]:
    system = load_prompt("niche_generate", count=count, today=date.today().isoformat())
    res = deps.llm.complete_json(system, f"Propose {count} niches now.", NicheList,
                                 temperature=cfg.s.llm.temperature)
    niches: list[str] = []
    for n in res.niches:
        n = " ".join(n.split())[:80]
        verdict = deps.blocklist.check(n)
        if n and verdict.ok and n.lower() not in (x.lower() for x in niches):
            niches.append(n)
        elif not verdict.ok:
            _log(deps, "blocked", {"niche": n, "hits": verdict.hits})
    if not niches:
        raise LLMError("the LLM proposed no usable niches")
    return niches[:10]


def request_briefs(niche: str, k: int, cfg: AppConfig, deps: Deps, avoid: list[str]) -> list[BriefDraft]:
    avoid_txt = "\n".join(f"- {a}" for a in avoid[-15:]) or "- (none yet)"
    user = (f'Niche: "{niche}"\n'
            f"Return exactly {k} distinct design briefs for this niche as JSON.\n"
            f"Each must be clearly different from each other and from these existing concepts:\n{avoid_txt}")
    batch = deps.llm.complete_json(concept_system(cfg), user, BriefBatch, temperature=cfg.s.llm.temperature)
    return batch.briefs


def post_process(drafts: list[BriefDraft], niche: str, cfg: AppConfig, blocklist: Blocklist,
                 accepted_vecs: list[np.ndarray], deps: Deps | None = None) -> list[DesignBrief]:
    """Safety filter, text policy, plain-background phrase, near-duplicate removal."""
    out: list[DesignBrief] = []
    for d in drafts:
        text_on = d.text_on_design if cfg.s.llm.allow_text_on_design else None
        joined = " | ".join(filter(None, [d.concept, d.prompt, d.style, text_on]))
        verdict = blocklist.check(joined)
        if not verdict.ok:
            log.info("dropped brief (safety): %s", verdict.hits)
            if deps is not None:
                _log(deps, "blocked", {"concept": d.concept, "hits": verdict.hits})
            continue
        try:
            brief = DesignBrief(
                brief_id="b" + uuid.uuid4().hex[:10], niche=niche, concept=d.concept.strip(),
                prompt=ensure_plain_background(d.prompt, d.bg_color if d.bg_color else "#ffffff"),
                style=d.style.strip() or "flat vector illustration", text_on_design=text_on,
                palette=d.palette or ["#333333"], bg_color=d.bg_color,
            )
        except ValueError as exc:
            log.info("dropped brief (invalid): %s", exc)
            continue
        vec = trigram_vector(brief.concept)
        if any(float(vec @ v) >= CONCEPT_DUP_THRESHOLD for v in accepted_vecs):
            log.info("dropped near-duplicate concept: %s", brief.concept)
            if deps is not None:
                _log(deps, "duplicate", {"concept": brief.concept})
            continue
        accepted_vecs.append(vec)
        out.append(brief)
    return out


def make_listing(brief: DesignBrief, cfg: AppConfig, deps: Deps) -> dict[str, Any] | None:
    m = cfg.s.metadata
    system = load_prompt("metadata_system", title_max=m.title_max, tags_max=m.tags_max,
                         tag_max_len=m.tag_max_len, desc_max=m.desc_max)
    user = f"Niche: {brief.niche}\nConcept: {brief.concept}\nStyle: {brief.style}"
    try:
        lst = deps.llm.complete_json(system, user, Listing, temperature=0.5)
    except LLMError as exc:
        log.warning("listing generation failed: %s", exc)
        return None
    bl = deps.blocklist
    title = " ".join(lst.title.split())[: m.title_max]
    if not bl.check(title).ok:
        title = kebab(brief.niche, 80).replace("-", " ").title()
    tags: list[str] = []
    for t in lst.tags:
        t = " ".join(t.lower().split())[: m.tag_max_len]
        if t and t not in tags and bl.check(t).ok:
            tags.append(t)
    desc = " ".join(lst.description.split())
    if not bl.check(desc).ok:
        desc = brief.concept
    suffix = f" {m.ai_disclosure}" if m.ai_disclosure else ""
    desc = desc[: max(0, m.desc_max - len(suffix))].rstrip() + suffix
    return {"title": title, "tags": tags[: m.tags_max], "description": desc}


# ------------------------------------------------------------------ entry
def generate(req: PromptRequest, cfg: AppConfig, deps: Deps) -> list[DesignState]:
    """Create up to ``req.n`` new DesignStates (capped by run.max_designs_per_run)."""
    n = min(req.n, cfg.s.run.max_designs_per_run)
    if n < req.n:
        log.warning("--n %d capped to max_designs_per_run=%d", req.n, n)
    per_call = cfg.s.run.briefs_per_niche
    if req.auto_niche:
        niches = propose_niches(cfg, deps, max(3, min(10, math.ceil(n / per_call) + 1)))
    elif req.niche and req.niche.strip():
        verdict = deps.blocklist.check(req.niche)
        if not verdict.ok:
            raise ValueError(f"niche refused by the safety filter: {verdict.hits}")
        niches = [req.niche.strip()]
    else:
        raise ValueError("give --niche or --auto-niche")

    briefs: list[DesignBrief] = []
    vecs: list[np.ndarray] = []
    max_calls = 2 * math.ceil(n / per_call) + 2 * len(niches)
    calls = 0
    while len(briefs) < n and calls < max_calls:
        niche = niches[calls % len(niches)]
        calls += 1
        k = min(per_call, n - len(briefs))
        try:
            drafts = request_briefs(niche, k, cfg, deps, [b.concept for b in briefs])
        except LLMError as exc:
            log.warning("brief request failed for %r: %s", niche, exc)
            _log(deps, "llm_error", {"niche": niche, "error": str(exc)})
            continue
        briefs.extend(post_process(drafts, niche, cfg, deps.blocklist, vecs, deps)[:k])
    if len(briefs) < n:
        log.warning("only %d of %d briefs survived after %d LLM calls", len(briefs), n, calls)

    slugs = set(req.existing_slugs)
    states: list[DesignState] = []
    for b in briefs:
        slug = make_slug(b.niche, slugs)
        slugs.add(slug)
        listing = make_listing(b, cfg, deps) if cfg.s.metadata.enabled else None
        states.append(DesignState(design_id="d" + uuid.uuid4().hex[:10], slug=slug, status=OUTPUT_STATUS,
                                  brief=b, listing=listing))
    return states
