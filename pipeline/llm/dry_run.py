"""Offline stand-in for Ollama used by --dry-run and tests (no network, deterministic)."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from pipeline.llm.base import T
from pipeline.schemas import BriefBatch, BriefDraft, Listing, NicheList, VisionQAResult

_ANGLES = (
    ("glowing round window and a tiny wooden door", "sunset orange, moss green and cream"),
    ("snail mailman delivering a letter at the gate", "teal, mustard yellow and brick red"),
    ("stacked teacups and a curl of steam from the chimney", "dusty rose, sage and chocolate brown"),
    ("fireflies circling a lantern on the porch", "midnight blue, lemon and soft violet"),
    ("tiny garden of radishes and a watering can", "tomato red, leaf green and oat beige"),
    ("frog reading a book on the doorstep", "lime, plum and warm sand"),
    ("winter snowcap with a wreath on the door", "icy blue, cranberry and white"),
    ("stack of pancakes cooling on the windowsill", "honey gold, berry pink and cocoa"),
)
_SETTINGS = ("on a sunny spring morning", "under a starry night sky", "during a gentle autumn breeze",
             "in a quiet snowy winter", "beside a babbling brook", "at a lively summer fair")
_PALETTES = (
    ["#e76f51", "#2a9d8f", "#264653"], ["#f4a261", "#e9c46a", "#6d597a"],
    ["#c1121f", "#669bbc", "#003049"], ["#606c38", "#dda15e", "#bc6c25"],
    ["#8338ec", "#ff006e", "#3a86ff"], ["#2b9348", "#f15bb5", "#00bbf9"],
)
_BGS = ("#fdf6e3", "#f1faee", "#fff8e7", "#eef4ed", "#fbeaeb", "#e8f1f8")
_NICHES = ["cozy mushroom cottage", "axolotl barista", "retro camping van at dusk",
           "cat astronaut picnic", "kawaii fruit market"]


def _h(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)


class DryRunLLM:
    model = "dry-run-stub"

    def __init__(self) -> None:
        self.calls = 0
        self.counter = 0
        self.unloaded = 0

    def complete_json(self, system: str, user: str, schema: type[T], *, model: str | None = None,
                      images: list[bytes] | None = None, temperature: float | None = None) -> T:
        self.calls += 1
        data: Any
        if schema is BriefBatch:
            data = self._briefs(user)
        elif schema is NicheList:
            m = re.search(r"Propose (\d+)", system + user)
            k = int(m.group(1)) if m else 3
            data = NicheList(niches=_NICHES[: max(3, min(k, len(_NICHES)))])
        elif schema is Listing:
            m = re.search(r"Concept: (.+)", user)
            concept = (m.group(1) if m else "Original design").strip().rstrip(".")
            words = [w for w in re.findall(r"[a-z]+", concept.lower()) if len(w) > 3]
            data = Listing(title=concept[:90], tags=sorted(set(words))[:10] or ["design"],
                           description=f"{concept}. A cheerful original illustration.")
        elif schema is VisionQAResult:
            data = VisionQAResult(severe_artifacts=False, artifacts=[], ocr_text="", ip_risk=0.05, notes="ok")
        else:
            raise TypeError(f"DryRunLLM cannot produce {schema.__name__}")
        return schema.model_validate(data.model_dump())

    def _briefs(self, user: str) -> BriefBatch:
        m = re.search(r"exactly (\d+)", user)
        k = int(m.group(1)) if m else 3
        nm = re.search(r'Niche: "([^"]+)"', user)
        niche = nm.group(1) if nm else "original design"
        out = []
        for _ in range(k):
            i = self.counter
            self.counter += 1
            detail, colours = _ANGLES[(i + _h(niche)) % len(_ANGLES)]
            setting = _SETTINGS[(i // len(_ANGLES)) % len(_SETTINGS)]
            bg = _BGS[i % len(_BGS)]
            concept = f"A {niche} with a {detail}, {setting}."
            prompt = (f"A charming flat vector illustration of a {niche} with a {detail}, {setting}. "
                      f"Bold clean outlines, simple shapes and a {colours} palette with soft flat shading. "
                      f"The subject is centred and fully visible with empty space around it. "
                      f"Isolated on a plain solid cream background.")
            out.append(BriefDraft(concept=concept, prompt=prompt, style="flat vector sticker",
                                  text_on_design=None, palette=_PALETTES[i % len(_PALETTES)], bg_color=bg))
        return BriefBatch(briefs=out)

    def unload(self, model: str | None = None) -> None:
        self.unloaded += 1
