"""Data models. Field names of the section-7 models are normative."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

Status = Literal[
    "new", "generated", "qa_failed", "qa_passed", "cutout_done",
    "awaiting_review", "approved", "rejected", "mastered",
    "exported", "error",
]
Decision = Literal["approve", "reject", "regenerate"]
TERMINAL_STATUSES: tuple[Status, ...] = ("exported", "rejected", "qa_failed", "error")


def utcnow() -> datetime:
    return datetime.now(UTC)


def _norm_hex(v: str) -> str:
    v = v.strip()
    if not v.startswith("#"):
        v = "#" + v
    if re.fullmatch(r"#[0-9a-fA-F]{3}", v):
        v = "#" + "".join(c * 2 for c in v[1:])
    if not HEX_RE.match(v):
        raise ValueError(f"not a hex colour: {v!r}")
    return v.lower()


# ------------------------------------------------------------------ section 7
class DesignBrief(BaseModel):
    brief_id: str
    niche: str
    concept: str                  # one sentence
    prompt: str                   # FLUX prompt, natural language, under 200 words
    style: str                    # e.g. "flat vector sticker"
    text_on_design: str | None = None
    palette: list[str]            # hex
    bg_color: str                 # hex; fills opaque products

    @field_validator("palette")
    @classmethod
    def _palette(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("palette must not be empty")
        return [_norm_hex(c) for c in v]

    @field_validator("bg_color")
    @classmethod
    def _bg(cls, v: str) -> str:
        return _norm_hex(v)

    @field_validator("prompt")
    @classmethod
    def _prompt_len(cls, v: str) -> str:
        if len(v.split()) >= 200:
            raise ValueError("prompt must be under 200 words")
        return v.strip()


class QAReport(BaseModel):
    passed: bool
    reasons: list[str] = []
    ip_risk: float | None = None
    similarity_matches: list[dict[str, Any]] = []   # {path, score}


class ExportRecord(BaseModel):
    product: str
    category: str
    width_px: int
    height_px: int
    fmt: Literal["png", "jpg"]
    transparent: bool
    scale_factor: float
    low_res: bool
    path: str
    bytes: int


class DesignState(BaseModel):
    design_id: str
    slug: str                     # kebab-case, max 40 chars, plus 2-digit index
    status: Status
    brief: DesignBrief | None = None
    seed: int | None = None
    raw_image: str | None = None
    cutout_image: str | None = None
    master_image: str | None = None
    qa: QAReport | None = None
    qa_retries: int = 0
    regen_count: int = 0
    exports: list[ExportRecord] = []
    listing: dict[str, Any] | None = None
    decision: Decision | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)

    def patch(self, **changes: Any) -> DesignState:
        """Return a copy with ``changes`` applied and updated_at refreshed (validated)."""
        data = self.model_dump()
        data.update(changes)
        data["updated_at"] = utcnow()
        return DesignState.model_validate(data)


# ---------------------------------------------------- LLM structured outputs
class _LLMModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class BriefDraft(_LLMModel):
    """What the LLM fills in; ids and niche are added by the prompt stage."""

    concept: str = Field(description="One sentence describing the original design idea.")
    prompt: str = Field(
        description="FLUX prompt: descriptive sentences, under 200 words, ends with the plain-background phrase.")
    style: str = Field(description="Short style label, e.g. 'flat vector sticker'.")
    text_on_design: str | None = Field(default=None, description="Must be null unless text is explicitly allowed.")
    palette: list[str] = Field(description="3 to 6 hex colours like #aabbcc.")
    bg_color: str = Field(description="Hex colour for the plain background / opaque products.")


class BriefBatch(_LLMModel):
    briefs: list[BriefDraft]


class NicheList(_LLMModel):
    niches: list[str] = Field(description="3 to 10 long-tail niches.")


class Listing(_LLMModel):
    title: str
    tags: list[str]
    description: str


class VisionQAResult(_LLMModel):
    severe_artifacts: bool = Field(description="True if the image has obvious generation defects.")
    artifacts: list[str] = Field(default_factory=list)
    ocr_text: str = Field(default="", description="Any readable text in the image, verbatim.")
    ip_risk: float = Field(
        ge=0, le=1, description="0-1 estimate that the image resembles a known brand, character or artwork.")
    notes: str = ""
