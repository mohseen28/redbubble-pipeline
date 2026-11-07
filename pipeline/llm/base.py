"""LLM client interface and prompt-file loading."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Protocol, TypeVar

from pydantic import BaseModel

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """The LLM backend failed or never returned schema-valid JSON."""


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: str, user: str, schema: type[T], *, model: str | None = None,
                      images: list[bytes] | None = None, temperature: float | None = None) -> T:
        """Return a validated instance of ``schema``."""
        ...

    def unload(self, model: str | None = None) -> None:
        """Ask the backend to free the model's (V)RAM."""
        ...


def load_prompt(name: str, **values: object) -> str:
    """Read pipeline/prompts/<name>.txt, drop '# ' header lines, substitute $placeholders."""
    path = PROMPTS_DIR / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    body = "\n".join(line for line in lines if not line.startswith("# ")).strip()
    return Template(body).substitute({k: str(v) for k, v in values.items()})


def prompt_version(name: str) -> str:
    first = (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8").splitlines()[0]
    return first.split("version:")[-1].strip() if "version:" in first else "?"
