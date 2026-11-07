"""Refuse text that names brands, characters, celebrities, artists or protected phrases.

check() is case-insensitive and accent-insensitive, matches whole words, also
matches the de-spaced form ("coca cola" vs "cocacola"), applies a fuzzy
Levenshtein match (distance 1-2 depending on length, see FUZZY_*), and flags
artist-style phrasing ("in the style of ...", "...-esque").

This is a coarse, advisory filter. It is not a legal clearance.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BLOCKLIST_FILES = ("brands.txt", "characters.txt", "celebrities.txt", "phrases.txt")
ALLOWLIST_FILE = "allowlist.txt"

# Fuzzy thresholds on the de-spaced entry length. Shorter entries match exactly
# only; tuned so common words ("people" ~ beeple, "channel" ~ chanel) stay clean.
FUZZY_MIN_LEN = 7          # single-word entries: distance 1 from 7 chars
FUZZY_D2_MIN_LEN = 10      # single-word entries: distance 2 from 10 chars
FUZZY_MULTI_MIN_LEN = 10   # multi-word entries: distance 1 from 10 chars
FUZZY_MULTI_D2_MIN_LEN = 12

ARTIST_STYLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("artist-style: 'in the style of'", re.compile(r"\bin\s+the\s+(?:art\s+)?style\s+of\b")),
    ("artist-style: 'in the manner of'", re.compile(r"\bin\s+the\s+manner\s+of\b")),
    ("artist-style: 'inspired by'", re.compile(r"\binspired\s+by\b")),
    ("artist-style: 'a la'", re.compile(r"(?:^|\s)(?:à|a)\s+la\s+[a-z]")),
    ("artist-style: 'by <artist>'", re.compile(r"\b(?:art|artwork|painting|illustration|drawing)\s+by\s+[a-z]")),
    ("artist-style: 'trending on'", re.compile(r"\btrending\s+on\b")),
)
_ESQUE_RE = re.compile(r"\b([a-z]{3,})esque\b")
_ESQUE_OK = {
    "pictur", "grot", "burl", "statu", "arab", "roman", "humor", "humour", "picar",
    "kafka",  # 'kafkaesque' is a common adjective, not an art-style request
}


@dataclass(frozen=True)
class SafetyResult:
    ok: bool
    hits: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def normalize(text: str) -> str:
    """Lowercase, strip accents, turn punctuation into spaces, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "").replace("\u2019", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def levenshtein(a: str, b: str, max_dist: int | None = None) -> int:
    """Edit distance with optional early exit (returns max_dist+1 when exceeded)."""
    if a == b:
        return 0
    if max_dist is not None and abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            val = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
            cur.append(val)
            row_min = min(row_min, val)
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class _Entry:
    original: str
    source: str
    tokens: tuple[str, ...]
    joined: str


class Blocklist:
    def __init__(self, entries: list[_Entry], allow: set[str]) -> None:
        self.entries = entries
        self.allow = allow
        self.by_ntokens: dict[int, list[_Entry]] = {}
        for e in entries:
            self.by_ntokens.setdefault(len(e.tokens), []).append(e)
        self.joined_exact = {e.joined: e for e in entries}

    @classmethod
    def from_dir(cls, directory: str | Path) -> Blocklist:
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"blocklist directory not found: {d}")
        entries: list[_Entry] = []
        for name in BLOCKLIST_FILES:
            path = d / name
            if not path.is_file():
                raise FileNotFoundError(f"missing blocklist file: {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                norm = normalize(line)
                if norm:
                    toks = tuple(norm.split())
                    entries.append(_Entry(line, name.removesuffix(".txt"), toks, "".join(toks)))
        allow: set[str] = set()
        allow_path = d / ALLOWLIST_FILE
        if allow_path.is_file():
            for line in allow_path.read_text(encoding="utf-8").splitlines():
                line = normalize(line.split("#", 1)[0])
                if line:
                    allow.add(line)
        return cls(entries, allow)

    def check(self, text: str) -> SafetyResult:
        hits: list[str] = []
        norm = normalize(text)
        tokens = norm.split()
        seen: set[str] = set()

        def add(msg: str, key: str) -> None:
            if key not in seen:
                seen.add(key)
                hits.append(msg)

        # 1) exact and fuzzy n-gram matches
        for n, entries in self.by_ntokens.items():
            for i in range(len(tokens) - n + 1):
                gram_toks = tokens[i:i + n]
                gram = " ".join(gram_toks)
                if gram in self.allow:
                    continue
                gram_joined = "".join(gram_toks)
                for e in entries:
                    if gram_toks == list(e.tokens) or gram_joined == e.joined:
                        add(f"{e.source}: '{e.original}'", e.joined)
                        continue
                    max_d = _max_distance(e)
                    if max_d == 0:
                        continue
                    if abs(len(gram_joined) - len(e.joined)) > max_d:
                        continue
                    if levenshtein(gram_joined, e.joined, max_d) <= max_d:
                        add(f"{e.source} (fuzzy): '{e.original}' ~ '{gram}'", e.joined)
        # 2) de-spaced multi-word entries split differently ("cocacola" vs "coca cola")
        for tok in tokens:
            e2 = self.joined_exact.get(tok)
            if e2 is not None and tok not in self.allow:
                add(f"{e2.source}: '{e2.original}'", e2.joined)
        # 3) artist-style patterns
        lowered = " " + normalize_keep_accents(text) + " "
        for label, pat in ARTIST_STYLE_PATTERNS:
            if pat.search(lowered):
                add(label, label)
        for m in _ESQUE_RE.finditer(norm):
            stem = m.group(1)
            if stem not in _ESQUE_OK and m.group(0) not in self.allow:
                add(f"artist-style: '{m.group(0)}'", m.group(0))
        return SafetyResult(ok=not hits, hits=hits)


def _max_distance(e: _Entry) -> int:
    n = len(e.joined)
    if len(e.tokens) == 1:
        return 2 if n >= FUZZY_D2_MIN_LEN else 1 if n >= FUZZY_MIN_LEN else 0
    return 2 if n >= FUZZY_MULTI_D2_MIN_LEN else 1 if n >= FUZZY_MULTI_MIN_LEN else 0


def normalize_keep_accents(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


@lru_cache(maxsize=8)
def _cached(directory: str) -> Blocklist:
    return Blocklist.from_dir(directory)


def load_blocklist(directory: str | Path) -> Blocklist:
    return _cached(str(Path(directory).resolve()))


def check(text: str | None, blocklist: Blocklist | str | Path) -> SafetyResult:
    """Module-level entry point used by the stages."""
    if not text:
        return SafetyResult(ok=True)
    bl = blocklist if isinstance(blocklist, Blocklist) else load_blocklist(blocklist)
    return bl.check(text)
