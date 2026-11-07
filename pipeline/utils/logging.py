"""Console logging setup. Per-design history goes to the SQLite events table."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str | int = "INFO") -> None:
    root = logging.getLogger("pipeline")
    if not any(getattr(h, "_pipeline", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
        handler._pipeline = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith("pipeline") else f"pipeline.{name}")
