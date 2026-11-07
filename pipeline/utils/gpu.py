"""GPU layout helpers. torch is imported lazily so CUDA_VISIBLE_DEVICES can be set first."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from typing import Literal

from pipeline.utils.logging import get_logger

log = get_logger("gpu")
Layout = Literal["dual", "single", "cpu"]


def count_gpus() -> int:
    """Count NVIDIA GPUs without importing torch (nvidia-smi -L)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=20, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip().startswith("GPU "))


def resolve_layout(configured: str, n_gpus: int | None = None) -> Layout:
    n = count_gpus() if n_gpus is None else n_gpus
    if n == 0:
        return "cpu"
    if configured == "dual":
        if n < 2:
            log.warning("gpu_layout=dual but only %d GPU found; using single", n)
            return "single"
        return "dual"
    if configured == "single":
        return "single"
    return "dual" if n >= 2 else "single"


def configure_process_devices(layout: Layout) -> None:
    """In dual mode the Python process uses GPU 1 (Ollama owns GPU 0).

    Must run before torch is imported; afterwards it only warns.
    """
    if layout != "dual":
        return
    if "torch" in sys.modules:
        log.warning("torch already imported; cannot pin the process to GPU 1 "
                    "(set CUDA_VISIBLE_DEVICES=1 before importing torch)")
        return
    current = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if current and "," not in current and current != "1":
        log.info("CUDA_VISIBLE_DEVICES explicitly set to %s; leaving it", current)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def torch_device(pref: str = "auto") -> str:
    if pref == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        if pref == "cuda":
            log.warning("torch not installed; falling back to CPU")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if pref == "cuda":
        log.warning("CUDA requested but not available; falling back to CPU")
    return "cpu"


def free_vram() -> None:
    gc.collect()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def vram_summary() -> str:
    if "torch" not in sys.modules:
        return "torch not loaded"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA"
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"cuda:{i} {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB")
    return ", ".join(parts)
