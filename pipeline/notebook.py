"""Helpers for notebooks/kaggle_run.ipynb (Kaggle only; nothing here runs in tests)."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from pipeline.config import load_config
from pipeline.db import DB
from pipeline.utils.gpu import count_gpus
from pipeline.utils.logging import get_logger

log = get_logger("notebook")
OLLAMA_HOST = "http://127.0.0.1:11434"


def load_kaggle_secrets(names: tuple[str, ...] = ("HF_TOKEN",)) -> list[str]:
    """Copy Kaggle Secrets into env vars. Missing secrets are skipped (HF_TOKEN is optional)."""
    loaded: list[str] = []
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return loaded
    client = UserSecretsClient()
    for name in names:
        try:
            value = client.get_secret(name)
        except Exception:  # secret not attached to this notebook
            continue
        if value:
            os.environ[name] = value
            loaded.append(name)
    return loaded


def install_ollama() -> None:
    """Install Ollama (needs Internet On). zstd is required by the current installer."""
    if shutil.which("ollama"):
        return
    subprocess.run("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw >/dev/null",
                   shell=True, check=False)
    res = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True,
                         capture_output=True, text=True, check=False)
    if res.returncode != 0 or not shutil.which("ollama"):
        raise RuntimeError("Ollama install failed. Log this exact error in DECISIONS.md and stop "
                           f"(spec section 4):\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def start_ollama(gpu: str | None = None, log_path: str = "/kaggle/working/ollama.log",
                 timeout: float = 60) -> subprocess.Popen[bytes] | None:
    """Start `ollama serve` in the background. Dual T4: GPU 0; single GPU: the only GPU."""
    if _alive():
        return None
    env = dict(os.environ)
    env["OLLAMA_MAX_LOADED_MODELS"] = "1"
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    if gpu is None:
        gpu = "0"
    env["CUDA_VISIBLE_DEVICES"] = gpu
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "ab")  # noqa: SIM115 - handle must outlive this function
    proc = subprocess.Popen(["ollama", "serve"], stdout=fh, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _alive():
            log.info("ollama serving on GPU %s (%d GPU(s) visible to the system)", gpu, count_gpus())
            return proc
        time.sleep(1)
    raise RuntimeError(f"ollama did not start within {timeout}s; see {log_path}")


def pull_model(model: str) -> None:
    res = subprocess.run(["ollama", "pull", model], capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ollama pull {model} failed:\n{res.stderr[-2000:]}")


def _alive() -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/version", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def review_gallery(config: str | None = None, work_dir: str | None = None) -> Any:
    """Show the review gallery for every design awaiting review."""
    from pipeline.stages.review import show_gallery

    overrides = {"runtime.work_dir": work_dir} if work_dir else None
    cfg = load_config(config, None, overrides)
    return show_gallery(DB(cfg.db_path), cfg)
