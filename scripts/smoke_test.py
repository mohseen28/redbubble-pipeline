#!/usr/bin/env python3
"""Phase 0 smoke test (Kaggle, GPU, Internet On).

Proves every model runs before the pipeline is trusted:
  1. Ollama returns one schema-valid JSON reply
  2. FLUX.1 schnell (NF4) generates one 1024x1024 image
  3. Real-ESRGAN upscales one tile
  4. rembg cuts the background out
  5. OpenCLIP embeds the image
Writes <work_dir>/smoke/report.json and the images. Exit code 0 only if every
step that was not --skip'ped passed. If Ollama fails, log the exact error in
DECISIONS.md and stop (spec section 4): do not improvise a different backend.

    python scripts/smoke_test.py --config config.yaml [--skip clip]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config
from pipeline.utils.gpu import configure_process_devices, resolve_layout, vram_summary

STEPS = ("ollama", "flux", "esrgan", "rembg", "clip")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--skip", nargs="*", default=[], choices=STEPS)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, None, {"runtime.work_dir": args.work_dir} if args.work_dir else None)
    layout = resolve_layout(cfg.s.runtime.gpu_layout)
    configure_process_devices(layout)            # must happen before torch is imported
    out_dir = cfg.work_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"layout": layout, "steps": {}}
    state: dict[str, Any] = {}

    def step(name: str, fn: Callable[[], Any]) -> None:
        if name in args.skip:
            report["steps"][name] = {"status": "skipped"}
            return
        t0 = time.time()
        print(f"--- {name} ...", flush=True)
        try:
            detail = fn()
            report["steps"][name] = {"status": "ok", "seconds": round(time.time() - t0, 1),
                                     "detail": detail, "vram": vram_summary()}
            print(f"    ok in {time.time() - t0:.1f}s: {detail}", flush=True)
        except Exception as exc:
            report["steps"][name] = {"status": "failed", "error": repr(exc), "traceback": traceback.format_exc()}
            print(f"    FAILED: {exc!r}", flush=True)

    def ollama() -> Any:
        from pipeline.llm.ollama_client import OllamaClient
        from pipeline.schemas import Listing

        client = OllamaClient.from_config(cfg.s.llm)
        if not client.healthy():
            raise RuntimeError("Ollama is not running (start it with pipeline.notebook.start_ollama())")
        res = client.complete_json("Reply with JSON only.",
                                   "Write a listing for an original sticker of a cozy mushroom cottage.", Listing)
        if layout == "single":
            client.unload()
        return {"model": client.model, "title": res.title}

    def flux() -> Any:
        from pipeline.imagegen.flux_schnell import FluxSchnellBackend

        g = cfg.s.imagegen
        backend = FluxSchnellBackend(g)
        img = backend.generate("A cute flat vector illustration of a mushroom cottage with a round window, "
                               "bold clean outlines. Isolated on a plain solid cream background.",
                               42, g.width, g.height)
        path = out_dir / "flux.png"
        img.save(path)
        state["image"] = img
        backend.unload()
        return {"size": img.size, "path": str(path)}

    def esrgan() -> Any:
        import numpy as np

        from pipeline.stages.upscale import RealESRGANUpscaler

        img = state.get("image")
        if img is None:
            from PIL import Image

            img = Image.effect_noise((256, 256), 30).convert("RGB")
        u = cfg.s.upscale
        up = RealESRGANUpscaler(u.model, cfg.models_dir, tile=0, device=u.device)
        tile = np.asarray(img.convert("RGB"))[:256, :256]
        out = up.upscale(tile)
        return {"in": list(tile.shape), "out": list(out.shape), "device": up.device}

    def rembg() -> Any:
        from pipeline.stages.cutout import RembgCutter
        from pipeline.utils.images import clean_alpha

        img = state.get("image")
        if img is None:
            raise RuntimeError("needs the FLUX image")
        cut = RembgCutter(cfg.s.cutout.model).remove(img)
        import numpy as np

        arr = np.array(cut)
        arr[..., 3] = clean_alpha(arr[..., 3])
        from PIL import Image

        Image.fromarray(arr, "RGBA").save(out_dir / "cutout.png")
        return {"opaque_fraction": round(float((arr[..., 3] > 128).mean()), 3)}

    def clip() -> Any:
        from pipeline.stages.qa import ClipEmbedder

        img = state.get("image")
        if img is None:
            raise RuntimeError("needs the FLUX image")
        v = ClipEmbedder(cfg.s.qa.clip_model, cfg.s.qa.clip_pretrained).embed(img)
        return {"dim": int(v.shape[0])}

    for name, fn in (("ollama", ollama), ("flux", flux), ("esrgan", esrgan), ("rembg", rembg), ("clip", clip)):
        step(name, fn)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    failed = [k for k, v in report["steps"].items() if v["status"] == "failed"]
    print(f"\nreport: {out_dir / 'report.json'}")
    print("SMOKE TEST PASSED" if not failed else f"SMOKE TEST FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
