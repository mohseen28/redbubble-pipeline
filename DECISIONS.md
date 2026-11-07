# DECISIONS.md

Deviations from, and interpretations of, *Build Specification v1.1 (Kaggle edition)*.
Where the spec was ambiguous the simpler option was chosen (spec section 2).

## Environment / verification
- **No GPU or Kaggle access in the build environment.** Phases 1-5 were built and tested on CPU
  (Python 3.12, 1 core). Phase 0 (`notebooks/kaggle_smoke_test.ipynb`, `scripts/smoke_test.py`) is written but
  has **not been executed**; the FLUX, Real-ESRGAN, rembg, OpenCLIP and live-Ollama code paths are covered only
  by mocked tests. Run Phase 0 first on Kaggle and record any failure here.
- `requirements-gpu.txt` pins are a known-compatible set, unverified on the current Kaggle image (Phase 0 checks
  them). `torch` and `numpy` are deliberately not pinned: Kaggle preinstalls a CUDA-matched build.

## Libraries and models
- **Real-ESRGAN via `spandrel`, not the `realesrgan`/`basicsr` packages.** `basicsr` imports
  `torchvision.transforms.functional_tensor`, removed in torchvision 0.17, so it fails on current Kaggle images.
  `spandrel` loads the *official* `RealESRGAN_x4plus_anime_6B.pth` / `RealESRGAN_x4plus.pth` weights (BSD-3).
  Tiling, padding and fp16 are implemented in `stages/upscale.py`; on CUDA OOM the tile is halved and retried.
- **Ollama installer needs `zstd`** (newer releases ship `.tar.zst`); `install_ollama()` apt-installs it first.
- FLUX fp16 can overflow on some prompts and give a black image; the backend raises `ImageGenError`, the design
  goes to `error` and `pipeline retry` re-queues it. If this happens often, set `imagegen.compute_dtype: float32`
  (slower) — verify in Phase 0.
- BRIA RMBG and FLUX.1 dev are refused in code (licences).

## Structure (additions to the section 6 layout)
- `pipeline/deps.py` — lazy, injectable backend container (one model load per stage batch; fakes in tests).
- `pipeline/validate.py` — validation logic shared by `pipeline validate` and `scripts/validate_exports.py`.
- `pipeline/archive.py` — `pipeline zip` (PNG/JPEG stored, not recompressed; checks free disk first).
- `pipeline/notebook.py` — notebook helpers (Ollama install/start/pull, Kaggle Secrets, review gallery).
- `notebooks/kaggle_smoke_test.ipynb` — Phase 0 notebook (spec calls it "first notebook").
- `assets/blocklist/allowlist.txt` — words that fuzzy matching must never flag (e.g. "people" ~ "beeple").
- `llm/dry_run.py` — offline stub LLM used by `--dry-run` and tests.
- Extra optional config keys with defaults (all validated): `llm.fallback_model`, `llm.max_json_retries`,
  `llm.request_timeout_s`, `llm.allow_text_on_design`, `imagegen.flux_repo/sdxl_repo/sdxl_steps/sdxl_guidance/
  cpu_offload`, `cutout.erode_px/feather_px/min_speck_area`, `upscale.general_model/tile_pad`,
  `qa.clip_model/clip_pretrained/min_resolution/bg_tolerance/min_subject_fraction/max_subject_fraction/
  border_px/max_border_fraction/max_noise`, `export.only_groups/alpha_bbox_threshold`, `safety.blocklist_dir`.

## Paths
- `export.output_dir: output` is **relative to work_dir** (= `/kaggle/working/output` on Kaggle) so local
  dry-runs with `--work-dir` don't write to `/kaggle`. `safety.blocklist_dir` is relative to `config.yaml`.
- `manifest.csv` `path` is relative to the run folder, so the zip is portable.
- The master is also listed in `manifest.csv` (category/product `master`), since "append every file".

## Pipeline behaviour
- **Runs.** `runs` table; `run_id = YYYYMMDD_runNN`. `pipeline run` with identical arguments resumes the latest
  unfinished run (stage 1 is never repeated); `--new-run` forces a new one. `pipeline run` with no niche
  processes existing designs only (used after Regenerate).
- **Stage 1 `--n`** is the number of designs; briefs are requested `briefs_per_niche` at a time until `n`
  survive filtering (bounded number of LLM calls). `n` is capped at `max_designs_per_run`.
- **Concept dedupe** uses a hashed character-trigram cosine (threshold 0.90) instead of a neural embedding:
  no extra model, and near-identical sentences are what it must catch.
- **Plain-background phrase** is appended to the FLUX prompt if the LLM left it out (colour named from bg_color).
- **text_on_design** is forced to `null` unless `llm.allow_text_on_design: true`.
- **Listing metadata is generated in stage 1** (while Ollama is loaded) and stored in `state.listing`;
  stage 7 writes `listing.txt`. This avoids reloading Ollama after FLUX in single-GPU mode. Title/tags/description
  pass through the blocklist (blocked tags dropped) and the AI-disclosure line is appended to the description.
- **pytrends** is not implemented (optional, off by default in the spec).
- **trademark_judge.txt** is shipped but only relevant when on-image text is enabled; not called in v1.
- **QA retry** sets the design back to `new` with `seed: null` (status `qa_failed` only after
  `max_qa_retries` regenerations, i.e. `1 + max_qa_retries` images). The image stage always draws a fresh seed.
- **Technical QA thresholds**: subject = pixels further than `bg_tolerance` from the border's median colour;
  fails if < 2 % (mostly empty), > 92 % (no plain background), > 2 % of a 6-px border ring (touches border), or
  median-filter residual > 12 (extreme noise).
- **CLIP dedupe** compares against embeddings of previously *passed* designs (stored in an `embeddings` table)
  and the optional `reference_dir`; rejected/regenerated designs' embeddings are deleted.
- **Vision QA**, when enabled, also fails on any OCR text while text is disallowed. In single-GPU mode it may
  compete with FLUX for VRAM; dual T4 is recommended if you enable it.
- **Cutout** additionally zeroes the 1-px outer ring so the four corners are always transparent.
- **Upscale model choice**: `general_model` if `brief.style` mentions photo/realistic/painterly/watercolour/
  oil painting/cinematic/3D render, else the anime/illustration model.
- **margin_pct** is applied per side as a percentage of each canvas dimension.
- **anchor** supports center, top, bottom, left, right and the four corners.
- **Re-export** of a design deletes its folder first and replaces its manifest rows (idempotent on resume).
- **Pillow pixel limit** is raised to `max_dim_px²` because poster_large (98 MP) exceeds Pillow's 89 MP default.
- **Errors**: a failing design gets status `error` with the traceback in events; `pipeline retry` resets it to
  the input status of the stage that failed.
- **`--dry-run` implies `review.mode: auto`** (override with `--review manual`). `pipeline finish` auto-detects
  dry-run when every pending design belongs to a dry-run run.
- **Dual GPU**: the CLI sets `CUDA_VISIBLE_DEVICES=1` unless it is already a single explicit device; a
  multi-device value such as `0,1` is overridden.

## Observed in testing
- With dry-run placeholders (subject ≈ 55-65 % of the frame), 8-9 of the 60 products are flagged `low_res`
  (scale factor 2.6-3.4 > 2.5). Expect similar with real FLUX output unless prompts make the subject fill more
  of the frame. Behaviour is per spec: exported, flagged in the manifest, warned in events.
- Full 60-product export ≈ 25 s per design on one CPU core; ~45 MB per placeholder design (real artwork with
  texture will be several times larger — plan for the spec's 300-400 MB).
