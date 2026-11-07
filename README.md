# Redbubble AI Design Pipeline v1.1 (Kaggle edition)

Generates original print-on-demand designs and exports a correctly sized, correctly named
image for **every Redbubble product** (60 products, `products.yaml`), plus a `manifest.csv`.

```
Ollama (qwen2.5) → FLUX.1 schnell NF4 → QA → rembg cutout → review → Real-ESRGAN x4 → export x60
   stage 1            stage 2           3        4             5          6              7
```

> **Before any upload:** check Redbubble's current policies on AI-generated content, trademarks and
> image requirements. This pipeline uploads nothing.
>
> **QA is advisory.** The blocklist, similarity check and optional vision check reduce risk but are
> **not a legal guarantee of originality** or of freedom from trademark/copyright claims.

## Kaggle setup (spec section 4)

1. New notebook → **Accelerator: GPU T4 x2** (one T4 or P100 also works) → **Internet: On**.
2. Optional: add an `HF_TOKEN` **Kaggle Secret** (FLUX.1 schnell is not gated; the token only avoids rate limits).
   Never paste tokens into the notebook.
3. Get the repo into the notebook: attach it as a private Kaggle Dataset (fastest) or `git clone` it.
4. First time on a Kaggle image: run `notebooks/kaggle_smoke_test.ipynb` (**Phase 0**). It installs Ollama,
   pulls the model, gets one JSON reply, makes one FLUX image, runs one Real-ESRGAN tile, rembg and CLIP,
   and writes `/kaggle/working/smoke/report.json`. If the Ollama install fails, record the exact error in
   `DECISIONS.md` and stop.
5. Then use `notebooks/kaggle_run.ipynb`. Its cell sequence:
   install pinned deps → install/start Ollama + pull model → `pipeline init` →
   `pipeline run --niche "..." --n 10` → review gallery → `pipeline finish` → `pipeline validate` → `pipeline zip`
   → download the zip from the **Output** panel.

**GPU layout** (`runtime.gpu_layout: auto`)

| Mode | What happens |
|---|---|
| dual (T4 x2) | Ollama on GPU 0 (`qwen2.5:14b`, ~9-10 GB). The pipeline process pins itself to GPU 1 (`CUDA_VISIBLE_DEVICES=1`, set before torch is imported) for FLUX NF4, CLIP, rembg and Real-ESRGAN. |
| single | Stage 1 runs, then the Ollama model is unloaded (`keep_alive 0`) before FLUX loads. Use `qwen2.5:7b`. |

T4 has no bfloat16: FLUX transformer and T5 run NF4 with float16 compute.

**Limits.** Everything is written under `/kaggle/working` (~20 GB). One design is roughly 300-400 MB of
exports, so `max_designs_per_run: 30` and the runner stops cleanly when free disk drops below
`min_free_disk_gb`. Sessions are ~12 h and GPU hours are weekly-capped: every command is **resumable**
(SQLite state) — after a restart re-run the setup cells and the same command. Optional speed-up: save
`/kaggle/working/models` and the HF cache as a private Dataset.

## Commands

```
pipeline init                                   # validate config + products, create folders + DB
pipeline run --niche "cozy mushroom cottage" --n 10 [--dry-run] [--until STAGE] [--review manual|auto]
pipeline run --auto-niche --n 10                # LLM proposes 3-10 seasonal/evergreen niches
pipeline run                                    # continue existing designs only (e.g. after regenerate)
pipeline review [--approve ID] [--reject ID] [--regenerate ID] [--approve-all]
pipeline finish                                 # stages 6-7 for approved designs
pipeline status | validate | zip [--all] | retry
```
Global options: `--config`, `--products`, `--work-dir`, `--set key.path=value` (e.g. `--set llm.text_model=qwen2.5:7b`).

Re-running `run` with the same arguments resumes the latest unfinished run instead of creating new designs
(`--new-run` forces a new one). In the notebook, review with:
```python
from pipeline.notebook import review_gallery
review_gallery(config="config.yaml")
```
Each cutout is shown on a checkerboard with niche, concept and QA notes. Approve / Reject / Regenerate are
written to SQLite and the events log immediately, so they survive a kernel restart. Rejected files move to
`data/rejected/`; Regenerate sends the design back to stage 2 with a new seed (run `pipeline run` again).

## Output (spec section 11)

```
/kaggle/working/output/20261106_run01/
  manifest.csv          # design_id, slug, category, product, width_px, height_px, format,
                        # transparent, scale_factor, low_res, bytes, path (relative to the run folder)
  cozy-mushroom-cottage-01/
    design.json         # brief, seed, model names, QA report
    listing.txt         # optional title / tags / description (+ AI disclosure line)
    master/cozy-mushroom-cottage-01_master_4096x4096.png
    apparel/cozy-mushroom-cottage-01_premium-tshirt_2875x3900.png
    stickers/... home/... wall_art/cozy-mushroom-cottage-01_poster-large_8310x11790.jpg
```
File names: `{slug}_{product-key}_{W}x{H}.{png|jpg}`. Transparent RGBA PNG for apparel, caps, cotton tote and
stickers; every other product is composited on the brief's `bg_color` and saved as JPEG q95. Every file has an
embedded sRGB profile and 300 DPI. Canvas sizes are exactly the minimums in `products.yaml` (× `scale`).

**`low_res` flag.** The master is 4096 px, but it is first trimmed to the subject's alpha bounding box. A
subject that fills ~55-65 % of the frame trims to ~2300-2700 px, so the biggest products (poster_large,
photo_print_xl, duvet/blanket/tapestry/shower curtain, jigsaw, A-line dress) need about 2.6-3.4× enlargement
and are exported with `low_res: true` in the manifest and a warning in the events log — never silently.
Options: prompt for subjects that fill more of the frame, raise `export.max_scale_factor`, or disable those
products. `pipeline validate` lists them as warnings.

## Configuration

`config.yaml` (runtime) and `products.yaml` (the only place product dimensions live). Loading fails fast with a
readable message on any missing/invalid value or unknown key. Trim output with `enabled: false` on a product or
in a group's `_`, or `export.only_groups: [apparel, stickers]`. Re-check Redbubble's "Product Dimension
Requirements" page (help.redbubble.com, article 202270679) before each season and edit only `products.yaml`.

## Local / CPU dry-run and tests

```
pip install -r requirements.txt -r requirements-dev.txt && pip install -e . --no-deps
pipeline --work-dir ./work run --niche "cozy mushroom cottage" --n 3 --dry-run   # ~75 s, no network
pipeline --work-dir ./work validate                                              # PASS: 3 designs, 180 files
pytest -m "not slow"        # 128 tests, ~40 s     |   pytest -m slow   # the 60-product definition-of-done run
ruff check . && mypy        # mypy --strict on pipeline/
```
`--dry-run` uses a stub LLM, placeholder shapes, a colour-key cutout, Lanczos upscaling and a thumbnail embedder
instead of the real models, so the whole pipeline (and `validate`) runs on CPU without network.

## Build status

| Phase | Status |
|---|---|
| 0 Smoke test | Notebook + `scripts/smoke_test.py` written. **Must be run on Kaggle** (no GPU in the build environment). |
| 1 Foundation | Done, tested. |
| 2 Image path | Done, tested with synthetic images. |
| 3 LLM and generation | Done. Ollama client fully tested with mocked HTTP; FLUX/SDXL tested with a fake pipeline. |
| 4 Runner, QA, review | Done, tested (resume, retries, caps, reject/regenerate, gallery). |
| 5 Hardening | Done: error isolation + `retry`, caps, metadata, zip, notebooks, docs. |

See `DECISIONS.md` for every deviation from the spec.
