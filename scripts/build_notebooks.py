#!/usr/bin/env python3
"""Regenerate notebooks/*.ipynb (run from the repo root). Edit this file, not the .ipynb JSON."""
import nbformat as nbf
from nbformat.v4 import new_code_cell as code
from nbformat.v4 import new_markdown_cell as md
from nbformat.v4 import new_notebook

COMMON_SETUP = [
md("""## 1. Settings
Accelerator: **GPU T4 x2** (preferred). Settings → **Internet: On**.
Optional: add an `HF_TOKEN` Kaggle Secret (never paste tokens in the notebook)."""),
code('''import os
import shutil
import subprocess

# --- edit these ---------------------------------------------------------
REPO_URL   = "https://github.com/<you>/redbubble-pipeline.git"  # or attach the repo as a Kaggle Dataset
DATASET    = "/kaggle/input/redbubble-pipeline"                  # used instead of git if it exists
REPO       = "/kaggle/working/redbubble-pipeline"
NICHE      = "cozy mushroom cottage"
N_DESIGNS  = 10
# ------------------------------------------------------------------------
if not os.path.exists(REPO):
    if os.path.isdir(DATASET):
        shutil.copytree(DATASET, REPO)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO], check=True)
os.chdir(REPO)
print(os.getcwd())'''),
md("## 2. Install pinned Python dependencies"),
code('''!pip install -q -r requirements.txt -r requirements-gpu.txt
!pip install -q -e . --no-deps'''),
md("""## 3. Install and start Ollama, pull the text model
Dual T4: Ollama runs on GPU 0 with `qwen2.5:14b`; the pipeline process uses GPU 1.
Single GPU: `qwen2.5:7b`, and the runner unloads it (keep_alive 0) before FLUX loads."""),
code('''from pipeline.notebook import install_ollama, load_kaggle_secrets, pull_model, start_ollama
from pipeline.utils.gpu import count_gpus

print("secrets loaded:", load_kaggle_secrets())
N_GPUS = count_gpus()
TEXT_MODEL = "qwen2.5:14b" if N_GPUS >= 2 else "qwen2.5:7b"
print(f"{N_GPUS} GPU(s) -> {TEXT_MODEL}")
install_ollama()          # if this fails: copy the error into DECISIONS.md and stop (spec section 4)
start_ollama(gpu="0")
pull_model(TEXT_MODEL)
P = f"pipeline --config {REPO}/config.yaml --set llm.text_model={TEXT_MODEL}"'''),
]

run_nb = new_notebook()
run_nb.cells = [
md("""# Redbubble AI Design Pipeline v1.1 (Kaggle)
Prompts (Ollama) → FLUX.1 schnell NF4 → QA → cutout → **review** → Real-ESRGAN → one file per Redbubble product.

Nothing is uploaded anywhere. Before any upload, check Redbubble's current policies on AI content,
trademarks and image requirements. QA is advisory, not a legal guarantee of originality.
Run the smoke-test notebook once per new Kaggle image before relying on this one."""),
*COMMON_SETUP,
md("## 4. Init (validates config + 60 products, creates folders and the database)"),
code("!{P} init"),
md("""## 5. Generate (stages 1-4). Resumable: if the session restarts, re-run the cells above and this
cell again; finished work is skipped."""),
code('!{P} run --niche "{NICHE}" --n {N_DESIGNS}'),
md("""## 6. Review in the gallery
Approve / Reject / Regenerate each design. Decisions are saved immediately (they survive a kernel restart)."""),
code('''from pipeline.notebook import review_gallery

review_gallery(config=f"{REPO}/config.yaml")'''),
md("Regenerated designs go back through stages 2-4. Re-run this, then review again (re-run the gallery cell)."),
code("!{P} run"),
md("## 7. Finish: upscale approved designs and export every product size (stages 6-7)"),
code("!{P} finish\n!{P} validate\n!{P} status"),
md("""## 8. Zip and download
The archive appears in the **Output** panel (`/kaggle/working/output/<run_id>.zip`).
About 300-400 MB per design: zip, download, then delete old run folders before the next big run."""),
code("!{P} zip\n!ls -lh /kaggle/working/output/*.zip\n!df -h /kaggle/working | tail -1"),
]

smoke_nb = new_notebook()
smoke_nb.cells = [
md("""# Phase 0 smoke test
Proves Ollama, FLUX.1 schnell (NF4), Real-ESRGAN, rembg and OpenCLIP all run on this Kaggle image.
If the Ollama install fails, log the exact error in DECISIONS.md and stop."""),
*COMMON_SETUP,
code("!{P} init\n!python scripts/smoke_test.py --config config.yaml"),
code('''from IPython.display import Image as Show
from IPython.display import display

display(Show("/kaggle/working/smoke/flux.png", width=384))
display(Show("/kaggle/working/smoke/cutout.png", width=384))
with open("/kaggle/working/smoke/report.json") as fh:
    print(fh.read()[:3000])'''),
md("Also run the CPU test-suite on Kaggle once:"),
code("!pip install -q -r requirements-dev.txt\n!python -m pytest -q -m 'not slow'"),
]
for nb, name in ((run_nb, "kaggle_run.ipynb"), (smoke_nb, "kaggle_smoke_test.ipynb")):
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"},
                   "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}}
    nbf.validate(nb)
    nbf.write(nb, f"notebooks/{name}")
print("ok")
