# POINTS-Reader MPS Runner (Mac)

End-to-end script to run Tencent **POINTS-Reader** locally on Apple Silicon (MPS), including:
- auto-download + patching for MPS (no FlashAttention2/CUDA),
- robust PDF→PNG rendering and clean RGB re-encode,
- single-file or whole-folder batch mode (loads model once),
- optional saving of per-page `.txt` and combined `.md`,
- safe re-run control with `--bruteforce`,
- dedupe via SHA-256 manifest.

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
pip install -r requirements.txt

# Single file
python next_steps_guide.py --pdf /path/to/invoice.pdf --save --debug

# Folder (loads model once)
python next_steps_guide.py --input-dir /path/to/invoices --save

# Force re-run for same input
python next_steps_guide.py --pdf /path/to/invoice.pdf --save --bruteforce


"""
================================================================================
POINTS-Reader on Mac (MPS) — End-to-End Runner with Batch Mode & Safe Re-runs
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This script runs the Tencent **POINTS-Reader** vision-language model locally on
macOS (Apple Silicon, MPS) to extract text from **PDFs and images (PNG/JPG)**.
It downloads and **patches** the model for MPS, converts PDFs to page PNGs,
re-encodes images to clean RGB (to avoid “blank output” issues), and runs
inference. It supports **single file** or **whole folder** processing and can
**skip reprocessing** files that haven’t changed. Outputs can be saved per page
and combined into one Markdown file.

WHY PATCHING WAS NEEDED (Background from our debugging)
-------------------------------------------------------
- Upstream code sometimes **forces FlashAttention-2** or **CUDA**; neither is
  available on Mac MPS. We patch the model’s remote code to:
  • Force `attn_implementation="eager"` (disable FA-2).
  • Remove `.cuda()`/`"cuda"` device moves (use `.to(device)` instead).
  • Fix **dtype → device** order to prevent float64 crashes on MPS.
  • Add a guard for “no images” so errors are informative.
- PDF renderers can output RGBA + color profiles. Some preprocessors then drop
  pixels → **empty/blank model output**. We always re-encode a clean 3-channel
  **RGB** `*_feed.png` with stripped metadata before inference.

KEY FEATURES
------------
1) **Model snapshot & patching** into `POINTS-Reader-local/` (one-time per model).
2) **Robust PDF→PNG** with PyMuPDF (preferred) or pdf2image (fallback).
3) **Image sanitization**: flatten alpha, strip ICC/EXIF, controlled resize, sharpen.
4) **Single file mode** (`--pdf` or `--png`) and **batch folder mode** (`--input-dir`)
   that loads the model **once** and processes everything sequentially.
5) **Safe re-run controls**:
   - Skips files already processed with the **same SHA-256** unless `--bruteforce`.
   - Writes a `manifest.json` **only when** outputs are saved (prevents “manifest-only”
     states).
6) **Deterministic outputs** (temperature=0.0 by default, unless you change it).

OUTPUTS & DIRECTORY LAYOUT
--------------------------
All runtime artifacts live under `./work/` (persisted across runs):
  ./work/input/<stem>/<original_file>
  ./work/pages/<stem>/<stem>_page_XXX.png
  ./work/output/<stem>/<stem>_page_XXX.txt
  ./work/output/<stem>/<stem>.md
  ./work/output/<stem>/manifest.json         (only when --save is used)

> `<stem>` is the filename without extension (e.g., `2308.pdf` → `2308`).

SETUP (one-time)
----------------
1) (Recommended) Create a virtualenv:
     python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
2) Install dependencies:
     pip install torch transformers huggingface_hub pillow pymupdf pdf2image
   If you plan to use pdf2image’s backend:
     brew install poppler
3) (Optional) Ensure MPS fallback is enabled:
     export PYTORCH_ENABLE_MPS_FALLBACK=1

USAGE (common examples)
-----------------------
Single file:
  python next_steps_guide.py --pdf /path/to/invoice.pdf --save --debug
  python next_steps_guide.py --png /path/to/page.png   --save

Whole folder (loads model once, processes all PDFs/PNGs recursively):
  python next_steps_guide.py --input-dir /path/to/invoices --save --debug

Force re-run of a previously processed input:
  python next_steps_guide.py --pdf /path/to/invoice.pdf --save --bruteforce

IMPORTANT FLAGS (CLI)
---------------------
--pdf / --png         Process a single file.
--input-dir           Process all PDFs/PNGs under a folder (recursively), reusing one model.
--local-dir           Local model directory (default: POINTS-Reader-local).
--refresh             Re-download & re-patch the model snapshot (useful after upstream updates).
--dpi                 PDF render DPI (default 300). Lower for speed, higher for tiny text.
--max-pages           Limit number of PDF pages to process.
--prompt              Text prompt (default: “Extract all text. Tables as HTML; other text as Markdown.”).
--max-new-tokens      Generation length cap (default 1200).
--save                Write per-page .txt files and a combined .md file to ./work/output/<stem>/.
--debug               Verbose logs, including original→feed image mode/size transitions.
--bruteforce          Clean prior work products for a given input and reprocess even if unchanged.

SKIP / RE-RUN LOGIC (how it decides to process)
-----------------------------------------------
A file is considered “already processed” only if:
  • `./work/output/<stem>/manifest.json` exists and has `"saved": true`, AND
  • `./work/output/<stem>/<stem>.md` exists, AND
  • the manifest’s `input_sha256` matches the current file’s SHA-256.

If all three are true and `--bruteforce` is NOT provided, the script **skips** it
and prints where the prior outputs live. Pass `--bruteforce` to force a clean
re-run; it removes `./work/input/<stem>/`, `./work/pages/<stem>/`, and
`./work/output/<stem>/` before reprocessing.

INFERENCE PIPELINE (high level)
-------------------------------
1) (If PDF) Render pages → PNGs at DPI (default 300).
2) Clean image: EXIF transpose → flatten to RGB → bounded resize → mild sharpen.
3) Re-encode as `*_feed.png` (RGB, stripped metadata) for stable preprocessing.
4) Run `model.chat(...)` per page with a deterministic config (`temperature=0.0`).
5) (If --save) Write per-page `.txt` and combined `.md`, then `manifest.json`.

TROUBLESHOOTING
---------------
- **Blank output (especially from PDFs)**:
  We now always feed a clean RGB `*_feed.png`. Run with `--debug` to see logs like:
  `[feed] 2308_page_001.png: mode=RGBA,size=(...) -> 2308_page_001_feed.png (RGB,(...))`
- **FlashAttention-2 or CUDA import errors**:
  Use `--refresh` to re-download and re-patch. Our patch forces `attn="eager"` and
  removes all CUDA calls for MPS.
- **“Cannot convert a MPS Tensor to float64”**:
  This script enforces dtype→device order; if you re-introduce code, keep that order.
- **Slow inference**:
  Lower `--dpi`, cap `--max-pages`, or reduce `--max-new-tokens`. Batch mode already
  avoids repeated model loads.
- **pdf2image errors**:
  Prefer PyMuPDF. If you must use pdf2image, ensure `poppler` is installed.

PERFORMANCE TIPS
----------------
- Batch mode (`--input-dir`) loads the model once.
- DPI 250–300 is a good balance; go lower if invoices are large and text is legible.
- Keep `temperature=0.0` for reproducibility; increase slightly only if outputs miss content.

CHANGELOG (highlights from this build)
--------------------------------------
- Patched POINTS-Reader for MPS (no FA-2/CUDA) + dtype→device fix + input guards.
- Added robust PDF→PNG and image sanitization with clean RGB re-encode.
- Added batch folder mode (single model load).
- Added `--save` outputs + manifest written **only when** outputs saved.
- Added SHA-256 dedupe & `--bruteforce` for safe re-runs.
- Added `--debug` image mode/size logging to diagnose blank outputs.

================================================================================
"""

