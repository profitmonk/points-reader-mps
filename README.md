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
