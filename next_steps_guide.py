#!/usr/bin/env python3
"""
(… header unchanged from your commented version …)
"""

import os, re, sys, gc, json, hashlib, shutil, argparse
from pathlib import Path
from typing import List, Tuple, Optional

# ---- Env hygiene ----
for var in ("HF_USE_FLASH_ATTENTION_2", "USE_FLASH_ATTENTION_2", "ATTN_IMPL"):
    os.environ.pop(var, None)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

def _require(mod, pip_name=None):
    try:
        return __import__(mod)
    except Exception:
        print(f"[ERR] Missing dependency '{mod}'. Install with: pip install {pip_name or mod}", file=sys.stderr)
        raise

torch = _require("torch", "torch")
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image, ImageOps, ImageFilter, PngImagePlugin

try:
    fitz = __import__("fitz")   # PyMuPDF
except Exception:
    fitz = None
try:
    pdf2image = __import__("pdf2image")
except Exception:
    pdf2image = None

torch.set_default_dtype(torch.float32)

# ---------------- Manifest helpers ----------------
def sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def manifest_path(work_root: Path, stem: str) -> Path:
    return work_root / "output" / stem / "manifest.json"

def write_manifest(work_root: Path, stem: str, src_path: Path, num_pages: int):
    out_dir = (work_root / "output" / stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "input_name": src_path.name,
        "input_sha256": sha256_file(src_path),
        "num_pages": num_pages,
        "saved": True,  # explicit marker that outputs were written
    }
    (out_dir / "manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def read_manifest(work_root: Path, stem: str) -> Optional[dict]:
    mp = manifest_path(work_root, stem)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def outputs_exist(work_root: Path, stem: str) -> bool:
    out_dir = (work_root / "output" / stem)
    combo = out_dir / f"{stem}.md"
    return combo.exists()

def already_processed(src_path: Path, work_root: Path, stem: str) -> bool:
    """Only true if manifest exists, saved==True, the hash matches, AND the combined .md exists."""
    m = read_manifest(work_root, stem)
    if not m or m.get("saved") is not True:
        return False
    cur = sha256_file(src_path)
    return (m.get("input_sha256") == cur) and outputs_exist(work_root, stem)

def cleanup_work_for_stem(work_root: Path, stem: str):
    for sub in ("input", "pages", "output"):
        p = work_root / sub / stem
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            print(f"[clean] Removed: {p}")

# ---------------- Cache ----------------
def clear_modules_cache(local_key: str):
    p = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / local_key
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
        print(f"[cache] Cleared modules cache: {p}")

# ---------------- Download + Patch ----------------
def download_local_model(local_dir: Path, refresh: bool = False) -> Path:
    if refresh and local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
        print(f"[download] Removed existing: {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="tencent/POINTS-Reader",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    model_dir = Path(path)
    print(f"[download] Snapshot at: {model_dir}")
    return model_dir

def patch_pointsv15_for_mps(model_dir: Path):
    target = model_dir / "modeling_pointsv15_chat.py"
    if not target.exists():
        raise FileNotFoundError(f"Expected code file not found: {target}")
    s = target.read_text()
    s = re.sub(r'config\.llm_config\._attn_implementation\s*=\s*["\']flash_attention_2["\']',
               'config.llm_config._attn_implementation = "eager"', s)
    s = re.sub(r'attn_implementation\s*=\s*["\']flash_attention_2["\']',
               'attn_implementation="eager"', s)
    s = s.replace(".to(self.vision_encoder.device).to(self.vision_encoder.dtype)",
                  ".to(self.vision_encoder.dtype).to(self.vision_encoder.device)")
    s = re.sub(r"\.cuda\(\)", ".to(self.vision_encoder.device)", s)
    s = re.sub(r"\.to\(\s*['\"]cuda['\"]\s*\)", ".to(self.vision_encoder.device)", s)
    s = re.sub(r"torch\.device\(\s*['\"]cuda['\"]\s*\)", "self.vision_encoder.device", s)
    s = re.sub(r"device\s*=\s*['\"]cuda['\"]", "device=self.vision_encoder.device", s)
    s = s.replace(".half().cuda()", ".to(self.vision_encoder.dtype).to(self.vision_encoder.device)")
    s = s.replace(
        "image_grid_thws = np.concatenate(image_grid_thws, axis=0)",
        "assert len(image_grid_thws) > 0, 'No images provided in messages.'\n        image_grid_thws = np.concatenate(image_grid_thws, axis=0)"
    )
    target.write_text(s)
    print(f"[patch] Patched for MPS: {target}")

# ---------------- Image prep ----------------
from PIL import Image, ImageOps, ImageFilter, PngImagePlugin

def _flatten_rgba_to_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1]); return bg
    if img.mode in ("LA", "P"):
        return img.convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

def _prep_invoice_image(img: Image.Image, min_long_side=1400, max_long_side=2000) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = _flatten_rgba_to_white(img)
    w, h = img.size; long_side = max(w, h)
    if long_side < min_long_side:
        scale = min_long_side / long_side
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    elif long_side > max_long_side:
        scale = max_long_side / long_side
        img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    return img

def _save_clean_png(img: Image.Image, out_path: Path):
    info = PngImagePlugin.PngInfo()
    img = _flatten_rgba_to_white(img)
    img.save(out_path, format="PNG", optimize=True, compress_level=3, pnginfo=info)

def pdf_to_clean_pngs(pdf_path: Path, pages_root: Path, stem: str, dpi: int = 300, max_pages: Optional[int] = None) -> List[Path]:
    subdir = (pages_root / stem); subdir.mkdir(parents=True, exist_ok=True)
    results: List[Path] = []
    try:
        if fitz is not None:
            doc = fitz.open(pdf_path.as_posix())
            page_count = doc.page_count
            n = page_count if max_pages is None else min(max_pages, page_count)
            zoom = dpi / 72.0; mat = fitz.Matrix(zoom, zoom)
            for i in range(n):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat, alpha=True)
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                img = _prep_invoice_image(img)
                p = (subdir / f"{stem}_page_{i+1:03d}.png").resolve()
                _save_clean_png(img, p); results.append(p)
            doc.close(); return results
    except Exception:
        pass
    if pdf2image is None:
        raise RuntimeError("No PDF renderer found. Install PyMuPDF or pdf2image + poppler.")
    pages = pdf2image.convert_from_path(pdf_path.as_posix(), dpi=dpi, fmt="png")
    n = len(pages) if max_pages is None else min(max_pages, len(pages))
    for i in range(n):
        img = _prep_invoice_image(pages[i])
        p = (subdir / f"{stem}_page_{i+1:03d}.png").resolve()
        _save_clean_png(img, p); results.append(p)
    return results

def prep_existing_png(png_path: Path, pages_root: Path, stem: str) -> List[Path]:
    subdir = (pages_root / stem); subdir.mkdir(parents=True, exist_ok=True)
    img = Image.open(png_path); img = _prep_invoice_image(img)
    p = (subdir / f"{stem}_page_001.png").resolve()
    _save_clean_png(img, p); return [p]

# ---------------- Model loading ----------------
def load_local_model(local_dir: Path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"[device] Using {device} with dtype {dtype}")
    tok  = AutoTokenizer.from_pretrained(local_dir.as_posix(), trust_remote_code=True)
    proc = Qwen2VLImageProcessor.from_pretrained(local_dir.as_posix())
    kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(local_dir.as_posix(), dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(local_dir.as_posix(), torch_dtype=dtype, **kwargs)
    model = model.to(device)
    print("attn impl (top):", getattr(model.config, "_attn_implementation", None))
    if getattr(model.config, "llm_config", None) is not None:
        print("attn impl (llm):", getattr(model.config.llm_config, "_attn_implementation", None))
    print("device:", next(model.parameters()).device)
    return model, tok, proc

# ---------------- Inference + Saving ----------------
def run_chat_on_images(model, tok, proc, image_paths: List[Path], prompt: str,
                       max_new_tokens: int = 1200, debug: bool = False) -> Tuple[List[str], str]:
    feed_paths: List[Path] = []
    for p in image_paths:
        assert p.exists(), f"Image missing: {p}"
        with Image.open(p) as im:
            mode_before, size_before = im.mode, im.size
            im = _flatten_rgba_to_white(im)
            im = _prep_invoice_image(im, min_long_side=1400, max_long_side=2000)
            feed = p.with_name(p.stem + "_feed.png")
            _save_clean_png(im, feed); feed_paths.append(feed.resolve())
        if debug:
            print(f"[feed] {p.name}: mode={mode_before}, size={size_before} -> {feed.name} (RGB,{im.size})")
    print("\n[run] Using image files:"); [print("  -", p) for p in feed_paths]
    page_texts: List[str] = []
    gen_cfg = dict(max_new_tokens=max_new_tokens, temperature=0.0)
    for idx, p in enumerate(feed_paths, 1):
        messages = [{"role": "user","content": [{"type": "image","image": p.as_posix()},{"type": "text","text": prompt}],}]
        with torch.inference_mode():
            page_out = model.chat(messages, tok, proc, gen_cfg)
        page_texts.append(page_out)
    combined = "\n\n".join(f"### Page {i}\n{t}" for i, t in enumerate(page_texts, 1))
    return page_texts, combined

def save_outputs(stem: str, image_paths: List[Path], page_texts: List[str], combined_md: str, work_root: Path) -> Path:
    out_dir = (work_root / "output" / stem); out_dir.mkdir(parents=True, exist_ok=True)
    for i, txt in enumerate(page_texts, start=1):
        (out_dir / f"{stem}_page_{i:03d}.txt").write_text(txt if txt else "", encoding="utf-8")
    combo = out_dir / f"{stem}.md"; combo.write_text(combined_md if combined_md else "", encoding="utf-8")
    print("\n[save] Wrote outputs to:", out_dir)
    return out_dir

# ---------------- Batch helpers ----------------
def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}

def _scan_inputs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_supported(p)])

def process_one_with_loaded_model(
    src_path: Path, *, work_root: Path, input_root: Path, pages_root: Path,
    model, tok, proc, dpi: int, max_pages: Optional[int], prompt: str,
    max_new_tokens: int, save: bool, debug: bool, bruteforce: bool,
):
    stem = src_path.stem

    # Skip if fully processed unless bruteforce
    if not bruteforce and already_processed(src_path, work_root, stem):
        out_dir = (work_root / "output" / stem)
        print(f"[skip] '{src_path.name}' already processed. Outputs at: {out_dir}")
        print("       If you want to re-run, pass --bruteforce to clean and reprocess.")
        return

    # Clean per-stem if bruteforce, or input hash changed
    if bruteforce:
        cleanup_work_for_stem(work_root, stem)
    else:
        m = read_manifest(work_root, stem)
        if m and m.get("input_sha256") != sha256_file(src_path):
            print(f"[info] Input '{src_path.name}' changed → cleaning pages/output for stem '{stem}'.")
            for sub in ("pages", "output"):
                p = work_root / sub / stem
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"[clean] Removed: {p}")

    # Copy to work/input/<stem>/
    input_dir = (input_root / stem); input_dir.mkdir(parents=True, exist_ok=True)
    copied = (input_dir / src_path.name).resolve()
    if not copied.exists() or copied.read_bytes() != src_path.read_bytes():
        shutil.copy2(src_path, copied)
    print(f"[work] Input copied to: {copied}")

    # Render/prepare images
    if copied.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image_paths = prep_existing_png(copied, pages_root, stem)
    else:
        image_paths = pdf_to_clean_pngs(copied, pages_root, stem, dpi=dpi, max_pages=max_pages)

    # Inference
    page_texts, combined_md = run_chat_on_images(model, tok, proc, image_paths, prompt, max_new_tokens, debug=debug)

    # Save + manifest (manifest only if outputs are saved)
    if save:
        out_dir = save_outputs(stem, image_paths, page_texts, combined_md, work_root)
        write_manifest(work_root, stem, copied, num_pages=len(image_paths))
    else:
        print("[save] DISABLED (pass --save to write files)")

    # Housekeeping
    del page_texts, combined_md
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="POINTS-Reader on Mac (MPS): single file or folder; persistent work dir; optional saving.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--png", type=str, help="Path to a PNG/JPG image page.")
    src.add_argument("--pdf", type=str, help="Path to a PDF file.")
    ap.add_argument("--input-dir", type=str, help="Process all PDFs/PNGs under this folder (recursively).")
    ap.add_argument("--local-dir", type=str, default="POINTS-Reader-local", help="Local model folder.")
    ap.add_argument("--refresh", action="store_true", help="Force re-download & re-patch the model.")
    ap.add_argument("--dpi", type=int, default=300, help="PDF render DPI (default 300).")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of PDF pages processed.")
    ap.add_argument("--prompt", type=str, default="Extract all text. Tables as HTML; other text as Markdown.")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--save", action="store_true", help="Write per-page .txt and combined .md under ./work/output/<stem>/")
    ap.add_argument("--debug", action="store_true", help="Verbose logging of image modes/sizes.")
    ap.add_argument("--bruteforce", action="store_true", help="Clean prior work-products for this input and re-run.")
    args = ap.parse_args()

    if not args.input_dir and not args.png and not args.pdf:
        print("[usage] Provide --png /path.png or --pdf /path.pdf, or --input-dir /folder", file=sys.stderr)
        sys.exit(2)

    print(f"[save] {'ENABLED' if args.save else 'DISABLED (pass --save to write files)'}")

    local_dir = Path(args.local_dir).resolve()
    local_key = local_dir.name

    base = Path.cwd().resolve()
    work_root  = base / "work"
    input_root = work_root / "input"
    pages_root = work_root / "pages"
    input_root.mkdir(parents=True, exist_ok=True)
    pages_root.mkdir(parents=True, exist_ok=True)

    clear_modules_cache(local_key)
    model_dir = download_local_model(local_dir, refresh=args.refresh)
    patch_pointsv15_for_mps(model_dir)
    clear_modules_cache(local_key)

    # Batch mode
    if args.input_dir:
        src_root = Path(args.input_dir).expanduser().resolve()
        files = _scan_inputs(src_root)
        if not files:
            print(f"[run] No PDFs/PNGs found under: {src_root}")
            return
        model, tok, proc = load_local_model(local_dir)
        print(f"[batch] Processing {len(files)} files sequentially…")
        for i, f in enumerate(files, 1):
            print(f"\n[batch] ({i}/{len(files)}) {f}")
            process_one_with_loaded_model(
                f, work_root=work_root, input_root=input_root, pages_root=pages_root,
                model=model, tok=tok, proc=proc,
                dpi=args.dpi, max_pages=args.max_pages,
                prompt=args.prompt, max_new_tokens=args.max_new_tokens,
                save=args.save, debug=args.debug, bruteforce=args.bruteforce,
            )
        print("\n[batch] Done.")
        return

    # Single-file mode
    src_path = Path(args.png or args.pdf).expanduser().resolve()
    stem = src_path.stem

    if not args.bruteforce and already_processed(src_path, work_root, stem):
        out_dir = (work_root / "output" / stem)
        print(f"[skip] '{src_path.name}' already processed. Outputs at: {out_dir}")
        print("       If you want to re-run, pass --bruteforce to clean and reprocess.")
        return

    if args.bruteforce:
        cleanup_work_for_stem(work_root, stem)
    else:
        m = read_manifest(work_root, stem)
        if m and m.get("input_sha256") != sha256_file(src_path):
            print(f"[info] Input '{src_path.name}' changed → cleaning pages/output for stem '{stem}'.")
            for sub in ("pages", "output"):
                p = work_root / sub / stem
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
                    print(f"[clean] Removed: {p}")

    input_dir = (input_root / stem); input_dir.mkdir(parents=True, exist_ok=True)
    copied = (input_dir / src_path.name).resolve()
    if not copied.exists() or copied.read_bytes() != src_path.read_bytes():
        shutil.copy2(src_path, copied)
    print(f"[work] Input copied to: {copied}")

    if copied.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image_paths = prep_existing_png(copied, pages_root, stem)
    else:
        image_paths = pdf_to_clean_pngs(copied, pages_root, stem, dpi=args.dpi, max_pages=args.max_pages)

    model, tok, proc = load_local_model(local_dir)
    page_texts, combined_md = run_chat_on_images(model, tok, proc, image_paths, args.prompt, args.max_new_tokens, debug=args.debug)

    print("\n===== MODEL OUTPUT =====\n")
    print(combined_md)
    print("\n========================\n")

    if args.save:
        out_dir = save_outputs(stem, image_paths, page_texts, combined_md, work_root)
        write_manifest(work_root, stem, copied, num_pages=len(image_paths))
    else:
        print("[save] DISABLED (pass --save to write files)")

if __name__ == "__main__":
    main()

