# run_points_reader_mps.py
import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

MODEL_PATH = "POINTS-Reader-local"  # your patched local folder from earlier
IMAGE_PATH = "./123.png"  # <- make sure this exists

tok  = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
proc = Qwen2VLImageProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

print("attn impl (top):", getattr(model.config, "_attn_implementation", None))
if getattr(model.config, "llm_config", None) is not None:
    print("attn impl (llm):", getattr(model.config.llm_config, "_attn_implementation", None))
print("device:", next(model.parameters()).device)

prompt = "Extract all text. Tables as HTML; other text as Markdown."
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": IMAGE_PATH},
        {"type": "text",  "text": prompt}
]}]

with torch.inference_mode():
    out = model.chat(messages, tok, proc, dict(max_new_tokens=512, temperature=0.0))
print(out)

