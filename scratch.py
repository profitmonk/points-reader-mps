import os, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

MODEL_ID = "tencent/POINTS-Reader"

def force_eager(conf):
    # hard-disable FA2 and pick eager everywhere the model/Transformers might look
    for k in ("use_flash_attention_2", "flash_attn_2_enabled"):
        if hasattr(conf, k):
            setattr(conf, k, False)
    for k in ("attn_implementation", "_attn_implementation", "_attn_implementation_internal"):
        if hasattr(conf, k):
            setattr(conf, k, "eager")

# 1) Load and sanitize configs BEFORE creating the model
cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, revision="main")
force_eager(cfg)
if getattr(cfg, "llm_config", None) is not None:
    force_eager(cfg.llm_config)

# 2) Load tokenizer & processor
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
proc = Qwen2VLImageProcessor.from_pretrained(MODEL_ID)

# 3) Create model with sanitized config (DON'T pass attn_implementation kwarg here)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    config=cfg,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)
model.to(device)

print("final attn (top):", getattr(model.config, "_attn_implementation", None),
      getattr(model.config, "_attn_implementation_internal", None),
      getattr(model.config, "use_flash_attention_2", None))
if getattr(model.config, "llm_config", None) is not None:
    print("final attn (llm):", getattr(model.config.llm_config, "_attn_implementation", None),
          getattr(model.config.llm_config, "_attn_implementation_internal", None),
          getattr(model.config.llm_config, "use_flash_attention_2", None))

# quick smoke run without images (just ensures forward graph binds)
messages = [{"role": "user", "content": [{"type":"text","text":"Say 'ready' if you loaded."}]}]
with torch.inference_mode():
    out = model.chat(messages, tok, proc, dict(max_new_tokens=4, temperature=0.0))
print(out)

