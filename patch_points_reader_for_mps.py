from huggingface_hub import snapshot_download
from pathlib import Path

local_dir = Path("POINTS-Reader-local")
local_dir.mkdir(exist_ok=True)
model_dir = Path(snapshot_download("tencent/POINTS-Reader", local_dir=local_dir.as_posix()))

code = (model_dir / "modeling_pointsv15_chat.py").read_text()

# 1) Hard-coded FA2 on the text LLM -> eager
code = code.replace(
    'config.llm_config._attn_implementation = "flash_attention_2"',
    'config.llm_config._attn_implementation = "eager"'
)

# 2) Hard-coded FA2 on the vision tower -> eager
code = code.replace(
    'attn_implementation="flash_attention_2"',
    'attn_implementation="eager"'
)

# 3) CUDA-only tensor move in chat(...) -> move to the vision encoder’s device
code = code.replace(
    ".cuda()\n\n    .long()",
    ".to(self.vision_encoder.device)\n\n    .long()"
)

(model_dir / "modeling_pointsv15_chat.py").write_text(code)
print("Patched:", model_dir)

