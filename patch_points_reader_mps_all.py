# patch_points_reader_mps_all.py
from pathlib import Path
import re

p = Path("POINTS-Reader-local") / "modeling_pointsv15_chat.py"
s = p.read_text()

# (A) Force eager attention everywhere (in case a new copy reintroduced FA2)
s = s.replace('config.llm_config._attn_implementation = "flash_attention_2"',
              'config.llm_config._attn_implementation = "eager"')
s = s.replace('attn_implementation="flash_attention_2"',
              'attn_implementation="eager"')

# (B) Fix dtype→device order (MPS can’t accept float64 on move)
s = s.replace(".to(self.vision_encoder.device).to(self.vision_encoder.dtype)",
              ".to(self.vision_encoder.dtype).to(self.vision_encoder.device)")

# (C) Remove ALL CUDA-only tensor/device moves
# 1) .cuda()  -->  .to(self.vision_encoder.device)
s = re.sub(r"\.cuda\(\)", ".to(self.vision_encoder.device)", s)

# 2) .to('cuda') / .to(\"cuda\")  -->  .to(self.vision_encoder.device)
s = re.sub(r"\.to\((['\"])cuda\1\)", ".to(self.vision_encoder.device)", s)

# 3) torch.device('cuda')  -->  self.vision_encoder.device
s = re.sub(r"torch\.device\((['\"])cuda\1\)", "self.vision_encoder.device", s)

# 4) device='cuda' kwargs  -->  device=self.vision_encoder.device
s = re.sub(r"device\s*=\s*(['\"])cuda\1", "device=self.vision_encoder.device", s)

# 5) common pattern: .half().cuda()  ->  .to(self.vision_encoder.dtype).to(self.vision_encoder.device)
s = s.replace(".half().cuda()",
              ".to(self.vision_encoder.dtype).to(self.vision_encoder.device)")

# (D) Optional guard: raise a clear error if no images given
s = s.replace(
    "image_grid_thws = np.concatenate(image_grid_thws, axis=0)",
    "assert len(image_grid_thws) > 0, 'No images provided in messages.'\n        image_grid_thws = np.concatenate(image_grid_thws, axis=0)"
)

p.write_text(s)
print("Patched:", p)

