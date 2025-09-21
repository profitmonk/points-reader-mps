# patch_fix_mps_dtype.py
from pathlib import Path

p = Path("POINTS-Reader-local") / "modeling_pointsv15_chat.py"
s = p.read_text()

# swap order: device->dtype  =>  dtype->device
s = s.replace(
    ".to(self.vision_encoder.device).to(self.vision_encoder.dtype)",
    ".to(self.vision_encoder.dtype).to(self.vision_encoder.device)"
)

# (optional) also catch any similar occurrences, just in case:
s = s.replace(
    ".to(self.vision_encoder.device).to(self.vision_encoder.dtype) # noqa",
    ".to(self.vision_encoder.dtype).to(self.vision_encoder.device) # noqa"
)

p.write_text(s)
print("Patched dtype/device order at:", p)

