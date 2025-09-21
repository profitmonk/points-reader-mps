# test_installation.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")

# Test model loading (this will download ~6GB)
try:
    tokenizer = AutoTokenizer.from_pretrained('tencent/POINTS-Reader', trust_remote_code=True)
    print("✅ Model components accessible")
except Exception as e:
    print(f"❌ Error: {e}")
