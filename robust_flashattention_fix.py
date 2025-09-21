#!/usr/bin/env python3
"""
Robust FlashAttention Fix for POINTS-Reader
This forcefully disables FlashAttention2 by modifying the config
"""

import torch
import os
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2VLImageProcessor, AutoConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_points_reader_robust(use_quantization=False):
    """
    Robust model loading that forcefully disables FlashAttention2
    """
    model_path = 'tencent/POINTS-Reader'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"🔧 Loading POINTS-Reader with robust FlashAttention fix...")
    print(f"   Device: {device}")
    print(f"   Quantization: {'Yes' if use_quantization else 'No'}")
    
    try:
        # METHOD 1: Load and modify config first
        print("📋 Step 1: Loading and modifying model configuration...")
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Forcefully disable FlashAttention in config
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'
            print("   ✅ Set config._attn_implementation = 'eager'")
        
        if hasattr(config, 'use_flash_attention_2'):
            config.use_flash_attention_2 = False
            print("   ✅ Set config.use_flash_attention_2 = False")
            
        # Also check nested configs (for multi-part models like POINTS)
        if hasattr(config, 'llm_config'):
            if hasattr(config.llm_config, '_attn_implementation'):
                config.llm_config._attn_implementation = 'eager'
                print("   ✅ Set config.llm_config._attn_implementation = 'eager'")
            if hasattr(config.llm_config, 'use_flash_attention_2'):
                config.llm_config.use_flash_attention_2 = False
                print("   ✅ Set config.llm_config.use_flash_attention_2 = False")
        
        # METHOD 2: Set environment variable as backup
        print("🌍 Step 2: Setting environment variable...")
        os.environ['TRANSFORMERS_ATTENTION_TYPE'] = 'eager'
        print("   ✅ Set TRANSFORMERS_ATTENTION_TYPE=eager")
        
        # METHOD 3: Load model with multiple fallback strategies
        print("📦 Step 3: Loading model with modified config...")
        
        model_kwargs = {
            'config': config,
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
            'attn_implementation': 'eager',  # Primary method
        }
        
        if use_quantization:
            try:
                import bitsandbytes
                model_kwargs.update({
                    'load_in_8bit': True,
                    'device_map': 'auto',
                    'torch_dtype': torch.float16,
                })
                print("   📉 Using 8-bit quantization")
            except ImportError:
                print("   ⚠️  bitsandbytes not installed, installing...")
                os.system("pip install bitsandbytes")
                import bitsandbytes
                model_kwargs.update({
                    'load_in_8bit': True,
                    'device_map': 'auto',
                    'torch_dtype': torch.float16,
                })
        else:
            # Standard loading for Mac
            torch_dtype = torch.float32 if device == 'mps' else torch.float16
            model_kwargs.update({
                'torch_dtype': torch_dtype,
                'device_map': device if device != 'mps' else None,
            })
        
        # Try loading with all our fixes
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Manual device placement for MPS
        if device == 'mps' and not use_quantization:
            model = model.to('mps')
            print("   📱 Moved model to MPS device")
        
        print("✅ Model loaded successfully!")
        
        # Load tokenizer and processor
        print("📝 Loading tokenizer and image processor...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
        print("✅ Tokenizer and image processor loaded!")
        
        return model, tokenizer, image_processor
        
    except Exception as e:
        print(f"❌ Robust loading failed: {e}")
        
        # FALLBACK METHOD: Try with torch.no_grad and manual config override
        print("\n🆘 Trying fallback method...")
        try:
            # Clear any cached modules that might have the wrong config
            import importlib
            import sys
            modules_to_clear = [name for name in sys.modules if 'flash' in name.lower()]
            for module_name in modules_to_clear:
                if module_name in sys.modules:
                    del sys.modules[module_name]
            
            # Force disable flash attention globally
            import transformers.models.qwen2.modeling_qwen2 as qwen2_modeling
            
            # Monkey patch to force eager attention
            original_init = qwen2_modeling.Qwen2Model.__init__
            
            def patched_init(self, config):
                # Force eager attention before calling original init
                if hasattr(config, '_attn_implementation'):
                    config._attn_implementation = 'eager'
                if hasattr(config, 'use_flash_attention_2'):
                    config.use_flash_attention_2 = False
                return original_init(self, config)
            
            qwen2_modeling.Qwen2Model.__init__ = patched_init
            
            print("🔧 Applied monkey patch to force eager attention")
            
            # Try loading again
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32 if device == 'mps' else torch.float16,
                low_cpu_mem_usage=True,
                device_map=device if device != 'mps' else None,
            )
            
            if device == 'mps':
                model = model.to('mps')
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            image_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
            
            print("✅ Fallback method successful!")
            return model, tokenizer, image_processor
            
        except Exception as fallback_error:
            print(f"❌ Fallback method also failed: {fallback_error}")
            raise

def test_model_loading():
    """Test the robust model loading"""
    print("🧪 Testing Robust POINTS-Reader Loading")
    print("=" * 50)
    
    # Check memory first
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"💾 Available memory: {available_gb:.1f} GB")
    
    # Decide on quantization
    use_quant = available_gb < 8
    if use_quant:
        print("📉 Using quantization due to limited memory")
    else:
        print("🚀 Using standard precision")
    
    try:
        # Load model with robust method
        model, tokenizer, image_processor = load_points_reader_robust(use_quantization=use_quant)
        
        print(f"\n✅ SUCCESS! Model loaded without FlashAttention errors")
        
        # Quick test
        print(f"\n🧪 Running quick test...")
        
        # Create test image
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), "Test Invoice\nTotal: $99.99", fill='black')
        test_path = "/tmp/test_robust.png"
        img.save(test_path)
        
        # Test inference
        content = [
            {'type': 'image', 'image': test_path},
            {'type': 'text', 'text': 'Extract the text from this image.'}
        ]
        messages = [{'role': 'user', 'content': content}]
        
        response = model.chat(
            messages,
            tokenizer,
            image_processor,
            {'max_new_tokens': 100, 'do_sample': False}
        )
        
        print(f"📄 Test result: {response}")
        print(f"\n🎉 Everything working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
        # Additional troubleshooting info
        print(f"\n🔍 Troubleshooting Information:")
        print(f"   Python version: {os.sys.version}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        # Check transformers version
        import transformers
        print(f"   Transformers version: {transformers.__version__}")
        
        if float(transformers.__version__.split('.')[0]) < 4 or float(transformers.__version__.split('.')[1]) < 55:
            print(f"   ⚠️  Consider upgrading transformers: pip install transformers>=4.55.0")
        
        return False

if __name__ == "__main__":
    success = test_model_loading()
    
    if success:
        print(f"\n📋 Next Steps:")
        print(f"   1. Copy the load_points_reader_robust() function")
        print(f"   2. Replace your model loading code with this function")
        print(f"   3. Your MCP app should work now!")
    else:
        print(f"\n💡 If this still fails, try:")
        print(f"   1. pip install --upgrade transformers")
        print(f"   2. pip install --upgrade torch")
        print(f"   3. Restart your Python environment")
        print(f"   4. Clear HuggingFace cache: rm -rf ~/.cache/huggingface/")
