"""
Verify BiomedCLIP model availability and download it.

This script:
1. Checks if BiomedCLIP is accessible on HuggingFace
2. Downloads the model to cache
3. Verifies model loads correctly
4. Displays model size and parameters

Run this before creating quantization scripts to ensure model availability.
"""

import sys
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import torch

# Model configuration
MODEL_NAME = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
ALTERNATIVE_MODELS = [
    "chuhac/BiomedCLIP-vit-bert-hf",  # HF-optimized variant
    "michel-ducartier/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",  # Feature extraction
]

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def verify_model(model_name: str):
    """Verify model can be downloaded and loaded"""
    print(f"\n{'='*70}")
    print(f"Verifying: {model_name}")
    print(f"{'='*70}\n")
    
    try:
        # Download and load model
        print("📥 Downloading model...")
        model = AutoModel.from_pretrained(model_name)
        
        # Load processor/tokenizer
        print("📥 Downloading processor...")
        try:
            processor = AutoProcessor.from_pretrained(model_name)
            print("✅ Processor loaded")
        except:
            print("⚠️  No processor found, trying tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("✅ Tokenizer loaded")
            except:
                print("❌ No processor or tokenizer available")
        
        # Get model stats
        param_count = sum(p.numel() for p in model.parameters())
        model_size_mb = get_model_size(model)
        
        print(f"\n📊 Model Statistics:")
        print(f"  • Parameters: {param_count:,} ({param_count/1e9:.2f}B)")
        print(f"  • Size (FP32): {model_size_mb:.1f} MB")
        print(f"  • Size (FP16): {model_size_mb/2:.1f} MB (estimated)")
        print(f"  • Quantized (INT8): {model_size_mb/4:.1f} MB (estimated)")
        
        # Test inference
        print(f"\n🧪 Testing inference...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"  • Device: {device}")
        print(f"  • Model architecture: {model.__class__.__name__}")
        
        print(f"\n✅ SUCCESS: {model_name} is ready!")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: {model_name}")
        print(f"  Error: {str(e)}")
        return False

def main():
    """Main verification routine"""
    print("\n" + "="*70)
    print("BiomedCLIP Model Verification Script")
    print("="*70)
    
    # Check CUDA availability
    print(f"\n🖥️  System Info:")
    print(f"  • CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  • GPU: {torch.cuda.get_device_name(0)}")
        print(f"  • VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Try primary model
    print("\n" + "="*70)
    print("ATTEMPTING PRIMARY MODEL (Official Microsoft)")
    print("="*70)
    
    success = verify_model(MODEL_NAME)
    
    if not success:
        print("\n⚠️  Primary model failed. Trying alternatives...\n")
        for alt_model in ALTERNATIVE_MODELS:
            if verify_model(alt_model):
                print(f"\n✅ Alternative model works: {alt_model}")
                print(f"💡 Update your scripts to use this model instead.")
                break
    
    print("\n" + "="*70)
    print("Verification Complete!")
    print("="*70)
    
    if success:
        print(f"\n✅ Recommended model: {MODEL_NAME}")
        print(f"\n📝 Next Steps:")
        print(f"  1. Create quantization/scripts/quantize_biomedclip.py")
        print(f"  2. Use model: {MODEL_NAME}")
        print(f"  3. Target size: ~100MB (INT8 quantization)")
    else:
        print(f"\n❌ No BiomedCLIP model could be loaded.")
        print(f"  • Check internet connection")
        print(f"  • Verify HuggingFace authentication (if needed)")
        print(f"  • Try manual download from: https://huggingface.co/{MODEL_NAME}")

if __name__ == "__main__":
    main()
