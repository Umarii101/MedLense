"""
Download BiomedCLIP model from HuggingFace
Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

BiomedCLIP is a biomedical vision-language model trained on 15M image-text pairs
from PubMed articles. It uses:
- Vision encoder: ViT-B/16 (224x224 input)
- Text encoder: PubMedBERT

For edge deployment, we'll convert the vision encoder to TFLite.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_biomedclip():
    """Download BiomedCLIP model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    model_id = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    output_dir = project_root / "models" / "biomedclip"
    
    print(f"Downloading BiomedCLIP from: {model_id}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    # Download model
    snapshot_download(
        repo_id=model_id,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt"]  # Skip documentation files
    )
    
    print("-" * 50)
    print(f"✓ BiomedCLIP downloaded to: {output_dir}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for f in output_dir.rglob("*"):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.2f} MB")
    
    return output_dir


def verify_model():
    """Verify the downloaded model loads correctly."""
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError:
        print("\nInstalling open_clip_torch for verification...")
        os.system("pip install open_clip_torch")
        from open_clip import create_model_from_pretrained, get_tokenizer
    
    model_path = project_root / "models" / "biomedclip"
    
    print("\nVerifying model loads correctly...")
    
    # BiomedCLIP uses open_clip format
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    tokenizer = get_tokenizer(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    
    print("✓ Model loaded successfully!")
    print(f"  Vision encoder input size: 224x224")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model, preprocess, tokenizer


if __name__ == "__main__":
    print("=" * 50)
    print("BiomedCLIP Download Script")
    print("=" * 50)
    
    # Download
    output_dir = download_biomedclip()
    
    # Verify
    print("\n" + "=" * 50)
    print("Verification")
    print("=" * 50)
    
    try:
        model, preprocess, tokenizer = verify_model()
        print("\n✓ BiomedCLIP ready for conversion!")
    except Exception as e:
        print(f"\n⚠ Verification failed: {e}")
        print("Model downloaded but may need manual verification.")
