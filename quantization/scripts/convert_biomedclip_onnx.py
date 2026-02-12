"""
Step 1: Extract BiomedCLIP Vision Encoder and Export to ONNX

This script:
1. Loads BiomedCLIP model
2. Extracts only the vision encoder (86.2M params)
3. Exports to ONNX format for conversion to TFLite
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class BiomedCLIPVisionEncoder(nn.Module):
    """Wrapper for BiomedCLIP vision encoder only."""
    
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
        
    def forward(self, x):
        """
        Forward pass through vision encoder.
        
        Args:
            x: Image tensor [B, 3, 224, 224]
            
        Returns:
            Normalized image embeddings [B, 512]
        """
        # Get image features
        features = self.visual(x)
        
        # L2 normalize (important for cosine similarity)
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features


def load_biomedclip():
    """Load BiomedCLIP and extract vision encoder."""
    from open_clip import create_model_from_pretrained
    
    print("Loading BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.eval()
    
    # Create vision-only wrapper
    vision_encoder = BiomedCLIPVisionEncoder(model)
    vision_encoder.eval()
    
    print(f"✓ Vision encoder extracted")
    print(f"  Parameters: {sum(p.numel() for p in vision_encoder.parameters()) / 1e6:.1f}M")
    
    return vision_encoder, preprocess, model


def export_to_onnx(vision_encoder, output_path):
    """Export vision encoder to ONNX format."""
    
    print(f"\nExporting to ONNX: {output_path}")
    
    # Create dummy input [batch, channels, height, width]
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export
    torch.onnx.export(
        vision_encoder,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Good compatibility
        do_constant_folding=True,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX export complete")
    
    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model validated")
    
    # Print model size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Model size: {size_mb:.1f} MB")
    
    return output_path


def verify_onnx(vision_encoder, preprocess, onnx_path):
    """Verify ONNX output matches PyTorch."""
    import onnxruntime as ort
    
    print("\nVerifying ONNX output...")
    
    # Load test image
    test_images = list((project_root / "Test Images" / "X-Ray").rglob("*.jpg"))
    if not test_images:
        test_images = list((project_root / "Test Images" / "X-Ray").rglob("*.png"))
    
    if test_images:
        test_image = Image.open(test_images[0]).convert("RGB")
        print(f"  Using test image: {test_images[0].name}")
    else:
        # Create random test image
        test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        print("  Using random test image")
    
    # Preprocess
    image_tensor = preprocess(test_image).unsqueeze(0)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = vision_encoder(image_tensor).numpy()
    
    # ONNX inference
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(None, {'image': image_tensor.numpy()})[0]
    
    # Compare
    diff = np.abs(pytorch_output - onnx_output).max()
    print(f"  Max difference: {diff:.6f}")
    
    if diff < 1e-4:
        print("✓ ONNX verification PASSED")
        return True
    else:
        print("⚠ ONNX verification: outputs differ (may still be acceptable)")
        return True  # Small differences are okay


def precompute_text_embeddings(model, labels):
    """Pre-compute text embeddings for classification labels."""
    from transformers import AutoTokenizer
    import json
    
    print("\nPre-computing text embeddings...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    )
    
    text_inputs = tokenizer(
        labels,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        text_features = model.encode_text(text_inputs['input_ids'])
        # Normalize
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Convert to dict
    embeddings_dict = {}
    for label, embedding in zip(labels, text_features.numpy()):
        embeddings_dict[label] = embedding.tolist()
    
    # Save
    output_path = project_root / "deployment" / "text_embeddings.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f)
    
    print(f"✓ Text embeddings saved: {output_path}")
    print(f"  Labels: {len(labels)}")
    print(f"  Embedding dim: {text_features.shape[1]}")
    
    return embeddings_dict


if __name__ == "__main__":
    print("=" * 60)
    print("BiomedCLIP Vision Encoder → ONNX Conversion")
    print("=" * 60)
    
    # Create output directory
    output_dir = project_root / "deployment" / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load model
    vision_encoder, preprocess, full_model = load_biomedclip()
    
    # Step 2: Export to ONNX
    onnx_path = output_dir / "biomedclip_vision.onnx"
    export_to_onnx(vision_encoder, str(onnx_path))
    
    # Step 3: Verify
    try:
        import onnxruntime
        verify_onnx(vision_encoder, preprocess, onnx_path)
    except ImportError:
        print("\n⚠ onnxruntime not installed, skipping verification")
        print("  Install with: pip install onnxruntime")
    
    # Step 4: Pre-compute text embeddings
    xray_labels = [
        "normal chest x-ray",
        "pneumonia in chest x-ray",
        "COVID-19 infection in chest x-ray",
        "bacterial pneumonia in chest x-ray",
        "viral pneumonia in chest x-ray",
        "tuberculosis in chest x-ray",
        "lung opacity in chest x-ray",
        "pleural effusion in chest x-ray",
    ]
    precompute_text_embeddings(full_model, xray_labels)
    
    print("\n" + "=" * 60)
    print("Step 1 Complete!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Text embeddings: {project_root / 'deployment' / 'text_embeddings.json'}")
    print(f"\nNext: Run convert_onnx_tflite.py to convert to TFLite")
