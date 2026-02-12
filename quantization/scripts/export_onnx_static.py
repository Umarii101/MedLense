"""
Export BiomedCLIP Vision Encoder to ONNX with Static Batch Size
Then convert to TFLite using onnx2tf
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

project_root = Path(__file__).parent.parent
deployment_dir = project_root / "deployment"
onnx_dir = deployment_dir / "onnx"
onnx_dir.mkdir(parents=True, exist_ok=True)


class VisionEncoderWrapper(nn.Module):
    """Wrapper for BiomedCLIP vision encoder."""
    
    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
    
    def forward(self, x):
        features = self.visual(x)
        features = features / features.norm(dim=-1, keepdim=True)
        return features


def export_onnx_static():
    """Export vision encoder to ONNX with static batch size."""
    from open_clip import create_model_from_pretrained
    
    print("Loading BiomedCLIP...")
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.eval()
    
    vision_encoder = VisionEncoderWrapper(model)
    vision_encoder.eval()
    
    params = sum(p.numel() for p in vision_encoder.parameters()) / 1e6
    print(f"✓ Vision encoder: {params:.1f}M parameters")
    
    # Static batch size = 1
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Test PyTorch
    with torch.no_grad():
        pytorch_output = vision_encoder(dummy_input)
    print(f"PyTorch output: {pytorch_output.shape}")
    
    # Export to ONNX with static batch
    onnx_path = onnx_dir / "biomedclip_vision_static.onnx"
    print(f"\nExporting to ONNX: {onnx_path.name}")
    
    torch.onnx.export(
        vision_encoder,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['embedding'],
        # NO dynamic axes - everything is static
    )
    
    size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✓ ONNX exported: {size_mb:.1f} MB")
    
    # Verify ONNX
    import onnx
    import onnxruntime as ort
    
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validated")
    
    # Test with ONNX Runtime
    session = ort.InferenceSession(str(onnx_path))
    ort_output = session.run(None, {'image': dummy_input.numpy()})[0]
    
    # Compare outputs
    diff = np.abs(pytorch_output.numpy() - ort_output).max()
    print(f"PyTorch vs ONNX diff: {diff:.6f}")
    
    return onnx_path


if __name__ == "__main__":
    print("=" * 60)
    print("BiomedCLIP ONNX Export (Static Batch)")
    print("=" * 60)
    
    onnx_path = export_onnx_static()
    
    print("\n" + "=" * 60)
    print("Next: Convert to TFLite")
    print("=" * 60)
    print(f"Run: onnx2tf -i {onnx_path} -o deployment/tflite_output -kat -osd")
