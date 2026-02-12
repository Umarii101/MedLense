"""
Test BiomedCLIP INT8 model for edge deployment.
Validates model loading, inference, and accuracy against FP32 baseline.

Usage:
    python tests/test_biomedclip.py
"""
import os
import sys
import time
import numpy as np

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "edge_deployment", "models", "biomedclip")
FP32_PATH = os.path.join(MODELS_DIR, "biomedclip_vision.onnx")
INT8_PATH = os.path.join(MODELS_DIR, "biomedclip_vision_int8.onnx")


def check_dependencies():
    """Check required packages."""
    try:
        import onnxruntime as ort
        print(f"  ONNX Runtime: {ort.__version__}")
        return True
    except ImportError:
        print("  ERROR: onnxruntime not installed")
        print("  Install: pip install onnxruntime")
        return False


def check_models():
    """Check model files exist."""
    models = {"FP32": FP32_PATH, "INT8": INT8_PATH}
    available = {}
    
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name}: {size_mb:.1f} MB")
            available[name] = path
        else:
            print(f"  {name}: NOT FOUND")
    
    return available


def test_inference(session, input_name, output_name, test_input, runs=10):
    """Run inference and return output with timing."""
    # Warmup
    _ = session.run([output_name], {input_name: test_input})
    
    # Benchmark
    start = time.time()
    for _ in range(runs):
        output = session.run([output_name], {input_name: test_input})[0]
    elapsed = (time.time() - start) / runs * 1000
    
    return output, elapsed


def cosine_similarity(a, b):
    """Compute cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    print("=" * 60)
    print("BiomedCLIP Edge Deployment Test")
    print("=" * 60)
    
    # Check dependencies
    print("\n[1/4] Checking dependencies...")
    if not check_dependencies():
        return 1
    
    import onnxruntime as ort
    
    # Check models
    print("\n[2/4] Checking models...")
    available = check_models()
    
    if "INT8" not in available:
        print("\n  ERROR: INT8 model required for testing")
        return 1
    
    # Load models
    print("\n[3/4] Loading models...")
    session_int8 = ort.InferenceSession(available["INT8"])
    input_name = session_int8.get_inputs()[0].name
    output_name = session_int8.get_outputs()[0].name
    
    print(f"  Input: {input_name} {session_int8.get_inputs()[0].shape}")
    print(f"  Output: {output_name} {session_int8.get_outputs()[0].shape}")
    
    session_fp32 = None
    if "FP32" in available:
        session_fp32 = ort.InferenceSession(available["FP32"])
    
    # Test inference
    print("\n[4/4] Running inference tests...")
    
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # INT8 inference
    output_int8, time_int8 = test_inference(
        session_int8, input_name, output_name, test_input
    )
    print(f"\n  INT8 Inference: {time_int8:.2f} ms")
    print(f"  Output shape: {output_int8.shape}")
    print(f"  Output range: [{output_int8.min():.4f}, {output_int8.max():.4f}]")
    
    # Compare with FP32 if available
    if session_fp32:
        output_fp32, time_fp32 = test_inference(
            session_fp32, input_name, output_name, test_input
        )
        
        cos_sim = cosine_similarity(output_fp32.flatten(), output_int8.flatten())
        max_diff = np.max(np.abs(output_fp32 - output_int8))
        
        print(f"\n  FP32 Inference: {time_fp32:.2f} ms")
        print(f"  Cosine Similarity: {cos_sim:.6f}")
        print(f"  Max Difference: {max_diff:.6f}")
        print(f"  Speedup: {time_fp32/time_int8:.2f}x")
        print(f"  Output fp32: min {output_fp32.min():.4f}, max {output_fp32.max():.4f}")
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    
    int8_size = os.path.getsize(available["INT8"]) / (1024 * 1024)
    
    passed = True
    if session_fp32:
        cos_sim = cosine_similarity(output_fp32.flatten(), output_int8.flatten())
        if cos_sim < 0.99:
            print(f"  [WARN] Accuracy degraded: {cos_sim:.4f}")
            passed = cos_sim > 0.95
        else:
            print(f"  [PASS] Accuracy excellent: {cos_sim:.6f}")
    
    print(f"  [INFO] Model size: {int8_size:.1f} MB")
    print(f"  [INFO] Inference time: {time_int8:.2f} ms")
    
    if passed:
        print("\n  STATUS: ALL TESTS PASSED")
        return 0
    else:
        print("\n  STATUS: TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
