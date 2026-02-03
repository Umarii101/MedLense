"""
Test MedGemma Q4_K_S quantized model for edge deployment.
Validates model loading and basic inference with llama-cpp-python.

Usage:
    python tests/test_medgemma.py
"""
import os
import sys
import time

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR, "edge_deployment", "models", "medgemma",
    "medgemma-4b-q4_k_s-final.gguf"
)


def check_dependencies():
    """Check required packages."""
    try:
        from llama_cpp import Llama
        print("  llama-cpp-python: OK")
        return True
    except ImportError:
        print("  ERROR: llama-cpp-python not installed")
        print("  Install: pip install llama-cpp-python")
        return False


def check_model():
    """Check model file exists."""
    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / (1024 ** 3)
        print(f"  Model: {size_gb:.2f} GB")
        return True
    else:
        print(f"  ERROR: Model not found at {MODEL_PATH}")
        return False


def test_inference():
    """Test basic inference."""
    from llama_cpp import Llama

    print("\n  Loading model (this may take 30-60 seconds)...")
    start = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )

    load_time = time.time() - start
    print(f"  Model loaded in {load_time:.1f}s")

    # Test prompt using Gemma 3 chat template
    # Note: llama.cpp adds <bos> automatically, so we don't include it
    prompt = """<start_of_turn>user
You are a medical AI assistant. A patient reports: "I have a persistent dry cough for 2 weeks."
Provide a brief assessment.<end_of_turn>
<start_of_turn>model
"""

    print("\n  Running inference...")
    start = time.time()

    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.7,
        stop=["<end_of_turn>", "<eos>"],
    )

    inference_time = time.time() - start
    response = output["choices"][0]["text"].strip()
    tokens = output["usage"]["completion_tokens"]

    print(f"  Inference time: {inference_time:.2f}s")
    print(f"  Tokens generated: {tokens}")
    print(f"  Speed: {tokens/inference_time:.1f} tok/s")

    print(f"\n  Response preview:")
    preview = response[:200] if len(response) > 200 else response
    print(f"  {preview}...")

    return {
        "load_time": load_time,
        "inference_time": inference_time,
        "tokens": tokens,
        "speed": tokens / inference_time,
    }


def main():
    print("=" * 60)
    print("MedGemma Q4_K_S Edge Deployment Test")
    print("=" * 60)

    # Check dependencies
    print("\n[1/3] Checking dependencies...")
    if not check_dependencies():
        return 1

    # Check model
    print("\n[2/3] Checking model...")
    if not check_model():
        return 1

    # Test inference
    print("\n[3/3] Testing inference...")
    try:
        results = test_inference()
    except Exception as e:
        print(f"  ERROR: {e}")
        return 1

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    model_size = os.path.getsize(MODEL_PATH) / (1024 ** 3)

    print(f"  [INFO] Model size: {model_size:.2f} GB")
    print(f"  [INFO] Load time: {results['load_time']:.1f}s")
    print(f"  [INFO] Speed: {results['speed']:.1f} tok/s")

    if results["speed"] > 1.0:
        print(f"  [PASS] Inference working")
        print("\n  STATUS: ALL TESTS PASSED")
        return 0
    else:
        print(f"  [WARN] Speed below expected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
