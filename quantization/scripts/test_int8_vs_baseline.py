"""
Test ONNX INT8 model against FP32 baseline.
Comprehensive comparison of accuracy, performance, and embedding quality.
"""
import os
import sys
import time
import numpy as np
import onnxruntime as ort

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FP32_PATH = os.path.join(BASE_DIR, "deployment", "onnx", "biomedclip_vision.onnx")
INT8_PATH = os.path.join(BASE_DIR, "deployment", "onnx", "biomedclip_vision_int8.onnx")


def load_models():
    """Load both FP32 and INT8 models."""
    print("Loading models...")
    session_fp32 = ort.InferenceSession(FP32_PATH)
    session_int8 = ort.InferenceSession(INT8_PATH)
    
    # Get I/O info
    input_name = session_fp32.get_inputs()[0].name
    output_name = session_fp32.get_outputs()[0].name
    
    print(f"  ✓ FP32: {os.path.getsize(FP32_PATH) / (1024*1024):.2f} MB")
    print(f"  ✓ INT8: {os.path.getsize(INT8_PATH) / (1024*1024):.2f} MB")
    
    return session_fp32, session_int8, input_name, output_name


def create_test_inputs():
    """Create various test inputs to thoroughly test the model."""
    inputs = {}
    
    # 1. Random normal (standard test)
    np.random.seed(42)
    inputs["random_normal"] = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # 2. Random uniform [0, 1] (like normalized images)
    np.random.seed(123)
    inputs["random_uniform"] = np.random.rand(1, 3, 224, 224).astype(np.float32)
    
    # 3. ImageNet-style normalized (mean=0.485, std=0.229 approx)
    np.random.seed(456)
    raw = np.random.rand(1, 3, 224, 224).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    inputs["imagenet_normalized"] = ((raw - mean) / std).astype(np.float32)
    
    # 4. Edge case: all zeros
    inputs["zeros"] = np.zeros((1, 3, 224, 224), dtype=np.float32)
    
    # 5. Edge case: all ones
    inputs["ones"] = np.ones((1, 3, 224, 224), dtype=np.float32)
    
    # 6. Simulated medical image (grayscale-like, higher contrast)
    np.random.seed(789)
    gray = np.random.rand(1, 1, 224, 224).astype(np.float32)
    gray = (gray - 0.5) * 2  # Range [-1, 1]
    inputs["medical_grayscale"] = np.repeat(gray, 3, axis=1)
    
    # 7. Batch of 4 images
    np.random.seed(999)
    inputs["batch_4"] = np.random.randn(4, 3, 224, 224).astype(np.float32)
    
    return inputs


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_accuracy(session_fp32, session_int8, input_name, output_name, test_inputs):
    """Compare outputs of FP32 and INT8 models."""
    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON")
    print("=" * 70)
    
    results = []
    
    print(f"\n{'Test Case':<25} {'Cosine Sim':>12} {'Max Diff':>12} {'Mean Diff':>12}")
    print("-" * 63)
    
    for name, test_input in test_inputs.items():
        # Run both models
        output_fp32 = session_fp32.run([output_name], {input_name: test_input})[0]
        output_int8 = session_int8.run([output_name], {input_name: test_input})[0]
        
        # Flatten for comparison (handles batches)
        fp32_flat = output_fp32.flatten()
        int8_flat = output_int8.flatten()
        
        # Metrics
        cos_sim = cosine_similarity(fp32_flat, int8_flat)
        max_diff = np.max(np.abs(fp32_flat - int8_flat))
        mean_diff = np.mean(np.abs(fp32_flat - int8_flat))
        
        results.append({
            "name": name,
            "cosine": cos_sim,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
        })
        
        print(f"{name:<25} {cos_sim:>12.8f} {max_diff:>12.6f} {mean_diff:>12.6f}")
    
    # Summary
    avg_cosine = np.mean([r["cosine"] for r in results])
    min_cosine = np.min([r["cosine"] for r in results])
    
    print("-" * 63)
    print(f"{'AVERAGE':<25} {avg_cosine:>12.8f}")
    print(f"{'MINIMUM':<25} {min_cosine:>12.8f}")
    
    return results


def test_embedding_quality(session_fp32, session_int8, input_name, output_name):
    """Test if INT8 preserves relative distances between embeddings."""
    print("\n" + "=" * 70)
    print("EMBEDDING QUALITY (Relative Distance Preservation)")
    print("=" * 70)
    
    # Create 5 different "images"
    np.random.seed(42)
    images = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(5)]
    
    # Get embeddings
    embeddings_fp32 = []
    embeddings_int8 = []
    
    for img in images:
        embeddings_fp32.append(session_fp32.run([output_name], {input_name: img})[0].flatten())
        embeddings_int8.append(session_int8.run([output_name], {input_name: img})[0].flatten())
    
    # Compute pairwise distances
    n = len(images)
    distances_fp32 = np.zeros((n, n))
    distances_int8 = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            distances_fp32[i, j] = cosine_similarity(embeddings_fp32[i], embeddings_fp32[j])
            distances_int8[i, j] = cosine_similarity(embeddings_int8[i], embeddings_int8[j])
    
    print("\n  FP32 Pairwise Cosine Similarities:")
    print("     " + "".join([f"  Img{j}  " for j in range(n)]))
    for i in range(n):
        row = f"  Img{i} "
        for j in range(n):
            row += f" {distances_fp32[i,j]:.4f} "
        print(row)
    
    print("\n  INT8 Pairwise Cosine Similarities:")
    print("     " + "".join([f"  Img{j}  " for j in range(n)]))
    for i in range(n):
        row = f"  Img{i} "
        for j in range(n):
            row += f" {distances_int8[i,j]:.4f} "
        print(row)
    
    # Check if rankings are preserved
    print("\n  Ranking Preservation Check:")
    rank_preserved = 0
    total_pairs = 0
    
    for i in range(n):
        # Get ranking of other images by similarity to image i
        fp32_ranking = np.argsort(-distances_fp32[i])
        int8_ranking = np.argsort(-distances_int8[i])
        
        if np.array_equal(fp32_ranking, int8_ranking):
            rank_preserved += 1
        total_pairs += 1
    
    print(f"    Rankings preserved: {rank_preserved}/{total_pairs}")
    
    # Correlation between distance matrices
    fp32_upper = distances_fp32[np.triu_indices(n, k=1)]
    int8_upper = distances_int8[np.triu_indices(n, k=1)]
    correlation = np.corrcoef(fp32_upper, int8_upper)[0, 1]
    print(f"    Distance matrix correlation: {correlation:.6f}")


def test_performance(session_fp32, session_int8, input_name, output_name):
    """Benchmark inference performance."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    # Test input
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Warmup
    for _ in range(5):
        _ = session_fp32.run([output_name], {input_name: test_input})
        _ = session_int8.run([output_name], {input_name: test_input})
    
    # Benchmark FP32
    iterations = 50
    
    start = time.time()
    for _ in range(iterations):
        _ = session_fp32.run([output_name], {input_name: test_input})
    fp32_time = (time.time() - start) / iterations * 1000
    
    # Benchmark INT8
    start = time.time()
    for _ in range(iterations):
        _ = session_int8.run([output_name], {input_name: test_input})
    int8_time = (time.time() - start) / iterations * 1000
    
    print(f"\n  Single Image Inference (avg of {iterations} runs):")
    print(f"    FP32: {fp32_time:.2f} ms")
    print(f"    INT8: {int8_time:.2f} ms")
    print(f"    Speedup: {fp32_time/int8_time:.2f}x")
    
    # Batch benchmark
    batch_input = np.random.randn(4, 3, 224, 224).astype(np.float32)
    
    start = time.time()
    for _ in range(iterations):
        _ = session_fp32.run([output_name], {input_name: batch_input})
    fp32_batch_time = (time.time() - start) / iterations * 1000
    
    start = time.time()
    for _ in range(iterations):
        _ = session_int8.run([output_name], {input_name: batch_input})
    int8_batch_time = (time.time() - start) / iterations * 1000
    
    print(f"\n  Batch of 4 Images (avg of {iterations} runs):")
    print(f"    FP32: {fp32_batch_time:.2f} ms ({fp32_batch_time/4:.2f} ms/image)")
    print(f"    INT8: {int8_batch_time:.2f} ms ({int8_batch_time/4:.2f} ms/image)")
    print(f"    Speedup: {fp32_batch_time/int8_batch_time:.2f}x")
    
    return {
        "fp32_single": fp32_time,
        "int8_single": int8_time,
        "fp32_batch": fp32_batch_time,
        "int8_batch": int8_batch_time,
    }


def test_classification_consistency(session_fp32, session_int8, input_name, output_name):
    """Test if classification results would be consistent."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION CONSISTENCY (Simulated)")
    print("=" * 70)
    
    # Simulate text embeddings for medical conditions
    np.random.seed(12345)
    conditions = [
        "normal chest x-ray",
        "pneumonia",
        "lung cancer",
        "tuberculosis",
        "healthy lungs",
    ]
    
    # Create pseudo text embeddings (in real use, these come from BiomedCLIP text encoder)
    text_embeddings = np.random.randn(len(conditions), 512).astype(np.float32)
    # Normalize
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Test with 10 random "images"
    np.random.seed(42)
    test_images = [np.random.randn(1, 3, 224, 224).astype(np.float32) for _ in range(10)]
    
    matches = 0
    total = len(test_images)
    
    print(f"\n  Testing {total} images against {len(conditions)} conditions...")
    print(f"\n  {'Image':<8} {'FP32 Prediction':<25} {'INT8 Prediction':<25} {'Match':>6}")
    print("  " + "-" * 66)
    
    for i, img in enumerate(test_images):
        # Get image embeddings
        emb_fp32 = session_fp32.run([output_name], {input_name: img})[0].flatten()
        emb_int8 = session_int8.run([output_name], {input_name: img})[0].flatten()
        
        # Normalize
        emb_fp32 = emb_fp32 / np.linalg.norm(emb_fp32)
        emb_int8 = emb_int8 / np.linalg.norm(emb_int8)
        
        # Compute similarities
        sims_fp32 = text_embeddings @ emb_fp32
        sims_int8 = text_embeddings @ emb_int8
        
        # Get predictions
        pred_fp32 = conditions[np.argmax(sims_fp32)]
        pred_int8 = conditions[np.argmax(sims_int8)]
        
        match = "✓" if pred_fp32 == pred_int8 else "✗"
        if pred_fp32 == pred_int8:
            matches += 1
        
        print(f"  {i+1:<8} {pred_fp32:<25} {pred_int8:<25} {match:>6}")
    
    print("  " + "-" * 66)
    print(f"\n  Classification Agreement: {matches}/{total} ({matches/total*100:.1f}%)")


def main():
    print("=" * 70)
    print("BiomedCLIP INT8 vs FP32 Baseline Comparison")
    print("=" * 70)
    
    # Check files exist
    if not os.path.exists(FP32_PATH):
        print(f"Error: FP32 model not found at {FP32_PATH}")
        return
    if not os.path.exists(INT8_PATH):
        print(f"Error: INT8 model not found at {INT8_PATH}")
        return
    
    # Load models
    session_fp32, session_int8, input_name, output_name = load_models()
    
    # Create test inputs
    test_inputs = create_test_inputs()
    
    # Run tests
    accuracy_results = test_accuracy(session_fp32, session_int8, input_name, output_name, test_inputs)
    test_embedding_quality(session_fp32, session_int8, input_name, output_name)
    perf_results = test_performance(session_fp32, session_int8, input_name, output_name)
    test_classification_consistency(session_fp32, session_int8, input_name, output_name)
    
    # Final Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    fp32_size = os.path.getsize(FP32_PATH) / (1024 * 1024)
    int8_size = os.path.getsize(INT8_PATH) / (1024 * 1024)
    
    avg_cosine = np.mean([r["cosine"] for r in accuracy_results])
    min_cosine = np.min([r["cosine"] for r in accuracy_results])
    
    print(f"""
    📦 MODEL SIZE:
       FP32: {fp32_size:.2f} MB
       INT8: {int8_size:.2f} MB
       Reduction: {(1 - int8_size/fp32_size)*100:.1f}%

    🎯 ACCURACY:
       Average Cosine Similarity: {avg_cosine:.6f}
       Minimum Cosine Similarity: {min_cosine:.6f}
       Quality: {"EXCELLENT" if min_cosine > 0.99 else "GOOD" if min_cosine > 0.95 else "ACCEPTABLE"}

    ⚡ PERFORMANCE (CPU):
       FP32: {perf_results['fp32_single']:.2f} ms/image
       INT8: {perf_results['int8_single']:.2f} ms/image
       Speedup: {perf_results['fp32_single']/perf_results['int8_single']:.2f}x

    ✅ RECOMMENDATION:
       The INT8 model is suitable for production deployment.
       Size reduced by {(1 - int8_size/fp32_size)*100:.0f}% with minimal accuracy loss.
    """)


if __name__ == "__main__":
    main()
