# Benchmark Results

Performance measurements for quantized edge deployment models.

## Desktop Benchmarks (Development Machine)

**Hardware**: Windows PC with NVIDIA GPU
**Runtime**: ONNX Runtime 1.23.2, llama-cpp-python 0.3.16

### BiomedCLIP INT8

| Metric | FP32 Baseline | INT8 Quantized | Change |
|--------|---------------|----------------|--------|
| Model Size | 329 MB | 84 MB | **-74%** |
| Load Time | ~500ms | ~200ms | -60% |
| Inference (CPU) | 117ms | 98ms | -16% |
| Inference (GPU) | 15ms | 12ms | -20% |
| Memory Usage | ~400 MB | ~120 MB | -70% |
| Output Accuracy | 1.000 | 0.9995 | **-0.05%** |

*Cosine similarity of 0.9995 indicates near-lossless quantization*

### MedGemma Q4_K_S

| Metric | FP16 (via transformers) | Q4_K_S GGUF | Change |
|--------|-------------------------|-------------|--------|
| Model Size | 8.6 GB | 2.2 GB | **-74%** |
| Load Time | ~30s | 2.2s | -93% |
| Speed (CPU, 4 threads) | ~3 tok/s | 9.0 tok/s | **+200%** |
| RAM Usage | ~10 GB | ~3 GB | -70% |
| Context Length | 131K | 2048 | Configured |

*Speed improvement due to optimized GGUF inference in llama.cpp*

## Target Device Specifications

**Device**: Realme GT Neo 6
**SoC**: Snapdragon 8s Gen 3

| Component | Specification |
|-----------|---------------|
| CPU | 1×Cortex-X4 @ 3.0 GHz + 4×Cortex-A720 @ 2.8 GHz + 3×Cortex-A520 @ 2.0 GHz |
| GPU | Adreno 735 |
| NPU | Hexagon NPU |
| RAM | 12 GB LPDDR5X |
| Storage | UFS 4.0 |

### Measured On-Device Performance

All measurements taken on the actual target device (CPU only — GPU/NPU offload not yet implemented):

| Model | Metric | Value |
|-------|--------|-------|
| BiomedCLIP INT8 | Model load | ~215 ms |
| BiomedCLIP INT8 | Inference | **~126 ms** (10-run average) |
| BiomedCLIP INT8 | Embedding dim | 512 |
| MedGemma Q4_K_S | Model load | 5–9 seconds |
| MedGemma Q4_K_S | Prompt processing | **32.8 tok/s** |
| MedGemma Q4_K_S | Token generation | **7.8 tok/s** |
| MedGemma Q4_K_S | Context window | 512 tokens |
| MedGemma Q4_K_S | Threads | 4 (big cores) |

### Future Targets (with hardware acceleration)

| Model | Target | Acceleration |
|-------|--------|--------------|
| BiomedCLIP INT8 | 30–50 ms | NNAPI (Hexagon NPU) |
| MedGemma Q4_K_S | 15–30 tok/s | Vulkan/OpenCL (Adreno 735 GPU) |

### Memory Budget

| Component | Allocation |
|-----------|------------|
| BiomedCLIP INT8 | ~150 MB |
| MedGemma Q4_K_S | ~2.5 GB |
| App + UI | ~200 MB |
| System Reserve | ~2 GB |
| **Total Required** | **~5 GB** |

*Fits comfortably in 12GB device RAM*

## Latency Breakdown (Measured)

**End-to-end clinical analysis with image (MedLens app):**

| Step | Time |
|------|------|
| Image capture & preprocessing | ~50 ms |
| BiomedCLIP inference | ~126 ms |
| Zero-shot classification (cosine sim) | <1 ms |
| MedGemma prompt construction | ~5 ms |
| MedGemma inference (~150 tokens) | ~19 s |
| **Total (image → complete response)** | **~20–25 seconds** |
| **Time to first token** | **~6–8 seconds** |

## Validation Commands

Run benchmarks locally:

```bash
# Full test suite
python tests/run_all_tests.py

# Individual model tests
python tests/test_biomedclip.py
python tests/test_medgemma.py
```

## Notes

1. **CPU only**: All on-device benchmarks are CPU-only. GPU (Vulkan) and NPU (NNAPI) offload are future work.
2. **Context length**: Set to 512 tokens on device to conserve KV cache memory. Configurable up to 2048.
3. **Batch size**: Single image/query (typical mobile use case)
4. **Warm-up**: All benchmarks exclude first inference (model loading)
5. **Debug vs Release**: Android builds use `assembleDebug` but CMake is forced to `-O3` via custom flags. See [DEPLOYMENT_TECHNICAL_REPORT.md](../android_app/DEPLOYMENT_TECHNICAL_REPORT.md).

---

*Benchmarks last updated: February 2026*
