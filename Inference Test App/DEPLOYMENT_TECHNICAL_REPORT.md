# Android Edge Deployment — Technical Report

> On-device deployment of MedGemma 4B-IT (Q4_K_S) and BiomedCLIP (INT8) on Android via llama.cpp JNI and ONNX Runtime.

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Android App (Kotlin)                    │
│              Jetpack Compose UI + Tabs                   │
├──────────────────────┬──────────────────────────────────┤
│   BiomedCLIP Tab     │        MedGemma Tab              │
│  (Image Analysis)    │   (Clinical Text Generation)     │
├──────────────────────┼──────────────────────────────────┤
│  BiomedClipInference │     MedGemmaInference            │
│  (Kotlin + ONNX RT)  │     (Kotlin JNI wrapper)         │
├──────────────────────┼──────────────────────────────────┤
│  ONNX Runtime Mobile │     medgemma_jni.cpp (C++)       │
│  (1.19.0, AAR)       │     ↕ llama.cpp (static)        │
├──────────────────────┼──────────────────────────────────┤
│         ARM CPU (NEON/dotprod/i8mm/fp16)                │
│       Snapdragon 8s Gen 3 — Realme GT Neo 6            │
└─────────────────────────────────────────────────────────┘
```

**Stack**: AGP 9.0.0, Kotlin 2.0.21, Jetpack Compose (BOM 2024.09.00), SDK 36, minSdk 26, NDK 28.2.13676358, CMake 3.22.1.

## 2. Model Deployment

### 2.1 BiomedCLIP — ONNX Runtime

| Property | Value |
|----------|-------|
| Model | BiomedCLIP Vision Encoder (INT8 ONNX) |
| Size | 83.6 MB |
| Runtime | ONNX Runtime Android 1.19.0 |
| Input | 224×224 RGB, ImageNet normalization |
| Output | 512-dimensional embedding vector |
| Load time | ~215 ms |
| Inference | ~126 ms average (10-run benchmark) |

**Integration path**: ONNX Runtime is distributed as a Gradle AAR dependency — no native build required. The Kotlin wrapper (`BiomedClipInference.kt`) handles:
- Image loading via Android `ContentResolver` + `BitmapFactory`
- Resize to 224×224, RGB extraction, NCHW layout
- ImageNet channel normalization (mean/std per channel)
- `OrtSession.run()` → float array → 512-dim embedding

### 2.2 MedGemma 4B-IT — llama.cpp via JNI

| Property | Value |
|----------|-------|
| Model | MedGemma 4B-IT (Q4_K_S GGUF) |
| Size | 2,267.8 MB (2.2 GB) |
| Runtime | llama.cpp (compiled from source, static linking) |
| Context | 512 tokens |
| Threads | 4 (targeting big cores: 1×Cortex-X4 + 3×Cortex-A720) |
| Load time | ~5–9 s (mmap=false, full RAM read) |
| Prompt processing | **32.8 tok/s** (23 tokens in 701 ms) |
| Token generation | **7.8 tok/s** (256 tokens in 32.7 s) |

**Integration path**: llama.cpp is compiled from source as a static library via CMake `add_subdirectory()`. A C++ JNI bridge (`medgemma_jni.cpp`) exposes five functions to Kotlin:
- `nativeInit()` — backend initialization
- `nativeLoadModel()` — load GGUF from external storage
- `nativeGenerate()` — streaming text generation with stop support
- `nativeGetPartialResult()` — poll partial results for live UI updates
- `nativeStopGeneration()` — abort generation mid-stream

**Streaming design**: The JNI layer writes tokens into a `std::string` buffer protected by a `std::mutex`. Kotlin polls `nativeGetPartialResult()` every 200 ms via a coroutine, parsing a `"tokens|speed|is_generating|text"` wire format for real-time UI updates (live token count, tok/s display, and incremental text rendering).

## 3. Challenges & Solutions

### Challenge 1: Abysmal Inference Speed — 0.2 tok/s

This was the **critical blocker** and took multiple debugging rounds to resolve. The fix required identifying **three independent root causes** that each contributed to the 50× slowdown.

#### Root Cause A: `CMAKE_BUILD_TYPE=Debug` → Implicit -O0

**Symptom**: All ggml matmul kernels compiled without optimization.

**Root cause**: Gradle's `assembleDebug` sets `CMAKE_BUILD_TYPE=Debug`. The NDK toolchain leaves `CMAKE_C_FLAGS_DEBUG` empty, resulting in implicit `-O0` (no optimization). Our initial `CMAKE_C_FLAGS CACHE FORCE` appeared to work in the CMake cache but **did not propagate** to subdirectory targets because llama.cpp's `add_subdirectory()` creates a child scope.

**Impact**: ~10–50× slower on compute-heavy ggml inner loops.

**Fix**: Three-mechanism approach to guarantee `-O3` on every source file:
```cmake
# 1. Directory property — inherited by ALL add_subdirectory() targets
add_compile_options(-O3 -DNDEBUG)

# 2. Override Debug config flags
set(CMAKE_C_FLAGS_DEBUG   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)

# 3. Append to global flags (belt-and-suspenders)
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG" CACHE STRING "" FORCE)
```

**Verification**: Audited all 188+ compile targets in `build.ninja` — confirmed `O3=True` on every line.

#### Root Cause B: `GGML_SYSTEM_ARCH=UNKNOWN` → ARM Kernel Files Never Compiled

**Symptom**: `llama_print_system_info()` showed NEON/DOTPROD/MATMUL_INT8 = 1 (compile-time checks passed) but performance didn't improve.

**Root cause**: The NDK's `android.toolchain.cmake` does **not** set `CMAKE_SYSTEM_PROCESSOR`. The ggml build system's `ggml_get_system_arch()` function checks `CMAKE_SYSTEM_PROCESSOR` to detect ARM, and without it, returns `"UNKNOWN"`. This means the ARM-optimized source files — `ggml-cpu/arch/arm/quants.c` and `ggml-cpu/arch/arm/repack.cpp` — were **never added to the build**. The model ran on generic scalar C fallback code.

**Impact**: ~10–30× slower for quantized matmul operations.

**Fix**:
```cmake
if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(CMAKE_SYSTEM_PROCESSOR "aarch64")  # enables ARM detection
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm+fp16" CACHE STRING "" FORCE)
    add_compile_options(-march=armv8.2-a+dotprod+i8mm+fp16)  # GLOBAL
endif()
```

**Verification**: Confirmed `arch/arm/quants.c` and `arch/arm/repack.cpp` appear in `build.ninja` link list for `libggml-cpu.a`.

#### Root Cause C: `-march` Only Applied to ggml-cpu, Not ggml-base

**Symptom**: ggml-cpu files had `-march=armv8.2-a+dotprod+i8mm+fp16` but ggml-base files (including the critical `ggml-quants.c` with core vectorized dequantization routines) did not.

**Root cause**: The `GGML_CPU_ARM_ARCH` variable only controls flags for the `ggml-cpu` CMake target. The `ggml-base` target (which contains `ggml-quants.c` — the file with `ggml_vec_dot_q4_K_q8_K` and other hot dequantization functions) does not inherit these flags.

**Impact**: Core dequantization routines compiled for baseline ARMv8.0 — no dotprod (`vdotq_s32`), no i8mm (`vmmlaq_s32`).

**Fix**: Added global `add_compile_options(-march=armv8.2-a+dotprod+i8mm+fp16)` before `add_subdirectory()`. This ensures ALL targets (ggml-base, ggml-cpu, llama, common) get the ARM extensions.

#### Root Cause D: mmap Page-Fault Thrashing

**Symptom**: Model "loaded" in 3 seconds but inference was extremely slow.

**Root cause**: With `use_mmap=true` (the default), the 2.2 GB model file is lazily memory-mapped. On each forward pass, random weight access triggers page faults that read from flash storage at ~0.7 GB/s. For a 4B parameter model, this means the entire model is effectively re-read from storage every few tokens.

**Impact**: ~5× slower due to I/O-bound inference.

**Fix**: Set `model_params.use_mmap = false` to force a sequential upfront read into malloc'd anonymous pages. Load time increases from 3s to ~5–9s, but inference runs at full memory bandwidth. Note: `use_mlock = true` was also attempted but Android apps lack `CAP_IPC_LOCK` (`RLIMIT_MEMLOCK` = 64 KB), so mlock silently fails.

### Challenge 2: KleidIAI FetchContent Failure

**Symptom**: CMake configure failed when `GGML_CPU_KLEIDIAI=ON`.

**Root cause**: KleidIAI uses `FetchContent` which downloads into the build tree. The workspace path contains spaces ("Kaggle Competitions"), which breaks `FetchContent`'s `ExternalProject_Add`.

**Fix**: `set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)`.

### Challenge 3: flash_attn API Change

**Symptom**: Compilation error — `flash_attn` is not a member of `llama_context_params`.

**Root cause**: llama.cpp changed the flash attention API from a `bool flash_attn` field to an enum `flash_attn_type` with values `LLAMA_FLASH_ATTN_TYPE_DISABLED`, `LLAMA_FLASH_ATTN_TYPE_ENABLED`, `LLAMA_FLASH_ATTN_TYPE_AUTO`.

**Fix**: Use `ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`.

## 4. Performance Progression

| Build | mmap | -O level | ARM arch | ARM kernels | Load (s) | PP (tok/s) | Gen (tok/s) |
|-------|------|----------|----------|-------------|----------|------------|-------------|
| 1 | ON | -O0 | baseline | Missing | 3 | 0.3 | 0.2 |
| 2 | OFF | -O0 | baseline | Missing | 5.3 | 0.2 | 0.2 |
| 3 | OFF | -O3 | armv8.2+dotprod+i8mm | ggml-cpu only | 29 | 0.7 | 0.3 |
| 4 | ON | -O3 (ggml-cpu only) | armv8.2+dotprod+i8mm | Present | 8.9 | 0.9 | 0.3 |
| **5** | **OFF** | **-O3 (ALL targets)** | **armv8.2+dotprod+i8mm (GLOBAL)** | **Present** | **4.9** | **32.8** | **7.8** |

**Net improvement**: Prompt processing **109×** faster, generation **39×** faster.

## 5. Final Build Configuration

```cmake
# Optimization: -O3 on EVERY source file (188+ targets)
add_compile_options(-O3 -DNDEBUG)
set(CMAKE_C_FLAGS_DEBUG   "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG "-O3 -DNDEBUG" CACHE STRING "" FORCE)

# ARM: set processor so ggml detects ARM, apply -march globally
set(CMAKE_SYSTEM_PROCESSOR "aarch64")
set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+i8mm+fp16" CACHE STRING "" FORCE)
add_compile_options(-march=armv8.2-a+dotprod+i8mm+fp16)

# Static build, OpenMP enabled, KleidIAI disabled
set(BUILD_SHARED_LIBS OFF)
set(GGML_OPENMP ON)
set(GGML_CPU_KLEIDIAI OFF)
```

**Runtime parameters**:
- `use_mmap = false` — full RAM load, no page-fault thrashing
- `n_ctx = 512` — minimal context for demo, saves ~100 MB KV cache
- `n_threads = 4` — targets big cores (1×X4 + 3×A720)
- `temp = 0.3` — low temperature for deterministic medical outputs

## 6. Test Device

| Property | Value |
|----------|-------|
| Device | Realme GT Neo 6 (RMX3852) |
| SoC | Snapdragon 8s Gen 3 |
| CPU | 1×Cortex-X4 (3.0 GHz) + 4×Cortex-A720 (2.8 GHz) + 3×Cortex-A520 (2.0 GHz) |
| GPU | Adreno 735 |
| RAM | 12 GB (11,296 MB available to userspace) |
| OS | Android 14 |
| Available RAM after model load | ~2.2 GB free of 11.3 GB |

## 7. APK Details

| Property | Value |
|----------|-------|
| Package | com.medgemma.edge |
| APK size | 57.2 MB |
| Native libs | libmedgemma-jni.so (llama.cpp static), ONNX Runtime |
| ABI | arm64-v8a, x86_64 |
| Min SDK | 26 (Android 8.0) |
| Target SDK | 36 |

## 8. File Structure

```
android_app/
├── .gitignore
├── build.gradle.kts                # Project-level Gradle config
├── settings.gradle.kts
├── gradle.properties
├── gradlew / gradlew.bat
│
├── app/
│   ├── build.gradle.kts            # App dependencies (ONNX Runtime, Compose, etc.)
│   │
│   └── src/main/
│       ├── AndroidManifest.xml     # Permissions (MANAGE_EXTERNAL_STORAGE, largeHeap)
│       │
│       ├── java/com/medgemma/edge/
│       │   ├── MainActivity.kt     # Compose UI: tabs, permission banner, streaming
│       │   └── inference/
│       │       ├── BiomedClipInference.kt   # ONNX Runtime wrapper
│       │       └── MedGemmaInference.kt     # JNI wrapper + streaming state
│       │
│       └── cpp/
│           ├── CMakeLists.txt      # Build config (O3, ARM, llama.cpp subdirectory)
│           └── medgemma_jni.cpp    # C++ JNI bridge (init, load, generate, bench)
│
├── DEPLOYMENT_TECHNICAL_REPORT.md  # This document
└── ROADMAP.md                      # Future optimization plans
```

## 9. Build & Run Instructions

### Prerequisites
- Android Studio (Ladybug+) with NDK 28.x and CMake 3.22.1
- llama.cpp source cloned adjacent to the android_app (see `CMakeLists.txt` for expected path)

### Build
```bash
# Clone llama.cpp (if not already present)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama_cpp_repo

# Open android_app/ in Android Studio → Build → assembleDebug
# Or from command line:
cd android_app
./gradlew assembleDebug
```

### Deploy Models to Phone
```bash
# Create model directory on device
adb shell mkdir -p /storage/emulated/0/MedGemmaEdge/

# Push quantized models
adb push biomedclip_vision_int8.onnx /storage/emulated/0/MedGemmaEdge/
adb push medgemma-4b-q4_k_s-final.gguf /storage/emulated/0/MedGemmaEdge/
```

### First Launch
1. Install APK via `adb install app-debug.apk` or Android Studio
2. Grant "All Files Access" when prompted (required for model loading)
3. Switch to BiomedCLIP tab → Load Model → pick an image → Run Inference
4. Switch to MedGemma tab → Load Model (5–9s) → type a prompt → Generate
