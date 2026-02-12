# MedGemma Edge: Offline Clinical AI for Low-Resource Settings

> **Kaggle MedGemma Impact Challenge** — Edge AI Submission

[![Edge AI](https://img.shields.io/badge/Edge%20AI-Optimized-green)]()
[![MedGemma](https://img.shields.io/badge/HAI--DEF-MedGemma%204B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 🎯 Project Overview

**MedGemma Edge** brings Google's healthcare AI models to mobile devices, enabling offline clinical decision support in environments without reliable internet access.

| Model | Original | Quantized | Reduction |
|-------|----------|-----------|-----------|
| **MedGemma 4B-IT** | 8.6 GB | 2.2 GB | 74% |
| **BiomedCLIP** | 329 MB | 84 MB | 74% |

**Target**: Android devices — includes **[MedLens](medlens/README.md)**, a production chat-based app with unified camera + gallery interface, combined BiomedCLIP → zero-shot classifier → MedGemma pipeline, streaming clinical assessments, and 30-condition medical classification — all running entirely on-device.

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Umarii101/The-MedGemma-Impact-Challenge.git
cd The-MedGemma-Impact-Challenge

# Install dependencies
pip install -r requirements.txt

#Download The quantized Models First, I have provided the link to the models below

# Run validation tests
python tests/run_all_tests.py
```

## 📥 Quantized Model Downloads

The Models that I have quantized are too large for Git. Download from Google Drive:

**[📁 Download Models (Google Drive)](https://drive.google.com/file/d/1JZmLMVmimPnL3tiSe0GkNxYENRP49qAP/view?usp=sharing)**


## 📁 Repository Structure

```
├── README.md                 # You are here
├── EDGE_DEPLOYMENT.md        # ⭐ Full edge deployment story
├── requirements.txt
│
├── medlens/                  # ⭐ MedLens — Production Android app
│   ├── README.md                      # App architecture, build & run instructions
│   ├── app/src/main/cpp/              # C++ JNI bridge (llama.cpp, static linked)
│   ├── app/src/main/java/             # Kotlin: ChatViewModel, inference wrappers, UI
│   ├── app/src/main/assets/           # Pre-computed text embeddings (30 conditions)
│   └── build.gradle.kts
│
├── android_app/              # Android PoC (test app — predecessor to MedLens)
│   ├── DEPLOYMENT_TECHNICAL_REPORT.md  # ⭐ Build & debugging story (0.2→7.8 tok/s)
│   ├── ROADMAP.md                      # Optimization roadmap
│   └── app/src/main/                   # 2-tab test harness
│
├── edge_deployment/          # Quantized models for mobile
│   ├── models/
│   │   ├── biomedclip/       # ONNX INT8 (84 MB)
│   │   └── medgemma/         # GGUF Q4_K_S (2.2 GB)
│   └── README.md
│
├── quantization/             # Quantization scripts & methodology
│   ├── scripts/              # ONNX export, INT8, GGUF, embedding generation
│   └── README.md
│
├── benchmarks/               # Performance measurements (desktop + on-device)
│   └── README.md
│
├── tests/                    # Validation test suite
│   ├── test_biomedclip.py
│   ├── test_medgemma.py
│   └── run_all_tests.py
│
├── models/                   # Desktop model loaders (MedGemma, BiomedCLIP, risk)
├── pipelines/                # Desktop analysis pipelines (text, image, multimodal)
├── schemas/                  # Pydantic output models
├── utils/                    # Safety checks, memory management
│
└── docs/                     # Additional documentation
    ├── DOCUMENTATION.md      # Technical deep dive
    ├── SETUP_GUIDE.md        # Development environment setup
    └── PROJECT_SUMMARY.md    # Executive summary
```

## 📊 Key Results

### On-Device Performance (Realme GT Neo 6, Snapdragon 8s Gen 3)

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| BiomedCLIP INT8 | 84 MB | 99.95% vs FP32 | 126 ms inference |
| MedGemma Q4_K_S | 2.2 GB | High quality | 32.8 tok/s pp, 7.8 tok/s gen |

### Validated Tests

```
[PASS] BiomedCLIP INT8 - Cosine similarity: 0.9995
[PASS] MedGemma Q4_K_S - Speed: 9.0 tok/s
ALL TESTS PASSED ✅
```

## 🌍 Impact

**Use Case**: Rural health clinics without internet access

1. Health worker captures patient symptoms + X-ray image
2. BiomedCLIP extracts visual features (100ms)
3. MedGemma provides clinical assessment (10-15s)
4. All processing happens **on-device** — no cloud required

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [medlens/README.md](medlens/README.md) | **⭐ MedLens app** — architecture, build instructions, pipeline details |
| [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md) | Full edge deployment story — quantization approach & rationale |
| [android_app/DEPLOYMENT_TECHNICAL_REPORT.md](android_app/DEPLOYMENT_TECHNICAL_REPORT.md) | Android build challenges & solutions (0.2 → 7.8 tok/s debugging) |
| [android_app/ROADMAP.md](android_app/ROADMAP.md) | Optimization roadmap & future targets |
| [quantization/README.md](quantization/README.md) | Quantization methodology & scripts |
| [benchmarks/README.md](benchmarks/README.md) | Performance measurements (desktop + on-device) |
| [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | Technical deep dive — pipelines, safety, output schema |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Development environment setup |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | Executive summary |

## 🔗 Links

- **Competition**: [Kaggle MedGemma Impact Challenge](https://kaggle.com/competitions/med-gemma-impact-challenge)
- **HAI-DEF Models**: [Google Health AI Developer Foundations](https://huggingface.co/google/medgemma-4b-it)
- **Video Demo**: *See [medlens/README.md](medlens/README.md) for app walkthrough*


## ⚠️ Medical Disclaimer

This system is for **assistive purposes only**. Not FDA approved. All outputs require validation by licensed healthcare providers.

*Built for the Kaggle MedGemma Impact Challenge — Edge AI*
