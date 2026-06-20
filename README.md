# MedLens — Cloud-Edge Collaborative Medical AI for Android

[![Edge AI](https://img.shields.io/badge/Edge%20AI-Optimized-green)](https://github.com/Umarii101/MedLense)
[![MedGemma](https://img.shields.io/badge/HAI--DEF-MedGemma%204B-blue)](https://github.com/Umarii101/MedLense)
[![BioMedClip](https://img.shields.io/badge/BiomedCLIP-PubMedBERT%20256vit-purple)](https://github.com/Umarii101/MedLense)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/Umarii101/MedLense)

**MedLens** is a production-ready Android application that brings Google’s MedGemma and BiomedCLIP models to mobile devices. It works **completely offline** (Edge mode) for privacy and speed, and can **escalate complex queries** to a cloud backend (Deep mode) via an intelligent routing strategy. This repository contains the **Edge/Android tier** — the full app source, quantized models, quantization scripts, and evaluation toolkit.

The **Cloud backend** (Django + llama.cpp) is hosted in a separate repository:
👉 [MedLensPlus — Cloud Backend](https://github.com/Umarii101/MedLensPlus)

---

## 🎯 Project Overview

- **Edge Tier** (this repo) — Runs MedGemma 4B-IT (Q4_K_S, 2.2 GB) and BiomedCLIP (INT8, 84 MB) entirely on-device.
  - Offline-first: no internet needed for common cases.
  - 30-condition zero-shot medical image classification.
  - Streaming clinical assessments in 10-15 seconds.

- **Cloud Tier** (separate repo) — Escalates when:
  - Classifier confidence is low (T < 0.70).
  - Query complexity is high (medical term density + sentence length).
  - Network is available.

- **Three-signal intelligent routing** ensures optimal trade-off between privacy, latency, and accuracy.

---

## 🚀 Quick Start (Edge + Android)

### 1. Clone this repository

```bash
git clone https://github.com/Umarii101/MedLense.git
cd MedLense
```

### 2. Download the quantized models

The quantized models are too large for Git. Download them from Kaggle:

- [MedGemma 4B Q4_K_S GGUF](https://www.kaggle.com/models/muhammadumar2001/medgemma-4b-q4-k-s-gguf)
- [BiomedCLIP Vision INT8 ONNX](https://www.kaggle.com/models/muhammadumar2001/biomedclip-vision-int8-onnx)

Place them in `edge_deployment/models/medgemma/` and `edge_deployment/models/biomedclip/` respectively.

### 3. Install dependencies (for desktop simulation)

```bash
pip install -r desktop_pipeline/requirements.txt
```

### 4. Validate the models

```bash
python tests/run_all_tests.py
```

### 5. Build and run the Android app

Open the `Medlens/` folder in Android Studio and build the APK, or directly install the pre-built APK from the `Medlens/APK/` directory.

📥 **Pre-built APK**

For immediate testing, download the latest debug APK from:
[Google Drive Link](https://drive.google.com/file/d/1gZ-i5k9q9-FefEVjtlYJP773gsaeQ0iV/view?usp=drive_link)

Simply install on any Android device (min SDK 26) and grant storage permissions. The app works offline immediately — no cloud setup required for Fast mode.

---

📁 **Repository Structure**

```text
MedLense/
├── README.md
├── EDGE_DEPLOYMENT.md          # Full edge deployment story
├── LICENSE
│
├── Medlens/                    # ⭐ Production Android app
│   ├── README.md               # Build & pipeline details
│   ├── APK/app-debug.apk       # Pre-built APK
│   ├── app/src/main/cpp/       # JNI bridge + llama.cpp static
│   ├── app/src/main/java/      # Kotlin UI + inference wrappers
│   └── app/src/main/assets/    # 30-condition embeddings
│
├── Inference Test App/         # Proof-of-concept test app (predecessor)
│   ├── DEPLOYMENT_TECHNICAL_REPORT.md
│   └── ROADMAP.md
│
├── edge_deployment/            # Quantized models (place them here)
│   ├── models/biomedclip/      # ONNX INT8
│   └── models/medgemma/        # GGUF Q4_K_S
│
├── quantization/               # Scripts: ONNX export, INT8, GGUF, embeddings
├── benchmarks/                 # Performance measurements
├── tests/                      # Validation test suite (run_all_tests.py)
├── evaluation/                 # Quality evaluation (accuracy, fidelity, clinical rubric)
├── desktop_pipeline/           # Desktop GPU prototype (RTX 3080)
└── docs/                       # Technical deep-dives, setup guide, summary
```

---

## 📊 Key Performance Results (Edge)

**Device:** Realme GT Neo 6 (Snapdragon 8s Gen 3, 12 GB RAM)

| Model | Size | Speed | Accuracy |
| :--- | :--- | :--- | :--- |
| BiomedCLIP INT8 | 84 MB | 126 ms inference | 0.9991 cosine fidelity vs FP32 |
| MedGemma 4B Q4_K_S | 2.2 GB | 7.8 tok/s gen, 32.8 tok/s prompt | High clinical quality |

### Model Quality Evaluation (400-image dataset — see `evaluation/README.md`)

| Evaluation Metric | Result |
| :--- | :--- |
| BiomedCLIP Zero-Shot (5 test images) | Top-5 clinical hit rate **80%** (4/5) |
| BiomedCLIP Zero-Shot (400 chest X-rays) | Top-5 clinical accuracy **98.8%** (395/400) |
| BiomedCLIP Zero-Shot (400 chest X-rays) | Top-3 clinical accuracy **87.5%** (350/400) |
| BiomedCLIP INT8 Fidelity | Cosine similarity vs FP32 **0.9991** |
| MedGemma Clinical Quality | Automated rubric (10-pt) **8.6/10 EXCELLENT** |
| MedGemma Safety | No absolute diagnostic claims **100%** (5/5 cases) |

### Validated Tests

```text
[PASS] BiomedCLIP INT8 - Cosine similarity: 0.9995
[PASS] MedGemma Q4_K_S - Speed: 9.0 tok/s
ALL TESTS PASSED ✅
```

---

## 🔄 Cloud Integration (Deep Mode)

The app can also offload complex queries to the **MedLensPlus** cloud backend. To enable this:

1. Clone and deploy the backend: [https://github.com/Umarii101/MedLensPlus](https://github.com/Umarii101/MedLensPlus)
2. Follow its `README.md` to start the Django + llama.cpp server.
3. In the Android app, enter the server’s IP address in settings.

The routing logic (classifier confidence, query complexity, network availability) automatically decides between Fast (edge) and Deep (cloud) mode.

---

## 📚 Documentation

| Document | Description |
| :--- | :--- |
| [Medlens/README.md](Medlens/README.md) | App architecture, build instructions, pipeline details |
| [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md) | Full edge deployment story — quantization approach & rationale |
| [Inference Test App/DEPLOYMENT_TECHNICAL_REPORT.md](Inference%20Test%20App/DEPLOYMENT_TECHNICAL_REPORT.md) | Android build challenges & solutions (0.2 — 7.8 tok/s) |
| [Inference Test App/ROADMAP.md](Inference%20Test%20App/ROADMAP.md) | Optimization roadmap & future targets |
| [quantization/README.md](quantization/README.md) | Quantization methodology & scripts |
| [benchmarks/README.md](benchmarks/README.md) | Performance measurements (desktop + on-device) |
| [desktop_pipeline/README.md](desktop_pipeline/README.md) | Desktop/GPU prototype — pipelines, models, safety |
| [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | Technical deep dive — pipelines, safety, output schema |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Development environment setup |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | Executive summary |
| [evaluation/README.md](evaluation/README.md) | Model quality evaluations — accuracy + clinical rubric |

---

## 🔗 Related Links

- **Competition**: [Kaggle MedGemma Impact Challenge](https://kaggle.com/competitions/med-gemma-impact-challenge)
- **Cloud Backend Repository**: [MedLensPlus](https://github.com/Umarii101/MedLensPlus)
- **Video Demo**: [YouTube](https://youtu.be/ZCZq52NL9NM)
- **HAI-DEF Models**: [Google Health AI Developer Foundations](https://huggingface.co/google/medgemma-4b-it)

---

## ⚠️ Medical Disclaimer

This system is for **assistive purposes only**. Not FDA approved. All outputs must be validated by licensed healthcare providers.

---

**License:** MIT
