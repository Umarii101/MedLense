# MedLens – Cloud-Edge Collaborative Medical AI for Android

**MedLens** is a production‑ready Android application that brings Google’s MedGemma and BiomedCLIP models to mobile devices. It works **completely offline** (Edge mode) for privacy and speed, and can **escalate complex queries** to a cloud backend (Deep mode) via an intelligent routing strategy. This repository contains the **Edge/Android tier** – the full app source, quantized models, quantization scripts, and evaluation toolkit.

The **Cloud backend** (Django + llama.cpp) is hosted in a separate repository:  
👉 [MedLensPlus – Cloud Backend](https://github.com/Umarii101/MedLensPlus)

---

## 🎯 Project Overview

- **Edge Tier** (this repo) – Runs MedGemma 4B‑IT (Q4_K_S, 2.2 GB) and BiomedCLIP (INT8, 84 MB) entirely on‑device.  
  - Offline‑first: no internet needed for common cases.  
  - 30‑condition zero‑shot medical image classification.  
  - Streaming clinical assessments in 10‑15 seconds.

- **Cloud Tier** (separate repo) – Escalates when:  
  - Classifier confidence is low (T < 0.70).  
  - Query complexity is high (medical term density + sentence length).  
  - Network is available.  

- **Three‑signal intelligent routing** ensures optimal trade‑off between privacy, latency, and accuracy.

---

## 🚀 Quick Start (Edge + Android)

### 1. Clone this repository
```bash
git clone https://github.com/Umarii101/MedLense.git
cd MedLense
```

### 2. Download the quantized models
The quantized models are too large for Git. Download them from Kaggle:
- **MedGemma 4B Q4_K_S GGUF**
- **BiomedCLIP Vision INT8 ONNX**

Place them in [edge_deployment/models/medgemma/](edge_deployment/models/medgemma/) and [edge_deployment/models/biomedclip/](edge_deployment/models/biomedclip/) respectively.

### 3. Install dependencies (for desktop simulation)
```bash
pip install -r desktop_pipeline/requirements.txt
```

### 4. Validate the models
```bash
python tests/run_all_tests.py
```

### 5. Build and run the Android app
Open the [Medlens/](Medlens/) folder in Android Studio and build the APK, or directly install the pre‑built APK from the [Medlens/APK/](Medlens/APK/) directory.

📥 **Pre‑built APK**
For immediate testing, download the latest debug APK from:
[Google Drive Link](#)

Simply install on any Android device (min SDK 26) and grant storage permissions. The app works offline immediately – no cloud setup required for Fast mode.

📁 **Repository Structure**
```text
MedLense/
├── README.md
├── EDGE_DEPLOYMENT.md          # Full edge deployment story
├── LICENSE
│
├── Medlens/                    # ⭐ Production Android app
│   ├── README.md               # Build & pipeline details
│   ├── APK/app-debug.apk       # Pre‑built APK
│   ├── app/src/main/cpp/       # JNI bridge + llama.cpp static
│   ├── app/src/main/java/      # Kotlin UI + inference wrappers
│   └── app/src/main/assets/    # 30‑condition embeddings
│
├── Inference Test App/         # Proof‑of‑concept test app (predecessor)
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
└── docs/                       # Technical deep‑dives, setup guide, summary
```
📊 Key Performance Results (Edge)
Device: Realme GT Neo 6 (Snapdragon 8s Gen 3, 12 GB RAM)

Model	Size	Speed	Accuracy
BiomedCLIP INT8	84 MB	126 ms inference	0.9991 cosine fidelity vs FP32
MedGemma 4B Q4_K_S	2.2 GB	7.8 tok/s generation, 32.8 tok/s prompt	High clinical quality
Zero‑shot top‑5 clinical accuracy: 98.8% (on 400 chest X‑rays)

MedGemma clinical quality rubric: 8.6/10 (EXCELLENT)

🔄 Cloud Integration (Deep Mode)
The app can also offload complex queries to the MedLensPlus cloud backend. To enable this:

Clone and deploy the backend: https://github.com/Umarii101/MedLensPlus

Follow its README.md to start the Django + llama.cpp server.

In the Android app, enter the server’s IP address in settings.

The routing logic (classifier confidence, query complexity, network availability) automatically decides between Fast (edge) and Deep (cloud) mode.

📚 Documentation

| Document | Description |
| :--- | :--- |
| [Medlens/README.md](Medlens/README.md) | App architecture, build instructions, pipeline details |
| [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md) | Full edge deployment story – quantization approach & rationale |
| [Inference Test App/DEPLOYMENT_TECHNICAL_REPORT.md](Inference%20Test%20App/DEPLOYMENT_TECHNICAL_REPORT.md) | Android build challenges & solutions (0.2 → 7.8 tok/s) |
| [Inference Test App/ROADMAP.md](Inference%20Test%20App/ROADMAP.md) | Optimization roadmap & future targets |
| [quantization/README.md](quantization/README.md) | Quantization methodology & scripts |
| [benchmarks/README.md](benchmarks/README.md) | Performance measurements (desktop + on‑device) |
| [desktop_pipeline/README.md](desktop_pipeline/README.md) | Desktop/GPU prototype – pipelines, models, safety |
| [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) | Technical deep dive – pipelines, safety, output schema |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Development environment setup |
| [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md) | Executive summary |
| [evaluation/README.md](evaluation/README.md) | Model quality evaluations – accuracy + clinical rubric |
🔗 Related Links
Cloud Backend Repository: MedLensPlus

Video Demo: YouTube

HAI‑DEF Models: Google Health AI Developer Foundations

⚠️ Medical Disclaimer
This system is for assistive purposes only. Not FDA approved. All outputs must be validated by licensed healthcare providers.

License: MIT
