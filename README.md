# MedGemma Impact Challenge - Healthcare AI Backend

## 🎯 Project Overview

Production-quality Python backend for offline clinical decision support using open-weight healthcare AI models. Built for the Kaggle MedGemma Impact Challenge.

**⚠️ ASSISTIVE TOOL ONLY - NOT FOR DIAGNOSIS**

This system assists healthcare professionals in low-resource settings. All outputs require clinical validation.

## 🏛️ Architecture

```
Input Layer (Text + Images + Metadata)
    → Image Encoder (CLIP/BiomedCLIP) → Embeddings
    → MedGemma 4B Reasoning Engine → Clinical Understanding
    → Safety & Framing Layer → Non-diagnostic Language
    → Structured JSON Output
```

## 🤖 Models Used

### Desktop/Server Deployment
| Model | Purpose | Size |
|-------|---------|------|
| **google/medgemma-4b-it** | Clinical reasoning (8-bit quantized) | ~4GB VRAM |
| **openai/clip-vit-large-patch14** | Medical image features | ~2GB VRAM |
| Rule-based + sklearn | Risk stratification | Minimal |

### Edge/Mobile Deployment
| Model | Format | Size | Target |
|-------|--------|------|--------|
| **BiomedCLIP Vision** | ONNX INT8 | 84 MB | Android image embeddings |
| **MedGemma 4B** | GGUF Q4_K_S | 2.2 GB | Android text generation |

## 💻 Hardware Requirements

### Desktop
- **GPU**: NVIDIA RTX 3060+ (10GB+ VRAM recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 30GB for models
- **CUDA**: 11.8+ with cuDNN

### Mobile (Edge Deployment)
- **Target**: Snapdragon 8s Gen 3 or equivalent
- **RAM**: 8GB+
- **Storage**: 3GB for quantized models

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

### Desktop Pipeline
```python
from pipelines.multimodal_pipeline import MultimodalPipeline

# Initialize system
pipeline = MultimodalPipeline()

# Run clinical text analysis
result = pipeline.analyze_clinical_text(
    clinical_note="Patient presents with persistent cough..."
)

print(result.model_dump_json(indent=2))
```

### Edge Deployment Tests
```bash
# Test BiomedCLIP INT8
python tests/test_biomedclip.py

# Test MedGemma Q4_K_S
python tests/test_medgemma.py

# Run all edge tests
python tests/run_all_tests.py
```

## 📂 Project Structure

```
Project 1/
├── models/                     # Desktop model loaders
│   ├── medgemma.py            # MedGemma 4B inference
│   ├── image_encoder.py       # CLIP/DINOv2 image features
│   └── risk_model.py          # Risk scoring
├── pipelines/                  # End-to-end workflows
│   ├── clinical_text_pipeline.py
│   ├── image_assist_pipeline.py
│   └── multimodal_pipeline.py
├── schemas/                    # Pydantic data models
│   └── outputs.py
├── utils/                      # Utilities
│   ├── safety.py              # Safety mechanisms
│   └── memory.py              # GPU memory management
├── edge_deployment/            # Mobile/edge models
│   ├── models/
│   │   ├── biomedclip/        # ONNX INT8 (84 MB)
│   │   └── medgemma/          # GGUF Q4_K_S (2.2 GB)
│   └── README.md
├── tests/                      # Validation tests
│   ├── test_biomedclip.py     # BiomedCLIP INT8 tests
│   ├── test_medgemma.py       # MedGemma Q4_K_S tests
│   └── run_all_tests.py       # Full test suite
├── test_images/                # Sample test images
├── examples/                   # Example data
├── main.py                     # Demo script
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
└── SETUP_GUIDE.md
```

## 🛡️ Safety Features

- Non-diagnostic language enforcement
- Confidence scoring
- Human-in-the-loop disclaimers
- Hallucination detection
- Clinical validation requirements

## 📊 Example Output

```json
{
  "summary": "Patient presents with respiratory symptoms requiring assessment",
  "key_findings": [
    "Persistent cough for 2 weeks",
    "No fever reported",
    "History of seasonal allergies"
  ],
  "risk_level": "Low",
  "confidence": 0.78,
  "recommendations": [
    "Consider pulmonary function test",
    "Review allergy medication compliance"
  ],
  "clinical_notes": "⚠️ ASSISTIVE ONLY - Requires clinical validation"
}
```

## 🎬 Competition Alignment

✅ Uses open-weight MedGemma model
✅ Runs offline on local GPU
✅ Edge deployment ready (Android)
✅ Suitable for low-resource healthcare settings
✅ Reproducible and well-documented

## 📄 License

MIT License - See LICENSE file

## ⚠️ Medical Disclaimer

This system is for **assistive purposes only**. It is NOT FDA approved and is NOT a substitute for professional medical judgment. All outputs require validation by licensed healthcare providers.
