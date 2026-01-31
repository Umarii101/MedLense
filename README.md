# MedGemma Impact Challenge - Healthcare AI Backend

## 🎯 Project Overview

Production-quality Python backend for offline clinical decision support using open-weight healthcare AI models. Built for the Kaggle MedGemma Impact Challenge.

**⚠️ ASSISTIVE TOOL ONLY - NOT FOR DIAGNOSIS**

This system assists healthcare professionals in low-resource settings. All outputs require clinical validation.

## 🏗️ Architecture

```
Input Layer (Text + Images + Metadata)
    ↓
Image Encoder (RAD-DINO) → Embeddings
    ↓
MedGemma 7B Reasoning Engine → Clinical Understanding
    ↓
Safety & Framing Layer → Non-diagnostic Language
    ↓
Structured JSON Output
```

## 🧠 Models Used

1. **Primary LLM**: `google/medgemma-7b` - Clinical reasoning
2. **Image Encoder**: `microsoft/rad-dino` - Medical image features
3. **Risk Model**: Lightweight sklearn baseline (optional)

## 💻 Hardware Requirements

- **GPU**: NVIDIA RTX 3080 (10GB VRAM) or better
- **RAM**: 16GB+ system RAM
- **Storage**: 30GB for models
- **CUDA**: 11.8+ with cuDNN

## 📦 Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🚀 Quick Start

```python
from pipelines.multimodal_pipeline import MultimodalPipeline

# Initialize system
pipeline = MultimodalPipeline()

# Run clinical text analysis
result = pipeline.analyze_clinical_text(
    clinical_note="Patient presents with persistent cough..."
)

# Run image-assisted analysis
result = pipeline.analyze_with_image(
    clinical_note="...",
    image_path="chest_xray.jpg"
)

print(result.model_dump_json(indent=2))
```

## 📁 Project Structure

```
Project 1/
├── models/
│   ├── medgemma.py
│   ├── image_encoder.py
│   └── risk_model.py
├── pipelines/
│   ├── clinical_text_pipeline.py
│   ├── image_assist_pipeline.py
│   └── multimodal_pipeline.py
├── schemas/
│   └── outputs.py
├── utils/
│   ├── safety.py
│   └── memory.py
├── examples/
│   └── example_data.py
├── main.py
├── requirements.txt
├── README.md
├── DOCUMENTATION.md
├── SETUP_GUIDE.md
├── PROJECT_SUMMARY.md
└── DELIVERY_SUMMARY.md
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
  "clinical_notes": "⚠️ ASSISTIVE ONLY - Requires clinical validation by licensed provider"
}
```

## 🎥 Competition Alignment

✅ Uses open-weight models only (no cloud APIs)
✅ Runs offline on local GPU
✅ Demonstrates MedGemma capabilities
✅ Suitable for low-resource healthcare settings
✅ Reproducible and well-documented

## 📄 License

MIT License - See LICENSE file

## ⚠️ Medical Disclaimer

This software is for research and assistive purposes only. Not FDA approved. Not a substitute for professional medical judgment. All outputs must be validated by licensed healthcare providers.
