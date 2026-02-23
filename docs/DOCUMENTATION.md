# MedGemma Impact Challenge — Technical Documentation

## 🎯 System Overview

This project delivers a healthcare AI system for the Kaggle MedGemma Impact Challenge, with two deployment targets:

1. **Desktop/Server** — Python pipelines for clinical text analysis, image analysis, and multimodal reasoning (this directory's code)
2. **Mobile/Edge** — MedLens Android app running quantized models entirely on-device (see [`medlens/`](../medlens/README.md))

### Core Principles

1. **Assistive, Not Diagnostic**: All outputs are framed as clinical decision support, requiring professional validation
2. **Offline-First**: Runs without cloud dependencies — desktop on local GPU, mobile on-device CPU
3. **Open-Weight Models**: Uses only Hugging Face models (no proprietary APIs)
4. **Safety-Focused**: Multiple layers of safety checks and non-diagnostic language enforcement
5. **Modular Design**: Clean architecture allowing independent component testing

## 🏗️ Architecture Deep Dive

### Desktop Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Multimodal Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │ Text Pipeline     │  │ Image Pipeline    │                │
│  ├──────────────────┤  ├──────────────────┤                │
│  │ • MedGemma 4B-IT │  │ • BiomedCLIP     │                │
│  │ • Entity Extract │  │ • Quality Check  │                │
│  │ • Risk Scoring   │  │ • Feature Extract│                │
│  └──────────────────┘  └──────────────────┘                │
│                                                               │
│  ┌─────────────────────────────────────────┐                │
│  │        Safety & Validation Layer         │                │
│  ├─────────────────────────────────────────┤                │
│  │ • Non-diagnostic language enforcement   │                │
│  │ • Hallucination detection               │                │
│  │ • Confidence calibration                 │                │
│  │ • Clinical validation                    │                │
│  └─────────────────────────────────────────┘                │
│                                                               │
│  ┌─────────────────────────────────────────┐                │
│  │        Structured Output (Pydantic)      │                │
│  └─────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Model Selection Rationale

#### Primary LLM: MedGemma 4B-IT
- **Why**: Purpose-built for medical applications with clinical knowledge
- **Size**: 4B parameters — fits comfortably on a single consumer GPU or mobile device
- **Desktop**: 8-bit quantization via bitsandbytes (~4 GB VRAM)
- **Mobile**: Q4_K_S GGUF via llama.cpp (2.2 GB)
- **Fallbacks**: Gemma-2-7B-IT → Llama-2-7B-Chat-HF (desktop only)

#### Image Encoder: BiomedCLIP
- **Model**: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Why**: Trained on PubMed biomedical image-text pairs; excellent medical image understanding
- **Architecture**: ViT-B/16
- **Output**: 512-dim embeddings for visual features
- **Desktop**: Full-precision PyTorch (~329 MB)
- **Mobile**: ONNX INT8 (84 MB), 99.95% cosine similarity vs FP32

#### Risk Scorer: Rule-Based + Sklearn
- **Why**: Interpretability and explainability
- **Method**: Weighted rule-based scoring with feature importance
- **Memory**: Negligible (~1 MB)

### Memory Management Strategy

**Desktop — RTX 3080 (10 GB VRAM):**

1. **8-bit Quantization**: Reduces MedGemma 4B from ~8 GB to ~4 GB VRAM
2. **Device Mapping**: Automatic offloading with `device_map="auto"`
3. **Gradient Disabled**: `.eval()` mode + `requires_grad=False`
4. **FP16 Inference**: Half-precision for image models
5. **Cache Clearing**: Explicit `torch.cuda.empty_cache()` calls
6. **Sequential Loading**: Load only needed models, offload when done

**Estimated Desktop Memory Usage:**
- MedGemma 4B-IT (8-bit): ~4 GB
- BiomedCLIP: ~1 GB
- Working Memory: 1–2 GB
- **Total**: ~6–7 GB (fits within 10 GB with headroom)

**Mobile — Snapdragon 8s Gen 3 (12 GB RAM):**
- MedGemma Q4_K_S: ~2.5 GB (model + KV cache)
- BiomedCLIP INT8: ~150 MB
- App + UI: ~200 MB
- **Total**: ~3 GB (leaves ~7 GB for Android OS and other apps)

## 📊 Output Schema Design

### Structured JSON Output

All system outputs use Pydantic models for:
- Type safety and validation
- Automatic JSON serialization
- IDE autocompletion
- Runtime validation

### Key Output Fields

```python
{
  "summary": str,                    # Clinical summary
  "key_findings": List[str],         # Important observations
  "extracted_symptoms": List[str],   # Parsed symptoms
  "extracted_conditions": List[str], # Historical conditions
  "medications_mentioned": List[str],# Current medications
  "risk_level": "Low|Medium|High",   # Risk stratification
  "confidence": float (0-1),         # Model confidence
  "recommendations": List[str],      # Clinical suggestions
  "safety_disclaimer": str           # Mandatory disclaimer
}
```

## 🛡️ Safety Mechanisms

### Layer 1: Language Framing
- Converts diagnostic terms → assistive observations
- Example: "diagnosed with" → "clinical history includes"
- Removes absolute certainty claims

### Layer 2: Confidence Calibration
- Caps maximum confidence at 0.95 (realistic uncertainty)
- Adds qualifiers based on confidence level
- Low confidence triggers additional warnings

### Layer 3: Hallucination Detection
- Checks for overly specific claims without data
- Flags missing hedges on uncertain topics
- Monitors for invented statistics

### Layer 4: Clinical Validation
- Validates output structure and content
- Ensures required disclaimers present
- Sanitizes problematic language

### Layer 5: Human-in-the-Loop
- All outputs explicitly state need for clinical validation
- Recommendations framed as suggestions, not directives
- Clear marking of AI-generated content

## 🔧 Pipeline Details

### Clinical Text Pipeline

**Input**: Clinical note text + optional patient age

**Steps**:
1. Text preprocessing and tokenization
2. MedGemma inference (clinical summarization)
3. Entity extraction (symptoms, conditions, medications)
4. Risk score calculation
5. Recommendation generation
6. Safety framing and validation

**Output**: `ClinicalTextOutput` (Pydantic model)

**Performance**: ~5-10 seconds on RTX 3080

### Image Assist Pipeline

**Input**: Medical image (PIL Image) + image type

**Steps**:
1. Image quality assessment
2. Preprocessing (resize to 224×224, ImageNet normalization)
3. Feature extraction (BiomedCLIP ViT-B/16 → 512-dim embedding)
4. Visual observation generation
5. Confidence calculation

**Output**: `ImageAnalysisOutput` (Pydantic model)

**Performance**: ~2-3 seconds on RTX 3080

### Multimodal Pipeline

**Input**: Clinical note + medical image + metadata

**Steps**:
1. Parallel text and image analysis
2. LLM-based finding integration
3. Cross-modal correlation detection
4. Unified risk assessment
5. Integrated recommendation generation

**Output**: `MultimodalOutput` (Pydantic model)

**Performance**: ~10-15 seconds on RTX 3080

## 📝 Usage Examples

### Basic Clinical Text Analysis

```python
from pipelines import MultimodalPipeline

pipeline = MultimodalPipeline()

result = pipeline.analyze_clinical_text(
    clinical_note="Patient presents with...",
    patient_age=65
)

print(result.model_dump_json(indent=2))
```

### Image Analysis

```python
from PIL import Image

image = Image.open("chest_xray.jpg")

result = pipeline.analyze_image(
    image=image,
    image_type="Chest X-Ray"
)

print(result.model_dump_json(indent=2))
```

### Multimodal Analysis

```python
result = pipeline.analyze_with_image(
    clinical_note="Patient with respiratory symptoms...",
    image=chest_xray_image,
    image_type="Chest X-Ray",
    patient_age=67,
    integrate_findings=True
)

print(result.clinical_summary)
print(result.integrated_findings)
print(result.next_steps)
```

### Target Use Cases

1. **Emergency Triage**: Quick patient assessment prioritization
2. **Rural/Remote Clinics**: Offline clinical decision support
3. **Developing Countries**: Low-resource healthcare settings
4. **Medical Education**: Case review and learning
5. **Clinical Handoffs**: Structured documentation
6. **Research**: Retrospective chart review

## ⚙️ Performance Optimization

### Inference Speed

**Optimization Techniques**:
- Batch processing for multiple cases
- Model caching (load once, use many times)
- KV-cache for faster generation
- Quantization (8-bit) for speed + memory

**Benchmarks (RTX 3080)**:
- Text analysis: 5-10 sec
- Image analysis: 2-3 sec
- Multimodal: 10-15 sec

### Memory Optimization

**Techniques**:
- Sequential model loading
- Explicit cache clearing
- Gradient computation disabled
- FP16 for image models
- 8-bit quantization for LLM

**Desktop Memory Footprint (RTX 3080):**
- Idle: ~1 GB
- Text pipeline: ~4 GB
- + Image pipeline: ~5 GB
- Peak: ~7 GB

## 🚀 Deployment Considerations

### Hardware Requirements

**Minimum**:
- NVIDIA GPU: RTX 3060 (12GB VRAM)
- RAM: 16GB
- Storage: 30GB

**Recommended**:
- NVIDIA GPU: RTX 3080/4080 (10-16GB VRAM)
- RAM: 32GB
- Storage: 50GB SSD

**Optimal**:
- NVIDIA GPU: RTX 4090 or A100
- RAM: 64GB
- Storage: 100GB NVMe SSD

### Software Requirements

- Python 3.10+
- CUDA 11.8+ with cuDNN
- PyTorch 2.1+
- Transformers 4.36+
- See `desktop_pipeline/requirements.txt` for full list

### Production Deployment

**Considerations**:
1. **Model versioning**: Track model versions in outputs
2. **Logging**: Comprehensive logging for audit trails
3. **Monitoring**: GPU utilization, inference time
4. **Error handling**: Graceful degradation
5. **Rate limiting**: Prevent GPU overload
6. **Batch processing**: Process multiple cases efficiently

## 📋 Limitations & Disclaimers

### Technical Limitations

- **Context window**: 2048 tokens for clinical notes
- **Image size**: Optimal at 224x224, max recommended 512x512
- **Batch size**: Limited by GPU memory
- **Languages**: Primarily English (model limitation)

### Clinical Limitations

- **NOT FDA approved**: Research/assistive use only
- **NOT diagnostic**: Requires professional validation
- **NOT real-time critical**: Not for emergency life/death decisions
- **NOT autonomous**: Human-in-the-loop mandatory

### Known Issues

- MedGemma 4B-IT requires a Hugging Face access token (gated model); fallbacks implemented
- 8-bit quantization may slightly reduce accuracy on desktop
- Q4_K_S quantization on mobile preserves medical reasoning quality well
- Complex multimodal reasoning can be inconsistent
- Confidence calibration is approximate

## 🔬 Future Enhancements

### Short-term ()
- Add support for more image modalities (MRI, CT)
- Improve multimodal integration prompts
- Add longitudinal tracking (patient history)
- Implement caching for repeated queries

### Medium-term ()
- Fine-tune on specific clinical domains
- Add retrieval-augmented generation (RAG)
- Implement active learning for model improvement
- Multi-GPU support for faster inference

### Long-term ()
- Clinical trial integration
- Real-world deployment studies
- Regulatory pathway exploration
- Multi-lingual support

## 📚 References

### Models Used
1. **MedGemma 4B-IT**: Google's medical-adapted Gemma model (`google/medgemma-4b-it`)
2. **BiomedCLIP**: Biomedical CLIP model (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`)
3. **CLIP ViT-Large**: Fallback image encoder (OpenAI)

### Datasets (for reference)
- MIMIC-III: Medical notes (not used directly)
- ChestX-ray14: Chest X-ray dataset (reference only)
- RadQA: Radiology question answering

### Papers
- "Towards Medical AI Assistants" (Google Research)
- "Self-Supervised Learning for Medical Image Analysis"
- "Effective Human-AI Teams in Clinical Settings"

## 📄 License & Ethics

### License
MIT License - See LICENSE file for details

### Ethical Considerations
- Patient privacy: No PHI in training or outputs
- Algorithmic bias: Regular auditing recommended
- Transparency: Open-source for scrutiny
- Accountability: Human responsibility for decisions

### Medical Ethics
- Beneficence: System designed to help clinicians
- Non-maleficence: Multiple safety mechanisms
- Autonomy: Preserves clinical judgment
- Justice: Improves access in low-resource settings

---
