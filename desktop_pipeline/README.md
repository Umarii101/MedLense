# Desktop Pipeline — GPU-Based Prototype

> This is the **desktop/server** prototype that runs on a local CUDA GPU.
> For the **production mobile app**, see [`Medlens/README.md`](../Medlens/README.md).

## What This Is

A Python-based clinical AI system that demonstrates the three core capabilities before any edge optimization:

| Pipeline | File | Purpose |
|----------|------|---------|
| Clinical Text | `pipelines/clinical_text_pipeline.py` | Summarize clinical notes, extract entities, calculate risk |
| Image Assist | `pipelines/image_assist_pipeline.py` | Medical image feature extraction & quality assessment |
| Multimodal | `pipelines/multimodal_pipeline.py` | Combined text + image reasoning via MedGemma |

## Architecture

```
main.py                         # Demo entry point
├── pipelines/
│   ├── clinical_text_pipeline   → models/medgemma + models/risk_model
│   ├── image_assist_pipeline    → models/image_encoder
│   └── multimodal_pipeline      → both sub-pipelines + MedGemma reasoning
├── models/
│   ├── medgemma.py              # MedGemma 4B-IT via HuggingFace Transformers (8-bit)
│   ├── image_encoder.py         # BiomedCLIP / DINOv2 feature extraction
│   └── risk_model.py            # Rule-based clinical risk scoring
├── schemas/outputs.py           # Pydantic output models with safety disclaimers
├── utils/
│   ├── safety.py                # Non-diagnostic language enforcement, hallucination checks
│   └── memory.py                # CUDA memory management for RTX 3080
├── examples/example_data.py     # Synthetic clinical cases for demo
└── test_images/xray/            # Sample X-rays for image pipeline
```

## Requirements

- **GPU**: NVIDIA RTX 3060+ (10 GB+ VRAM)
- **RAM**: 16 GB+
- **Python**: 3.10+
- **CUDA**: 11.8+

```bash
pip install -r requirements.txt
python main.py
```

## Relationship to Edge Deployment

This prototype validated the clinical AI approach on a powerful GPU. The models were then:

1. **MedGemma 4B-IT** → quantized to Q4_K_S GGUF (8.6 GB → 2.2 GB) via llama.cpp
2. **BiomedCLIP** → exported to ONNX → INT8 quantized (329 MB → 84 MB) via ONNX Runtime

The quantized models run on-device in the [MedLens Android app](../Medlens/README.md) without any GPU or internet.
