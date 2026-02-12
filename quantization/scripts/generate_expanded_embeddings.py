"""
Generate expanded text embeddings for MedLens zero-shot classification.

Uses BiomedCLIP's text encoder to pre-compute 512-dim embeddings for
~30 medical conditions across chest X-ray, dermatology, ophthalmology,
and general categories. These embeddings are shipped as an app asset.
"""

import os
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

# ── Expanded label set ──────────────────────────────────────────────────────
LABELS = {
    "chest_xray": [
        "normal chest x-ray",
        "pneumonia in chest x-ray",
        "COVID-19 infection in chest x-ray",
        "bacterial pneumonia in chest x-ray",
        "viral pneumonia in chest x-ray",
        "tuberculosis in chest x-ray",
        "lung opacity in chest x-ray",
        "pleural effusion in chest x-ray",
        "cardiomegaly in chest x-ray",
        "pulmonary edema in chest x-ray",
        "lung nodule in chest x-ray",
        "pneumothorax in chest x-ray",
        "atelectasis in chest x-ray",
    ],
    "dermatology": [
        "melanoma skin lesion",
        "benign nevus on skin",
        "basal cell carcinoma on skin",
        "squamous cell carcinoma on skin",
        "actinic keratosis on skin",
        "dermatofibroma on skin",
        "vascular lesion on skin",
        "psoriasis skin condition",
        "eczema or dermatitis on skin",
        "normal healthy skin",
    ],
    "ophthalmology": [
        "normal retinal fundus image",
        "diabetic retinopathy in retinal image",
        "glaucoma in retinal image",
        "age-related macular degeneration in retinal image",
    ],
    "general": [
        "normal healthy medical image",
        "fracture in medical image",
        "tumor or mass in medical image",
    ],
}


def main():
    from open_clip import create_model_from_pretrained
    from transformers import AutoTokenizer

    print("Loading BiomedCLIP model...")
    model, _ = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    model.eval()

    print("Loading BiomedBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
    )

    # Flatten all labels
    all_labels = []
    label_categories = {}
    for category, labels in LABELS.items():
        for label in labels:
            all_labels.append(label)
            label_categories[label] = category

    print(f"\nGenerating embeddings for {len(all_labels)} conditions...")

    text_inputs = tokenizer(
        all_labels,
        padding='max_length',
        truncation=True,
        max_length=256,
        return_tensors='pt'
    )

    with torch.no_grad():
        text_features = model.encode_text(text_inputs['input_ids'])
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Build output dict with category metadata
    embeddings_dict = {}
    for label, embedding in zip(all_labels, text_features.numpy()):
        embeddings_dict[label] = embedding.tolist()

    # Save expanded embeddings
    output_path = project_root / "deployment" / "text_embeddings_expanded.json"
    with open(output_path, 'w') as f:
        json.dump(embeddings_dict, f)

    print(f"✓ Saved: {output_path}")
    print(f"  Labels: {len(all_labels)}")
    print(f"  Dim: {text_features.shape[1]}")

    # Also save category mapping for the app
    categories_path = project_root / "deployment" / "label_categories.json"
    with open(categories_path, 'w') as f:
        json.dump(label_categories, f, indent=2)
    print(f"✓ Saved: {categories_path}")

    # Verify: print norms (should all be ~1.0)
    norms = text_features.norm(dim=-1)
    print(f"\n  Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")

    # Print labels by category
    for cat, labels in LABELS.items():
        print(f"\n  [{cat}] ({len(labels)} labels)")
        for label in labels:
            print(f"    - {label}")

    print(f"\nTotal: {len(all_labels)} conditions across {len(LABELS)} categories")


if __name__ == "__main__":
    main()
