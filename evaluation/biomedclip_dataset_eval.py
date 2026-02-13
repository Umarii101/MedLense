"""
BiomedCLIP Large-Scale Zero-Shot Classification Evaluation
===========================================================
Evaluates BiomedCLIP INT8 ONNX on the COVID-19 Radiography Dataset
(400 images, 4 classes) with proper per-class metrics, confusion matrix,
and statistical analysis.

Dataset classes → taxonomy label mapping:
    covid19         → "COVID-19 infection in chest x-ray"
    normal          → "normal chest x-ray"
    lung_opacity    → "lung opacity in chest x-ray"
    viral_pneumonia → "viral pneumonia in chest x-ray"

Usage:
    python evaluation/biomedclip_dataset_eval.py

Requires:
    pip install onnxruntime numpy Pillow
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_INT8 = REPO_ROOT / "edge_deployment" / "models" / "biomedclip" / "biomedclip_vision_int8.onnx"
EMBEDDINGS_PATH = REPO_ROOT / "Medlens" / "app" / "src" / "main" / "assets" / "text_embeddings.json"
DATASET_DIR = REPO_ROOT / "evaluation" / "test_data" / "chest_xray_dataset"
RESULTS_PATH = REPO_ROOT / "evaluation" / "results" / "biomedclip_dataset_results.json"

# BiomedCLIP preprocessing (CLIP-standard)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_SIZE = 224

# Ground-truth: dataset folder → exact match labels + clinically acceptable labels
GROUND_TRUTH = {
    "covid19": {
        "exact": ["COVID-19 infection in chest x-ray"],
        "clinical_top3": [
            "COVID-19 infection in chest x-ray",
            "viral pneumonia in chest x-ray",
            "pneumonia in chest x-ray",
            "lung opacity in chest x-ray",
            "pulmonary edema in chest x-ray",
        ],
        "clinical_top5": [
            "COVID-19 infection in chest x-ray",
            "viral pneumonia in chest x-ray",
            "pneumonia in chest x-ray",
            "lung opacity in chest x-ray",
            "pulmonary edema in chest x-ray",
            "pleural effusion in chest x-ray",
            "bacterial pneumonia in chest x-ray",
        ],
    },
    "normal": {
        "exact": ["normal chest x-ray"],
        "clinical_top3": [
            "normal chest x-ray",
            "normal healthy medical image",
        ],
        "clinical_top5": [
            "normal chest x-ray",
            "normal healthy medical image",
        ],
    },
    "lung_opacity": {
        "exact": ["lung opacity in chest x-ray"],
        "clinical_top3": [
            "lung opacity in chest x-ray",
            "pneumonia in chest x-ray",
            "bacterial pneumonia in chest x-ray",
            "viral pneumonia in chest x-ray",
            "pulmonary edema in chest x-ray",
            "pleural effusion in chest x-ray",
        ],
        "clinical_top5": [
            "lung opacity in chest x-ray",
            "pneumonia in chest x-ray",
            "bacterial pneumonia in chest x-ray",
            "viral pneumonia in chest x-ray",
            "pulmonary edema in chest x-ray",
            "pleural effusion in chest x-ray",
            "atelectasis in chest x-ray",
            "COVID-19 infection in chest x-ray",
        ],
    },
    "viral_pneumonia": {
        "exact": ["viral pneumonia in chest x-ray"],
        "clinical_top3": [
            "viral pneumonia in chest x-ray",
            "pneumonia in chest x-ray",
            "COVID-19 infection in chest x-ray",
            "lung opacity in chest x-ray",
        ],
        "clinical_top5": [
            "viral pneumonia in chest x-ray",
            "pneumonia in chest x-ray",
            "COVID-19 infection in chest x-ray",
            "lung opacity in chest x-ray",
            "bacterial pneumonia in chest x-ray",
            "pulmonary edema in chest x-ray",
        ],
    },
}


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image identically to Android app (BiomedClipInference.kt)."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return np.expand_dims(arr, 0).astype(np.float32)


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    with open(path) as f:
        data = json.load(f)
    labels = list(data.keys())
    embeddings = np.array([data[l] for l in labels], dtype=np.float32)
    return labels, embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_norm @ b_norm.T).squeeze()


def discover_images(dataset_dir: Path) -> List[Tuple[str, str]]:
    """Discover images as dataset_dir/<class>/<file>."""
    images = []
    for class_dir in sorted(dataset_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("_"):
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append((str(img_file), class_dir.name))
    return images


def compute_metrics(results: list) -> dict:
    """Compute per-class and aggregate metrics."""
    classes = sorted(set(r["true_class"] for r in results))

    # Aggregate counters
    agg = {
        "top1_exact": 0, "top3_clinical": 0, "top5_clinical": 0,
        "total": len(results),
    }
    per_class = {c: {"top1_exact": 0, "top3_clinical": 0, "top5_clinical": 0, "total": 0}
                 for c in classes}

    # Confusion matrix: true → predicted top-1 (mapped to 4 classes + "other")
    label_to_class = {
        "COVID-19 infection in chest x-ray": "covid19",
        "normal chest x-ray": "normal",
        "lung opacity in chest x-ray": "lung_opacity",
        "viral pneumonia in chest x-ray": "viral_pneumonia",
        "pneumonia in chest x-ray": "pneumonia_generic",
        "bacterial pneumonia in chest x-ray": "bacterial_pneumonia",
    }
    confusion_labels = classes + ["pneumonia_generic", "bacterial_pneumonia", "other"]
    confusion = {true: {pred: 0 for pred in confusion_labels} for true in classes}

    for r in results:
        tc = r["true_class"]
        per_class[tc]["total"] += 1

        if r["top1_exact"]:
            agg["top1_exact"] += 1
            per_class[tc]["top1_exact"] += 1
        if r["top3_clinical"]:
            agg["top3_clinical"] += 1
            per_class[tc]["top3_clinical"] += 1
        if r["top5_clinical"]:
            agg["top5_clinical"] += 1
            per_class[tc]["top5_clinical"] += 1

        # Confusion matrix
        pred_label = r["top1_label"]
        pred_class = label_to_class.get(pred_label, "other")
        if pred_class in confusion_labels:
            confusion[tc][pred_class] += 1
        else:
            confusion[tc]["other"] += 1

    # Calculate rates
    n = agg["total"]
    metrics = {
        "aggregate": {
            "total_images": n,
            "top1_exact_accuracy": round(agg["top1_exact"] / n * 100, 1),
            "top3_clinical_accuracy": round(agg["top3_clinical"] / n * 100, 1),
            "top5_clinical_accuracy": round(agg["top5_clinical"] / n * 100, 1),
            "top1_exact_count": agg["top1_exact"],
            "top3_clinical_count": agg["top3_clinical"],
            "top5_clinical_count": agg["top5_clinical"],
        },
        "per_class": {},
        "confusion_matrix": confusion,
    }

    for c in classes:
        pc = per_class[c]
        t = pc["total"]
        metrics["per_class"][c] = {
            "total": t,
            "top1_exact": round(pc["top1_exact"] / t * 100, 1) if t else 0,
            "top3_clinical": round(pc["top3_clinical"] / t * 100, 1) if t else 0,
            "top5_clinical": round(pc["top5_clinical"] / t * 100, 1) if t else 0,
        }

    return metrics


def main():
    import onnxruntime as ort

    print("=" * 80)
    print("  BiomedCLIP Large-Scale Zero-Shot Classification Evaluation")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset not found at {DATASET_DIR}")
        print("Run first: python evaluation/download_eval_dataset.py")
        sys.exit(1)

    if not MODEL_INT8.exists():
        print(f"\nERROR: INT8 model not found at {MODEL_INT8}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load resources
    # ------------------------------------------------------------------
    print(f"\n[1/3] Loading resources...")
    labels, text_embeddings = load_embeddings(EMBEDDINGS_PATH)
    print(f"       Text embeddings: {len(labels)} labels, {text_embeddings.shape[1]}-dim")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = os.cpu_count()
    session = ort.InferenceSession(str(MODEL_INT8), sess_opts)
    print(f"       Model: {MODEL_INT8.name} ({MODEL_INT8.stat().st_size / 1e6:.1f} MB)")

    # ------------------------------------------------------------------
    # Discover images
    # ------------------------------------------------------------------
    print(f"\n[2/3] Discovering images in {DATASET_DIR.name}/...")
    test_images = discover_images(DATASET_DIR)
    class_counts = defaultdict(int)
    for _, c in test_images:
        class_counts[c] += 1

    for c, n in sorted(class_counts.items()):
        print(f"       {c}: {n} images")
    print(f"       Total: {len(test_images)} images")

    if not test_images:
        print("\nERROR: No images found! Run download_eval_dataset.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------
    print(f"\n[3/3] Running evaluation...")
    print("-" * 80)

    results = []
    total_ms = 0
    errors = 0

    for i, (img_path, true_class) in enumerate(test_images):
        gt = GROUND_TRUTH.get(true_class)
        if gt is None:
            continue

        try:
            t0 = time.perf_counter()
            input_tensor = preprocess_image(img_path)
            embedding = session.run(None, {"image": input_tensor})[0]
            similarities = cosine_similarity(embedding, text_embeddings)
            elapsed_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR on {Path(img_path).name}: {e}")
            continue

        ranked = sorted(zip(labels, similarities.tolist()), key=lambda x: -x[1])
        top1_label = ranked[0][0]
        top3_labels = [l for l, _ in ranked[:3]]
        top5_labels = [l for l, _ in ranked[:5]]

        top1_exact = top1_label in gt["exact"]
        top3_clinical = any(l in gt["clinical_top3"] for l in top3_labels)
        top5_clinical = any(l in gt["clinical_top5"] for l in top5_labels)

        total_ms += elapsed_ms

        results.append({
            "image": Path(img_path).name,
            "true_class": true_class,
            "top1_label": top1_label,
            "top1_score": round(ranked[0][1], 4),
            "top1_exact": top1_exact,
            "top3_clinical": top3_clinical,
            "top5_clinical": top5_clinical,
            "inference_ms": round(elapsed_ms, 1),
        })

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(test_images):
            pct = (i + 1) / len(test_images) * 100
            running_top1 = sum(1 for r in results if r["top1_exact"]) / len(results) * 100
            running_top5 = sum(1 for r in results if r["top5_clinical"]) / len(results) * 100
            print(f"  [{i+1:>4}/{len(test_images)}] ({pct:5.1f}%)  "
                  f"Top-1: {running_top1:5.1f}%  Top-5 Clin: {running_top5:5.1f}%  "
                  f"Avg: {total_ms / len(results):.0f} ms/img")

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    metrics = compute_metrics(results)
    agg = metrics["aggregate"]

    avg_ms = total_ms / len(results) if results else 0

    print(f"\n  RESULTS ({agg['total_images']} images)")
    print(f"  {'─' * 50}")
    print(f"  Top-1 Exact Accuracy:     {agg['top1_exact_accuracy']:5.1f}%  ({agg['top1_exact_count']}/{agg['total_images']})")
    print(f"  Top-3 Clinical Accuracy:  {agg['top3_clinical_accuracy']:5.1f}%  ({agg['top3_clinical_count']}/{agg['total_images']})")
    print(f"  Top-5 Clinical Accuracy:  {agg['top5_clinical_accuracy']:5.1f}%  ({agg['top5_clinical_count']}/{agg['total_images']})")
    print(f"  Average Inference:        {avg_ms:.0f} ms/image")
    if errors:
        print(f"  Errors:                   {errors}")

    print(f"\n  PER-CLASS BREAKDOWN")
    print(f"  {'Class':<20} {'N':>5} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}")
    print(f"  {'─' * 52}")
    for c, pc in sorted(metrics["per_class"].items()):
        print(f"  {c:<20} {pc['total']:>5} {pc['top1_exact']:>7.1f}% {pc['top3_clinical']:>7.1f}% {pc['top5_clinical']:>7.1f}%")

    print(f"\n  CONFUSION MATRIX (Top-1 Predictions)")
    cm = metrics["confusion_matrix"]
    pred_cols = sorted({p for row in cm.values() for p, count in row.items() if count > 0})
    cm_header = "True / Pred"
    header = f"  {cm_header:<20}" + "".join(f"{p[:12]:>13}" for p in pred_cols)
    print(header)
    print(f"  {'─' * (20 + 13 * len(pred_cols))}")
    for true_c in sorted(cm.keys()):
        row = f"  {true_c:<20}"
        for pred_c in pred_cols:
            count = cm[true_c].get(pred_c, 0)
            row += f"{count:>13}"
        print(row)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "model": MODEL_INT8.name,
            "dataset": str(DATASET_DIR.name),
            "total_images": len(results),
            "errors": errors,
            "avg_inference_ms": round(avg_ms, 1),
        },
        "metrics": metrics,
        "per_image_results": results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Save human-readable txt report
    # ------------------------------------------------------------------
    TXT_PATH = RESULTS_PATH.parent / "biomedclip_dataset_results.txt"
    with open(TXT_PATH, "w") as f:
        f.write("BiomedCLIP Large-Scale Zero-Shot Classification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:    {MODEL_INT8.name}\n")
        f.write(f"Dataset:  {DATASET_DIR.name}\n")
        f.write(f"Images:   {len(results)}\n")
        f.write(f"Errors:   {errors}\n")
        f.write(f"Avg Inference: {avg_ms:.0f} ms/image\n\n")

        f.write("AGGREGATE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-1 Exact Accuracy:    {agg['top1_exact_accuracy']:5.1f}%  ({agg['top1_exact_count']}/{agg['total_images']})\n")
        f.write(f"Top-3 Clinical Accuracy: {agg['top3_clinical_accuracy']:5.1f}%  ({agg['top3_clinical_count']}/{agg['total_images']})\n")
        f.write(f"Top-5 Clinical Accuracy: {agg['top5_clinical_accuracy']:5.1f}%  ({agg['top5_clinical_count']}/{agg['total_images']})\n\n")

        f.write("PER-CLASS BREAKDOWN\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<20} {'N':>5} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8}\n")
        f.write("-" * 52 + "\n")
        for c, pc in sorted(metrics["per_class"].items()):
            f.write(f"{c:<20} {pc['total']:>5} {pc['top1_exact']:>7.1f}% {pc['top3_clinical']:>7.1f}% {pc['top5_clinical']:>7.1f}%\n")

        f.write(f"\nCONFUSION MATRIX (Top-1 Predictions)\n")
        f.write("-" * 60 + "\n")
        cm = metrics["confusion_matrix"]
        pred_cols = sorted({p for row in cm.values() for p, count in row.items() if count > 0})
        cm_header = "True / Pred"
        header = f"{cm_header:<20}" + "".join(f"{p[:12]:>13}" for p in pred_cols)
        f.write(header + "\n")
        f.write("-" * (20 + 13 * len(pred_cols)) + "\n")
        for true_c in sorted(cm.keys()):
            row = f"{true_c:<20}"
            for pred_c in pred_cols:
                count = cm[true_c].get(pred_c, 0)
                row += f"{count:>13}"
            f.write(row + "\n")

        f.write(f"\n\nGROUND TRUTH DEFINITIONS\n")
        f.write("-" * 60 + "\n")
        for cls, gt in sorted(GROUND_TRUTH.items()):
            f.write(f"\n{cls}:\n")
            f.write(f"  Exact match: {gt['exact']}\n")
            f.write(f"  Clinical top-3 acceptable: {gt['clinical_top3']}\n")
            f.write(f"  Clinical top-5 acceptable: {gt['clinical_top5']}\n")

    print(f"  TXT report saved to: {TXT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
