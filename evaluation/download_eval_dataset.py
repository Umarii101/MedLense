"""
Download COVID-19 Radiography Database from Kaggle and sample a balanced subset
for BiomedCLIP evaluation.

Dataset: tawsifurrahman/covid19-radiography-database
Classes: COVID, Normal, Lung_Opacity, Viral Pneumonia
Source: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Usage:
    python download_eval_dataset.py [--samples-per-class 100]

Requires: kaggle API credentials in ~/.kaggle/kaggle.json
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

DATASET = "tawsifurrahman/covid19-radiography-database"

# Map dataset folder names -> our taxonomy labels
CLASS_MAP = {
    "COVID": "covid19",
    "Normal": "normal",
    "Lung_Opacity": "lung_opacity",
    "Viral Pneumonia": "viral_pneumonia",
}

EVAL_DIR = Path(__file__).parent.parent / "evaluation" / "test_data" / "chest_xray_dataset"


def download_dataset(dest: Path):
    """Download full dataset via Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {DATASET}...")
    print(f"Destination: {dest}")
    api.dataset_download_files(DATASET, path=str(dest), unzip=True)
    print("Download complete.")


def find_image_dirs(dest: Path) -> dict:
    """Find the image directories for each class."""
    # The dataset extracts to: dest/COVID-19_Radiography_Dataset/<class>/images/
    base = None
    for candidate in [
        dest / "COVID-19_Radiography_Dataset",
        dest / "COVID-19 Radiography Database",
        dest,
    ]:
        if candidate.exists() and any(candidate.iterdir()):
            base = candidate
            break

    if base is None:
        print(f"ERROR: Could not find extracted dataset in {dest}")
        sys.exit(1)

    class_dirs = {}
    for folder_name, label in CLASS_MAP.items():
        # Try with /images/ subfolder first (newer dataset versions)
        images_dir = base / folder_name / "images"
        if not images_dir.exists():
            images_dir = base / folder_name
        if images_dir.exists():
            class_dirs[label] = images_dir
        else:
            print(f"WARNING: Class directory not found: {base / folder_name}")

    return class_dirs


def sample_and_copy(class_dirs: dict, samples_per_class: int, seed: int = 42):
    """Sample N images per class and copy to evaluation directory."""
    random.seed(seed)

    if EVAL_DIR.exists():
        shutil.rmtree(EVAL_DIR)

    total = 0
    summary = {}

    for label, src_dir in sorted(class_dirs.items()):
        # Collect image files
        extensions = {".png", ".jpg", ".jpeg"}
        images = [f for f in src_dir.iterdir() if f.suffix.lower() in extensions]

        available = len(images)
        n = min(samples_per_class, available)

        sampled = random.sample(images, n)

        # Copy to eval directory
        dest_dir = EVAL_DIR / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sampled:
            shutil.copy2(img_path, dest_dir / img_path.name)

        summary[label] = {"available": available, "sampled": n}
        total += n
        print(f"  {label}: {n}/{available} images sampled")

    print(f"\nTotal: {total} images -> {EVAL_DIR}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Download chest X-ray evaluation dataset")
    parser.add_argument("--samples-per-class", type=int, default=100,
                        help="Number of images to sample per class (default: 100)")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to download the full dataset (default: temp in eval dir)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, just resample from existing download")
    args = parser.parse_args()

    download_dir = Path(args.download_dir) if args.download_dir else (
        Path(__file__).parent.parent / "evaluation" / "_raw_dataset"
    )

    if not args.skip_download:
        download_dir.mkdir(parents=True, exist_ok=True)
        download_dataset(download_dir)
    else:
        print(f"Skipping download, using existing data in {download_dir}")

    print("\nFinding image directories...")
    class_dirs = find_image_dirs(download_dir)
    if not class_dirs:
        print("ERROR: No class directories found!")
        sys.exit(1)

    print(f"Found {len(class_dirs)} classes: {list(class_dirs.keys())}")
    print(f"\nSampling {args.samples_per_class} images per class...")
    summary = sample_and_copy(class_dirs, args.samples_per_class)

    # Save summary to txt
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    txt_path = results_dir / "download_summary.txt"
    with open(txt_path, "w") as f:
        f.write("COVID-19 Radiography Dataset - Sampling Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source: tawsifurrahman/covid19-radiography-database\n")
        f.write(f"Samples per class: {args.samples_per_class}\n")
        f.write(f"Random seed: 42\n\n")
        total = 0
        for label, info in sorted(summary.items()):
            f.write(f"  {label}: {info['sampled']}/{info['available']} sampled\n")
            total += info["sampled"]
        f.write(f"\nTotal: {total} images\n")
        f.write(f"Output: {EVAL_DIR}\n")
    print(f"\nSummary saved to: {txt_path}")

    print("\nDone! Run the evaluation with:")
    print("  python evaluation/biomedclip_dataset_eval.py")


if __name__ == "__main__":
    main()
