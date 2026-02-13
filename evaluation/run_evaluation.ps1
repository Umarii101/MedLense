# MedLens Evaluation Runner
# Run from: Project 1/ root directory
# Prerequisites: venv activated, dataset extracted to evaluation/_raw_dataset/

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MedLens - BiomedCLIP Evaluation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Step 1: Sample dataset
Write-Host "`n[Step 1/2] Sampling 100 images per class from raw dataset..." -ForegroundColor Yellow
python evaluation/download_eval_dataset.py --skip-download --samples-per-class 100

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Dataset sampling failed. Make sure you extracted the dataset to evaluation/_raw_dataset/" -ForegroundColor Red
    exit 1
}

# Step 2: Run BiomedCLIP evaluation
Write-Host "`n[Step 2/2] Running BiomedCLIP zero-shot classification on 400 images..." -ForegroundColor Yellow
python evaluation/biomedclip_dataset_eval.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Evaluation failed." -ForegroundColor Red
    exit 1
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  DONE! Results saved to:" -ForegroundColor Green
Write-Host "    evaluation/results/biomedclip_dataset_results.txt" -ForegroundColor White
Write-Host "    evaluation/results/biomedclip_dataset_results.json" -ForegroundColor White
Write-Host "    evaluation/results/download_summary.txt" -ForegroundColor White
Write-Host "========================================" -ForegroundColor Green
