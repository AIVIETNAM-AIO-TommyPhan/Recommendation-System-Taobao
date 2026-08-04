<#
.SYNOPSIS
    First-time setup: creates a virtualenv, installs dependencies, and checks
    that the raw data files are in place.

.DESCRIPTION
    Run this once (or again after pulling changes to requirements.txt).
    It does three things, in order, printing what it's doing at each step:
      1. Create a .venv/ virtual environment (skipped if it already exists).
      2. Install everything from requirements.txt into it.
      3. Check input_data/ has the 4 files the pipeline reads from — this is
         gitignored data you must copy in yourself, so we warn loudly if
         anything is missing rather than letting the pipeline fail later
         with a confusing FileNotFoundError.

.EXAMPLE
    .\scripts\setup.ps1
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== 1/3: Virtual environment ===" -ForegroundColor Cyan
if (Test-Path ".venv") {
    Write-Host "  .venv already exists, skipping creation."
} else {
    Write-Host "  Creating .venv ..."
    python -m venv .venv
}

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Could not find $VenvPython — venv creation must have failed."
}

Write-Host "`n=== 2/3: Installing dependencies ===" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip | Out-Null
& $VenvPython -m pip install -r requirements.txt
Write-Host "  Installed. Activate the venv in your own shell with:"
Write-Host "    .venv\Scripts\Activate.ps1" -ForegroundColor Yellow

Write-Host "`n=== 3/3: Checking input_data/ ===" -ForegroundColor Cyan
$RequiredFiles = @(
    "input_data\train.csv",
    "input_data\test.csv",
    "input_data\ad_feature.csv.zip",
    "input_data\user_profile.csv.zip"
)
$Missing = $RequiredFiles | Where-Object { -not (Test-Path $_) }
if ($Missing.Count -eq 0) {
    Write-Host "  All 4 required files found in input_data\." -ForegroundColor Green
} else {
    Write-Host "  Missing files (pipeline will fail without these):" -ForegroundColor Red
    $Missing | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
    Write-Host "  Copy the Taobao Display Ad Click dataset files into input_data\ before running the pipeline."
    Write-Host "  See README.md section 4-5 for what each file is."
}

Write-Host "`nSetup done. Next steps:" -ForegroundColor Cyan
Write-Host "  .\scripts\run_pipeline.ps1     # run Bronze -> Silver -> Gold"
Write-Host "  .\scripts\run_tests.ps1        # run the test suite"
Write-Host "  .\scripts\airflow_up.ps1       # or: run the pipeline via Airflow instead"
