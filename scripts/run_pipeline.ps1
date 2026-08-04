<#
.SYNOPSIS
    Runs the Bronze -> Silver -> Gold medallion pipeline locally (no Airflow,
    no Docker — just plain Python), and explains where the output lands.

.DESCRIPTION
    Thin wrapper around `python medallion/run_pipeline.py`. Useful when you
    want to iterate on medallion/ code directly and see print() output
    immediately, instead of going through the Airflow UI (scripts/airflow_up.ps1
    is the alternative for that).

    Requires input_data/ to already have the 4 raw files — run
    scripts/setup.ps1 first if you haven't.

.PARAMETER Layer
    Optional. Run only one layer instead of the full pipeline:
    "bronze", "silver", or "gold". Each layer reads whatever the previous
    layer already wrote to data/, so running "gold" alone requires that
    data/silver/ already exists from a prior run.

.EXAMPLE
    .\scripts\run_pipeline.ps1
    Runs all three layers end to end.

.EXAMPLE
    .\scripts\run_pipeline.ps1 -Layer gold
    Re-runs only Gold (e.g. after editing medallion/gold/features.py),
    reusing the Silver output already on disk.
#>

param(
    [ValidateSet("bronze", "silver", "gold")]
    [string]$Layer
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else {
    Write-Host "No .venv found, falling back to system 'python'. Run .\scripts\setup.ps1 first for an isolated env." -ForegroundColor Yellow
    "python"
}

if ($Layer) {
    $Module = switch ($Layer) {
        "bronze" { "medallion.bronze.ingest" }
        "silver" { "medallion.silver.clean" }
        "gold"   { "medallion.gold.features" }
    }
    Write-Host "=== Running layer: $Layer ($Module) ===" -ForegroundColor Cyan
    & $Python -m $Module
} else {
    Write-Host "=== Running full pipeline: Bronze -> Silver -> Gold ===" -ForegroundColor Cyan
    & $Python medallion\run_pipeline.py
}

Write-Host "`nOutput written to:" -ForegroundColor Cyan
Write-Host "  data\bronze\   raw, day-partitioned events + full ad/user catalogs"
Write-Host "  data\silver\   deduped fact_events + dim_ad + dim_user"
Write-Host "  data\gold\     training_features.parquet + item_coclick_snapshot.pkl"
Write-Host "`nNext: open experiment\EDA.ipynb or experiment\Full_Training_Model.ipynb"
