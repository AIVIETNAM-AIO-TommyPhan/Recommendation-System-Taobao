<#
.SYNOPSIS
    Runs the medallion/ test suite (16 tests across bronze/silver/gold).

.DESCRIPTION
    Thin wrapper around `pytest`. Tests use hand-built DataFrames (see
    tests/conftest.py) rather than reading input_data/, so this works even
    before you've copied in the raw dataset.

.PARAMETER Verbose
    Pass -Verbose to run pytest in verbose mode (-v, shows each test name)
    instead of the default quiet mode (-q).

.EXAMPLE
    .\scripts\run_tests.ps1

.EXAMPLE
    .\scripts\run_tests.ps1 -Verbose
#>

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$VenvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$PytestArgs = if ($Verbose) { "-v" } else { "-q" }

Write-Host "=== Running pytest $PytestArgs ===" -ForegroundColor Cyan
& $Python -m pytest $PytestArgs
