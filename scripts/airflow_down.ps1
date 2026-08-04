<#
.SYNOPSIS
    Stops the Airflow stack started by scripts\airflow_up.ps1.

.DESCRIPTION
    Wraps `docker compose -f airflow/docker-compose.yaml down`. By default
    this stops and removes the containers but keeps the Postgres volume
    (so Airflow's DAG run history/metadata survives a restart).

.PARAMETER Wipe
    Also remove the Postgres volume (docker compose down -v) — use this for
    a completely clean slate, e.g. if Airflow's metadata DB got into a bad
    state. You'll need to re-create the admin user on next
    scripts\airflow_up.ps1 (handled automatically by the airflow-init step).

.EXAMPLE
    .\scripts\airflow_down.ps1

.EXAMPLE
    .\scripts\airflow_down.ps1 -Wipe
#>

param(
    [switch]$Wipe
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $DefaultDockerBin = "C:\Program Files\Docker\Docker\resources\bin"
    if (Test-Path (Join-Path $DefaultDockerBin "docker.exe")) {
        $env:PATH = "$DefaultDockerBin;$env:PATH"
    }
}

if ($Wipe) {
    Write-Host "=== Stopping Airflow and wiping the Postgres volume ===" -ForegroundColor Cyan
    docker compose -f airflow/docker-compose.yaml down -v
} else {
    Write-Host "=== Stopping Airflow (metadata volume kept) ===" -ForegroundColor Cyan
    docker compose -f airflow/docker-compose.yaml down
}

Write-Host "`nDone. Restart with: .\scripts\airflow_up.ps1" -ForegroundColor Green
