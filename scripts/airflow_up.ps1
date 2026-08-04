<#
.SYNOPSIS
    Builds and starts the Airflow stack (Postgres + webserver + scheduler)
    that runs the medallion pipeline as a DAG, then waits until it's ready.

.DESCRIPTION
    Wraps `docker compose -f airflow/docker-compose.yaml up -d --build`.
    Requires Docker Desktop to be installed and running — see
    airflow/README.md for install instructions if `docker` isn't found.

    This script also works around a common gotcha on a fresh Docker Desktop
    install: its `resources\bin` folder (where docker.exe and the
    credential helper live) isn't always on PATH yet for the current shell
    session, which makes plain `docker ...` fail even though Docker Desktop
    is running. If `docker` isn't already on PATH, we add the default
    install location for this session only (no permanent PATH change).

.PARAMETER Rebuild
    Force a rebuild even if you haven't changed the Dockerfile
    (equivalent to always passing --build; this script already does that
    by default, so this flag mainly exists for symmetry/documentation).

.EXAMPLE
    .\scripts\airflow_up.ps1
    Then open http://localhost:8080 and log in with admin / admin.
#>

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=== 1/3: Checking Docker ===" -ForegroundColor Cyan
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    $DefaultDockerBin = "C:\Program Files\Docker\Docker\resources\bin"
    if (Test-Path (Join-Path $DefaultDockerBin "docker.exe")) {
        Write-Host "  'docker' not on PATH for this session — adding $DefaultDockerBin"
        $env:PATH = "$DefaultDockerBin;$env:PATH"
    } else {
        throw "Docker not found. Install Docker Desktop first: winget install Docker.DockerDesktop (see airflow/README.md)."
    }
}
docker info *>$null
if ($LASTEXITCODE -ne 0) {
    throw "Docker CLI found but the daemon isn't responding — open Docker Desktop and wait for it to finish starting, then re-run this script."
}
Write-Host "  Docker is running." -ForegroundColor Green

Write-Host "`n=== 2/3: Building and starting the stack ===" -ForegroundColor Cyan
Write-Host "  (first run downloads the Airflow base image + installs deps — a few minutes)"
docker compose -f airflow/docker-compose.yaml up -d --build

Write-Host "`n=== 3/3: Waiting for the webserver to become healthy ===" -ForegroundColor Cyan
$Healthy = $false
for ($i = 0; $i -lt 24; $i++) {
    $Status = docker compose -f airflow/docker-compose.yaml ps airflow-webserver --format "{{.Status}}" 2>$null
    Write-Host "  $Status"
    if ($Status -match "healthy") { $Healthy = $true; break }
    Start-Sleep -Seconds 5
}

if ($Healthy) {
    Write-Host "`nAirflow is up." -ForegroundColor Green
} else {
    Write-Host "`nWebserver isn't reporting healthy yet — it may just need more time." -ForegroundColor Yellow
    Write-Host "Check with: docker compose -f airflow/docker-compose.yaml ps"
}

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Open http://localhost:8080  (login: admin / admin)"
Write-Host "  2. Find the 'medallion_pipeline' DAG and click the trigger (>) button"
Write-Host "  3. Watch progress in the Grid view; click a task box -> Logs for details"
Write-Host "`nTo stop: .\scripts\airflow_down.ps1"
