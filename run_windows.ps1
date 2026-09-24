# Starts the web app at http://127.0.0.1:5001 on native Windows.
# The first launch downloads the Detectron2 keypoint model (~240 MB).
# Usage:  powershell -ExecutionPolicy Bypass -File .\run_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "No .venv found. Run setup first:  powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1"
    exit 1
}
if (-not $env:PORT) { $env:PORT = "5001" }
Write-Host "Loading models... then open http://127.0.0.1:$($env:PORT)"
& $VenvPython app.py
