# One-time setup for native Windows 10/11 (64-bit).
# WSL2 is the recommended way to run this project on Windows (see README); use this script
# if you prefer a native install.
# Usage, from the repo folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
# Then start the app with:
#   powershell -ExecutionPolicy Bypass -File .\run_windows.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Detectron2Commit = "a2f4a8771ab77e8411c26b27f24f9489a28a2453"   # the Detectron2 version we tested
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

Write-Host "==> 1/5 Checking build tools (Detectron2 is compiled from source)"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Git is missing. Install it with:  winget install --id Git.Git -e"
    Write-Host "Then open a new PowerShell window and re-run this script."
    exit 1
}
$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = $null
if (Test-Path $vswhere) {
    $vsPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}
if (-not $vsPath) {
    Write-Host "The Microsoft C++ Build Tools are missing. Install them with:"
    Write-Host '  winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"'
    Write-Host "Then re-run this script."
    exit 1
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"

Write-Host "==> 2/5 Checking for uv (installs Python and packages)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "==> 3/5 Creating .venv with Python 3.11"
uv venv --python 3.11 --allow-existing .venv
if ($LASTEXITCODE -ne 0) { throw "Creating the virtual environment failed." }

Write-Host "==> 4/5 Installing Python packages"
# PyTorch from PyPI is the CPU build on Windows.
uv pip install --python $VenvPython -r requirements-modern.txt
if ($LASTEXITCODE -ne 0) { throw "Installing packages failed." }

Write-Host "==> 5/5 Building Detectron2 (takes 5-15 minutes)"
$ErrorActionPreference = "Continue"   # a failed import writes to stderr; don't treat that as fatal
& $VenvPython -c "import detectron2" 2>$null
$alreadyInstalled = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if ($alreadyInstalled) {
    Write-Host "Detectron2 already installed, skipping."
} else {
    # Build inside the Visual Studio x64 developer environment via a small batch file
    # (avoids cmd.exe quoting problems with paths that contain spaces).
    $bat = Join-Path $env:TEMP "build_detectron2.bat"
    Set-Content -Path $bat -Encoding ASCII -Value @(
        "@echo off",
        "call `"$vcvars`"",
        "set DISTUTILS_USE_SDK=1",
        "uv pip install --python `"$VenvPython`" --no-build-isolation git+https://github.com/facebookresearch/detectron2.git@$Detectron2Commit",
        "exit /b %errorlevel%"
    )
    & $bat
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "Detectron2 failed to build natively. The WSL2 route in the README avoids this."
        exit 1
    }
}

& $VenvPython -c "import torch, detectron2, cv2, flask, pytorch_lightning as pl; print('OK  torch', torch.__version__, '| detectron2', detectron2.__version__, '| opencv', cv2.__version__, '| lightning', pl.__version__)"
Write-Host "==> Building the React frontend (needs Node.js 20.19+)"
$nodeOk = $false
if (Get-Command node -ErrorAction SilentlyContinue) {
    $v = [version]((node -v).TrimStart('v'))
    $nodeOk = ($v.Major -gt 20) -or ($v.Major -eq 20 -and $v.Minor -ge 19)
}
if ($nodeOk) {
    Push-Location frontend
    npm ci
    if ($LASTEXITCODE -eq 0) { npm run build }
    if ($LASTEXITCODE -ne 0) { Write-Host "Frontend build failed; the app will use the existing frontend\dist if there is one." }
    Pop-Location
} else {
    Write-Host "Node.js 20.19+ not found. Skipping the frontend build (winget install OpenJS.NodeJS.LTS to get it)."
    if (Test-Path "frontend\dist\index.html") { Write-Host "Using the existing build in frontend\dist." }
    else { Write-Host "Until then the app serves the original page." }
}

Write-Host ""
Write-Host "Setup complete. Start the app with:  powershell -ExecutionPolicy Bypass -File .\run_windows.ps1"
