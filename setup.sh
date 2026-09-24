#!/usr/bin/env bash
# One-time setup for the Human Activity Detection web app.
# Works on: macOS (Apple Silicon), Linux (Ubuntu/Debian), and Windows through WSL2 (Ubuntu).
# Usage (from the repo folder):   bash setup.sh
# Then start the app with:         bash run.sh
set -euo pipefail
cd "$(dirname "$0")"

DETECTRON2_COMMIT=a2f4a8771ab77e8411c26b27f24f9489a28a2453   # the Detectron2 version we tested
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "==> 1/5 Checking build tools (Detectron2 is compiled from source)"
if [ "$OS" = "Darwin" ]; then
  if [ "$ARCH" != "arm64" ]; then
    echo "Intel Macs are not supported: current PyTorch releases only ship Apple Silicon wheels for macOS."
    exit 1
  fi
  if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install || true
    echo "Finish the Command Line Tools install dialog, then re-run: bash setup.sh"
    exit 1
  fi
  export CC=clang CXX=clang++ ARCHFLAGS="-arch arm64"
else
  missing=""
  for tool in gcc g++ git curl; do
    command -v "$tool" >/dev/null 2>&1 || missing="$missing $tool"
  done
  if [ -n "$missing" ]; then
    echo "Missing:$missing"
    echo "Install them with:  sudo apt update && sudo apt install -y build-essential git curl"
    exit 1
  fi
fi

echo "==> 2/5 Checking for uv (installs Python and packages)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> 3/5 Creating .venv with Python 3.11"
uv venv --python 3.11 --allow-existing .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> 4/5 Installing Python packages"
if [ "$OS" = "Linux" ]; then
  # CPU-only PyTorch: a ~200 MB download instead of several GB of CUDA libraries.
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    || uv pip install torch torchvision
fi
uv pip install -r requirements-modern.txt

echo "==> 5/5 Building Detectron2 (takes 3-10 minutes)"
if python -c "import detectron2" >/dev/null 2>&1; then
  echo "Detectron2 already installed, skipping."
else
  MAX_JOBS="${MAX_JOBS:-4}" uv pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"
fi

python -c "import torch, detectron2, cv2, flask, pytorch_lightning as pl; print('OK  torch', torch.__version__, '| detectron2', detectron2.__version__, '| opencv', cv2.__version__, '| lightning', pl.__version__)"
echo
echo "Setup complete. Start the app with:  bash run.sh"
