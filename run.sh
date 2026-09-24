#!/usr/bin/env bash
# Starts the web app at http://127.0.0.1:5001 (macOS, Linux, WSL2).
# The first launch downloads the Detectron2 keypoint model (~240 MB).
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .venv/bin/activate ]; then
  echo "No .venv found. Run setup first:  bash setup.sh"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PORT="${PORT:-5001}"
echo "Loading models... then open http://127.0.0.1:${PORT}"
python app.py
