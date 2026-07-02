#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SKILL_DIR="$ROOT"
VENV_DIR="$SKILL_DIR/.venv"

echo "Deploying icebag-counter skill..."

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$SKILL_DIR/requirements_cuda.txt" || {
  echo "pip install failed — inspect and install platform-specific wheels manually"
  exit 1
}

# Optional: try to build TensorRT engine from ONNX if trtexec present
MODEL_ONNX="$SKILL_DIR/${MODEL_PATH:-models/icebag_yolo26.onnx}"
ENGINE_OUT="${MODEL_ONNX%.*}.trt"
if [ -f "$MODEL_ONNX" ] && command -v trtexec >/dev/null 2>&1; then
  echo "Attempting to build TensorRT engine with trtexec..."
  trtexec --onnx="$MODEL_ONNX" --saveEngine="$ENGINE_OUT" --explicitBatch --fp16 || echo "trtexec failed (ok to continue)"
fi

echo "Deploy complete. Use the skill via Aegis or run scripts/detect.py manually."
