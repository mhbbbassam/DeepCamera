#!/usr/bin/env bash
set -euo pipefail

# Enhanced deploy script for icebag-counter
# Purpose: make it easy to use an existing venv (or create one) and install dependencies.
# It can also optionally install a TensorRT wheel if TENSORRT_WHEEL is set.

SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SKILL_DIR"

# Allow user to specify an existing venv via environment variable VENV_PATH
# Examples:
#  export VENV_PATH="D:/commange/venv"   # on windows paths allowed when running in msys/Cygwin
#  export VENV_PATH="$SKILL_DIR/.venv"
VENV_PATH=${VENV_PATH:-"$SKILL_DIR/.venv"}

echo "Deploying icebag-counter skill in: $SKILL_DIR"
echo "Using VENV_PATH=$VENV_PATH"

# On POSIX, activate by sourcing activate script
if [ -f "$VENV_PATH/bin/activate" ]; then
  echo "Activating existing venv at $VENV_PATH"
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
elif [ -f "$VENV_PATH/Scripts/activate" ]; then
  echo "Detected Windows-style venv at $VENV_PATH (bash mode)"
  # shellcheck source=/dev/null
  source "$VENV_PATH/Scripts/activate"
else
  echo "No existing venv found at $VENV_PATH. Creating a new venv at $VENV_PATH"
  python3 -m venv "$VENV_PATH"
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate" || source "$VENV_PATH/Scripts/activate"
fi

python -m pip install --upgrade pip setuptools wheel

REQ_FILE="$SKILL_DIR/requirements_cuda.txt"
if [ -f "$REQ_FILE" ]; then
  echo "Installing python requirements from $REQ_FILE"
  pip install -r "$REQ_FILE" || {
    echo "pip install failed — inspect output. Trying to continue."
  }
else
  echo "No requirements file found at $REQ_FILE"
fi

# Optional: install TensorRT wheel if provided via env var
if [ -n "${TENSORRT_WHEEL:-}" ]; then
  if [ -f "$TENSORRT_WHEEL" ]; then
    echo "Installing TensorRT wheel: $TENSORRT_WHEEL"
    pip install "$TENSORRT_WHEEL" || echo "Failed to pip install TensorRT wheel"
  else
    echo "TENSORRT_WHEEL is set but file not found: $TENSORRT_WHEEL"
  fi
fi

# If running on Windows (Git Bash/MSYS) try to call trtexec to build engine from onnx if present
MODEL_ONNX="$SKILL_DIR/${MODEL_PATH:-models/icebag_yolo26.onnx}"
if [ -f "$MODEL_ONNX" ] && command -v trtexec >/dev/null 2>&1; then
  echo "Attempting to build TensorRT engine with trtexec from $MODEL_ONNX"
  ENGINE_OUT="${MODEL_ONNX%.*}.trt"
  trtexec --onnx="$MODEL_ONNX" --saveEngine="$ENGINE_OUT" --explicitBatch --fp16 || echo "trtexec failed (OK to continue)"
fi

# If running packaged (PyInstaller), set up GStreamer paths if packaged in the skill
if [ "${PYINSTALLER_APP:-false}" = "true" ]; then
  BASE_DIR="$SKILL_DIR"
  gst_bin_path="$BASE_DIR/bin"
  gst_plugin_path="$BASE_DIR/lib/gstreamer-1.0"
  if [ -d "$gst_bin_path" ]; then
    export PATH="$gst_bin_path:$PATH"
    echo "Added GStreamer bin path: $gst_bin_path"
  fi
  if [ -d "$gst_plugin_path" ]; then
    export GST_PLUGIN_PATH="$gst_plugin_path"
    echo "Set GST_PLUGIN_PATH=$gst_plugin_path"
  fi
fi

echo "Deploy complete. To run the skill under your preferred venv, either:
  - use the run_skill.bat (Windows) or run_skill.ps1 (PowerShell)
  - or ensure VENV_PATH is exported before starting the skill runner"
