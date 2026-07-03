---
name: icebag-counter
description: "Icebag Counting Skill — custom YOLO26 detector + ByteTrack + line/zone counting"
version: 1.0.0
icon: assets/icon.png
entry: scripts/run_skill.bat
deploy: deploy.sh

requirements:
  python: ">=3.9"
  platforms: ["linux", "macos", "windows"]

parameters:
  - name: model_path
    label: "Model Path"
    type: string
    default: "models/icebag_yolo26.onnx"
    description: "Path to ONNX/TensorRT/PT model file"
    group: Model

  - name: model_format
    label: "Model Format"
    type: select
    options: ["auto","tensorrt","onnx","pytorch"]
    default: "auto"
    description: "auto = detect best format; tensorrt prefers .trt/.engine"
    group: Model

  - name: device
    label: "Inference Device"
    type: select
    options: ["auto","cuda","cpu"]
    default: "auto"
    group: Performance

  - name: confidence
    label: "Confidence Threshold"
    type: number
    min: 0.01
    max: 1.0
    default: 0.5
    group: Model

  - name: iou
    label: "NMS IoU"
    type: number
    min: 0.01
    max: 1.0
    default: 0.45
    group: Model

  - name: fps
    label: "Processing FPS"
    type: select
    options: [0.5,1,3,5,10,15]
    default: 5
    group: Performance

  - name: tracker_type
    label: "Tracker"
    type: select
    options: ["bytetrack","deepocsort","strongsort","botsort","ocsort","none"]
    default: "bytetrack"
    group: Features

  - name: venv_path
    label: "Python virtualenv path"
    type: string
    default: "D:\\commange\\venv"
    description: "Full path to the Python venv to use for this skill (Windows). If empty, skill will try to use an internal .venv or create one."
    group: Environment

  - name: line_coords
    label: "Counting line (x1,y1;x2,y2)"
    type: string
    default: ""
    description: "Example: 100,200;600,200"
    group: Counting

capabilities:
  live_detection:
    script: scripts/run_skill.bat
    description: "Detects icebag objects, tracks them and emits counting events (runs under configured venv)"
---
