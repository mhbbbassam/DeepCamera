# Icebag Counter Skill

Purpose: count icebag objects crossing a line or entering a zone using a custom YOLO26 model and ByteTrack.

Quick start (Windows with GPU):
1. Place your model at `skills/detection/icebag-counter/models/icebag_yolo26.onnx` (or .trt/.engine/.pt)
2. Edit `config.yaml` to set model_path, tracker type and counting zones.
3. Run deploy (to install deps):
   - Windows: `deploy.bat`
   - Linux/macOS: `./deploy.sh`
4. Start skill manually for testing:
   python scripts/detect.py --config config.yaml

Integration with Aegis:
- The skill reads JSONL `frame` events on stdin and writes `ready`, `detections`, `count`, and `perf_stats` messages to stdout.
- Example `frame` message:
  `{"event":"frame","frame_id":42,"camera_id":"front_door","timestamp":"...","frame_path":"C:/tmp/frame.jpg"}`
