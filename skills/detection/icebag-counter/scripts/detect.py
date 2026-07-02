#!/usr/bin/env python3
import sys, os, json, time, argparse, traceback
from pathlib import Path

# Ensure repo lib on path if running inside skills folder
HERE = Path(__file__).resolve().parent.parent
LIB = HERE / "lib"
if LIB.exists():
    sys.path.insert(0, str(LIB))

import yaml

from model_loader import load_model
from tracker import build_tracker
from counter import CounterManager

# helper emitter
def emit(obj):
    print(json.dumps(obj), flush=True)

def load_config(path=None, overrides=None):
    cfg = {}
    if path and Path(path).exists():
        with open(path,'r') as f:
            cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg.update(overrides)
    # normalize nested cfg expected by submodules
    full = {
        "model": cfg.get("model", {}),
        "tracking": cfg.get("tracking", {}),
        "filtering": cfg.get("filtering", {}),
        "counting": cfg.get("counting", {}),
        "logging": cfg.get("logging", {}),
    }
    # copy top-level keys into model for backward compatibility
    top_keys = ["model_path","model_format","device","confidence","iou"]
    for k in top_keys:
        if cfg.get(k) is not None:
            full["model"][k] = cfg[k]
    return full

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=str(HERE/"config.yaml"))
    parser.add_argument("--overrides", help="JSON overrides for quick testing", default=None)
    args = parser.parse_args()

    overrides = None
    if args.overrides:
        overrides = json.loads(args.overrides)

    cfg = load_config(args.config, overrides)

    # load model
    try:
        model = load_model(cfg["model"])
    except Exception as e:
        sys.stderr.write(f"Model load failed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    tracker = build_tracker(cfg["tracking"]) if cfg["tracking"].get("enabled", True) else None
    counter = CounterManager(cfg["counting"] or {})

    # emit ready
    emit({"event":"ready","model":cfg["model"].get("model_path"), "device": cfg["model"].get("device","auto"), "backend":"custom", "format": cfg["model"].get("model_format","auto")})

    # main loop: read JSONL from stdin
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            data = json.loads(line)
            if data.get("event") == "frame":
                frame_path = data.get("frame_path")
                frame_id = data.get("frame_id")
                camera_id = data.get("camera_id")
                ts = data.get("timestamp", time.time())

                if not frame_path or not Path(frame_path).exists():
                    sys.stderr.write(f"Frame missing or not found: {frame_path}\n")
                    continue

                t0 = time.time()
                # inference
                detections = model(frame_path, conf=cfg["model"].get("confidence",0.5), classes=cfg["model"].get("class_ids"))
                # detections: list of dicts {"x1","y1","x2","y2","conf","cls"}

                # apply filtering sizes
                filt = []
                fcfg = cfg.get("filtering", {})
                for d in detections:
                    w = d["x2"] - d["x1"]
                    h = d["y2"] - d["y1"]
                    if w >= fcfg.get("min_w", 0) and w <= fcfg.get("max_w", 99999) and \
                       h >= fcfg.get("min_h", 0) and h <= fcfg.get("max_h", 99999):
                        filt.append(d)
                import numpy as np
                dets_np = np.array([[d["x1"],d["y1"],d["x2"],d["y2"],d["conf"],d["cls"]] for d in filt]) if filt else np.empty((0,6))

                # tracking
                if tracker and dets_np.shape[0] > 0:
                    tracks = tracker.update(dets_np, frame_path)
                else:
                    # no tracker -> convert detections to track-like dicts (track_id None)
                    tracks = []
                    for i,d in enumerate(filt):
                        tracks.append({"x1":int(d["x1"]),"y1":int(d["y1"]),"x2":int(d["x2"]),"y2":int(d["y2"]),"track_id": None, "conf": float(d["conf"]), "class": d["cls"]})

                # counter events
                events = counter.update_and_check(tracks)

                # prepare output objects (for Aegis overlay)
                objects_out = []
                for t in tracks:
                    objects_out.append({
                        "class": t.get("class","icebag"),
                        "confidence": float(t.get("conf",0.0)),
                        "bbox": [int(t["x1"]), int(t["y1"]), int(t["x2"]), int(t["y2"])],
                        "track_id": t.get("track_id")
                    })

                emit({
                    "event":"detections",
                    "frame_id": frame_id,
                    "camera_id": camera_id,
                    "timestamp": ts,
                    "objects": objects_out
                })

                # emit count events separately so Aegis can surface them
                for ev in events:
                    emit({
                        "event":"count",
                        "camera_id": camera_id,
                        "zone_id": ev["zone_id"],
                        "direction": ev["direction"],
                        "track_id": ev["track_id"],
                        "timestamp": ev["timestamp"]
                    })

                # perf_stats (simple)
                t1 = time.time()
                emit({"event":"perf_stats","frame_id":frame_id,"camera_id":camera_id,"timings_ms":{"total":(t1-t0)*1000}})

            elif data.get("command") == "stop":
                emit({"event":"stopped"})
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            sys.stderr.write("Error in main loop: " + str(e) + "\n")
            traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
