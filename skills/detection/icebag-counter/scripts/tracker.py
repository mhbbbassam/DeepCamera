"""
Tracker wrapper: exposes a standardized update(detections, frame) -> list of tracks
Each returned track is a dict: {"x1","y1","x2","y2","track_id"}
"""
from pathlib import Path
import numpy as np

class DummyTracker:
    def update(self, detections, frame):
        # detections: numpy array Nx6 [x1,y1,x2,y2,conf,cls]
        ret = []
        for i, d in enumerate(detections):
            x1,y1,x2,y2,conf,cls = d
            ret.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "track_id": i})
        return ret

def build_tracker(cfg: dict):
    ttype = cfg.get("type", "bytetrack")
    if ttype == "none":
        return DummyTracker()
    try:
        if ttype == "bytetrack":
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            tr_cfg = cfg.get("bytetrack", {})
            tracker = ByteTrack(frame_rate=tr_cfg.get("frame_rate", 30),
                                track_thresh=tr_cfg.get("track_thresh", 0.5),
                                track_buffer=tr_cfg.get("track_buffer", 30),
                                match_thresh=tr_cfg.get("match_thresh", 0.6))
            # Wrap to standard interface
            class BTWrapper:
                def __init__(self, impl):
                    self.impl = impl
                def update(self, detections, frame):
                    """
                    Expect detections Nx6 numpy [x1,y1,x2,y2,conf,cls]
                    Underlying ByteTrack impl may return array-like rows [x1,y1,x2,y2,track_id]
                    Convert to list of dicts.
                    """
                    out = self.impl.update(detections, frame)
                    tracks = []
                    # Normalize outputs
                    if isinstance(out, np.ndarray):
                        for row in out:
                            x1,y1,x2,y2,tid = row[:5]
                            tracks.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "track_id": int(tid)})
                    else:
                        # assume list of dicts or objects
                        for item in out:
                            try:
                                tracks.append({"x1": int(item[0]), "y1": int(item[1]), "x2": int(item[2]), "y2": int(item[3]), "track_id": int(item[4])})
                            except Exception:
                                # item may already be dict
                                if isinstance(item, dict):
                                    tracks.append(item)
                    return tracks
            return BTWrapper(tracker)
    except Exception:
        # if ByteTrack not available, fallback to DummyTracker
        return DummyTracker()
