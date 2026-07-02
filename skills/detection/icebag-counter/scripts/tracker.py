"""
Tracker wrapper: supports multiple trackers (ByteTrack, StrongSort, DeepOcSort, BoTSORT, OcSort)
Provides build_tracker(cfg) -> tracker with update(detections, frame) -> list[dict]
Each returned dict: {"x1","y1","x2","y2","track_id","conf","class"}
"""
from pathlib import Path
import numpy as np

class DummyTracker:
    def update(self, detections, frame):
        # detections: numpy array Nx6 [x1,y1,x2,y2,conf,cls]
        out = []
        for i, d in enumerate(detections):
            x1,y1,x2,y2,conf,cls = d
            out.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "track_id": i, "conf": float(conf), "class": int(cls)})
        return out


def _wrap_listlike(out):
    tracks = []
    if out is None:
        return tracks
    # numpy array rows
    if isinstance(out, np.ndarray):
        for row in out:
            vals = row.tolist()
            if len(vals) >= 5:
                x1,y1,x2,y2,tid = vals[:5]
                tracks.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "track_id": int(tid)})
    else:
        # list of dicts or objects
        for it in out:
            if isinstance(it, dict):
                tracks.append(it)
            elif hasattr(it, '__len__'):
                try:
                    x1,y1,x2,y2,tid = it[:5]
                    tracks.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "track_id": int(tid)})
                except Exception:
                    pass
    return tracks


def build_tracker(cfg: dict):
    ttype = cfg.get("type", "bytetrack")
    # config for trackers
    tcfg = cfg.get(ttype, {}) if isinstance(cfg, dict) else {}

    if ttype == "none":
        return DummyTracker()

    # ByteTrack
    if ttype == "bytetrack":
        try:
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack
            tr = ByteTrack(frame_rate=tcfg.get('frame_rate', 30),
                           track_thresh=tcfg.get('track_thresh', 0.5),
                           track_buffer=tcfg.get('track_buffer', 30),
                           match_thresh=tcfg.get('match_thresh', 0.6))
            class BTWrap:
                def __init__(self, impl):
                    self.impl = impl
                def update(self, detections, frame):
                    out = self.impl.update(detections, frame)
                    return _wrap_listlike(out)
            return BTWrap(tr)
        except Exception:
            return DummyTracker()

    # StrongSort
    if ttype == 'strongsort':
        try:
            from boxmot.trackers.strongsort.strongsort import StrongSort
            reid = Path(cfg.get('reid_weights', 'models/reid_resnet50_triplet.pt'))
            device = cfg.get('device', 'cuda:0')
            tracker = StrongSort(reid_weights=reid, device=device, half=tcfg.get('half', False))
            class SSWrap:
                def __init__(self, impl): self.impl = impl
                def update(self, detections, frame):
                    out = self.impl.update(detections, frame)
                    return _wrap_listlike(out)
            return SSWrap(tracker)
        except Exception:
            return DummyTracker()

    # DeepOcSort
    if ttype == 'deepocsort':
        try:
            from boxmot.trackers.deepocsort.deepocsort import DeepOcSort
            reid = Path(cfg.get('reid_weights', 'models/reid_resnet50_triplet.pt'))
            device = cfg.get('device', 'cuda:0')
            tracker = DeepOcSort(reid_weights=reid, device=device, half=tcfg.get('half', False), embedding_off=tcfg.get('embedding_off', True))
            class DOWrap:
                def __init__(self, impl): self.impl = impl
                def update(self, detections, frame):
                    out = self.impl.update(detections, frame)
                    return _wrap_listlike(out)
            return DOWrap(tracker)
        except Exception:
            return DummyTracker()

    # BoTSORT
    if ttype == 'botsort':
        try:
            from boxmot.trackers.botsort.botsort import BotSort
            reid = Path(cfg.get('reid_weights', 'models/reid_resnet50_triplet.pt'))
            device = cfg.get('device', 'cuda:0')
            tracker = BotSort(reid_weights=reid, device=device, half=tcfg.get('half', False))
            class BWrap:
                def __init__(self, impl): self.impl = impl
                def update(self, detections, frame):
                    out = self.impl.update(detections, frame)
                    return _wrap_listlike(out)
            return BWrap(tracker)
        except Exception:
            return DummyTracker()

    # OcSort
    if ttype == 'ocsort':
        try:
            from boxmot.trackers.ocsort.ocsort import OcSort
            tracker = OcSort(det_thresh=tcfg.get('det_thresh', 0.10), max_age=tcfg.get('max_age', 60), min_hits=tcfg.get('min_hits', 3), asso_func=tcfg.get('asso_func','iou'))
            class OWrap:
                def __init__(self, impl): self.impl = impl
                def update(self, detections, frame):
                    out = self.impl.update(detections, frame)
                    return _wrap_listlike(out)
            return OWrap(tracker)
        except Exception:
            return DummyTracker()

    return DummyTracker()
