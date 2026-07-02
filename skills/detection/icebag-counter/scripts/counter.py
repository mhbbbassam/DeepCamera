"""
CounterManager: manages line/polygon zones and determines crossing events.
Events returned: list of dicts: {"zone_id","direction","track_id","timestamp"}
"""

import time
import math
import numpy as np
import cv2

class CounterManager:
    def __init__(self, cfg):
        # cfg.zones: list of {id,type,coords,direction}
        self.zones = cfg.get("zones", []) if cfg else []
        # track history: track_id -> list of centroids
        self.history = {}
        # per-zone counted track ids
        self.counted = {z['id']: set() for z in self.zones}
        self.debounce_frames = cfg.get("debounce_frames", 10) if cfg else 10

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def _segment_intersect(self, a,b,c,d):
        # reuse simple orientation-based intersection
        def orient(p,q,r):
            return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
        a,b,c,d = tuple(a),tuple(b),tuple(c),tuple(d)
        o1 = orient(a,b,c)
        o2 = orient(a,b,d)
        o3 = orient(c,d,a)
        o4 = orient(c,d,b)
        return (o1*o2 < 0 and o3*o4 < 0)

    def update_and_check(self, tracks):
        """
        tracks: list of dicts {"x1","y1","x2","y2","track_id"}
        returns: events list of crossing events
        """
        events = []
        for t in tracks:
            tid = t['track_id']
            bbox = [t['x1'], t['y1'], t['x2'], t['y2']]
            centroid = self._centroid(bbox)
            hist = self.history.setdefault(tid, [])
            hist.append(centroid)
            if len(hist) > 10:
                hist.pop(0)

            # check all zones
            for z in self.zones:
                zid = z['id']
                ztype = z['type']
                coords = z['coords']
                direction = z.get('direction', 'both')
                if ztype == 'line' and len(coords)>=2 and len(hist) >= 2:
                    p_prev = hist[-2]
                    p_curr = hist[-1]
                    if self._segment_intersect(p_prev, p_curr, tuple(coords[0]), tuple(coords[1])):
                        if tid not in self.counted.get(zid, set()):
                            # determine simple direction using cross product sign
                            vx = coords[1][0] - coords[0][0]
                            vy = coords[1][1] - coords[0][1]
                            move_x = p_curr[0] - p_prev[0]
                            move_y = p_curr[1] - p_prev[1]
                            dot = move_x * (-vy) + move_y * vx
                            dir_str = "unknown"
                            if dot > 0:
                                dir_str = "in"
                            else:
                                dir_str = "out"
                            if direction in ("both","any") or direction == dir_str:
                                self.counted.setdefault(zid, set()).add(tid)
                                events.append({"zone_id": zid, "direction": dir_str, "track_id": tid, "timestamp": time.time()})
                elif ztype == 'polygon':
                    poly = np.array(coords, dtype=np.int32)
                    inside = (cv2.pointPolygonTest(poly, (centroid[0], centroid[1]), False) >= 0)
                    if inside and tid not in self.counted.get(zid, set()):
                        self.counted.setdefault(zid, set()).add(tid)
                        events.append({"zone_id": zid, "direction": "enter", "track_id": tid, "timestamp": time.time()})
        return events
