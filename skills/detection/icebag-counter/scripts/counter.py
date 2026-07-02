"""
Counter manager with geometry helpers and zone support.
Zones config example:
  zones:
    - id: 'door_line'
      type: 'line'
      coords: [[100,200],[600,200]]
      direction: 'both'
    - id: 'drop_zone'
      type: 'polygon'
      coords: [[10,10],[100,10],[100,100],[10,100]]

update_and_check(tracks) expects tracks list of dicts with x1,y1,x2,y2,track_id
Returns list of events: {zone_id, direction, track_id, timestamp}
"""
import time
import math
import numpy as np
import cv2


def point_in_poly(pt, poly):
    # cv2.pointPolygonTest expects numpy int32 array
    try:
        arr = np.array(poly, dtype=np.int32)
        return cv2.pointPolygonTest(arr, (pt[0], pt[1]), False) >= 0
    except Exception:
        return False


def segment_intersect(a,b,c,d):
    def orient(p,q,r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    a,b,c,d = tuple(a),tuple(b),tuple(c),tuple(d)
    o1 = orient(a,b,c)
    o2 = orient(a,b,d)
    o3 = orient(c,d,a)
    o4 = orient(c,d,b)
    return (o1*o2 < 0 and o3*o4 < 0)


def get_point_side(line_p1, line_p2, point):
    try:
        x1,y1 = line_p1
        x2,y2 = line_p2
        px,py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    except Exception:
        return 0

class CounterManager:
    def __init__(self, cfg):
        cfg = cfg or {}
        self.zones = cfg.get('zones', [])
        self.history = {}  # track_id -> list of centroids
        self.counted = {z['id']: set() for z in self.zones}
        self.debounce_frames = cfg.get('debounce_frames', 10)
        self.max_history = cfg.get('history_len', 16)

    def _centroid(self, bbox):
        x1,y1,x2,y2 = bbox
        return ((x1+x2)/2.0, (y1+y2)/2.0)

    def update_and_check(self, tracks):
        events = []
        for t in tracks:
            tid = t.get('track_id')
            if tid is None:
                # assign temporary negative id to keep history per-frame
                tid = f"tmp_{id(t)}_{int(time.time()*1000)}"
            bbox = [t['x1'], t['y1'], t['x2'], t['y2']]
            centroid = self._centroid(bbox)
            hist = self.history.setdefault(tid, [])
            hist.append(centroid)
            if len(hist) > self.max_history:
                hist.pop(0)

            # check zones
            for z in self.zones:
                zid = z.get('id')
                ztype = z.get('type')
                coords = z.get('coords')
                direction = z.get('direction','both')
                if ztype == 'line' and len(coords) >= 2 and len(hist) >= 2:
                    p_prev = hist[-2]
                    p_curr = hist[-1]
                    if segment_intersect(p_prev, p_curr, tuple(coords[0]), tuple(coords[1])):
                        if tid not in self.counted.get(zid, set()):
                            # simple direction detection via cross product
                            vx = coords[1][0] - coords[0][0]
                            vy = coords[1][1] - coords[0][1]
                            move_x = p_curr[0] - p_prev[0]
                            move_y = p_curr[1] - p_prev[1]
                            dot = move_x * (-vy) + move_y * vx
                            dir_str = 'in' if dot > 0 else 'out'
                            if direction in ('both','any') or direction == dir_str:
                                self.counted.setdefault(zid, set()).add(tid)
                                events.append({'zone_id': zid, 'direction': dir_str, 'track_id': tid, 'timestamp': time.time()})
                elif ztype == 'polygon' and coords:
                    inside = point_in_poly(centroid, coords)
                    if inside and tid not in self.counted.get(zid, set()):
                        self.counted.setdefault(zid, set()).add(tid)
                        events.append({'zone_id': zid, 'direction': 'enter', 'track_id': tid, 'timestamp': time.time()})
        return events
