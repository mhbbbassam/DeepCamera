"""
Optional Excel logger wrapper (asynchronous writer) reusing Mainapp logic.
If pandas/openpyxl not available the logger becomes a noop.
"""
import threading
import time
import os
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False

class ExcelLogger:
    def __init__(self, filename):
        self.filename = filename
        self._pending = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if PANDAS_AVAILABLE:
            self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filename):
            try:
                df = pd.DataFrame(columns=['Timestamp','Source','Direction','ObjectID','TotalCount','Width','Height'])
                df.to_excel(self.filename, index=False, engine='openpyxl')
            except Exception:
                pass

    def log_crossing(self, source_name, direction, object_id, total_count, width=0, height=0):
        entry = {
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'Source': source_name,
            'Direction': direction,
            'ObjectID': object_id,
            'TotalCount': total_count,
            'Width': width,
            'Height': height
        }
        with self._lock:
            self._pending.append(entry)

    def _loop(self):
        while not self._stop.is_set():
            self._flush()
            self._stop.wait(2)
        self._flush()

    def _flush(self):
        with self._lock:
            if not self._pending:
                return
            entries = self._pending[:]
            self._pending.clear()
        if not PANDAS_AVAILABLE:
            return
        try:
            try:
                df = pd.read_excel(self.filename, engine='openpyxl')
            except Exception:
                df = pd.DataFrame(columns=['Timestamp','Source','Direction','ObjectID','TotalCount','Width','Height'])
            new = pd.DataFrame(entries)
            df = pd.concat([df, new], ignore_index=True)
            df.to_excel(self.filename, index=False, engine='openpyxl')
        except Exception:
            pass

    def stop(self):
        self._stop.set()
        self._thread.join()
