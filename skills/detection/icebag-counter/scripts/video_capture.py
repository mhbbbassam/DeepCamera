"""
Video capture helper with RTSP/GStreamer support and robust grab/retrieve loop.
Provides VideoSource class with read() -> (ok, frame) and automatic reopen on failure.
"""
import cv2
import time


def get_gst_pipeline(source_url: str) -> str:
    # uses rtspsrc with low latency and appsink
    return (
        f"rtspsrc location={source_url} latency=0 ! "
        "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false drop=0"
    )

class VideoSource:
    def __init__(self, source, use_gst_if_rtsp=True, open_timeout=5.0):
        self.source = source
        self.use_gst = use_gst_if_rtsp and str(source).lower().startswith('rtsp://')
        self.open_timeout = open_timeout
        self.cap = None
        self.open()

    def open(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        try:
            if self.use_gst:
                pipeline = get_gst_pipeline(self.source)
                self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            else:
                self.cap = cv2.VideoCapture(self.source)
            t0 = time.time()
            while not self.cap.isOpened() and (time.time() - t0) < self.open_timeout:
                time.sleep(0.1)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {self.source}")
        except Exception as e:
            self.cap = None
            raise

    def read(self):
        if not self.cap:
            try:
                self.open()
            except Exception:
                return False, None
        try:
            grabbed = self.cap.grab()
            if not grabbed:
                return False, None
            ret, frame = self.cap.retrieve()
            if not ret:
                return False, None
            return True, frame
        except Exception:
            return False, None

    def release(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
