"""Video sources: recorded files or live RTSP/USB cameras.

LiveSource reads the stream in a background thread and always keeps only the
LATEST frame, so a slow analysis loop never builds up delay. If the stream
drops, it reconnects automatically.
"""

import os
import threading
import time

import cv2

# lower RTSP latency / use TCP (more reliable than UDP over Wi-Fi / VPN)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")


def open_capture(src):
    """src: file path, rtsp/http URL, or webcam index (int or digit string)."""
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        return cv2.VideoCapture(int(src))
    if isinstance(src, str) and src.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    return cv2.VideoCapture(str(src))


class LiveSource:
    def __init__(self, src, name=""):
        self.src, self.name = src, name
        self.frame, self.frame_no = None, 0
        self.connected = False
        self.lock = threading.Lock()
        self.stop_flag = False
        self.w = self.h = 0
        self.fps = 25.0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_flag:
            cap = open_capture(self.src)
            if not cap.isOpened():
                print(f"[{self.name}] cannot open stream, retrying in 5 s ...")
                time.sleep(5)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.connected = True
            print(f"[{self.name}] connected")
            fails = 0
            # a FILE used as a test "live" source: play it at its real speed
            is_file = not (isinstance(self.src, int) or str(self.src).isdigit() or
                           str(self.src).startswith(("rtsp://", "rtsps://", "http")))
            next_t = time.time()
            while not self.stop_flag:
                if is_file:
                    next_t += 1.0 / self.fps
                    time.sleep(max(0.0, next_t - time.time()))
                ok, f = cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        break
                    time.sleep(0.02)
                    continue
                fails = 0
                with self.lock:
                    self.frame = f
                    self.frame_no += 1
                    self.h, self.w = f.shape[:2]
            cap.release()
            self.connected = False
            if not self.stop_flag:
                print(f"[{self.name}] stream lost, reconnecting ...")
                time.sleep(2)

    def latest(self, have_no=-1):
        """(frame_no, frame) - frame is None if nothing newer than have_no."""
        with self.lock:
            if self.frame is None or self.frame_no == have_no:
                return self.frame_no, None
            return self.frame_no, self.frame.copy()

    def wait_first_frame(self, timeout=20):
        t0 = time.time()
        while self.frame is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        return self.frame is not None

    def release(self):
        self.stop_flag = True
