import cv2
import numpy as np
import sys
import threading
import time
import os
from collections import deque

class Config:
    HAAR_FACE      = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    CAMERA_INDEX   = 0
    CAMERA_WIDTH   = 1280
    CAMERA_HEIGHT  = 720

    RED_LOW1       = np.array([0,   120,  70])
    RED_HIGH1      = np.array([10,  255, 255])
    RED_LOW2       = np.array([170, 120,  70])
    RED_HIGH2      = np.array([180, 255, 255])
    MIN_RED_AREA   = 500

    CROSS_COLOR    = (0, 0, 255)
    RECT_COLOR     = (0, 165, 255)
    FACE_COLOR     = (0, 220, 0)
    CROSS_THICK    = 3
    CROSS_LEN      = 30

    FPS_SAMPLES    = 30

    WARNING_IMG    = "warning.jpg"
    LOCKSCREEN_IMG = "lockscreen.jpg"

    STATE_CONFIRM_FRAMES = 10


# Lockscreeny

class OverlayLoader:
    def __init__(self, width: int, height: int):
        self.width  = width
        self.height = height
        self.warning    = self._load(Config.WARNING_IMG,    "warning")
        self.lockscreen = self._load(Config.LOCKSCREEN_IMG, "lockscreen")

    def _load(self, path: str, name: str) -> np.ndarray:
        if not os.path.exists(path):
            return self._fallback(name)
        img = cv2.imread(path)
        if img is None:
            return self._fallback(name)
        return cv2.resize(img, (self.width, self.height))

    def _fallback(self, name: str) -> np.ndarray:
        fb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return fb


# Obraz z kamery

class CameraReader:
    def __init__(self):
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        if not self.cap.isOpened():
            sys.exit(1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._frame = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="CameraReader")
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


# Detekce obličejů

class FaceDetector:

    def __init__(self):
        self._cascade = cv2.CascadeClassifier(Config.HAAR_FACE)
        if self._cascade.empty():
            sys.exit(1)

        self._faces       = None
        self._input_frame = None
        self._lock_in     = threading.Lock()
        self._lock_out    = threading.Lock()
        self._event       = threading.Event()
        self._stop        = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="FaceDetector")
        self._thread.start()

    def submit(self, frame: np.ndarray):
        with self._lock_in:
            self._input_frame = frame
        self._event.set()

    def get_faces(self):
        with self._lock_out:
            return list(self._faces) if self._faces is not None else None

    def _run(self):
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.05):
                continue
            self._event.clear()
            with self._lock_in:
                frame = self._input_frame
            if frame is None:
                continue
            gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray   = cv2.equalizeHist(gray)
            result = self._cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = list(result) if len(result) > 0 else []
            with self._lock_out:
                self._faces = faces

    def stop(self):
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=2)


# 

class StateSmoother:

    def __init__(self, confirm: int = Config.STATE_CONFIRM_FRAMES):
        self._confirm       = confirm
        self._stable_state  = 0
        self._candidate     = 0
        self._streak        = 0

    def _bucket(self, count: int) -> int:
        if count == 0:   return 0
        if count == 1:   return 1
        return 2

    def update(self, count: int) -> int:
        bucket = self._bucket(count)
        if bucket == self._candidate:
            self._streak += 1
        else:
            self._candidate = bucket
            self._streak    = 1

        if self._streak >= self._confirm:
            self._stable_state = self._candidate

        return self._stable_state

# Červený objekt

class RedObjectDetector:
    def __init__(self):
        self._kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self._center      = None
        self._bbox        = None
        self._input_frame = None
        self._lock_in     = threading.Lock()
        self._lock_out    = threading.Lock()
        self._event       = threading.Event()
        self._stop        = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="RedDetector")
        self._thread.start()

    def submit(self, frame: np.ndarray):
        with self._lock_in:
            self._input_frame = frame
        self._event.set()

    def get_result(self):
        with self._lock_out:
            return self._center, self._bbox

    def _detect(self, frame: np.ndarray):
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, Config.RED_LOW1, Config.RED_HIGH1)
        mask2 = cv2.inRange(hsv, Config.RED_LOW2, Config.RED_HIGH2)
        mask  = cv2.bitwise_or(mask1, mask2)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < Config.MIN_RED_AREA:
            return None, None
        x, y, w, h = cv2.boundingRect(largest)
        return (x + w // 2, y + h // 2), (x, y, w, h)

    def _run(self):
        while not self._stop.is_set():
            if not self._event.wait(timeout=0.05):
                continue
            self._event.clear()
            with self._lock_in:
                frame = self._input_frame
            if frame is None:
                continue
            center, bbox = self._detect(frame)
            with self._lock_out:
                self._center = center
                self._bbox   = bbox

    def stop(self):
        self._stop.set()
        self._event.set()
        self._thread.join(timeout=2)


# FPS

class FPSCounter:
    def __init__(self, samples: int = Config.FPS_SAMPLES):
        self._times: deque = deque(maxlen=samples)

    def tick(self):
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class Renderer:
    @staticmethod
    def draw_cross(frame: np.ndarray, center: tuple):
        cx, cy = center
        h, w = frame.shape[:2]
        x1 = max(cx - Config.CROSS_LEN, 0);  x2 = min(cx + Config.CROSS_LEN, w - 1)
        y1 = max(cy - Config.CROSS_LEN, 0);  y2 = min(cy + Config.CROSS_LEN, h - 1)
        cv2.line(frame, (x1, cy), (x2, cy), Config.CROSS_COLOR, Config.CROSS_THICK)
        cv2.line(frame, (cx, y1), (cx, y2), Config.CROSS_COLOR, Config.CROSS_THICK)

    @staticmethod
    def draw_hud(frame: np.ndarray, face_count: int, red_found: bool, fps: float):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (265, 95), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:5.1f}",          (10, 26), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Obliceje: {face_count}",    (10, 52), font, 0.7, (0, 255, 0),   2, cv2.LINE_AA)
        cv2.putText(frame, f"Cerveny: {'ANO' if red_found else 'NE'}", (10, 78), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    @staticmethod
    def draw_fps_overlay(frame: np.ndarray, fps: float):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (160, 38), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


# Main

class App:
    def __init__(self):
        self.camera    = CameraReader()
        w = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.overlays  = OverlayLoader(w, h)
        self.face_det  = FaceDetector()
        self.red_det   = RedObjectDetector()
        self.fps_ctr   = FPSCounter()
        self.renderer  = Renderer()
        self.smoother  = StateSmoother()
        self._frame_n  = 0
        self._det_skip = 2

    def run(self):

        last_faces = []
        last_frame = None

        while True:
            frame = self.camera.read()
            if frame is not None:
                last_frame = frame
            elif last_frame is None:
                time.sleep(0.001)
                continue
            else:
                frame = last_frame

            self._frame_n += 1
            if self._frame_n % self._det_skip == 0:
                self.face_det.submit(frame)
                self.red_det.submit(frame)

            detected = self.face_det.get_faces()
            if detected is not None:
                last_faces = detected

            raw_count    = len(last_faces)
            stable_state = self.smoother.update(raw_count)

            fps = self.fps_ctr.fps
            self.fps_ctr.tick()

            if stable_state == 0:
                display = self.overlays.warning.copy()
                self.renderer.draw_fps_overlay(display, fps)

            elif stable_state == 1:
                display = frame
                for (x, y, fw, fh) in last_faces:
                    cv2.rectangle(display, (x, y), (x + fw, y + fh), Config.FACE_COLOR, 2)
                center, bbox = self.red_det.get_result()
                if center is not None:
                    bx, by, bw, bh = bbox
                    cv2.rectangle(display, (bx, by), (bx + bw, by + bh), Config.RECT_COLOR, 2)
                    self.renderer.draw_cross(display, center)
                self.renderer.draw_hud(display, raw_count, center is not None, fps)

            else:
                display = self.overlays.lockscreen.copy()
                self.renderer.draw_fps_overlay(display, fps)

            cv2.imshow("ICP DETECTION", display)

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

        self._shutdown()

    def _shutdown(self):
        self.camera.stop()
        self.face_det.stop()
        self.red_det.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    App().run()