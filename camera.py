"""
AI Driver Drowsiness Detection & Alert System
Webcam & Video Capture Module
"""

import time
import threading
from typing import Optional, Tuple
import cv2
import numpy as np
from config import DEFAULT_CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, TARGET_FPS


class Camera:
    """
    Thread-safe webcam capture manager.
    Captures video frames in a background thread to prevent GUI freezing.
    """

    def __init__(self, camera_index: int = DEFAULT_CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame: Optional[np.ndarray] = None
        self.ret = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.fps = 0.0
        self._frame_count = 0
        self._fps_start_time = time.time()
        self.error_message: Optional[str] = None

    def start(self) -> bool:
        """
        Open the camera device and launch the background capture thread.
        Returns True if successful, False otherwise.
        """
        with self._lock:
            if self.is_running:
                return True

            self.error_message = None

            # Open with DirectShow backend on Windows for fast startup
            try:
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(self.camera_index)
            except Exception as e:
                self.error_message = f"Failed to initialize webcam: {e}"
                print(f"[Camera] {self.error_message}")
                return False

            if not self.cap.isOpened():
                self.error_message = f"Webcam #{self.camera_index} could not be opened or is busy."
                print(f"[Camera] {self.error_message}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                return False

            # Configure camera capture properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

            self.is_running = True
            self._fps_start_time = time.time()
            self._frame_count = 0

            # Start worker thread
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            print(f"[Camera] Webcam #{self.camera_index} started successfully.")
            return True

    def _capture_loop(self):
        """Continuously grab frames from webcam inside background thread."""
        frame_interval = 1.0 / TARGET_FPS

        while self.is_running:
            start_loop = time.time()
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        # Mirror horizontally for natural webcam selfie view
                        self.frame = cv2.flip(frame, 1)
                        self.ret = True
                        self._frame_count += 1
                        
                        # Calculate FPS every 1 second
                        elapsed = time.time() - self._fps_start_time
                        if elapsed >= 1.0:
                            self.fps = round(self._frame_count / elapsed, 1)
                            self._frame_count = 0
                            self._fps_start_time = time.time()
                else:
                    with self._lock:
                        self.ret = False
                        self.error_message = "Failed to capture frame from webcam."
            else:
                break

            # Sleep remaining time to throttle frame rate and keep CPU usage low
            delay = frame_interval - (time.time() - start_loop)
            if delay > 0.001:
                time.sleep(delay)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest captured frame (ret, frame) safely without blocking.
        """
        with self._lock:
            if not self.is_running or not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        """Stop capture thread and release webcam resources."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        with self._lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.ret = False
            self.frame = None
            self.fps = 0.0
            print(f"[Camera] Webcam #{self.camera_index} stopped.")

    def set_camera_index(self, index: int) -> bool:
        """Switch to a different camera index."""
        was_running = self.is_running
        self.stop()
        self.camera_index = index
        if was_running:
            return self.start()
        return True

    @staticmethod
    def list_available_cameras(max_check: int = 4) -> list[int]:
        """Discover available webcam indexes on the computer."""
        available = []
        for i in range(max_check):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                available.append(i)
                cap.release()
        if not available:
            available = [0]  # Default to index 0
        return available

    @staticmethod
    def create_placeholder_frame(
        width: int = 640,
        height: int = 480,
        message: str = "CAMERA OFFLINE",
        sub_message: str = "Click 'START MONITORING' to begin"
    ) -> np.ndarray:
        """Generate a sleek, dark aesthetic placeholder frame when camera is idle or disconnected."""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (20, 24, 32)  # Dark cockpit slate (#141820)

        # Draw tech border & grid lines
        cv2.rectangle(img, (15, 15), (width - 15, height - 15), (45, 55, 75), 2)
        cv2.line(img, (width // 2 - 40, height // 2 - 40), (width // 2 + 40, height // 2 - 40), (60, 75, 100), 2)
        cv2.circle(img, (width // 2, height // 2 - 40), 25, (60, 75, 100), 2)
        cv2.circle(img, (width // 2, height // 2 - 40), 8, (60, 75, 100), -1)

        # Main message
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        color = (200, 210, 225)
        text_size = cv2.getTextSize(message, font, font_scale, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2 + 30
        cv2.putText(img, message, (text_x, text_y), font, font_scale, color, 2, cv2.LINE_AA)

        # Subtitle message
        sub_scale = 0.5
        sub_color = (120, 135, 155)
        sub_size = cv2.getTextSize(sub_message, font, sub_scale, 1)[0]
        sub_x = (width - sub_size[0]) // 2
        sub_y = text_y + 35
        cv2.putText(img, sub_message, (sub_x, sub_y), font, sub_scale, sub_color, 1, cv2.LINE_AA)

        return img
