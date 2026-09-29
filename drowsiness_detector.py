"""
AI Driver Drowsiness Detection & Alert System
Computer Vision & Drowsiness Detection Engine
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np

from config import (
    FACE_CASCADE_PATH,
    EYE_CASCADE_PATH,
    EYE_GLASSES_CASCADE_PATH,
    DEFAULT_CLOSED_THRESHOLD,
    FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBORS,
    EYE_SCALE_FACTOR,
    EYE_MIN_NEIGHBORS,
    EYE_ROI_Y_START_RATIO,
    EYE_ROI_Y_END_RATIO,
    EYE_ROI_X_MARGIN_RATIO,
    ensure_cascades_exist
)


@dataclass
class DetectionResult:
    """Telemetry data structure returned by detector for each video frame."""
    face_detected: bool
    face_box: Optional[Tuple[int, int, int, int]]
    eye_status: str             # "OPEN", "CLOSED", or "SEARCHING"
    driver_status: str          # "AWAKE" or "DROWSY"
    is_drowsy: bool             # True if alert should sound
    closed_duration: float      # Elapsed continuous closure time in seconds
    closed_threshold: float     # Active threshold setting in seconds
    event_count: int            # Total lifetime drowsiness episodes detected
    eyes_detected_count: int    # Number of open eyes detected in current frame
    annotated_frame: np.ndarray # Video frame with tech HUD overlays


class DrowsinessDetector:
    """
    OpenCV-based Driver Drowsiness Detector.
    Tracks face presence, monitors eye open/close state, and measures continuous eye closure.
    """

    def __init__(self, closed_threshold: float = DEFAULT_CLOSED_THRESHOLD):
        self.closed_threshold = closed_threshold
        self.drowsiness_events = 0
        
        # State tracking
        self.is_drowsy = False
        self.closed_start_time: Optional[float] = None
        self.continuous_closed_duration: float = 0.0
        self.consecutive_closed_frames = 0
        self.consecutive_open_frames = 0
        self.last_face_time: float = 0.0

        # Flashing alert banner animation counter
        self._flash_counter = 0

        # Load cascade classifiers
        ensure_cascades_exist()
        self.face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
        self.eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)
        self.eye_glasses_cascade = cv2.CascadeClassifier(EYE_GLASSES_CASCADE_PATH)

        if self.face_cascade.empty():
            print(f"[Detector] Error: Face cascade failed to load from {FACE_CASCADE_PATH}")
        if self.eye_cascade.empty():
            print(f"[Detector] Error: Eye cascade failed to load from {EYE_CASCADE_PATH}")

    def set_threshold(self, threshold: float):
        """Update eye closure threshold in seconds."""
        self.closed_threshold = max(0.5, float(threshold))

    def reset_events(self):
        """Reset drowsiness event counter and active timer."""
        self.drowsiness_events = 0
        self.continuous_closed_duration = 0.0
        self.closed_start_time = None
        self.is_drowsy = False
        self.consecutive_closed_frames = 0
        self.consecutive_open_frames = 0

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Process a single BGR video frame to detect face and eye closure state.
        Returns a comprehensive DetectionResult object.
        """
        if frame is None or frame.size == 0:
            return DetectionResult(
                face_detected=False,
                face_box=None,
                eye_status="SEARCHING",
                driver_status="AWAKE",
                is_drowsy=False,
                closed_duration=0.0,
                closed_threshold=self.closed_threshold,
                event_count=self.drowsiness_events,
                eyes_detected_count=0,
                annotated_frame=frame
            )

        annotated = frame.copy()
        h, w = frame.shape[:2]
        current_time = time.time()
        self._flash_counter += 1

        # Preprocessing: convert to grayscale and normalize contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        # 1. Face Detection
        faces = self.face_cascade.detectMultiScale(
            equalized,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(int(w * 0.2), int(h * 0.2))
        )

        face_detected = len(faces) > 0
        largest_face = None
        eyes_found: List[Tuple[int, int, int, int]] = []
        eye_status = "SEARCHING"

        if face_detected:
            self.last_face_time = current_time
            # Select the largest face (closest to camera)
            largest_face = max(faces, key=lambda b: b[2] * b[3])
            fx, fy, fw, fh = largest_face

            # 2. Extract Eye Region of Interest (ROI)
            roi_y1 = max(0, fy + int(fh * EYE_ROI_Y_START_RATIO))
            roi_y2 = min(h, fy + int(fh * EYE_ROI_Y_END_RATIO))
            roi_x1 = max(0, fx + int(fw * EYE_ROI_X_MARGIN_RATIO))
            roi_x2 = min(w, fx + int(fw * (1.0 - EYE_ROI_X_MARGIN_RATIO)))

            roi_gray = equalized[roi_y1:roi_y2, roi_x1:roi_x2]

            # 3. Eye Detection inside ROI
            if roi_gray.size > 0:
                raw_eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=EYE_SCALE_FACTOR,
                    minNeighbors=EYE_MIN_NEIGHBORS,
                    minSize=(int(fw * 0.1), int(fh * 0.08)),
                    maxSize=(int(fw * 0.45), int(fh * 0.35))
                )

                # If no eyes detected, try eyeglasses cascade fallback
                if len(raw_eyes) == 0 and not self.eye_glasses_cascade.empty():
                    raw_eyes = self.eye_glasses_cascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=EYE_SCALE_FACTOR,
                        minNeighbors=EYE_MIN_NEIGHBORS,
                        minSize=(int(fw * 0.1), int(fh * 0.08)),
                        maxSize=(int(fw * 0.45), int(fh * 0.35))
                    )

                # Filter out overlapping or spurious detections
                for (ex, ey, ew, eh) in raw_eyes:
                    eyes_found.append((roi_x1 + ex, roi_y1 + ey, ew, eh))

            # 4. State Determination: Open vs Closed
            if len(eyes_found) > 0:
                # Eyes detected open
                self.consecutive_open_frames += 1
                self.consecutive_closed_frames = 0
                
                # Debounce open condition (requires 1-2 open frames to reset closure)
                if self.consecutive_open_frames >= 1:
                    eye_status = "OPEN"
                    self.closed_start_time = None
                    self.continuous_closed_duration = 0.0
                    self.is_drowsy = False
            else:
                # Face detected, but 0 eyes found in eye ROI -> Eyes are closed
                self.consecutive_closed_frames += 1
                self.consecutive_open_frames = 0

                # Micro-debounce to ignore momentary single-frame blinks (< 80ms)
                if self.consecutive_closed_frames >= 2:
                    eye_status = "CLOSED"
                    if self.closed_start_time is None:
                        self.closed_start_time = current_time

                    self.continuous_closed_duration = current_time - self.closed_start_time

                    # Check if duration crosses threshold
                    if self.continuous_closed_duration >= self.closed_threshold:
                        if not self.is_drowsy:
                            # New episode transition -> increment event count
                            self.drowsiness_events += 1
                            self.is_drowsy = True
                else:
                    eye_status = "OPEN"
        else:
            # No face detected
            eye_status = "SEARCHING"
            # Pause closure timer or slowly decay to avoid false trigger when driver turns head
            self.closed_start_time = None
            self.continuous_closed_duration = 0.0
            self.consecutive_closed_frames = 0
            self.is_drowsy = False

        driver_status = "DROWSY" if self.is_drowsy else "AWAKE"

        # 5. Draw Visual Annotations and Cockpit HUD on frame
        self._render_hud(
            annotated=annotated,
            face_box=largest_face,
            eyes=eyes_found,
            eye_status=eye_status,
            driver_status=driver_status,
            is_drowsy=self.is_drowsy,
            closed_duration=self.continuous_closed_duration,
            face_detected=face_detected
        )

        return DetectionResult(
            face_detected=face_detected,
            face_box=largest_face,
            eye_status=eye_status,
            driver_status=driver_status,
            is_drowsy=self.is_drowsy,
            closed_duration=round(self.continuous_closed_duration, 2),
            closed_threshold=self.closed_threshold,
            event_count=self.drowsiness_events,
            eyes_detected_count=len(eyes_found),
            annotated_frame=annotated
        )

    def _render_hud(
        self,
        annotated: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]],
        eyes: List[Tuple[int, int, int, int]],
        eye_status: str,
        driver_status: str,
        is_drowsy: bool,
        closed_duration: float,
        face_detected: bool
    ):
        """Draw aesthetic cockpit overlay, tech brackets, and warning banner."""
        h, w = annotated.shape[:2]

        # Colors (BGR)
        color_green = (129, 215, 16)      # Emerald Awake
        color_red = (45, 45, 235)         # Crimson Drowsy
        color_amber = (11, 158, 245)      # Amber Caution
        color_cyan = (240, 200, 30)       # Cyan Eye Tracker
        color_card_bg = (20, 24, 34)

        theme_color = color_red if is_drowsy else (color_green if face_detected else color_amber)

        # A. Draw Face Bounding Box with sleek corner brackets
        if face_box is not None:
            fx, fy, fw, fh = face_box
            
            # Subtle bounding box
            cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), theme_color, 1)

            # High-tech corner accents
            corner_len = int(min(fw, fh) * 0.18)
            t = 3  # corner thickness
            # Top-Left
            cv2.line(annotated, (fx, fy), (fx + corner_len, fy), theme_color, t)
            cv2.line(annotated, (fx, fy), (fx, fy + corner_len), theme_color, t)
            # Top-Right
            cv2.line(annotated, (fx + fw, fy), (fx + fw - corner_len, fy), theme_color, t)
            cv2.line(annotated, (fx + fw, fy), (fx + fw, fy + corner_len), theme_color, t)
            # Bottom-Left
            cv2.line(annotated, (fx, fy + fh), (fx + corner_len, fy + fh), theme_color, t)
            cv2.line(annotated, (fx, fy + fh), (fx, fy + fh - corner_len), theme_color, t)
            # Bottom-Right
            cv2.line(annotated, (fx + fw, fy + fh), (fx + fw - corner_len, fy + fh), theme_color, t)
            cv2.line(annotated, (fx + fw, fy + fh), (fx + fw, fy + fh - corner_len), theme_color, t)

            # Draw Eye Bounding Boxes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(annotated, (ex, ey), (ex + ew, ey + eh), color_cyan, 2)
                cv2.circle(annotated, (ex + ew // 2, ey + eh // 2), 2, color_cyan, -1)

            # Label above face
            label = "DRIVER [AWAKE]" if not is_drowsy else "DRIVER [DROWSY!]"
            cv2.putText(annotated, label, (fx, max(25, fy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, theme_color, 2, cv2.LINE_AA)

        # B. Drowsiness Warning Banner (Flashing Large Alert)
        if is_drowsy:
            # Flashing effect based on cycle
            is_flash_bright = (self._flash_counter % 12) < 7
            banner_bg = (30, 20, 180) if is_flash_bright else (20, 15, 120)
            
            # Semi-transparent red banner at top of camera frame
            banner_height = 68
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_height), banner_bg, -1)
            cv2.rectangle(overlay, (0, banner_height - 3), (w, banner_height), (0, 0, 255), 3)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Alert Text
            alert_text = "! WARNING: DROWSINESS DETECTED !"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 0.85
            text_size = cv2.getTextSize(alert_text, font, scale, 2)[0]
            tx = (w - text_size[0]) // 2
            ty = 42
            cv2.putText(annotated, alert_text, (tx, ty), font, scale, (255, 255, 255), 2, cv2.LINE_AA)

        # C. Lower Telemetry Mini-Bar
        bar_h = 32
        bar_y = h - bar_h
        cv2.rectangle(annotated, (0, bar_y), (w, h), (15, 18, 25), -1)
        cv2.line(annotated, (0, bar_y), (w, bar_y), (35, 45, 60), 1)

        # Telemetry pills
        status_text = f"EYES: {eye_status}"
        cv2.putText(annotated, status_text, (15, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 215, 230), 1, cv2.LINE_AA)

        timer_text = f"CLOSED: {closed_duration:.1f}s / {self.closed_threshold:.1f}s"
        timer_color = color_red if closed_duration >= self.closed_threshold else (
            color_amber if closed_duration > 0.5 else (180, 195, 210)
        )
        cv2.putText(annotated, timer_text, (180, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, timer_color, 1, cv2.LINE_AA)

        face_status = "FACE: TRACKED" if face_detected else "FACE: NONE"
        cv2.putText(annotated, face_status, (w - 140, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (color_green if face_detected else color_amber), 1, cv2.LINE_AA)
