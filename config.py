"""
AI Driver Drowsiness Detection & Alert System
Configuration & Constants Module
"""

import os
import sys
import urllib.request

# ==========================================
# APPLICATION INFORMATION
# ==========================================
APP_NAME = "AI Driver Drowsiness Detection & Alert System"
APP_VERSION = "1.0.0"
WINDOW_WIDTH = 1260
WINDOW_HEIGHT = 800

# ==========================================
# DIRECTORIES & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADES_DIR = os.path.join(BASE_DIR, "cascades")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(CASCADES_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Haar Cascade File Paths
FACE_CASCADE_PATH = os.path.join(CASCADES_DIR, "haarcascade_frontalface_default.xml")
EYE_CASCADE_PATH = os.path.join(CASCADES_DIR, "haarcascade_eye.xml")
EYE_GLASSES_CASCADE_PATH = os.path.join(CASCADES_DIR, "haarcascade_eye_tree_eyeglasses.xml")

# Sound File Path
ALARM_SOUND_FILE = os.path.join(ASSETS_DIR, "alert.wav")

# ==========================================
# DEFAULT DETECTION SETTINGS
# ==========================================
DEFAULT_CLOSED_THRESHOLD = 2.0  # seconds eyes must remain closed to trigger DROWSY
MIN_CLOSED_THRESHOLD = 1.0     # min threshold configurable via slider
MAX_CLOSED_THRESHOLD = 5.0     # max threshold configurable via slider

DEFAULT_CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# Face and Eye Cascade Parameters
FACE_SCALE_FACTOR = 1.2
FACE_MIN_NEIGHBORS = 5
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 3

# Eye Region of Interest (ROI) ratios relative to face bounding box
EYE_ROI_Y_START_RATIO = 0.22
EYE_ROI_Y_END_RATIO = 0.58
EYE_ROI_X_MARGIN_RATIO = 0.08

# ==========================================
# SERIAL COMMUNICATION (ESP32)
# ==========================================
DEFAULT_BAUD_RATE = 115200
SERIAL_HEARTBEAT_INTERVAL = 1.5  # seconds between repeated status messages
SERIAL_CMD_AWAKE = "NORMAL\n"
SERIAL_CMD_DROWSY = "DROWSY\n"

# ==========================================
# UI COLOR PALETTE (Dark Cockpit Theme)
# ==========================================
COLORS = {
    "bg_dark": "#0a0e17",          # Deep space / cockpit background
    "bg_card": "#131a29",          # Panel / card background
    "bg_card_secondary": "#1a2438",# Secondary card background
    "border": "#24324d",           # Border & separators
    "accent_blue": "#2563eb",      # Tech blue accent
    "accent_cyan": "#06b6d4",      # Secondary cyan accent
    
    # Status Colors
    "status_awake": "#10b981",     # Vibrant Emerald Green (Awake)
    "status_awake_bg": "#064e3b",  # Dark green container
    "status_drowsy": "#ef4444",    # Crimson Red (Drowsy)
    "status_drowsy_bg": "#7f1d1d", # Dark red container
    "status_caution": "#f59e0b",   # Amber / Caution (No face / low confidence)
    "status_caution_bg": "#78350f",# Dark amber container
    "status_idle": "#64748b",      # Slate grey (Monitoring stopped)
    
    # Typography
    "text_primary": "#f8fafc",     # High contrast bright white
    "text_secondary": "#94a3b8",   # Soft secondary grey
    "text_muted": "#64748b",       # Muted label grey
    
    # Button states
    "btn_start": "#059669",
    "btn_start_hover": "#047857",
    "btn_stop": "#dc2626",
    "btn_stop_hover": "#b91c1c",
    "btn_reset": "#475569",
    "btn_reset_hover": "#334155",
}

# ==========================================
# CASCADE DOWNLOAD HELPER
# ==========================================
CASCADE_URLS = {
    FACE_CASCADE_PATH: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
    EYE_CASCADE_PATH: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
    EYE_GLASSES_CASCADE_PATH: "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
}

def ensure_cascades_exist():
    """Ensure all required Haar Cascade XML files are present, downloading if missing."""
    for local_path, url in CASCADE_URLS.items():
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1000:
            try:
                print(f"[Config] Downloading cascade: {os.path.basename(local_path)}...")
                urllib.request.urlretrieve(url, local_path)
                print(f"[Config] Successfully downloaded {os.path.basename(local_path)}")
            except Exception as e:
                print(f"[Config] Warning: Could not download {os.path.basename(local_path)}: {e}")
