"""
AI Driver Drowsiness Detection & Alert System
Main Desktop GUI Application
"""

import sys
import os
import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

# Local project modules
from config import (
    APP_NAME,
    APP_VERSION,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    DEFAULT_CLOSED_THRESHOLD,
    MIN_CLOSED_THRESHOLD,
    MAX_CLOSED_THRESHOLD,
    COLORS
)
from camera import Camera
from drowsiness_detector import DrowsinessDetector, DetectionResult
from alarm import alarm_manager
from serial_communication import serial_manager

# Set CustomTkinter theme and appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DrowsinessDetectionApp(ctk.CTk):
    """
    Primary GUI Application for AI Driver Drowsiness Detection & Alert System.
    Provides a real-time dark automotive cockpit dashboard.
    """

    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(1100, 720)
        self.configure(fg_color=COLORS["bg_dark"])

        # Core Subsystems
        self.camera = Camera(camera_index=0)
        self.detector = DrowsinessDetector(closed_threshold=DEFAULT_CLOSED_THRESHOLD)
        self.is_monitoring = False
        self.alarm_enabled = True

        # Video dimensions for GUI display
        self.display_width = 640
        self.display_height = 480
        self.current_tk_image: Optional[ImageTk.PhotoImage] = None

        # Flash animation state for alert banner
        self.alert_flash_state = False

        # Build UI Components
        self._create_header()
        self._create_main_layout()
        self._create_footer()

        # Connect Serial Callback for live TX telemetry
        serial_manager.register_tx_callback(self._on_serial_tx)

        # Initialize Video Canvas with Sleek Offline Placeholder
        self._display_placeholder()

        # Handle window close cleanly
        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Scheduled update loop for UI clock and alerts
        self._schedule_gui_ticks()

    # =========================================================================
    # UI CONSTRUCTION
    # =========================================================================

    def _create_header(self):
        """Top header bar containing title, system status, and live clock."""
        header_frame = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_card"],
            corner_radius=0,
            height=65,
            border_width=1,
            border_color=COLORS["border"]
        )
        header_frame.pack(fill="x", side="top", padx=0, pady=0)
        header_frame.pack_propagate(False)

        # Title container
        title_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        title_box.pack(side="left", padx=20, pady=10)

        title_lbl = ctk.CTkLabel(
            title_box,
            text="🛡️  AI DRIVER DROWSINESS DETECTION & ALERT SYSTEM",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        title_lbl.pack(anchor="w")

        subtitle_lbl = ctk.CTkLabel(
            title_box,
            text="Real-time Computer Vision & Embedded Safety Telemetry",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=COLORS["text_secondary"]
        )
        subtitle_lbl.pack(anchor="w")

        # Right header telemetry
        right_box = ctk.CTkFrame(header_frame, fg_color="transparent")
        right_box.pack(side="right", padx=20, pady=12)

        self.sys_status_badge = ctk.CTkLabel(
            right_box,
            text="● SYSTEM STANDBY",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=COLORS["bg_card_secondary"],
            text_color=COLORS["status_idle"],
            corner_radius=8,
            padx=14,
            pady=6
        )
        self.sys_status_badge.pack(side="left", padx=10)

        self.clock_lbl = ctk.CTkLabel(
            right_box,
            text="00:00:00",
            font=ctk.CTkFont(family="Consolas", size=14, weight="bold"),
            text_color=COLORS["accent_cyan"]
        )
        self.clock_lbl.pack(side="left", padx=5)

    def _create_main_layout(self):
        """Construct the two-column dashboard layout."""
        # Top Warning Banner (Hidden / Neutral by default)
        self.warning_banner = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_card"],
            corner_radius=8,
            height=45,
            border_width=1,
            border_color=COLORS["border"]
        )
        self.warning_banner.pack(fill="x", padx=18, pady=(12, 6))
        self.warning_banner.pack_propagate(False)

        self.warning_text = ctk.CTkLabel(
            self.warning_banner,
            text="STATUS: SYSTEM READY — START MONITORING TO COMMENCE SURVEILLANCE",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLORS["text_secondary"]
        )
        self.warning_text.pack(expand=True)

        # Content Splitter (Left: Video, Right: Dashboard Controls)
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=18, pady=6)
        content_frame.grid_columnconfigure(0, weight=6)
        content_frame.grid_columnconfigure(1, weight=4)
        content_frame.grid_rowconfigure(0, weight=1)

        # ---------------------------------------------------------------------
        # LEFT COLUMN: Live Camera Feed & Primary Action Controls
        # ---------------------------------------------------------------------
        left_panel = ctk.CTkFrame(
            content_frame,
            fg_color=COLORS["bg_card"],
            corner_radius=12,
            border_width=1,
            border_color=COLORS["border"]
        )
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        # Panel Header
        feed_header = ctk.CTkFrame(left_panel, fg_color="transparent", height=35)
        feed_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        
        feed_title = ctk.CTkLabel(
            feed_header,
            text="LIVE CAMERA STREAM",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        feed_title.pack(side="left")

        self.fps_indicator = ctk.CTkLabel(
            feed_header,
            text="FPS: 0.0",
            font=ctk.CTkFont(family="Consolas", size=12),
            text_color=COLORS["accent_cyan"]
        )
        self.fps_indicator.pack(side="right")

        # Camera Display Canvas
        self.camera_canvas = tk.Canvas(
            left_panel,
            width=self.display_width,
            height=self.display_height,
            bg=COLORS["bg_dark"],
            highlightthickness=0
        )
        self.camera_canvas.grid(row=1, column=0, padx=16, pady=4, sticky="nsew")

        # Primary Buttons Row
        btn_bar = ctk.CTkFrame(left_panel, fg_color="transparent")
        btn_bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(10, 16))

        self.btn_start = ctk.CTkButton(
            btn_bar,
            text="▶  START MONITORING",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=COLORS["btn_start"],
            hover_color=COLORS["btn_start_hover"],
            height=40,
            command=self.start_monitoring
        )
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_stop = ctk.CTkButton(
            btn_bar,
            text="⏹  STOP MONITORING",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=COLORS["btn_stop"],
            hover_color=COLORS["btn_stop_hover"],
            height=40,
            state="disabled",
            command=self.stop_monitoring
        )
        self.btn_stop.pack(side="left", fill="x", expand=True, padx=6)

        self.btn_reset = ctk.CTkButton(
            btn_bar,
            text="↺  RESET",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color=COLORS["btn_reset"],
            hover_color=COLORS["btn_reset_hover"],
            width=100,
            height=40,
            command=self.reset_system
        )
        self.btn_reset.pack(side="left", padx=6)

        self.btn_exit = ctk.CTkButton(
            btn_bar,
            text="✕  EXIT",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            fg_color="#334155",
            hover_color="#1e293b",
            width=90,
            height=40,
            command=self.on_exit
        )
        self.btn_exit.pack(side="left", padx=(6, 0))

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: Driver Telemetry, ESP32 Hub & Settings
        # ---------------------------------------------------------------------
        right_panel = ctk.CTkScrollableFrame(
            content_frame,
            fg_color="transparent",
            label_text="MONITORING TELEMETRY & CONTROLS",
            label_font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            label_text_color=COLORS["text_secondary"]
        )
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=0)

        # 1. DRIVER STATUS CARD
        status_card = ctk.CTkFrame(
            right_panel,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        status_card.pack(fill="x", pady=(0, 10), padx=4)

        ctk.CTkLabel(
            status_card,
            text="DRIVER STATUS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=16, pady=(12, 4))

        self.driver_status_badge = ctk.CTkLabel(
            status_card,
            text="AWAKE",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            fg_color=COLORS["status_awake_bg"],
            text_color=COLORS["status_awake"],
            corner_radius=8,
            height=48
        )
        self.driver_status_badge.pack(fill="x", padx=16, pady=(4, 14))

        # 2. EYE STATUS & CLOSURE DURATION CARD
        eye_card = ctk.CTkFrame(
            right_panel,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        eye_card.pack(fill="x", pady=6, padx=4)

        eye_top_row = ctk.CTkFrame(eye_card, fg_color="transparent")
        eye_top_row.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            eye_top_row,
            text="EYE TRACKING STATE",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(side="left")

        self.eye_state_badge = ctk.CTkLabel(
            eye_top_row,
            text="OPEN",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            fg_color=COLORS["status_awake_bg"],
            text_color=COLORS["status_awake"],
            corner_radius=6,
            padx=10,
            pady=2
        )
        self.eye_state_badge.pack(side="right")

        # Continuous closure timer display
        timer_row = ctk.CTkFrame(eye_card, fg_color="transparent")
        timer_row.pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(
            timer_row,
            text="Closure Duration:",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_secondary"]
        ).pack(side="left")

        self.closure_timer_lbl = ctk.CTkLabel(
            timer_row,
            text=f"0.00 s / {self.detector.closed_threshold:.1f} s",
            font=ctk.CTkFont(family="Consolas", size=15, weight="bold"),
            text_color=COLORS["text_primary"]
        )
        self.closure_timer_lbl.pack(side="right")

        # Real-time closure progress bar towards threshold
        self.closure_progress = ctk.CTkProgressBar(
            eye_card,
            height=10,
            corner_radius=5,
            fg_color=COLORS["bg_dark"],
            progress_color=COLORS["status_awake"]
        )
        self.closure_progress.pack(fill="x", padx=16, pady=(6, 14))
        self.closure_progress.set(0.0)

        # 3. METRICS ROW: Drowsiness Events & Face Tracking
        metrics_row = ctk.CTkFrame(right_panel, fg_color="transparent")
        metrics_row.pack(fill="x", pady=6, padx=4)
        metrics_row.grid_columnconfigure((0, 1), weight=1)

        # Events Counter Box
        events_box = ctk.CTkFrame(
            metrics_row,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        events_box.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkLabel(
            events_box,
            text="DROWSY EVENTS",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=12, pady=(10, 2))

        self.events_count_lbl = ctk.CTkLabel(
            events_box,
            text="0",
            font=ctk.CTkFont(family="Consolas", size=26, weight="bold"),
            text_color=COLORS["accent_cyan"]
        )
        self.events_count_lbl.pack(anchor="w", padx=12, pady=(0, 10))

        # Face Presence Box
        face_box = ctk.CTkFrame(
            metrics_row,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        face_box.grid(row=0, column=1, sticky="ew", padx=(4, 0))

        ctk.CTkLabel(
            face_box,
            text="FACE TRACKING",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=12, pady=(10, 2))

        self.face_status_lbl = ctk.CTkLabel(
            face_box,
            text="NO FACE",
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
            text_color=COLORS["status_caution"]
        )
        self.face_status_lbl.pack(anchor="w", padx=12, pady=(4, 10))

        # 4. HARDWARE & ESP32 SERIAL SECTION
        serial_card = ctk.CTkFrame(
            right_panel,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        serial_card.pack(fill="x", pady=6, padx=4)

        serial_header = ctk.CTkFrame(serial_card, fg_color="transparent")
        serial_header.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            serial_header,
            text="ESP32 SERIAL HARDWARE",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(side="left")

        self.serial_badge = ctk.CTkLabel(
            serial_header,
            text="● STANDALONE",
            font=ctk.CTkFont(family="Segoe UI", size=10, weight="bold"),
            text_color=COLORS["status_idle"]
        )
        self.serial_badge.pack(side="right")

        # Serial controls row
        serial_ctl_row = ctk.CTkFrame(serial_card, fg_color="transparent")
        serial_ctl_row.pack(fill="x", padx=16, pady=4)

        # Port Selector Dropdown
        ports = serial_manager.get_available_ports()
        port_values = ports if ports else ["No Ports Found"]
        self.port_dropdown = ctk.CTkComboBox(
            serial_ctl_row,
            values=port_values,
            width=130,
            height=30
        )
        if ports:
            self.port_dropdown.set(ports[0])
        else:
            self.port_dropdown.set("No Ports")
        self.port_dropdown.pack(side="left", padx=(0, 6))

        self.btn_serial_connect = ctk.CTkButton(
            serial_ctl_row,
            text="CONNECT",
            width=80,
            height=30,
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            command=self.toggle_serial_connection
        )
        self.btn_serial_connect.pack(side="left", padx=4)

        btn_refresh_ports = ctk.CTkButton(
            serial_ctl_row,
            text="🔄",
            width=32,
            height=30,
            command=self.refresh_com_ports
        )
        btn_refresh_ports.pack(side="left", padx=4)

        # Serial telemetry monitor line
        self.serial_tx_lbl = ctk.CTkLabel(
            serial_card,
            text="Last TX: NONE (ESP32 Ready)",
            font=ctk.CTkFont(family="Consolas", size=10),
            text_color=COLORS["text_secondary"]
        )
        self.serial_tx_lbl.pack(anchor="w", padx=16, pady=(4, 12))

        # 5. DETECTION SETTINGS & THRESHOLD SLIDER
        settings_card = ctk.CTkFrame(
            right_panel,
            fg_color=COLORS["bg_card"],
            corner_radius=10,
            border_width=1,
            border_color=COLORS["border"]
        )
        settings_card.pack(fill="x", pady=6, padx=4)

        ctk.CTkLabel(
            settings_card,
            text="CALIBRATION & SETTINGS",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=COLORS["text_muted"]
        ).pack(anchor="w", padx=16, pady=(12, 6))

        # Threshold slider
        thresh_info = ctk.CTkFrame(settings_card, fg_color="transparent")
        thresh_info.pack(fill="x", padx=16, pady=(2, 2))
        
        ctk.CTkLabel(
            thresh_info,
            text="Eye Closure Threshold:",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=COLORS["text_secondary"]
        ).pack(side="left")

        self.slider_val_lbl = ctk.CTkLabel(
            thresh_info,
            text=f"{self.detector.closed_threshold:.1f} s",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color=COLORS["accent_cyan"]
        )
        self.slider_val_lbl.pack(side="right")

        self.threshold_slider = ctk.CTkSlider(
            settings_card,
            from_=MIN_CLOSED_THRESHOLD,
            to=MAX_CLOSED_THRESHOLD,
            number_of_steps=40,
            command=self._on_threshold_slider_change
        )
        self.threshold_slider.set(self.detector.closed_threshold)
        self.threshold_slider.pack(fill="x", padx=16, pady=4)

        # Audio and Camera Options
        settings_toggles = ctk.CTkFrame(settings_card, fg_color="transparent")
        settings_toggles.pack(fill="x", padx=16, pady=(8, 14))

        self.alarm_switch = ctk.CTkSwitch(
            settings_toggles,
            text="Laptop Audio Alarm",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            command=self._on_alarm_toggle
        )
        self.alarm_switch.select()
        self.alarm_switch.pack(side="left")

        btn_test_alarm = ctk.CTkButton(
            settings_toggles,
            text="Test Alarm",
            width=90,
            height=28,
            font=ctk.CTkFont(family="Segoe UI", size=11),
            command=self.test_alarm_sound
        )
        btn_test_alarm.pack(side="right")

    def _create_footer(self):
        """Bottom status bar with system information."""
        footer_frame = ctk.CTkFrame(
            self,
            fg_color=COLORS["bg_card"],
            corner_radius=0,
            height=28,
            border_width=1,
            border_color=COLORS["border"]
        )
        footer_frame.pack(fill="x", side="bottom", padx=0, pady=0)
        footer_frame.pack_propagate(False)

        footer_left = ctk.CTkLabel(
            footer_frame,
            text="  OpenCV Vision Engine Active  |  Haar Cascade Tracking  |  DirectShow I/O",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_muted"]
        )
        footer_left.pack(side="left", padx=10)

        self.footer_right = ctk.CTkLabel(
            footer_frame,
            text="ESP32 Mode: Standalone  |  Status: Idle  ",
            font=ctk.CTkFont(family="Segoe UI", size=10),
            text_color=COLORS["text_secondary"]
        )
        self.footer_right.pack(side="right", padx=10)

    # =========================================================================
    # CORE LOGIC & MONITORING LOOP
    # =========================================================================

    def start_monitoring(self):
        """Initiate webcam stream and continuous detection pipeline."""
        if self.is_monitoring:
            return

        success = self.camera.start()
        if not success:
            err = self.camera.error_message or "Could not access the laptop webcam."
            messagebox.showerror(
                "Camera Error",
                f"Failed to start webcam:\n{err}\n\nPlease check camera permissions or select another camera."
            )
            return

        self.is_monitoring = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.sys_status_badge.configure(
            text="● MONITORING ACTIVE",
            fg_color=COLORS["status_awake_bg"],
            text_color=COLORS["status_awake"]
        )
        self.warning_text.configure(
            text="STATUS: MONITORING ACTIVE — DRIVER ATTENTIVENESS UNDER SURVEILLANCE",
            text_color=COLORS["text_secondary"]
        )

        # Kick off frame capture and detection loop
        self._video_loop()

    def stop_monitoring(self):
        """Halt camera feed and detection."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        self.camera.stop()
        alarm_manager.stop_alarm()

        # Send NORMAL to microcontroller upon stopping
        serial_manager.send_driver_status(is_drowsy=False, force=True)

        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.sys_status_badge.configure(
            text="● SYSTEM STANDBY",
            fg_color=COLORS["bg_card_secondary"],
            text_color=COLORS["status_idle"]
        )
        self._set_awake_ui()
        self._display_placeholder()
        self.fps_indicator.configure(text="FPS: 0.0")

    def reset_system(self):
        """Reset drowsiness event counter, timer, and active warnings."""
        self.detector.reset_events()
        alarm_manager.stop_alarm()
        self.events_count_lbl.configure(text="0")
        self.closure_timer_lbl.configure(
            text=f"0.00 s / {self.detector.closed_threshold:.1f} s"
        )
        self.closure_progress.set(0.0)
        self._set_awake_ui()
        serial_manager.send_driver_status(is_drowsy=False, force=True)

    def _video_loop(self):
        """Recursive video polling loop running at ~30 FPS on the Tkinter main thread."""
        if not self.is_monitoring:
            return

        ret, frame = self.camera.read()

        if ret and frame is not None:
            # Process video frame with drowsiness detector
            result = self.detector.process_frame(frame)

            # Update GUI dashboard with detection telemetry
            self._update_telemetry(result)

            # Convert BGR frame to RGB for Tkinter rendering
            rgb_frame = cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Scale frame proportionally to canvas
            canvas_w = self.camera_canvas.winfo_width()
            canvas_h = self.camera_canvas.winfo_height()
            if canvas_w > 50 and canvas_h > 50:
                render_w, render_h = self._compute_aspect_fit(
                    frame.shape[1], frame.shape[0], canvas_w, canvas_h
                )
            else:
                render_w, render_h = (self.display_width, self.display_height)

            resized_img = Image.fromarray(rgb_frame).resize(
                (render_w, render_h), Image.Resampling.BILINEAR
            )
            self.current_tk_image = ImageTk.PhotoImage(image=resized_img)

            self.camera_canvas.delete("all")
            self.camera_canvas.create_image(
                canvas_w // 2, canvas_h // 2, image=self.current_tk_image, anchor="center"
            )

            # Update Camera FPS display
            self.fps_indicator.configure(text=f"FPS: {self.camera.fps:.1f}")

        # Schedule next iteration (aiming for ~30ms interval)
        if self.is_monitoring:
            self.after(25, self._video_loop)

    def _update_telemetry(self, res: DetectionResult):
        """Update dashboard badges, timers, warning banners, audio, and serial communication."""
        # 1. Closure Duration & Progress Bar
        self.closure_timer_lbl.configure(
            text=f"{res.closed_duration:.2f} s / {res.closed_threshold:.1f} s"
        )
        progress = min(1.0, res.closed_duration / max(0.1, res.closed_threshold))
        self.closure_progress.set(progress)

        # 2. Eye State Badge
        if res.eye_status == "OPEN":
            self.eye_state_badge.configure(
                text="OPEN",
                fg_color=COLORS["status_awake_bg"],
                text_color=COLORS["status_awake"]
            )
            self.closure_progress.configure(progress_color=COLORS["status_awake"])
        elif res.eye_status == "CLOSED":
            self.eye_state_badge.configure(
                text="CLOSED",
                fg_color=COLORS["status_drowsy_bg"],
                text_color=COLORS["status_drowsy"]
            )
            # Tint progress bar amber then red as threshold nears
            bar_color = COLORS["status_drowsy"] if progress > 0.7 else COLORS["status_caution"]
            self.closure_progress.configure(progress_color=bar_color)
        else:
            self.eye_state_badge.configure(
                text="SEARCHING",
                fg_color=COLORS["status_caution_bg"],
                text_color=COLORS["status_caution"]
            )

        # 3. Face Presence Indicator
        if res.face_detected:
            self.face_status_lbl.configure(
                text="TRACKED",
                text_color=COLORS["status_awake"]
            )
        else:
            self.face_status_lbl.configure(
                text="NO FACE",
                text_color=COLORS["status_caution"]
            )

        # 4. Drowsiness Event Counter
        self.events_count_lbl.configure(text=str(res.event_count))

        # 5. Drowsy State Transition & Hardware / Audio Triggers
        if res.is_drowsy:
            self._set_drowsy_ui()
            # Trigger Laptop Alarm
            if self.alarm_enabled:
                alarm_manager.start_alarm()
            # Transmit to ESP32 Serial
            serial_manager.send_driver_status(is_drowsy=True)
        else:
            self._set_awake_ui()
            # Silence Laptop Alarm
            alarm_manager.stop_alarm()
            # Transmit to ESP32 Serial
            serial_manager.send_driver_status(is_drowsy=False)

    def _set_drowsy_ui(self):
        """Update GUI to high-urgency danger warning state."""
        self.driver_status_badge.configure(
            text="⚠  DROWSY DETECTED",
            fg_color=COLORS["status_drowsy_bg"],
            text_color=COLORS["status_drowsy"]
        )

        # Flashing alert banner
        banner_bg = COLORS["status_drowsy_bg"] if self.alert_flash_state else "#991b1b"
        self.warning_banner.configure(fg_color=banner_bg, border_color=COLORS["status_drowsy"])
        self.warning_text.configure(
            text="🚨  WARNING: DROWSINESS DETECTED! WAKE UP! TAKE A BREAK!  🚨",
            text_color="#ffffff"
        )

    def _set_awake_ui(self):
        """Update GUI to safe, normal awake state."""
        self.driver_status_badge.configure(
            text="AWAKE",
            fg_color=COLORS["status_awake_bg"],
            text_color=COLORS["status_awake"]
        )
        if self.is_monitoring:
            self.warning_banner.configure(
                fg_color=COLORS["bg_card"],
                border_color=COLORS["border"]
            )
            self.warning_text.configure(
                text="STATUS: NORMAL — DRIVER AWAKE AND ATTENTIVE",
                text_color=COLORS["status_awake"]
            )

    # =========================================================================
    # HELPERS, CALIBRATION & EVENTS
    # =========================================================================

    def _compute_aspect_fit(self, img_w: int, img_h: int, box_w: int, box_h: int):
        """Calculate scaled dimensions preserving original camera aspect ratio."""
        scale = min(box_w / img_w, box_h / img_h)
        return int(img_w * scale), int(img_h * scale)

    def _display_placeholder(self):
        """Render stylish standby card on video canvas."""
        placeholder_bgr = Camera.create_placeholder_frame(
            self.display_width, self.display_height,
            message="MONITORING OFFLINE",
            sub_message="Click 'START MONITORING' to activate camera feed"
        )
        rgb = cv2.cvtColor(placeholder_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        canvas_w = max(50, self.camera_canvas.winfo_width())
        canvas_h = max(50, self.camera_canvas.winfo_height())
        fit_w, fit_h = self._compute_aspect_fit(
            self.display_width, self.display_height, canvas_w, canvas_h
        )
        pil_img = pil_img.resize((fit_w, fit_h), Image.Resampling.BILINEAR)

        self.current_tk_image = ImageTk.PhotoImage(image=pil_img)
        self.camera_canvas.delete("all")
        self.camera_canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self.current_tk_image, anchor="center"
        )

    def _on_threshold_slider_change(self, val: float):
        """Handle eye closure threshold calibration slider."""
        rounded = round(val, 1)
        self.detector.set_threshold(rounded)
        self.slider_val_lbl.configure(text=f"{rounded:.1f} s")
        self.closure_timer_lbl.configure(
            text=f"{self.detector.continuous_closed_duration:.2f} s / {rounded:.1f} s"
        )

    def _on_alarm_toggle(self):
        """Enable or disable audio siren alert."""
        self.alarm_enabled = self.alarm_switch.get() == 1
        alarm_manager.set_enabled(self.alarm_enabled)

    def test_alarm_sound(self):
        """Test speaker output with a 1-second pulse."""
        alarm_manager.test_alarm(duration_s=1.0)

    def refresh_com_ports(self):
        """Rescan available USB serial ports on laptop."""
        ports = serial_manager.get_available_ports()
        if ports:
            self.port_dropdown.configure(values=ports)
            self.port_dropdown.set(ports[0])
        else:
            self.port_dropdown.configure(values=["No Ports Found"])
            self.port_dropdown.set("No Ports Found")

    def toggle_serial_connection(self):
        """Connect or disconnect ESP32 USB Serial port."""
        if serial_manager.is_connected():
            serial_manager.disconnect()
            self.btn_serial_connect.configure(text="CONNECT", fg_color=COLORS["accent_blue"])
            self.serial_badge.configure(text="● STANDALONE", text_color=COLORS["status_idle"])
            self.footer_right.configure(text="ESP32 Mode: Standalone  |  Status: Disconnected")
        else:
            selected_port = self.port_dropdown.get()
            if not selected_port or "No Ports" in selected_port:
                messagebox.showwarning(
                    "No Port Selected",
                    "No USB Serial COM port detected.\n\nThe application will continue operating in Standalone Mode."
                )
                return

            success, msg = serial_manager.connect(selected_port)
            if success:
                self.btn_serial_connect.configure(text="DISCONNECT", fg_color=COLORS["btn_stop"])
                self.serial_badge.configure(text=f"● {selected_port}", text_color=COLORS["status_awake"])
                self.footer_right.configure(text=f"ESP32 Mode: Connected ({selected_port})  |  Status: Live")
            else:
                messagebox.showerror("Serial Connection Error", msg)

    def _on_serial_tx(self, cmd: str, status_desc: str):
        """Callback to display sent serial command on GUI dashboard."""
        self.serial_tx_lbl.configure(
            text=f"TX >> [{cmd}]  ({status_desc})"
        )

    def _schedule_gui_ticks(self):
        """Periodic clock updates and alert flashing animation (runs every 500ms)."""
        # Update Digital Clock
        current_time = time.strftime("%H:%M:%S")
        self.clock_lbl.configure(text=current_time)

        # Toggle flash state for danger alerts
        self.alert_flash_state = not self.alert_flash_state

        # Repeat tick in 500ms
        self.after(500, self._schedule_gui_ticks)

    def on_exit(self):
        """Gracefully release camera, audio, and serial hardware upon shutdown."""
        self.stop_monitoring()
        serial_manager.disconnect()
        self.destroy()
        sys.exit(0)


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """Main launcher."""
    app = DrowsinessDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
