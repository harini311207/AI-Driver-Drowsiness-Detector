# AI Driver Drowsiness Detection & Alert System 🛡️

A modern, high-performance desktop application for real-time driver attentiveness monitoring and drowsiness alerting. Powered by OpenCV computer vision, threaded camera processing, native audio alarms, and an embedded-ready USB serial interface for ESP32 and Arduino microcontrollers.

---

## 📸 Key Features

- **Real-Time Webcam Face & Eye Tracking**: Continuously captures live video from the laptop's built-in webcam at 30 FPS.
- **Accurate Eye Closure Analysis**: Extracts the upper facial Region of Interest (ROI) and analyzes eye presence and closure using OpenCV Haar Cascade models.
- **Continuous Closure Stopwatch**: Accurately tracks continuous eye closure in seconds. If the eyes remain closed past the configured threshold (default **2.0 seconds**), the system triggers a **DROWSY** state.
- **Immediate Recovery to AWAKE**: As soon as the driver opens their eyes, the alarm is silenced instantly, and the system resets back to **AWAKE**.
- **Audible Laptop Alarm**: High-urgency automotive pulsing alert siren generated using Windows native audio APIs (no external audio compilation required).
- **Cockpit Dashboard GUI**: Sleek, modern dark-mode automotive interface built with CustomTkinter:
  - Large live camera viewport with head & eye tracking bounding boxes
  - Flashing full-width warning banner: **"⚠️ DROWSINESS DETECTED! WAKE UP!"**
  - Driver Status Badge (**AWAKE** in Emerald Green / **DROWSY** in Crimson Red)
  - Eye Tracking Status (**OPEN** / **CLOSED** / **SEARCHING**)
  - Real-time continuous closure progress bar and digital timer
  - Total Drowsiness Episode Counter
  - Face tracking indicator (handles missing face gracefully without false alarms)
  - Camera FPS telemetry
- **Embedded ESP32 Serial Hub**:
  - Connects to an ESP32 or Arduino over USB Serial (`115200` baud)
  - Sends `NORMAL\n` when the driver is awake
  - Sends `DROWSY\n` when drowsiness is detected
  - Fully functional in **Standalone Mode** when no ESP32 is plugged in
- **Interactive Calibration & Settings**:
  - Eye-closure duration threshold slider (`1.0 s` to `5.0 s`)
  - Audio alarm toggle switch & "Test Alarm" speaker verification
  - Available COM port scanner and connector

---

## 📁 Project Structure

```text
esd project/
│
├── app.py                      # Main GUI desktop application (CustomTkinter)
├── camera.py                   # Thread-safe webcam capture with DirectShow backend
├── drowsiness_detector.py      # OpenCV face/eye detection engine & closure timer
├── serial_communication.py     # ESP32 USB Serial communication manager
├── alarm.py                    # Windows native audio alarm generator & player
├── config.py                   # Central settings, color palettes, and file paths
├── requirements.txt            # Python dependencies
│
├── cascades/                   # Haar Cascade XML models
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_eye.xml
│   └── haarcascade_eye_tree_eyeglasses.xml
│
├── assets/                     # Audio assets (auto-generated siren WAV)
│   └── alert.wav
│
└── esp32_firmware/             # Embedded microcontroller firmware
    └── esp32_firmware.ino      # Ready-to-flash Arduino/ESP32 sketch
```

---

## 🚀 Quick Start Guide (Windows)

### 1. Prerequisites
Ensure you have Python 3.10+ installed on Windows. (Python 3.10 through 3.14 are supported).

### 2. Open Command Prompt / PowerShell
Navigate to the project directory:
```powershell
cd "c:\Users\LENOVO\OneDrive\Desktop\esd project"
```

### 3. Install Dependencies
Run:
```powershell
pip install -r requirements.txt
```
*(Dependencies: `opencv-python`, `customtkinter`, `pillow`, `pyserial`, `numpy`)*

### 4. Run the Application
Launch the desktop application:
```powershell
python app.py
```

---

## 🎮 How to Use the Application

1. **Start Monitoring**:
   - Click the green **`START MONITORING`** button.
   - Your laptop's webcam will turn on, and you will see your face tracked inside the camera feed.
   - Green bounding brackets will follow your face, and cyan boxes will track your eyes.

2. **Testing Drowsiness Detection**:
   - Close both eyes for **2.0 seconds** (or your chosen threshold).
   - The status badge changes to **`DROWSY`**, the large **DROWSINESS DETECTED** banner flashes, the laptop alarm sounds, and the event counter increments.
   - Open your eyes: the alarm immediately silences, and the status returns to **`AWAKE`**.

3. **Adjusting Sensitivity**:
   - Adjust the **Eye Closure Threshold** slider in the calibration panel to make the detector faster (e.g. `1.5 s`) or more forgiving (e.g. `2.5 s`).

4. **Testing Audio**:
   - Click the **"Test Alarm"** button in the settings panel to verify speaker volume.
   - You can toggle the audio alarm ON or OFF using the switch.

5. **Reset & Exit**:
   - Click **`RESET`** to clear the episode counter and active timers.
   - Click **`STOP MONITORING`** to pause the webcam.
   - Click **`EXIT`** to cleanly close all threads and devices.

---

## 🔌 Embedded ESP32 Hardware Integration (Next Stage)

The software is pre-engineered for seamless USB serial communication with an ESP32 or Arduino board.

### Serial Protocol
- Baud rate: **`115200`**
- Data sent from laptop to ESP32:
  - **`NORMAL\n`**: Driver is awake and attentive.
  - **`DROWSY\n`**: Drowsiness threshold reached.

### Hardware Wiring Diagram
| Component | Pin on ESP32 | Description |
| :--- | :--- | :--- |
| **Active Buzzer (+)** | **GPIO 18** | Positive terminal of 5V/3.3V buzzer |
| **Buzzer (-)** | **GND** | Ground |
| **Red Warning LED (+)** | **GPIO 19** | In series with a 220Ω resistor |
| **Green Safe LED (+)** | **GPIO 21** | In series with a 220Ω resistor |
| **LED (-) Cathodes** | **GND** | Common ground |
| **ESP32 USB** | **Laptop USB Port** | Supplies power & serial communication |

### Connecting ESP32 via GUI
1. Plug the ESP32 into a USB port on your laptop.
2. In the desktop application, locate the **ESP32 SERIAL HARDWARE** card on the right panel.
3. Click the refresh button (`🔄`) to discover the port (e.g. `COM3` or `COM4`).
4. Click **`CONNECT`**.
5. The application will automatically stream `NORMAL` and `DROWSY` states directly to the microcontroller!
