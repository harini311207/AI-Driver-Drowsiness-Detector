"""
AI Driver Drowsiness Detection & Alert System
Alarm & Audio Notification Module
"""

import os
import math
import struct
import wave
import threading
import time
import winsound
from config import ALARM_SOUND_FILE, ASSETS_DIR


def generate_alarm_wave(filepath=ALARM_SOUND_FILE, duration_s=1.2, sample_rate=44100):
    """
    Synthesize a high-urgency alternating two-tone automotive alert siren WAV file.
    Creates alternating 950 Hz and 1350 Hz sound pulses.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        return filepath

    num_samples = int(duration_s * sample_rate)
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)        # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            # 4-stage pulsed cadence (0.15s pulse, 0.15s pulse, alternating pitch)
            cycle = int(t / 0.15) % 4
            if cycle == 0:
                freq = 950.0
                envelope = min(1.0, (t % 0.15) / 0.01) * max(0.0, (0.15 - (t % 0.15)) / 0.01)
                val = int(28000 * envelope * math.sin(2 * math.pi * freq * t))
            elif cycle == 2:
                freq = 1350.0
                envelope = min(1.0, (t % 0.15) / 0.01) * max(0.0, (0.15 - (t % 0.15)) / 0.01)
                val = int(28000 * envelope * math.sin(2 * math.pi * freq * t))
            else:
                val = 0
            frames.extend(struct.pack('<h', max(-32767, min(32767, val))))
            
        wf.writeframes(frames)
    return filepath


class AlarmManager:
    """
    Thread-safe Audio Alert Manager using native Windows sound APIs.
    Plays a continuous looping siren during drowsiness and terminates instantly on recovery.
    """

    def __init__(self, sound_file=ALARM_SOUND_FILE):
        self.sound_file = sound_file
        self.enabled = True
        self.is_playing = False
        self._lock = threading.Lock()
        self._beep_thread = None
        self._stop_beep_flag = threading.Event()

        # Ensure the audio asset exists
        try:
            generate_alarm_wave(self.sound_file)
        except Exception as e:
            print(f"[Alarm] Warning: could not synthesize sound file: {e}")

    def set_enabled(self, enabled: bool):
        """Enable or disable alarm playback."""
        with self._lock:
            self.enabled = enabled
            if not enabled and self.is_playing:
                self._stop_audio()

    def start_alarm(self):
        """Start playing the alarm sound in a non-blocking loop."""
        with self._lock:
            if not self.enabled or self.is_playing:
                return

            self.is_playing = True

            # Attempt native asynchronous looping playback
            try:
                if os.path.exists(self.sound_file):
                    flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
                    winsound.PlaySound(self.sound_file, flags)
                else:
                    self._start_beep_fallback()
            except Exception as e:
                print(f"[Alarm] PlaySound error: {e}. Falling back to Beep.")
                self._start_beep_fallback()

    def stop_alarm(self):
        """Stop playing the alarm sound immediately."""
        with self._lock:
            if not self.is_playing:
                return
            self._stop_audio()

    def _stop_audio(self):
        """Internal helper to halt all audio output."""
        self.is_playing = False
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

        # Stop fallback thread if active
        if self._beep_thread and self._beep_thread.is_alive():
            self._stop_beep_flag.set()
            self._beep_thread = None

    def _start_beep_fallback(self):
        """Fallback thread that loops winsound.Beep if WAV playback is unavailable."""
        self._stop_beep_flag.clear()
        self._beep_thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._beep_thread.start()

    def _beep_loop(self):
        """Thread loop for winsound.Beep fallback."""
        while not self._stop_beep_flag.is_set():
            try:
                winsound.Beep(1200, 180)
                time.sleep(0.08)
                if self._stop_beep_flag.is_set():
                    break
                winsound.Beep(950, 180)
                time.sleep(0.12)
            except Exception:
                break

    def test_alarm(self, duration_s: float = 1.0):
        """Play a brief alarm sound test for a specified duration."""
        def _test():
            self.start_alarm()
            time.sleep(duration_s)
            self.stop_alarm()

        threading.Thread(target=_test, daemon=True).start()


# Global shared instance
alarm_manager = AlarmManager()
