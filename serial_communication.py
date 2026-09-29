"""
AI Driver Drowsiness Detection & Alert System
ESP32 USB Serial Communication Module
"""

import time
import threading
from typing import List, Optional, Callable
import serial
import serial.tools.list_ports
from config import (
    DEFAULT_BAUD_RATE,
    SERIAL_CMD_AWAKE,
    SERIAL_CMD_DROWSY,
    SERIAL_HEARTBEAT_INTERVAL
)


class SerialManager:
    """
    Manages USB Serial Communication with an external ESP32 / Arduino microcontroller.
    Operates safely in standalone mode when hardware is not connected.
    """

    def __init__(self, baud_rate: int = DEFAULT_BAUD_RATE):
        self.baud_rate = baud_rate
        self.port: Optional[str] = None
        self.connection: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        
        # State tracking
        self.last_sent_status: Optional[str] = None
        self.last_send_time: float = 0.0
        self.tx_callback: Optional[Callable[[str, str], None]] = None
        self.status_message: str = "Standalone Mode (Hardware Optional)"

    @staticmethod
    def get_available_ports() -> List[str]:
        """Return a list of available COM port names."""
        try:
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return sorted(ports)
        except Exception as e:
            print(f"[Serial] Error enumerating ports: {e}")
            return []

    def is_connected(self) -> bool:
        """Check if serial connection is currently open and valid."""
        with self._lock:
            return self.connection is not None and self.connection.is_open

    def connect(self, port_name: str, baud_rate: Optional[int] = None) -> tuple[bool, str]:
        """
        Open a serial connection to the specified port.
        Returns: (success: bool, message: str)
        """
        with self._lock:
            if baud_rate is not None:
                self.baud_rate = baud_rate

            self.port = port_name

            # Close previous connection if open
            if self.connection and self.connection.is_open:
                try:
                    self.connection.close()
                except Exception:
                    pass

            try:
                self.connection = serial.Serial(
                    port=port_name,
                    baudrate=self.baud_rate,
                    timeout=0.5,
                    write_timeout=0.5
                )
                # Brief wait for Arduino/ESP32 reset upon DTR toggle
                time.sleep(0.1)
                self.status_message = f"Connected: {port_name} @ {self.baud_rate}"
                print(f"[Serial] Successfully connected to {port_name} @ {self.baud_rate} baud")
                return True, self.status_message
            except Exception as e:
                self.connection = None
                self.status_message = f"Connection Failed: {str(e)}"
                print(f"[Serial] Connection error on {port_name}: {e}")
                return False, self.status_message

    def disconnect(self):
        """Safely close active serial connection."""
        with self._lock:
            if self.connection:
                try:
                    self.connection.close()
                except Exception:
                    pass
                self.connection = None
            self.status_message = "Disconnected (Standalone Mode)"
            print("[Serial] Port closed.")

    def register_tx_callback(self, callback: Callable[[str, str], None]):
        """
        Register a callback function to receive transmission updates.
        Signature: callback(command: str, port_status: str)
        """
        self.tx_callback = callback

    def send_driver_status(self, is_drowsy: bool, force: bool = False):
        """
        Send driver state to ESP32:
        - Sends 'DROWSY\\n' when drowsiness is detected
        - Sends 'NORMAL\\n' when driver is awake
        
        Transmits immediately when state changes, or at heartbeat intervals.
        """
        cmd = SERIAL_CMD_DROWSY if is_drowsy else SERIAL_CMD_AWAKE
        now = time.time()

        # Send immediately on state transition or when heartbeat interval elapses
        should_send = (
            force
            or (cmd != self.last_sent_status)
            or (now - self.last_send_time >= SERIAL_HEARTBEAT_INTERVAL)
        )

        if not should_send:
            return

        clean_cmd = cmd.strip()

        with self._lock:
            if self.connection and self.connection.is_open:
                try:
                    self.connection.write(cmd.encode("utf-8"))
                    self.connection.flush()
                    self.last_sent_status = cmd
                    self.last_send_time = now
                    if self.tx_callback:
                        self.tx_callback(clean_cmd, f"Active ({self.port})")
                except Exception as e:
                    print(f"[Serial] Write error: {e}")
                    self.status_message = f"Error sending to {self.port}: {e}"
                    # Auto close failed connection
                    try:
                        self.connection.close()
                    except Exception:
                        pass
                    self.connection = None
            else:
                # Standalone simulation: update timestamp and trigger UI callback
                self.last_sent_status = cmd
                self.last_send_time = now
                if self.tx_callback:
                    self.tx_callback(clean_cmd, "Standalone (Simulated)")


# Global shared instance
serial_manager = SerialManager()
