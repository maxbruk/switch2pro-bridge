"""Core manager orchestrating USB, BLE, and Virtual Xbox Controller."""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any

from report_parser import ControllerState
from usb_connector import UsbProConnector, enumerate_usb_controllers
from ble_connector import BleProConnector
from virtual_xinput import VirtualXboxController, check_vigem_driver_available, install_vigem_driver

logger = logging.getLogger(__name__)


class ControllerManager:
    """Unified manager providing single-button connection across USB-C and Bluetooth."""

    def __init__(self):
        self.virtual_gamepad = VirtualXboxController()
        self.usb_connector = UsbProConnector(
            on_state_change=self._on_controller_state,
            on_disconnect=self._on_transport_disconnected,
        )
        self.ble_connector = BleProConnector(
            on_state_change=self._on_controller_state,
            on_disconnect=self._on_transport_disconnected,
            on_status=self._on_status,
        )

        self.active_transport: Optional[str] = None  # "USB" or "Bluetooth"
        self.active_device_name: str = "Nintendo Switch 2 Pro Controller"
        self.is_connected = False
        self.is_connecting = False
        self.auto_connect_enabled = True
        self.button_layout = "nintendo"  # "nintendo" or "xbox"
        self.deadzone = 0.05

        # Callbacks for GUI or CLI
        self.on_connection_changed: Optional[Callable[[bool, str, str, int], None]] = None
        self.on_input_update: Optional[Callable[[ControllerState], None]] = None
        self.on_status_message: Optional[Callable[[str], None]] = None

        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()
        self._lock = threading.Lock()

    def start_auto_detector(self):
        """Start auto-detection background thread to connect on cable plug or BLE wake."""
        if self._auto_thread and self._auto_thread.is_alive():
            return
        self._auto_stop.clear()
        self._auto_thread = threading.Thread(target=self._auto_connect_loop, name="AutoDetector", daemon=True)
        self._auto_thread.start()

    def stop_auto_detector(self):
        """Stop auto-detection thread."""
        self._auto_stop.set()
        if self._auto_thread and self._auto_thread.is_alive():
            self._auto_thread.join(timeout=1.0)

    def _auto_connect_loop(self):
        """Background poll for controller presence."""
        while not self._auto_stop.is_set():
            if self.auto_connect_enabled and not self.is_connected and not self.is_connecting:
                # Fast check for wired USB controller
                usb_devs = enumerate_usb_controllers()
                if usb_devs:
                    logger.info("Auto-detector found wired Switch controller: %s", usb_devs[0]["name"])
                    self.connect_single_click()
            time.sleep(1.0)

    def connect_single_click(self) -> bool:
        """The core single-action connection routine.
        
        Attempts instant USB-C connection first; if no USB cable is plugged in,
        seamlessly starts Bluetooth BLE scanning and pairing.
        """
        with self._lock:
            if self.is_connected or self.is_connecting:
                return True
            self.is_connecting = True

        self._on_status("Checking USB connection...")

        # 1. Try USB-C first (instant & most stable)
        usb_controllers = enumerate_usb_controllers()
        if usb_controllers:
            dev = usb_controllers[0]
            self._on_status(f"Found {dev['name']} via USB. Initializing...")
            self.usb_connector.deadzone = self.deadzone
            if self.usb_connector.connect(dev):
                self.active_transport = "USB"
                self.active_device_name = dev["name"]
                self.is_connected = True
                self.is_connecting = False

                # Start virtual Xbox gamepad
                self.virtual_gamepad.layout = self.button_layout
                self.virtual_gamepad.connect()

                self._on_status(f"Connected: {self.active_device_name} (USB-C)")
                self._notify_connection(True, "USB-C", self.active_device_name, 100)
                return True

        # 2. If USB not present, attempt Bluetooth LE
        self._on_status("No USB controller detected. Scanning Bluetooth LE...")
        self.ble_connector.deadzone = self.deadzone
        self.ble_connector.start_connection()

        # Let BLE scan run asynchronously in the background
        def check_ble_result():
            for _ in range(70):
                if self.ble_connector.is_connected:
                    with self._lock:
                        self.active_transport = "Bluetooth"
                        self.active_device_name = (
                            self.ble_connector.device_info.get("name", "Switch 2 Pro Controller")
                            if self.ble_connector.device_info else "Switch 2 Pro Controller"
                        )
                        self.is_connected = True
                        self.is_connecting = False

                    self.virtual_gamepad.layout = self.button_layout
                    self.virtual_gamepad.connect()

                    self._on_status(f"Connected: {self.active_device_name} (Bluetooth)")
                    self._notify_connection(True, "Bluetooth", self.active_device_name, 85)
                    return
                time.sleep(0.5)

            # Timed out
            with self._lock:
                self.is_connecting = False
            self._on_status("Ready. Plug in USB or press the Sync button on controller.")

        threading.Thread(target=check_ble_result, daemon=True).start()
        return True

    def disconnect(self):
        """Disconnect controller and release virtual gamepad."""
        with self._lock:
            self.is_connecting = False
            self.is_connected = False
            transport = self.active_transport
            self.active_transport = None

        if transport == "USB":
            self.usb_connector.disconnect()
        elif transport == "Bluetooth":
            self.ble_connector.stop_connection()

        self.virtual_gamepad.disconnect()
        self._on_status("Controller disconnected.")
        self._notify_connection(False, "None", "Disconnected", 0)

    def set_button_layout(self, layout: str):
        """Toggle button layout ('nintendo' or 'xbox')."""
        self.button_layout = layout
        self.virtual_gamepad.layout = layout

    def set_deadzone(self, deadzone: float):
        """Set stick deadzone (e.g. 0.05)."""
        self.deadzone = deadzone
        self.usb_connector.deadzone = deadzone
        self.ble_connector.deadzone = deadzone

    def set_gl_mapping(self, mapping: str):
        """Set Grip Left (GL) paddle mapping."""
        self.virtual_gamepad.gl_mapping = mapping

    def set_gr_mapping(self, mapping: str):
        """Set Grip Right (GR) paddle mapping."""
        self.virtual_gamepad.gr_mapping = mapping


    def _on_controller_state(self, state: ControllerState):
        """Received new input state from USB or BLE."""
        # Update virtual Xbox 360 controller
        self.virtual_gamepad.update_from_state(state, layout=self.button_layout)

        # Notify UI for live visualizer
        if self.on_input_update:
            self.on_input_update(state)

    def _on_transport_disconnected(self):
        """Transport layer reported connection drop."""
        logger.info("Transport layer disconnected.")
        self.disconnect()

    def _on_status(self, msg: str):
        if self.on_status_message:
            self.on_status_message(msg)

    def _notify_connection(self, connected: bool, transport: str, name: str, battery: int):
        if self.on_connection_changed:
            self.on_connection_changed(connected, transport, name, battery)
