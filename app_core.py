"""Core manager orchestrating USB, BLE, Virtual Controllers (Xbox/DS4), and Cemuhook DSU."""

import time
import logging
import threading
from typing import Optional, Callable, Dict, Any

from report_parser import ControllerState
from usb_connector import UsbProConnector, enumerate_usb_controllers
from ble_connector import BleProConnector
from virtual_gamepad import BaseVirtualController
from virtual_xinput import VirtualXboxController, check_vigem_driver_available, install_vigem_driver
from virtual_ds4 import VirtualDS4Controller
from dsu_server import DsuServer

logger = logging.getLogger(__name__)


class ControllerManager:
    """Unified manager providing single-button connection across USB-C and Bluetooth,
    supporting Xbox 360 and DualShock 4 virtual gamepads and Cemuhook DSU motion streaming."""

    def __init__(self, dsu_host: str = "127.0.0.1", dsu_port: int = 26760):
        self._output_mode: str = "xbox"  # "xbox" or "ds4"
        self.button_layout: str = "nintendo"  # "nintendo" or "xbox"
        self.gl_mapping: str = "L3"
        self.gr_mapping: str = "R3"
        self.deadzone: float = 0.05
        self.rumble_enabled: bool = True
        self.rumble_intensity: float = 1.0

        # Initialize default virtual controller (Xbox 360)
        self.virtual_gamepad: BaseVirtualController = VirtualXboxController()
        self.virtual_gamepad.layout = self.button_layout
        self.virtual_gamepad.gl_mapping = self.gl_mapping
        self.virtual_gamepad.gr_mapping = self.gr_mapping
        self.virtual_gamepad.on_rumble = self.trigger_rumble

        # Cemuhook DSU Motion Server
        self.dsu_server: DsuServer = DsuServer(host=dsu_host, port=dsu_port)
        self._dsu_enabled: bool = False

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

        # Callbacks for GUI or CLI
        self.on_connection_changed: Optional[Callable[[bool, str, str, int], None]] = None
        self.on_input_update: Optional[Callable[[ControllerState], None]] = None
        self.on_status_message: Optional[Callable[[str], None]] = None

        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop = threading.Event()
        self._lock = threading.Lock()

    @property
    def output_mode(self) -> str:
        """Active virtual controller output mode ('xbox' or 'ds4')."""
        return self._output_mode

    @output_mode.setter
    def output_mode(self, mode: str) -> None:
        self.set_output_mode(mode)

    def set_output_mode(self, mode: str) -> None:
        """Dynamically switch virtual controller output backend ('xbox' or 'ds4').

        Thread-safe: seamlessly disconnects old backend and initializes new backend
        while preserving button layout, paddle mappings, and connection state.
        """
        if not isinstance(mode, str):
            raise ValueError(f"Output mode must be a string, got {type(mode).__name__}")

        clean_mode = mode.lower().strip()
        if clean_mode not in ("xbox", "ds4"):
            raise ValueError(f"Invalid output mode: '{mode}'. Must be 'xbox' or 'ds4'")

        with self._lock:
            if clean_mode == self._output_mode:
                return

            self._output_mode = clean_mode
            old_gamepad = self.virtual_gamepad

            # Instantiate new virtual controller backend
            if clean_mode == "ds4":
                new_gamepad: BaseVirtualController = VirtualDS4Controller()
            else:
                new_gamepad = VirtualXboxController()

            # Preserve button layout, grip paddle mappings, and rumble forwarding
            new_gamepad.layout = self.button_layout
            new_gamepad.gl_mapping = self.gl_mapping
            new_gamepad.gr_mapping = self.gr_mapping
            new_gamepad.on_rumble = self.trigger_rumble

            # If physical controller is connected or previous backend was active, connect new backend
            should_connect = self.is_connected or (old_gamepad is not None and getattr(old_gamepad, "is_connected", False))
            if should_connect:
                connected = new_gamepad.connect()
                if not connected:
                    logger.warning("Failed to connect virtual %s controller.", clean_mode.upper())
                    self._on_status(f"Failed to connect virtual {clean_mode.upper()} controller.")

            # Atomically switch active virtual gamepad reference
            self.virtual_gamepad = new_gamepad

        # Disconnect old backend outside manager lock to prevent latency spikes
        if old_gamepad is not None:
            try:
                old_gamepad.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting old virtual gamepad: %s", e)

        logger.info("Switched output controller mode to %s", clean_mode.upper())
        self._on_status(f"Output mode: {clean_mode.upper()}")

    @property
    def dsu_enabled(self) -> bool:
        """Whether Cemuhook DSU motion server is enabled."""
        return self._dsu_enabled

    @dsu_enabled.setter
    def dsu_enabled(self, enabled: bool) -> None:
        self.set_dsu_enabled(enabled)

    def set_dsu_enabled(self, enabled: bool) -> None:
        """Enable or disable the background Cemuhook DSU motion server.

        Thread-safe and idempotent. When enabled, DsuServer listens on UDP port 26760
        and broadcasts 6-axis motion frames to emulators (Cemu, Dolphin, Yuzu, Ryujinx).
        """
        target = bool(enabled)
        with self._lock:
            if target == self._dsu_enabled:
                return

            if target:
                if self.dsu_server is None:
                    self.dsu_server = DsuServer()
                if not getattr(self.dsu_server, "is_running", False):
                    started = self.dsu_server.start()
                    if not started:
                        self._dsu_enabled = False
                        logger.warning("Failed to start DSU server on port %d", getattr(self.dsu_server, "port", 26760))
                        self._on_status(f"Failed to start DSU server (port {getattr(self.dsu_server, 'port', 26760)} in use)")
                        return
                self._dsu_enabled = True
                logger.info("DSU server started on port %d", self.dsu_server.port)
                self._on_status(f"DSU Motion Server active (port {self.dsu_server.port})")
            else:
                self._dsu_enabled = False
                if self.dsu_server is not None and getattr(self.dsu_server, "is_running", False):
                    try:
                        self.dsu_server.stop()
                    except Exception as e:
                        logger.debug("Error stopping DSU server: %s", e)
                    logger.info("DSU server stopped.")
                    self._on_status("DSU Motion Server disabled.")

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

                # Start active virtual gamepad
                self.virtual_gamepad.layout = self.button_layout
                self.virtual_gamepad.gl_mapping = self.gl_mapping
                self.virtual_gamepad.gr_mapping = self.gr_mapping
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
                    self.virtual_gamepad.gl_mapping = self.gl_mapping
                    self.virtual_gamepad.gr_mapping = self.gr_mapping
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

        if self.virtual_gamepad is not None:
            try:
                self.virtual_gamepad.disconnect()
            except Exception as e:
                logger.debug("Error disconnecting virtual gamepad: %s", e)
        self._on_status("Controller disconnected.")
        self._notify_connection(False, "None", "Disconnected", 0)

    def shutdown(self):
        """Cleanly terminate auto-detector, controller connections, and DSU server."""
        self.stop_auto_detector()
        self.disconnect()
        with self._lock:
            if self.dsu_server is not None and getattr(self.dsu_server, "is_running", False):
                try:
                    self.dsu_server.stop()
                except Exception as e:
                    logger.debug("Error stopping DSU server during shutdown: %s", e)
                self._dsu_enabled = False

    def set_button_layout(self, layout: str):
        """Toggle button layout ('nintendo' or 'xbox')."""
        with self._lock:
            self.button_layout = layout
            if self.virtual_gamepad is not None:
                self.virtual_gamepad.layout = layout

    def set_deadzone(self, deadzone: float):
        """Set stick deadzone (e.g. 0.05)."""
        self.deadzone = deadzone
        self.usb_connector.deadzone = deadzone
        self.ble_connector.deadzone = deadzone

    def set_gl_mapping(self, mapping: str):
        """Set Grip Left (GL) paddle mapping."""
        with self._lock:
            self.gl_mapping = mapping
            if self.virtual_gamepad is not None:
                self.virtual_gamepad.gl_mapping = mapping

    def set_gr_mapping(self, mapping: str):
        """Set Grip Right (GR) paddle mapping."""
        with self._lock:
            self.gr_mapping = mapping
            if self.virtual_gamepad is not None:
                self.virtual_gamepad.gr_mapping = mapping

    def set_rumble_enabled(self, enabled: bool) -> None:
        """Enable or disable forwarding virtual gamepad haptics to the physical controller."""
        self.rumble_enabled = bool(enabled)
        if not self.rumble_enabled:
            self.usb_connector.set_rumble(0, 0)
            self.ble_connector.set_rumble(0, 0)

    def set_rumble_intensity(self, intensity: float) -> None:
        """Set haptic intensity scale in [0.0, 1.0]."""
        self.rumble_intensity = max(0.0, min(1.0, float(intensity)))

    def trigger_rumble(self, large_motor: int, small_motor: int) -> None:
        """Forward large/small motor rumble values (0..255) to the active physical transport."""
        if not self.rumble_enabled:
            if getattr(self.usb_connector, "is_connected", False):
                self.usb_connector.set_rumble(0, 0)
            if getattr(self.ble_connector, "is_connected", False):
                self.ble_connector.set_rumble(0, 0)
            return

        if getattr(self.usb_connector, "is_connected", False):
            self.usb_connector.set_rumble(large_motor, small_motor, intensity=self.rumble_intensity)
        if getattr(self.ble_connector, "is_connected", False):
            self.ble_connector.set_rumble(large_motor, small_motor, intensity=self.rumble_intensity)

    def test_rumble_pulse(self, duration_sec: float = 0.20) -> None:
        """Send a brief non-blocking haptic vibration test pulse to the connected controller."""
        if not self.rumble_enabled:
            return

        def _pulse():
            self.trigger_rumble(180, 180)
            time.sleep(max(0.05, float(duration_sec)))
            self.trigger_rumble(0, 0)

        threading.Thread(target=_pulse, name="RumbleTestPulse", daemon=True).start()

    def _on_controller_state(self, state: Optional[ControllerState]):
        """Received new input state from USB or BLE (invoked at ~250 Hz)."""
        if state is None:
            return

        # 1. Update active virtual gamepad (lock-free reference snapshot)
        gp = self.virtual_gamepad
        if gp is not None:
            try:
                gp.update_from_state(state, layout=self.button_layout)
            except Exception as e:
                logger.debug("Error updating virtual gamepad: %s", e)

        # 2. Dual-dispatch: broadcast 6-axis motion to Cemuhook DSU server if enabled
        if self._dsu_enabled and self.dsu_server is not None:
            try:
                self.dsu_server.update_state(state)
            except Exception as e:
                logger.debug("Error updating DSU server: %s", e)

        # 3. Notify UI for live visualizer
        cb = self.on_input_update
        if cb is not None:
            try:
                cb(state)
            except Exception as e:
                logger.debug("Error in on_input_update callback: %s", e)

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
