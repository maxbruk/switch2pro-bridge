"""Unit tests for ControllerManager, layout mapping, output mode switching, and DSU streaming."""

import socket
import struct
import threading
import time
import unittest.mock as mock
import zlib
from typing import List, Optional, Set, Tuple

import pytest

from app_core import ControllerManager
from report_parser import ControllerState
from constants import (
    NINTENDO_VENDOR_ID,
    SWITCH2_PRO_PID,
    SWITCH1_PRO_PID,
    NSO_GAMECUBE_PID,
    USB_STARTUP_COMMANDS,
)
from virtual_xinput import VirtualXboxController
from virtual_ds4 import VirtualDS4Controller
from virtual_gamepad import BaseVirtualController
from dsu_server import DsuServer
from tests.test_virtual_controllers import MockVX360Gamepad, MockVDS4Gamepad, make_state



def test_constants_definitions():
    assert NINTENDO_VENDOR_ID == 0x057E
    assert SWITCH2_PRO_PID == 0x2069
    assert SWITCH1_PRO_PID == 0x2009
    assert NSO_GAMECUBE_PID == 0x2073
    assert len(USB_STARTUP_COMMANDS) == 5


def test_manager_defaults():
    manager = ControllerManager()
    assert manager.is_connected is False
    assert manager.auto_connect_enabled is True
    assert manager.button_layout == "nintendo"
    assert manager.deadzone == 0.05


def test_manager_layout_switch():
    manager = ControllerManager()
    manager.set_button_layout("xbox")
    assert manager.button_layout == "xbox"
    assert manager.virtual_gamepad.layout == "xbox"

    manager.set_button_layout("nintendo")
    assert manager.button_layout == "nintendo"
    assert manager.virtual_gamepad.layout == "nintendo"


def test_manager_deadzone():
    manager = ControllerManager()
    manager.set_deadzone(0.12)
    assert manager.deadzone == 0.12
    assert manager.usb_connector.deadzone == 0.12
    assert manager.ble_connector.deadzone == 0.12


def test_manager_callbacks():
    manager = ControllerManager()
    captured = {}

    def on_conn(connected, transport, name, batt):
        captured["connected"] = connected
        captured["transport"] = transport
        captured["name"] = name
        captured["batt"] = batt

    manager.on_connection_changed = on_conn
    manager._notify_connection(True, "USB-C", "Switch 2 Pro Controller", 100)

    assert captured["connected"] is True
    assert captured["transport"] == "USB-C"
    assert captured["name"] == "Switch 2 Pro Controller"
    assert captured["batt"] == 100


def test_manager_gl_gr_mapping():
    manager = ControllerManager()
    assert manager.virtual_gamepad.gl_mapping == "L3"
    assert manager.virtual_gamepad.gr_mapping == "R3"

    manager.set_gl_mapping("LB")
    manager.set_gr_mapping("RB")
    assert manager.virtual_gamepad.gl_mapping == "LB"
    assert manager.virtual_gamepad.gr_mapping == "RB"

    manager.set_gl_mapping("Disabled")
    manager.set_gr_mapping("RT")
    assert manager.virtual_gamepad.gl_mapping == "Disabled"
    assert manager.virtual_gamepad.gr_mapping == "RT"


class MockBtn:
    XUSB_GAMEPAD_LEFT_THUMB = 1
    XUSB_GAMEPAD_RIGHT_THUMB = 2
    XUSB_GAMEPAD_LEFT_SHOULDER = 4
    XUSB_GAMEPAD_RIGHT_SHOULDER = 8
    XUSB_GAMEPAD_A = 16
    XUSB_GAMEPAD_B = 32
    XUSB_GAMEPAD_X = 64
    XUSB_GAMEPAD_Y = 128


class MockGamepad:
    def __init__(self):
        self.pressed = []
        self.left_trigger = 0.0
        self.right_trigger = 0.0

    def press_button(self, btn):
        self.pressed.append(btn)

    def left_trigger_float(self, value_float):
        self.left_trigger = value_float

    def right_trigger_float(self, value_float):
        self.right_trigger = value_float


def test_virtual_gamepad_apply_mapped_actions():
    manager = ControllerManager()
    vg = manager.virtual_gamepad
    mock_btn = MockBtn()

    # Test L3 & R3
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "L3", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_LEFT_THUMB]

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "R3", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_RIGHT_THUMB]

    # Test LB & RB
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "LB", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_LEFT_SHOULDER]

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "RB", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_RIGHT_SHOULDER]

    # Test Face buttons
    for code, attr in [("A", "XUSB_GAMEPAD_A"), ("B", "XUSB_GAMEPAD_B"), ("X", "XUSB_GAMEPAD_X"), ("Y", "XUSB_GAMEPAD_Y")]:
        gp = MockGamepad()
        vg._apply_mapped_action(gp, code, mock_btn)
        assert gp.pressed == [getattr(MockBtn, attr)]

    # Test LT & RT triggers
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "LT", mock_btn)
    assert gp.left_trigger == 1.0

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "RT", mock_btn)
    assert gp.right_trigger == 1.0

    # Test Disabled
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "Disabled", mock_btn)
    assert gp.pressed == []
    assert gp.left_trigger == 0.0
    assert gp.right_trigger == 0.0


# ============================================================================
# Mock Infrastructure
# ============================================================================

class MockDsuServer:
    """Zero-network, thread-safe mock for DsuServer testing in ControllerManager."""

    def __init__(self, host: str = "127.0.0.1", port: int = 26760):
        self.host = host
        self.port = port
        self.is_running = False
        self.start_count = 0
        self.stop_count = 0
        self.states_received: List[ControllerState] = []
        self.fail_bind = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            self.start_count += 1
            if self.fail_bind:
                self.is_running = False
                return False
            self.is_running = True
            return True

    def stop(self) -> None:
        with self._lock:
            self.stop_count += 1
            self.is_running = False

    def update_state(self, state: ControllerState) -> None:
        with self._lock:
            if self.is_running and state is not None:
                self.states_received.append(state)


# ============================================================================
# Suite 1: Initialization & Defaults
# ============================================================================

class TestManagerInitialization:
    """Verifies baseline manager initialization and default attributes."""

    def test_manager_defaults(self):
        manager = ControllerManager()
        assert manager.is_connected is False
        assert manager.auto_connect_enabled is True
        assert manager.button_layout == "nintendo"
        assert manager.deadzone == 0.05
        # M4 Additions
        assert manager.output_mode == "xbox"
        assert manager.dsu_enabled is False
        assert isinstance(manager.virtual_gamepad, VirtualXboxController)
        assert manager.virtual_gamepad.is_connected is False
        assert hasattr(manager, "dsu_server")
        assert manager.dsu_server.is_running is False


# ============================================================================
# Suite 2: Output Mode Switching (Disconnected State)
# ============================================================================

class TestManagerOutputModeDisconnected:
    """Verifies output mode transitions when controller is disconnected."""

    def test_set_output_mode_ds4(self):
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        assert manager.output_mode == "ds4"
        assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
        assert manager.virtual_gamepad.is_connected is False

    def test_set_output_mode_back_to_xbox(self):
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        manager.set_output_mode("xbox")
        assert manager.output_mode == "xbox"
        assert isinstance(manager.virtual_gamepad, VirtualXboxController)
        assert manager.virtual_gamepad.is_connected is False

    def test_set_output_mode_preserves_layout_and_paddles(self):
        manager = ControllerManager()
        manager.set_button_layout("xbox")
        manager.set_gl_mapping("LB")
        manager.set_gr_mapping("RB")

        manager.set_output_mode("ds4")
        assert manager.virtual_gamepad.layout == "xbox"
        assert manager.virtual_gamepad.gl_mapping == "LB"
        assert manager.virtual_gamepad.gr_mapping == "RB"

        manager.set_output_mode("xbox")
        assert manager.virtual_gamepad.layout == "xbox"
        assert manager.virtual_gamepad.gl_mapping == "LB"
        assert manager.virtual_gamepad.gr_mapping == "RB"

    def test_set_output_mode_idempotent(self):
        manager = ControllerManager()
        initial_gp = manager.virtual_gamepad
        manager.set_output_mode("xbox")
        # Instance should not be unnecessarily churned
        assert manager.virtual_gamepad is initial_gp

    def test_set_output_mode_case_insensitive(self):
        manager = ControllerManager()
        manager.set_output_mode("DS4")
        assert manager.output_mode == "ds4"
        assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
        manager.set_output_mode("XBOX")
        assert manager.output_mode == "xbox"
        assert isinstance(manager.virtual_gamepad, VirtualXboxController)

    def test_set_output_mode_invalid_raises_value_error(self):
        manager = ControllerManager()
        with pytest.raises(ValueError, match="Invalid output mode"):
            manager.set_output_mode("switch_pro")
        assert manager.output_mode == "xbox"
        assert isinstance(manager.virtual_gamepad, VirtualXboxController)


# ============================================================================
# Suite 3: Output Mode Switching (Connected State)
# ============================================================================

class TestManagerOutputModeConnected:
    """Verifies live controller replacement when physical device is connected."""

    def test_mode_switch_while_connected_transitions_virtual_controller(self):
        manager = ControllerManager()
        mock_xbox_gp = MockVX360Gamepad()
        mock_ds4_gp = MockVDS4Gamepad()

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox_gp), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4_gp):

            # Simulate initial connection
            manager.is_connected = True
            manager.virtual_gamepad.connect()
            assert manager.virtual_gamepad.is_connected is True
            assert isinstance(manager.virtual_gamepad, VirtualXboxController)

            # Switch to DS4 while connected
            manager.set_output_mode("ds4")

            # Verify old Xbox controller disconnected
            assert mock_xbox_gp.reset_count >= 1
            # Verify new DS4 controller connected
            assert manager.output_mode == "ds4"
            assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
            assert manager.virtual_gamepad.is_connected is True
            assert manager.is_connected is True

            # Switch back to Xbox while connected
            manager.set_output_mode("xbox")
            assert mock_ds4_gp.reset_count >= 1
            assert manager.output_mode == "xbox"
            assert isinstance(manager.virtual_gamepad, VirtualXboxController)
            assert manager.virtual_gamepad.is_connected is True

    def test_mode_switch_connect_failure_handled_gracefully(self):
        manager = ControllerManager()
        statuses = []
        manager.on_status_message = lambda msg: statuses.append(msg)

        manager.is_connected = True
        manager.virtual_gamepad.is_connected = True

        # Simulate ViGEm failure on spawning DS4
        with mock.patch("vgamepad.VDS4Gamepad", side_effect=RuntimeError("ViGEmBus failed")):
            manager.set_output_mode("ds4")
            assert manager.output_mode == "ds4"
            assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
            assert manager.virtual_gamepad.is_connected is False
            assert any("Failed" in s for s in statuses)


# ============================================================================
# Suite 4: DSU Server Lifecycle
# ============================================================================

class TestManagerDsuLifecycle:
    """Verifies start, stop, and failure handling of Cemuhook DSU server."""

    def test_dsu_lifecycle_toggle(self):
        manager = ControllerManager()
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu

        # Toggle True
        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is True
        assert mock_dsu.is_running is True
        assert mock_dsu.start_count == 1

        # Toggle False
        manager.set_dsu_enabled(False)
        assert manager.dsu_enabled is False
        assert mock_dsu.is_running is False
        assert mock_dsu.stop_count == 1

    def test_dsu_lifecycle_idempotent(self):
        manager = ControllerManager()
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu

        manager.set_dsu_enabled(True)
        manager.set_dsu_enabled(True)
        assert mock_dsu.start_count == 1
        assert manager.dsu_enabled is True

        manager.set_dsu_enabled(False)
        manager.set_dsu_enabled(False)
        assert mock_dsu.stop_count == 1
        assert manager.dsu_enabled is False

    def test_dsu_bind_failure_handled_gracefully(self):
        manager = ControllerManager()
        mock_dsu = MockDsuServer()
        mock_dsu.fail_bind = True
        manager.dsu_server = mock_dsu
        statuses = []
        manager.on_status_message = lambda msg: statuses.append(msg)

        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is False
        assert mock_dsu.is_running is False
        assert any("Failed" in s for s in statuses)


# ============================================================================
# Suite 5: Dual-Stream Dispatch
# ============================================================================

class TestManagerDualStreamDispatch:
    """Verifies simultaneous routing to Virtual Controller, DSU, and GUI."""

    def test_dispatch_xbox_mode_dsu_disabled(self):
        manager = ControllerManager()
        mock_gp = MockVX360Gamepad()
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu
        manager.dsu_enabled = False

        ui_updates = []
        manager.on_input_update = lambda s: ui_updates.append(s)

        st = make_state(a=True, left_stick_x=0.5, accel_z=1.0)
        manager._on_controller_state(st)

        assert mock_gp.update_count == 1
        assert len(mock_dsu.states_received) == 0
        assert len(ui_updates) == 1
        assert ui_updates[0] is st

    def test_dispatch_xbox_mode_dsu_enabled_dual_stream(self):
        manager = ControllerManager()
        mock_gp = MockVX360Gamepad()
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu
        manager.set_dsu_enabled(True)

        st = make_state(b=True, accel_z=1.0, gyro_x=45.0)
        manager._on_controller_state(st)

        assert mock_gp.update_count == 1
        assert len(mock_dsu.states_received) == 1
        assert mock_dsu.states_received[0] is st

    def test_dispatch_ds4_mode_dsu_disabled(self):
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        mock_gp = MockVDS4Gamepad()
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu
        manager.dsu_enabled = False

        st = make_state(accel_z=1.0, gyro_x=90.0)
        manager._on_controller_state(st)

        assert mock_gp.update_ex_count == 1
        assert mock_gp.last_report_snapshot.wAccelZ == 8192
        assert len(mock_dsu.states_received) == 0

    def test_dispatch_ds4_mode_dsu_enabled_dual_motion(self):
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        mock_gp = MockVDS4Gamepad()
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu
        manager.set_dsu_enabled(True)

        st = make_state(accel_z=1.0, gyro_x=90.0)
        manager._on_controller_state(st)

        assert mock_gp.update_ex_count == 1
        assert len(mock_dsu.states_received) == 1
        assert mock_dsu.states_received[0] is st

    def test_dispatch_with_null_callbacks_and_none_state(self):
        manager = ControllerManager()
        manager.on_input_update = None
        # Safe no-ops
        manager._on_controller_state(None)
        manager._on_controller_state(make_state())


# ============================================================================
# Suite 6: Multi-Threaded Concurrency & Thread Safety
# ============================================================================

class TestManagerConcurrencyAndThreadSafety:
    """Stress tests simulating high-frequency input dispatch concurrent with GUI toggles."""

    def test_concurrent_state_dispatch_and_mode_switching(self):
        manager = ControllerManager()
        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):
            manager.is_connected = True
            manager.virtual_gamepad.connect()

            stop_event = threading.Event()
            exceptions = []

            def producer():
                st = make_state(a=True, accel_z=1.0)
                try:
                    for _ in range(500):
                        manager._on_controller_state(st)
                        time.sleep(0.001)
                except Exception as e:
                    exceptions.append(e)

            def switcher():
                try:
                    for i in range(50):
                        target = "ds4" if (i % 2 == 0) else "xbox"
                        manager.set_output_mode(target)
                        time.sleep(0.005)
                except Exception as e:
                    exceptions.append(e)

            t1 = threading.Thread(target=producer, name="SimulatedInputThread")
            t2 = threading.Thread(target=switcher, name="SimulatedGuiThread")

            t1.start()
            t2.start()
            t1.join(timeout=5.0)
            t2.join(timeout=5.0)

            assert len(exceptions) == 0, f"Exceptions occurred during concurrency test: {exceptions}"
            assert not t1.is_alive()
            assert not t2.is_alive()
            assert manager.output_mode in ("xbox", "ds4")

    def test_concurrent_state_dispatch_and_dsu_toggling(self):
        manager = ControllerManager()
        mock_dsu = MockDsuServer()
        manager.dsu_server = mock_dsu

        exceptions = []

        def producer():
            st = make_state(accel_z=1.0)
            try:
                for _ in range(500):
                    manager._on_controller_state(st)
                    time.sleep(0.001)
            except Exception as e:
                exceptions.append(e)

        def dsu_toggler():
            try:
                for i in range(50):
                    manager.set_dsu_enabled(i % 2 == 0)
                    time.sleep(0.005)
            except Exception as e:
                exceptions.append(e)

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=dsu_toggler)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert len(exceptions) == 0
        assert not t1.is_alive()
        assert not t2.is_alive()


# ============================================================================
# Suite 7: Disconnection & Teardown
# ============================================================================

class TestManagerDisconnectTeardown:
    """Verifies clean teardown on disconnect and settings preservation."""

    def test_disconnect_cleans_up_active_controller(self):
        manager = ControllerManager()
        mock_ds4 = MockVDS4Gamepad()

        with mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):
            manager.set_output_mode("ds4")
            manager.is_connected = True
            manager.virtual_gamepad.connect()

            manager.disconnect()
            assert manager.is_connected is False
            assert manager.virtual_gamepad.is_connected is False
            assert mock_ds4.reset_count >= 1

    def test_disconnect_retains_output_mode_preference(self):
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        manager.disconnect()
        assert manager.output_mode == "ds4"

    def test_transport_disconnect_triggers_clean_teardown(self):
        manager = ControllerManager()
        with mock.patch.object(manager, "disconnect") as mock_disc:
            manager._on_transport_disconnected()
            mock_disc.assert_called_once()


# ============================================================================
# Suite 8: Real Socket Ephemeral DSU Integration
# ============================================================================

class TestManagerRealDsuIntegration:
    """Verifies real UDP network integration using dynamic ephemeral port 0."""

    def test_real_dsu_ephemeral_streaming(self):
        manager = ControllerManager()
        # Bind ephemeral port 0 to prevent port 26760 collision
        real_dsu = DsuServer(host="127.0.0.1", port=0)
        manager.dsu_server = real_dsu

        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is True
        bound_port = manager.dsu_server.port
        assert bound_port > 0

        # Send subscription request (0x100002) from test UDP client
        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_sock.settimeout(1.5)

        try:
            req = bytearray(20)
            struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 4, 0, 0x12345678)
            struct.pack_into("<I", req, 16, 0x100002)
            crc = zlib.crc32(req) & 0xFFFFFFFF
            struct.pack_into("<I", req, 8, crc)
            client_sock.sendto(req, ("127.0.0.1", bound_port))
            time.sleep(0.05)  # Allow listener loop to register subscription

            # Dispatch motion state through manager
            st = make_state(accel_x=0.25, accel_y=-0.5, accel_z=0.98, gyro_x=45.0, motion_timestamp=123456)
            manager._on_controller_state(st)

            # Receive 100-byte DSU data packet
            packet, _ = client_sock.recvfrom(1024)
            assert len(packet) == 100
            magic, ver, length, crc_recv, sender = struct.unpack_from("<4sHHII", packet, 0)
            assert magic == b"DSUS"
            assert ver == 1001
            assert length == 84

            # Verify CRC32
            temp = bytearray(packet)
            temp[8:12] = b"\x00\x00\x00\x00"
            assert crc_recv == (zlib.crc32(temp) & 0xFFFFFFFF)

            # Verify Motion Floats at bytes 76..99
            ax, ay, az, gx, gy, gz = struct.unpack_from("<ffffff", packet, 76)
            assert abs(ax - 0.25) < 1e-4
            assert abs(ay - (-0.5)) < 1e-4
            assert abs(az - 0.98) < 1e-4
            assert abs(gx - 45.0) < 1e-4

        finally:
            client_sock.close()
            manager.set_dsu_enabled(False)
            assert manager.dsu_enabled is False


# ============================================================================
# Suite 9: Haptic Feedback / HD Rumble Forwarding
# ============================================================================

class TestManagerRumbleForwarding:
    """Verifies ViGEmBus force-feedback callback registration and transport forwarding."""

    def test_trigger_rumble_forwards_to_active_usb_and_ble(self):
        manager = ControllerManager()
        manager.usb_connector.is_connected = True
        manager.ble_connector.is_connected = False

        with mock.patch.object(manager.usb_connector, "set_rumble", return_value=True) as usb_rumble, \
             mock.patch.object(manager.ble_connector, "set_rumble", return_value=True) as ble_rumble:
            manager.set_rumble_intensity(0.75)
            manager.trigger_rumble(200, 120)
            usb_rumble.assert_called_once_with(200, 120, intensity=0.75)
            ble_rumble.assert_not_called()

            manager.set_rumble_enabled(False)
            assert manager.rumble_enabled is False
            manager.trigger_rumble(255, 255)
            usb_rumble.assert_called_with(0, 0)

    def test_virtual_controllers_forward_vigem_notification_to_manager(self):
        manager = ControllerManager()
        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()
        captured_xbox_cb = None
        captured_ds4_cb = None

        def reg_xbox(cb):
            nonlocal captured_xbox_cb
            captured_xbox_cb = cb

        def reg_ds4(cb):
            nonlocal captured_ds4_cb
            captured_ds4_cb = cb

        mock_xbox.register_notification = reg_xbox  # type: ignore
        mock_xbox.unregister_notification = lambda: None  # type: ignore
        mock_ds4.register_notification = reg_ds4  # type: ignore
        mock_ds4.unregister_notification = lambda: None  # type: ignore

        manager.usb_connector.is_connected = True
        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4), \
             mock.patch.object(manager.usb_connector, "set_rumble", return_value=True) as usb_rumble:
            manager.is_connected = True
            # Verify Xbox notification hook
            manager.virtual_gamepad.connect()
            assert captured_xbox_cb is not None
            captured_xbox_cb(None, None, 150, 75, 0, None)
            usb_rumble.assert_called_with(150, 75, intensity=1.0)

            # Switch to DS4 and verify DS4 notification hook preserves rumble wiring
            manager.set_output_mode("ds4")
            assert captured_ds4_cb is not None
            captured_ds4_cb(None, None, 220, 40, 0, None)
            usb_rumble.assert_called_with(220, 40, intensity=1.0)



