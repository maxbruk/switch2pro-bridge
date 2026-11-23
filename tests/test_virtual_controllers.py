"""Comprehensive unit and integration tests for virtual controllers (Milestone 3).

Covers BaseVirtualController polymorphic contracts, VirtualXboxController,
VirtualDS4Controller extended report packaging, 6-axis IMU conversion,
Sony DirectInput stick Y inversion, 8-way D-pad hat encoding, button layouts,
grip paddles, and driver resilience.
"""

import abc
import math
import sys
import unittest.mock as mock
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import pytest

from report_parser import ControllerState
import virtual_gamepad
from virtual_gamepad import BaseVirtualController
import virtual_xinput
from virtual_xinput import VirtualXboxController, check_vigem_driver_available
import virtual_ds4
from virtual_ds4 import VirtualDS4Controller


# ============================================================================
# Mock Infrastructure & Snapshots
# ============================================================================

@dataclass
class DS4ReportExSnapshot:
    """Immutable snapshot of ctypes DS4_SUB_REPORT_EX fields for deterministic assertions."""
    bThumbLX: int
    bThumbLY: int
    bThumbRX: int
    bThumbRY: int
    wButtons: int
    bSpecial: int
    bTriggerL: int
    bTriggerR: int
    wTimestamp: int
    bBatteryLvl: int
    wGyroX: int
    wGyroY: int
    wGyroZ: int
    wAccelX: int
    wAccelY: int
    wAccelZ: int


class MockVDS4Gamepad:
    """Mock for vgamepad.VDS4Gamepad capturing extended report ctypes structures."""

    def __init__(self):
        self.reset_count = 0
        self.update_count = 0
        self.update_ex_count = 0
        self.last_report_snapshot: Optional[DS4ReportExSnapshot] = None
        self.history: List[DS4ReportExSnapshot] = []

    def reset(self):
        self.reset_count += 1

    def update(self):
        self.update_count += 1

    def update_extended_report(self, ex_report):
        self.update_ex_count += 1
        sub = ex_report.Report
        snapshot = DS4ReportExSnapshot(
            bThumbLX=int(sub.bThumbLX),
            bThumbLY=int(sub.bThumbLY),
            bThumbRX=int(sub.bThumbRX),
            bThumbRY=int(sub.bThumbRY),
            wButtons=int(sub.wButtons),
            bSpecial=int(sub.bSpecial),
            bTriggerL=int(sub.bTriggerL),
            bTriggerR=int(sub.bTriggerR),
            wTimestamp=int(sub.wTimestamp),
            bBatteryLvl=int(sub.bBatteryLvl),
            wGyroX=int(sub.wGyroX),
            wGyroY=int(sub.wGyroY),
            wGyroZ=int(sub.wGyroZ),
            wAccelX=int(sub.wAccelX),
            wAccelY=int(sub.wAccelY),
            wAccelZ=int(sub.wAccelZ),
        )
        self.last_report_snapshot = snapshot
        self.history.append(snapshot)


class MockVX360Gamepad:
    """Mock for vgamepad.VX360Gamepad tracking XInput button presses and analog values."""

    def __init__(self):
        self.reset_count = 0
        self.update_count = 0
        self.pressed_buttons: Set[int] = set()
        self.left_trigger: float = 0.0
        self.right_trigger: float = 0.0
        self.left_stick: Tuple[float, float] = (0.0, 0.0)
        self.right_stick: Tuple[float, float] = (0.0, 0.0)

    def reset(self):
        self.reset_count += 1
        self.pressed_buttons.clear()
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.left_stick = (0.0, 0.0)
        self.right_stick = (0.0, 0.0)

    def press_button(self, button):
        self.pressed_buttons.add(button)

    def left_trigger_float(self, value_float: float):
        self.left_trigger = float(value_float)

    def right_trigger_float(self, value_float: float):
        self.right_trigger = float(value_float)

    def left_joystick_float(self, x_value_float: float, y_value_float: float):
        self.left_stick = (float(x_value_float), float(y_value_float))

    def right_joystick_float(self, x_value_float: float, y_value_float: float):
        self.right_stick = (float(x_value_float), float(y_value_float))

    def update(self):
        self.update_count += 1


def make_state(**kwargs) -> ControllerState:
    """Helper to instantiate a ControllerState with sensible neutral defaults."""
    defaults = dict(
        timestamp=0,
        buttons_raw=0,
        a=False, b=False, x=False, y=False,
        plus=False, minus=False, home=False, capture=False,
        l=False, r=False, l_thumb=False, r_thumb=False,
        dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
        zl_analog=0.0, zr_analog=0.0,
        left_stick_x=0.0, left_stick_y=0.0,
        right_stick_x=0.0, right_stick_y=0.0,
        gl=False, gr=False,
        battery_percentage=100,
        battery_voltage=3.8,
        accel_x=0.0, accel_y=0.0, accel_z=0.0,
        gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        raw_accel_x=0, raw_accel_y=0, raw_accel_z=0,
        raw_gyro_x=0, raw_gyro_y=0, raw_gyro_z=0,
        motion_timestamp=0,
        imu_temperature=25.0,
        has_motion=False,
    )
    defaults.update(kwargs)
    return ControllerState(**defaults)


# ============================================================================
# 1. BaseVirtualController Polymorphic Contract Tests
# ============================================================================

class TestBaseVirtualControllerContract:
    """Verifies interface invariants, ABC enforcement, and PEP 562 lazy re-exports."""

    def test_base_virtual_controller_is_abc(self):
        assert issubclass(BaseVirtualController, abc.ABC)

    def test_base_virtual_controller_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseVirtualController()  # type: ignore

    def test_subclass_missing_methods_raises(self):
        class IncompleteController(BaseVirtualController):
            def connect(self) -> bool:
                return True
            # disconnect and update_from_state missing

        with pytest.raises(TypeError):
            IncompleteController()  # type: ignore

    def test_polymorphic_properties(self):
        xbox = VirtualXboxController()
        ds4 = VirtualDS4Controller()

        for c in (xbox, ds4):
            assert isinstance(c, BaseVirtualController)
            assert hasattr(c, "is_connected")
            assert hasattr(c, "layout")
            assert hasattr(c, "gl_mapping")
            assert hasattr(c, "gr_mapping")
            assert c.layout == "nintendo"
            assert c.gl_mapping == "L3"
            assert c.gr_mapping == "R3"
            assert c.is_connected is False

    def test_polymorphic_dispatch(self):
        def dispatch_test(controller: BaseVirtualController, state: ControllerState):
            controller.update(state)

        xbox = VirtualXboxController()
        ds4 = VirtualDS4Controller()
        st = make_state()

        # Calling update when disconnected should be safe no-op
        dispatch_test(xbox, st)
        dispatch_test(ds4, st)

    def test_virtual_gamepad_lazy_imports(self):
        assert virtual_gamepad.VirtualXboxController is VirtualXboxController
        assert virtual_gamepad.VirtualDS4Controller is VirtualDS4Controller
        assert "BaseVirtualController" in dir(virtual_gamepad)
        assert "VirtualXboxController" in dir(virtual_gamepad)
        assert "VirtualDS4Controller" in dir(virtual_gamepad)

        with pytest.raises(AttributeError):
            _ = virtual_gamepad.NonExistentClass

    def test_update_alias_calls_update_from_state(self):
        class ConcreteController(BaseVirtualController):
            def __init__(self):
                super().__init__()
                self.called_with = None

            def connect(self) -> bool:
                return True

            def disconnect(self) -> None:
                pass

            def update_from_state(self, state: ControllerState, layout: Optional[str] = None) -> None:
                self.called_with = state

        c = ConcreteController()
        st = make_state(a=True)
        c.update(st)
        assert c.called_with is st


# ============================================================================
# 2. Virtual Xbox Controller Unit Tests
# ============================================================================

class TestVirtualXboxControllerUnit:
    """Isolated unit tests for VirtualXboxController with MockVX360Gamepad."""

    def test_xbox_connect_disconnect_lifecycle(self):
        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_gp):
            assert c.connect() is True
            assert c.is_connected is True
            assert c._gamepad is mock_gp

            # Repeated connect is idempotent
            assert c.connect() is True

            c.disconnect()
            assert c.is_connected is False
            assert c._gamepad is None
            assert mock_gp.reset_count >= 1

    def test_xbox_driver_absence_graceful_return(self):
        c = VirtualXboxController()
        with mock.patch("vgamepad.VX360Gamepad", side_effect=RuntimeError("Driver missing")):
            assert c.connect() is False
            assert c.is_connected is False

    def test_xbox_null_or_disconnected_update_safe(self):
        c = VirtualXboxController()
        # Disconnected: safe no-op
        c.update_from_state(make_state())
        # Connected with mock, None state: safe no-op
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.update_from_state(None)
        assert mock_gp.update_count == 0

    def test_xbox_nintendo_layout_face_buttons(self):
        import vgamepad as vg
        btn = vg.XUSB_BUTTON

        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "nintendo"

        # B -> A
        c.update_from_state(make_state(b=True))
        assert btn.XUSB_GAMEPAD_A in mock_gp.pressed_buttons

        # A -> B
        c.update_from_state(make_state(a=True))
        assert btn.XUSB_GAMEPAD_B in mock_gp.pressed_buttons

        # Y -> X
        c.update_from_state(make_state(y=True))
        assert btn.XUSB_GAMEPAD_X in mock_gp.pressed_buttons

        # X -> Y
        c.update_from_state(make_state(x=True))
        assert btn.XUSB_GAMEPAD_Y in mock_gp.pressed_buttons

    def test_xbox_letter_layout_face_buttons(self):
        import vgamepad as vg
        btn = vg.XUSB_BUTTON

        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "xbox"

        c.update_from_state(make_state(a=True))
        assert btn.XUSB_GAMEPAD_A in mock_gp.pressed_buttons

        c.update_from_state(make_state(b=True))
        assert btn.XUSB_GAMEPAD_B in mock_gp.pressed_buttons

        c.update_from_state(make_state(x=True))
        assert btn.XUSB_GAMEPAD_X in mock_gp.pressed_buttons

        c.update_from_state(make_state(y=True))
        assert btn.XUSB_GAMEPAD_Y in mock_gp.pressed_buttons

    def test_xbox_bumpers_and_sticks(self):
        import vgamepad as vg
        btn = vg.XUSB_BUTTON

        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(
            l=True, r=True,
            plus=True, minus=True, home=True,
            l_thumb=True, r_thumb=True,
        )
        c.update_from_state(st)

        assert btn.XUSB_GAMEPAD_LEFT_SHOULDER in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_RIGHT_SHOULDER in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_START in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_BACK in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_GUIDE in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_LEFT_THUMB in mock_gp.pressed_buttons
        assert btn.XUSB_GAMEPAD_RIGHT_THUMB in mock_gp.pressed_buttons

    def test_xbox_dpad_buttons(self):
        import vgamepad as vg
        btn = vg.XUSB_BUTTON

        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        c.update_from_state(make_state(dpad_up=True))
        assert btn.XUSB_GAMEPAD_DPAD_UP in mock_gp.pressed_buttons

        c.update_from_state(make_state(dpad_down=True))
        assert btn.XUSB_GAMEPAD_DPAD_DOWN in mock_gp.pressed_buttons

        c.update_from_state(make_state(dpad_left=True))
        assert btn.XUSB_GAMEPAD_DPAD_LEFT in mock_gp.pressed_buttons

        c.update_from_state(make_state(dpad_right=True))
        assert btn.XUSB_GAMEPAD_DPAD_RIGHT in mock_gp.pressed_buttons

    def test_xbox_analog_triggers_and_sticks(self):
        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        c.update_from_state(make_state(
            zl_analog=0.75, zr_analog=1.0,
            left_stick_x=-0.5, left_stick_y=0.8,
            right_stick_x=0.25, right_stick_y=-0.9,
        ))

        assert abs(mock_gp.left_trigger - 0.75) < 1e-4
        assert abs(mock_gp.right_trigger - 1.0) < 1e-4
        assert mock_gp.left_stick == (-0.5, 0.8)
        assert mock_gp.right_stick == (0.25, -0.9)


# ============================================================================
# 3. Virtual DS4 Controller Lifecycle & Driver Fallback Tests
# ============================================================================

class TestVirtualDS4ControllerLifecycle:
    """Verifies connection lifecycle, idempotence, and driver fault tolerance."""

    def test_ds4_connect_success(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        with mock.patch("vgamepad.VDS4Gamepad", return_value=mock_gp):
            assert c.connect() is True
            assert c.is_connected is True
            assert c._gamepad is mock_gp

    def test_ds4_idempotent_connect(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        with mock.patch("vgamepad.VDS4Gamepad", return_value=mock_gp):
            assert c.connect() is True
            assert c.connect() is True
            assert c.is_connected is True

    def test_ds4_disconnect_resets_and_cleans(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        c.disconnect()
        assert c.is_connected is False
        assert c._gamepad is None
        assert mock_gp.reset_count >= 1

    def test_ds4_idempotent_disconnect(self):
        c = VirtualDS4Controller()
        c.disconnect()
        c.disconnect()
        assert c.is_connected is False

    def test_ds4_driver_exception_handling(self):
        c = VirtualDS4Controller()
        with mock.patch("vgamepad.VDS4Gamepad", side_effect=Exception("ViGEmBus not found")):
            assert c.connect() is False
            assert c.is_connected is False
            assert c._gamepad is None

    def test_ds4_update_when_disconnected_noop(self):
        c = VirtualDS4Controller()
        # Should execute silently without exception
        c.update_from_state(make_state(a=True))

    def test_ds4_update_with_none_state_noop(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.update_from_state(None)
        assert mock_gp.update_ex_count == 0


# ============================================================================
# 4. Virtual DS4 Button Mappings & Special Controls Tests
# ============================================================================

class TestVirtualDS4ButtonMappings:
    """Verifies Nintendo vs Xbox face layout, bumpers, clicks, and special buttons."""

    def test_ds4_nintendo_layout_face_buttons(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "nintendo"

        # Switch B -> DS4 Cross (0x0020)
        c.update_from_state(make_state(b=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0020 != 0
        assert mock_gp.last_report_snapshot.wButtons & 0x0040 == 0

        # Switch A -> DS4 Circle (0x0040)
        c.update_from_state(make_state(a=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0040 != 0

        # Switch Y -> DS4 Square (0x0010)
        c.update_from_state(make_state(y=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0010 != 0

        # Switch X -> DS4 Triangle (0x0080)
        c.update_from_state(make_state(x=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0080 != 0

    def test_ds4_xbox_layout_face_buttons(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "xbox"

        # Switch A -> DS4 Cross (0x0020)
        c.update_from_state(make_state(a=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0020 != 0

        # Switch B -> DS4 Circle (0x0040)
        c.update_from_state(make_state(b=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0040 != 0

        # Switch X -> DS4 Square (0x0010)
        c.update_from_state(make_state(x=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0010 != 0

        # Switch Y -> DS4 Triangle (0x0080)
        c.update_from_state(make_state(y=True))
        assert mock_gp.last_report_snapshot.wButtons & 0x0080 != 0

    def test_ds4_bumpers_and_clicks(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(l=True, r=True, l_thumb=True, r_thumb=True)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wButtons & 0x0100 != 0  # L1
        assert snap.wButtons & 0x0200 != 0  # R1
        assert snap.wButtons & 0x4000 != 0  # L3
        assert snap.wButtons & 0x8000 != 0  # R3

    def test_ds4_center_and_special_buttons(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(plus=True, minus=True, home=True, capture=True)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wButtons & 0x2000 != 0  # Options
        assert snap.wButtons & 0x1000 != 0  # Share
        assert snap.bSpecial & 0x01 != 0    # PS button
        assert snap.bSpecial & 0x02 != 0    # Touchpad click

    def test_ds4_triggers(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Below 0.5 threshold: analog byte set, digital bits not set
        c.update_from_state(make_state(zl_analog=0.4, zr_analog=0.5))
        snap = mock_gp.last_report_snapshot
        assert snap.bTriggerL == int(0.4 * 255)
        assert snap.bTriggerR == int(0.5 * 255)
        assert snap.wButtons & 0x0400 == 0
        assert snap.wButtons & 0x0800 == 0

        # Above 0.5 threshold: digital click bits set
        c.update_from_state(make_state(zl_analog=0.75, zr_analog=1.0))
        snap = mock_gp.last_report_snapshot
        assert snap.bTriggerL == int(0.75 * 255)
        assert snap.bTriggerR == 255
        assert snap.wButtons & 0x0400 != 0
        assert snap.wButtons & 0x0800 != 0

    def test_ds4_simultaneous_button_chords(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "nintendo"

        # Chord: A (Circle), B (Cross), L (L1), R (R1), Plus (Options)
        c.update_from_state(make_state(a=True, b=True, l=True, r=True, plus=True))
        snap = mock_gp.last_report_snapshot
        expected = 0x0040 | 0x0020 | 0x0100 | 0x0200 | 0x2000
        assert (snap.wButtons & ~0xF) == expected


# ============================================================================
# 5. Virtual DS4 D-Pad 8-Way Hat Direction Tests
# ============================================================================

class TestVirtualDS4Dpad:
    """Verifies 8-way directional hat encoding in lower 4 bits of wButtons."""

    @pytest.mark.parametrize(
        "up,down,left,right,expected_hat",
        [
            (False, False, False, False, 8),  # NONE
            (True, False, False, False, 0),   # NORTH
            (True, False, False, True, 1),    # NORTHEAST
            (False, False, False, True, 2),   # EAST
            (False, True, False, True, 3),    # SOUTHEAST
            (False, True, False, False, 4),   # SOUTH
            (False, True, True, False, 5),    # SOUTHWEST
            (False, False, True, False, 6),   # WEST
            (True, False, True, False, 7),    # NORTHWEST
            # Opposing input cancellation
            (True, True, False, False, 8),    # UP+DOWN -> NONE
            (False, False, True, True, 8),    # LEFT+RIGHT -> NONE
            (True, True, False, True, 2),     # UP+DOWN cancels -> EAST
            (True, True, True, True, 8),      # ALL 4 cancel -> NONE
        ],
    )
    def test_dpad_hat_directions(self, up, down, left, right, expected_hat):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(dpad_up=up, dpad_down=down, dpad_left=left, dpad_right=right)
        c.update_from_state(st)
        assert (mock_gp.last_report_snapshot.wButtons & 0xF) == expected_hat


# ============================================================================
# 6. Virtual DS4 Stick Axis Mapping & Sony Stick Y Inversion Tests
# ============================================================================

class TestVirtualDS4SticksAndInversion:
    """Verifies horizontal stick mapping and Sony DirectInput stick Y inversion."""

    def test_neutral_center(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        c.update_from_state(make_state(left_stick_x=0.0, left_stick_y=0.0, right_stick_x=0.0, right_stick_y=0.0))
        snap = mock_gp.last_report_snapshot

        # Center should be ~127 or 128
        assert abs(snap.bThumbLX - 127) <= 1
        assert abs(snap.bThumbLY - 127) <= 1
        assert abs(snap.bThumbRX - 127) <= 1
        assert abs(snap.bThumbRY - 127) <= 1

    def test_horizontal_sticks(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Left stick full left (-1.0 -> 0), Right stick full right (+1.0 -> 255)
        c.update_from_state(make_state(left_stick_x=-1.0, right_stick_x=1.0))
        snap = mock_gp.last_report_snapshot
        assert snap.bThumbLX == 0
        assert snap.bThumbRX == 255

    def test_sony_stick_y_inversion(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Full UP: +1.0 must map to 0 in Sony convention
        c.update_from_state(make_state(left_stick_y=1.0, right_stick_y=1.0))
        snap = mock_gp.last_report_snapshot
        assert snap.bThumbLY == 0
        assert snap.bThumbRY == 0

        # Full DOWN: -1.0 must map to 255 in Sony convention
        c.update_from_state(make_state(left_stick_y=-1.0, right_stick_y=-1.0))
        snap = mock_gp.last_report_snapshot
        assert snap.bThumbLY == 255
        assert snap.bThumbRY == 255

        # Half UP (+0.5): (1.0 - 0.5) * 127.5 = 63.75 -> 63 or 64
        c.update_from_state(make_state(left_stick_y=0.5))
        assert abs(mock_gp.last_report_snapshot.bThumbLY - 63) <= 1

        # Half DOWN (-0.5): (1.0 - (-0.5)) * 127.5 = 191.25 -> 191
        c.update_from_state(make_state(left_stick_y=-0.5))
        assert abs(mock_gp.last_report_snapshot.bThumbLY - 191) <= 1

    def test_stick_out_of_bounds_clamping(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Over-range Y +1.5 -> clamped to 0 (Up)
        c.update_from_state(make_state(left_stick_y=1.5, left_stick_x=-1.5))
        snap = mock_gp.last_report_snapshot
        assert snap.bThumbLY == 0
        assert snap.bThumbLX == 0

        # Under-range Y -1.5 -> clamped to 255 (Down)
        c.update_from_state(make_state(left_stick_y=-1.5, left_stick_x=1.5))
        snap = mock_gp.last_report_snapshot
        assert snap.bThumbLY == 255
        assert snap.bThumbLX == 255


# ============================================================================
# 7. Virtual DS4 Motion Packaging & Saturation Tests
# ============================================================================

class TestVirtualDS4MotionPackaging:
    """Verifies 6-axis IMU scaling, saturation clamping to [-32768, 32767], and telemetry."""

    def test_resting_flat_1g_gravity(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(accel_x=0.0, accel_y=0.0, accel_z=1.0)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelZ == 8192
        assert snap.wAccelX == 0
        assert snap.wAccelY == 0
        assert snap.wGyroX == 0
        assert snap.wGyroY == 0
        assert snap.wGyroZ == 0

    def test_dynamic_angular_rate(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(gyro_x=90.0, gyro_y=-45.0, gyro_z=180.0)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wGyroX == int(90.0 * 16.0)    # 1440
        assert snap.wGyroY == int(-45.0 * 16.0)   # -720
        assert snap.wGyroZ == int(180.0 * 16.0)   # 2880

    def test_raw_imu_values(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(raw_accel_x=1234, raw_accel_y=-5678, raw_gyro_z=9999)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 2468
        assert snap.wAccelY == -11356
        assert snap.wGyroZ == 9765

    def test_extreme_saturation_clamping(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Acceleration saturation (+15g and -15g)
        st = make_state(accel_x=15.0, accel_y=-15.0, gyro_x=5000.0, gyro_y=-5000.0)
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 32767
        assert snap.wAccelY == -32768
        assert snap.wGyroX == 32767
        assert snap.wGyroY == -32768

    def test_non_finite_float_handling(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(
            accel_x=float("inf"),
            accel_y=float("-inf"),
            gyro_x=float("nan"),
        )
        c.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 32767
        assert snap.wAccelY == -32768
        assert snap.wGyroX == 0

    def test_timestamp_rollover(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        st = make_state(motion_timestamp=70000)
        c.update_from_state(st)
        assert mock_gp.last_report_snapshot.wTimestamp == (70000 & 0xFFFF)

    def test_battery_level(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        c.update_from_state(make_state(battery_percentage=100))
        assert mock_gp.last_report_snapshot.bBatteryLvl == 10

        c.update_from_state(make_state(battery_percentage=45))
        assert mock_gp.last_report_snapshot.bBatteryLvl == 4

        c.update_from_state(make_state(battery_percentage=0))
        assert mock_gp.last_report_snapshot.bBatteryLvl == 0


# ============================================================================
# 8. Virtual DS4 Grip Paddle Remapping Tests
# ============================================================================

class TestVirtualDS4GripPaddles:
    """Verifies GL and GR grip paddle actions remapped to DS4 buttons/triggers."""

    def test_paddle_thumb_clicks(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.gl_mapping = "L3"
        c.gr_mapping = "R3"

        c.update_from_state(make_state(gl=True, gr=True))
        snap = mock_gp.last_report_snapshot
        assert snap.wButtons & 0x4000 != 0  # L3
        assert snap.wButtons & 0x8000 != 0  # R3

    def test_paddle_bumpers(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.gl_mapping = "LB"
        c.gr_mapping = "RB"

        c.update_from_state(make_state(gl=True, gr=True))
        snap = mock_gp.last_report_snapshot
        assert snap.wButtons & 0x0100 != 0  # L1
        assert snap.wButtons & 0x0200 != 0  # R1

    def test_paddle_triggers(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.gl_mapping = "LT"
        c.gr_mapping = "RT"

        c.update_from_state(make_state(gl=True, gr=True))
        snap = mock_gp.last_report_snapshot
        assert snap.bTriggerL == 255
        assert snap.bTriggerR == 255
        assert snap.wButtons & 0x0400 != 0  # L2 digital
        assert snap.wButtons & 0x0800 != 0  # R2 digital

    def test_paddle_face_buttons_nintendo_layout(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.layout = "nintendo"
        c.gl_mapping = "A"
        c.gr_mapping = "B"

        c.update_from_state(make_state(gl=True, gr=True))
        snap = mock_gp.last_report_snapshot
        assert snap.wButtons & 0x0040 != 0  # Circle (A in nintendo)
        assert snap.wButtons & 0x0020 != 0  # Cross (B in nintendo)

    def test_paddle_disabled(self):
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True
        c.gl_mapping = "Disabled"
        c.gr_mapping = "Disabled"

        c.update_from_state(make_state(gl=True, gr=True))
        snap = mock_gp.last_report_snapshot
        assert snap.wButtons == 0x0008  # Neutral D-pad only


# ============================================================================
# 9. Live ViGEmBus Integration Tests (Guarded by Driver Check)
# ============================================================================

class TestVirtualControllersLiveViGEm:
    """Live ViGEmBus driver tests running conditionally when host driver is present."""

    @pytest.mark.skipif(not check_vigem_driver_available(), reason="ViGEmBus driver not installed")
    def test_live_xbox_lifecycle_and_update(self):
        c = VirtualXboxController()
        assert c.connect() is True
        try:
            st = make_state(a=True, left_stick_x=0.5)
            c.update_from_state(st)
        finally:
            c.disconnect()

    @pytest.mark.skipif(not check_vigem_driver_available(), reason="ViGEmBus driver not installed")
    def test_live_ds4_lifecycle_and_motion(self):
        c = VirtualDS4Controller()
        assert c.connect() is True
        try:
            st = make_state(b=True, accel_z=1.0, gyro_x=45.0)
            c.update_from_state(st)
        finally:
            c.disconnect()


# ============================================================================
# 10. Regression Tests for Milestone 3 Remediation
# ============================================================================

def test_ds4_with_parsed_controller_state_1g_gravity_waccelz_8192():
    """Regression Test 1: Real parsed Switch 2 report with 1.0g resting gravity produces 8192 LSB in DS4 wAccelZ."""
    from report_parser import parse_input_report
    from tests.test_e2e_motion import build_switch2_usb_report

    # Switch 2 USB input report with resting gravity on Z (raw 4095 at +/-8g)
    raw_packet = build_switch2_usb_report(ax_raw=0, ay_raw=0, az_raw=4095, gx_raw=0, gy_raw=0, gz_raw=0)
    state = parse_input_report(raw_packet)
    assert state is not None
    assert state.has_motion is True
    assert abs(state.accel_z - 1.0) < 0.02
    assert state.raw_accel_z == 4095

    controller = VirtualDS4Controller()
    mock_gp = MockVDS4Gamepad()
    controller._gamepad = mock_gp
    controller.is_connected = True

    controller.update_from_state(state)
    snap = mock_gp.last_report_snapshot
    assert snap is not None

    # DS4 requires 8192 LSB/g for 1.0g:
    assert abs(snap.wAccelZ - 8192) <= 100, (
        f"DS4 wAccelZ was {snap.wAccelZ}, expected ~8192 LSB (1.0g). "
        f"Raw count (4095) must not bypass normalized physical scaling."
    )
    assert abs(snap.wAccelX) <= 100
    assert abs(snap.wAccelY) <= 100


def test_ds4_gl_mapped_to_rt_actuates_right_trigger():
    """Regression Test 2: GL mapped to RT actuates Right Trigger analog (255) and digital button without contaminating Left Trigger."""
    controller = VirtualDS4Controller()
    mock_gp = MockVDS4Gamepad()
    controller._gamepad = mock_gp
    controller.is_connected = True

    controller.gl_mapping = "RT"
    st = make_state(gl=True, zl_analog=0.0, zr_analog=0.0)
    controller.update_from_state(st)
    snap = mock_gp.last_report_snapshot

    assert snap is not None
    assert snap.bTriggerR == 255, f"Expected bTriggerR == 255, got {snap.bTriggerR}"
    assert snap.bTriggerL == 0, f"Expected bTriggerL == 0, got {snap.bTriggerL}"
    assert bool(snap.wButtons & 0x0800), "DS4_BUTTON_TRIGGER_RIGHT bit missing from wButtons"
    assert not bool(snap.wButtons & 0x0400), "DS4_BUTTON_TRIGGER_LEFT bit must not be set"


def test_ds4_gr_mapped_to_lt_actuates_left_trigger():
    """Regression Test 3: GR mapped to LT actuates Left Trigger analog (255) and digital button without contaminating Right Trigger."""
    controller = VirtualDS4Controller()
    mock_gp = MockVDS4Gamepad()
    controller._gamepad = mock_gp
    controller.is_connected = True

    controller.gr_mapping = "LT"
    st = make_state(gr=True, zl_analog=0.0, zr_analog=0.0)
    controller.update_from_state(st)
    snap = mock_gp.last_report_snapshot

    assert snap is not None
    assert snap.bTriggerL == 255, f"Expected bTriggerL == 255, got {snap.bTriggerL}"
    assert snap.bTriggerR == 0, f"Expected bTriggerR == 0, got {snap.bTriggerR}"
    assert bool(snap.wButtons & 0x0400), "DS4_BUTTON_TRIGGER_LEFT bit missing from wButtons"
    assert not bool(snap.wButtons & 0x0800), "DS4_BUTTON_TRIGGER_RIGHT bit must not be set"


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_virtual_xbox_controller_non_finite_values_do_not_crash(bad_val):
    """Regression Test 4: VirtualXboxController safely sanitizes non-finite stick and trigger floats without crashing."""
    controller = VirtualXboxController()
    mock_gp = MockVX360Gamepad()
    controller._gamepad = mock_gp
    controller.is_connected = True

    # 1. Non-finite values on analog sticks
    st_sticks = make_state(
        left_stick_x=bad_val,
        left_stick_y=bad_val,
        right_stick_x=bad_val,
        right_stick_y=bad_val,
    )
    # Must execute cleanly without raising ValueError or OverflowError
    controller.update_from_state(st_sticks)
    assert mock_gp.update_count == 1
    assert mock_gp.left_stick == (0.0, 0.0)
    assert mock_gp.right_stick == (0.0, 0.0)

    # 2. Non-finite values on analog triggers
    st_triggers = make_state(
        zl_analog=bad_val,
        zr_analog=bad_val,
    )
    controller.update_from_state(st_triggers)
    assert mock_gp.update_count == 2
    assert mock_gp.left_trigger == 0.0
    assert mock_gp.right_trigger == 0.0


def test_ds4_stick_neutral_centering_directinput_128():
    """Regression Test 5: Neutral stick inputs (0.0) map cleanly to DirectInput standard 128, not truncated 127."""
    from virtual_ds4 import _norm_stick_to_byte, _norm_stick_y_inverted_to_byte

    # Direct helper function checks
    assert _norm_stick_to_byte(0.0) == 128
    assert _norm_stick_y_inverted_to_byte(0.0) == 128

    # Controller snapshot checks
    controller = VirtualDS4Controller()
    mock_gp = MockVDS4Gamepad()
    controller._gamepad = mock_gp
    controller.is_connected = True

    st_neutral = make_state(left_stick_x=0.0, left_stick_y=0.0, right_stick_x=0.0, right_stick_y=0.0)
    controller.update_from_state(st_neutral)
    snap = mock_gp.last_report_snapshot

    assert snap is not None
    assert snap.bThumbLX == 128
    assert snap.bThumbLY == 128
    assert snap.bThumbRX == 128
    assert snap.bThumbRY == 128

