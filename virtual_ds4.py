"""Virtual DualShock 4 controller emulation with 6-axis motion via ViGEmBus."""

import logging
import math
import sys
import threading
from typing import Optional, Tuple

from report_parser import ControllerState
from virtual_gamepad import BaseVirtualController

logger = logging.getLogger(__name__)

try:
    import vgamepad as vg
    import vgamepad.win.vigem_commons as vcom
except ImportError:
    vg = None
    vcom = None


def _clamp_int16(val: float | int) -> int:
    """Clamp value to signed 16-bit integer range [-32768, 32767]."""
    if isinstance(val, float):
        if not math.isfinite(val):
            return 32767 if val > 0 else (-32768 if val < 0 else 0)
    return max(-32768, min(32767, int(round(val))))


def _accel_to_int16(raw: int, g_val: float) -> int:
    """Convert accelerometer reading to DS4 signed 16-bit integer (8192 LSB/g).

    Prioritizes normalized physical value (g_val in g) scaled to DS4 sensitivity (8192 LSB/g).
    If g_val is 0.0 but raw is non-zero (synthetic test input), converts raw Switch 2 counts
    (4096 LSB/g at +/-8g FSR) to DS4 scale (8192 LSB/g) via 2x multiplier.
    """
    if g_val is None or not isinstance(g_val, (int, float)):
        g_val = 0.0
    if not math.isfinite(g_val):
        return _clamp_int16(g_val)
    if g_val != 0.0:
        return _clamp_int16(g_val * 8192.0)
    if raw is not None and isinstance(raw, (int, float)) and raw != 0:
        return _clamp_int16(raw * 2)
    return 0


def _gyro_to_int16(raw: int, dps_val: float) -> int:
    """Convert gyroscope reading to DS4 signed 16-bit integer (16.0 LSB/(deg/s)).

    Prioritizes normalized physical value (dps_val in deg/s) scaled to DS4 sensitivity (16.0 LSB/dps).
    If dps_val is 0.0 but raw is non-zero (synthetic test input), converts raw Switch 2 counts
    (16.384 LSB/dps at +/-2000 dps FSR) to DS4 scale (16.0 LSB/dps) via (16.0 / 16.384) factor.
    """
    if dps_val is None or not isinstance(dps_val, (int, float)):
        dps_val = 0.0
    if not math.isfinite(dps_val):
        return _clamp_int16(dps_val)
    if dps_val != 0.0:
        return _clamp_int16(dps_val * 16.0)
    if raw is not None and isinstance(raw, (int, float)) and raw != 0:
        return _clamp_int16(raw * (16.0 / 16.384))
    return 0


def _norm_stick_to_byte(norm_val: float) -> int:
    """Map normalized [-1.0, 1.0] axis to [0, 255] byte with neutral center at 128."""
    if not (isinstance(norm_val, (int, float)) and math.isfinite(norm_val)):
        return 128
    return max(0, min(255, int(round((norm_val + 1.0) * 127.5))))


def _norm_stick_y_inverted_to_byte(norm_y: float) -> int:
    """Sony DirectInput stick Y inversion: +1.0 (UP) -> 0, -1.0 (DOWN) -> 255 with neutral center at 128."""
    if not (isinstance(norm_y, (int, float)) and math.isfinite(norm_y)):
        return 128
    return max(0, min(255, int(round((1.0 - norm_y) * 127.5))))


def _norm_trigger_to_byte(norm_val: float) -> int:
    """Map normalized [0.0, 1.0] trigger value to [0, 255] byte."""
    if not (isinstance(norm_val, (int, float)) and math.isfinite(norm_val)):
        return 0
    return max(0, min(255, int(norm_val * 255.0)))


def _get_dpad_direction(up: bool, down: bool, left: bool, right: bool) -> int:
    """Calculate DS4 8-way directional hat value (0..8) with opposing input cancellation."""
    if up and down:
        up = down = False
    if left and right:
        left = right = False

    if up and right:
        return 0x1  # NORTHEAST
    elif down and right:
        return 0x3  # SOUTHEAST
    elif down and left:
        return 0x5  # SOUTHWEST
    elif up and left:
        return 0x7  # NORTHWEST
    elif up:
        return 0x0  # NORTH
    elif right:
        return 0x2  # EAST
    elif down:
        return 0x4  # SOUTH
    elif left:
        return 0x6  # WEST
    else:
        return 0x8  # NONE


class VirtualDS4Controller(BaseVirtualController):
    """Manages an emulated DualShock 4 controller with 6-axis motion via ViGEmBus."""

    def __init__(self):
        super().__init__()
        self._gamepad = None

    def connect(self) -> bool:
        """Create virtual DualShock 4 gamepad."""
        global vg, vcom
        with self._lock:
            if self._gamepad is not None:
                return True
            try:
                if vg is None:
                    import vgamepad as vg_mod
                    import vgamepad.win.vigem_commons as vcom_mod
                    vg = vg_mod
                    vcom = vcom_mod
                self._gamepad = vg.VDS4Gamepad()
                self._register_rumble_notification(self._gamepad)
                self.is_connected = True
                logger.info("Virtual DualShock 4 controller spawned successfully.")
                return True
            except Exception as e:
                logger.error("Failed to spawn virtual DualShock 4 controller: %s", e)
                self._gamepad = None
                self.is_connected = False
                return False

    def disconnect(self):
        """Destroy virtual DualShock 4 gamepad and reset inputs."""
        with self._lock:
            if self._gamepad is not None:
                self._unregister_rumble_notification(self._gamepad)
                try:
                    self._gamepad.reset()
                    self._gamepad.update()
                except Exception:
                    pass
                self._gamepad = None
            self.is_connected = False
            logger.info("Virtual DualShock 4 controller disconnected.")

    def update_from_state(self, state: ControllerState, layout: Optional[str] = None):
        """Update virtual DS4 inputs and 6-axis motion from ControllerState."""
        global vg, vcom
        if state is None:
            return

        with self._lock:
            if self._gamepad is None or not self.is_connected:
                return

            try:
                if vcom is None or vg is None:
                    import vgamepad as vg_mod
                    import vgamepad.win.vigem_commons as vcom_mod
                    vg = vg_mod
                    vcom = vcom_mod

                rep = vcom.DS4_REPORT_EX()
                vcom.DS4_REPORT_INIT(rep.Report)
                sub = rep.Report

                # 1. Analog Sticks with Sony Y-Inversion
                sub.bThumbLX = _norm_stick_to_byte(state.left_stick_x)
                sub.bThumbLY = _norm_stick_y_inverted_to_byte(state.left_stick_y)
                sub.bThumbRX = _norm_stick_to_byte(state.right_stick_x)
                sub.bThumbRY = _norm_stick_y_inverted_to_byte(state.right_stick_y)

                # 2. Analog Triggers
                sub.bTriggerL = _norm_trigger_to_byte(state.zl_analog)
                sub.bTriggerR = _norm_trigger_to_byte(state.zr_analog)

                # 3. Digital Buttons & Layout Mapping
                active_layout = layout or self.layout
                btn = vg.DS4_BUTTONS
                btn_mask = 0

                if active_layout == "nintendo":
                    # Physical Position Match:
                    # Switch B (Bottom) -> DS4 Cross (Bottom)
                    # Switch A (Right)  -> DS4 Circle (Right)
                    # Switch Y (Left)   -> DS4 Square (Left)
                    # Switch X (Top)    -> DS4 Triangle (Top)
                    if state.b: btn_mask |= btn.DS4_BUTTON_CROSS
                    if state.a: btn_mask |= btn.DS4_BUTTON_CIRCLE
                    if state.y: btn_mask |= btn.DS4_BUTTON_SQUARE
                    if state.x: btn_mask |= btn.DS4_BUTTON_TRIANGLE
                else:
                    # Xbox Letter Match:
                    # Switch A (Xbox A = Bottom) -> DS4 Cross
                    # Switch B (Xbox B = Right)  -> DS4 Circle
                    # Switch X (Xbox X = Left)   -> DS4 Square
                    # Switch Y (Xbox Y = Top)    -> DS4 Triangle
                    if state.a: btn_mask |= btn.DS4_BUTTON_CROSS
                    if state.b: btn_mask |= btn.DS4_BUTTON_CIRCLE
                    if state.x: btn_mask |= btn.DS4_BUTTON_SQUARE
                    if state.y: btn_mask |= btn.DS4_BUTTON_TRIANGLE

                # Bumpers, stick clicks, and center buttons
                if state.l: btn_mask |= btn.DS4_BUTTON_SHOULDER_LEFT
                if state.r: btn_mask |= btn.DS4_BUTTON_SHOULDER_RIGHT
                if state.plus: btn_mask |= btn.DS4_BUTTON_OPTIONS
                if state.minus: btn_mask |= btn.DS4_BUTTON_SHARE
                if state.l_thumb: btn_mask |= btn.DS4_BUTTON_THUMB_LEFT
                if state.r_thumb: btn_mask |= btn.DS4_BUTTON_THUMB_RIGHT

                # Digital trigger clicks
                if state.zl_analog > 0.5: btn_mask |= btn.DS4_BUTTON_TRIGGER_LEFT
                if state.zr_analog > 0.5: btn_mask |= btn.DS4_BUTTON_TRIGGER_RIGHT

                # 4. Special Buttons (PS & Touchpad)
                special_mask = 0
                sp = vg.DS4_SPECIAL_BUTTONS
                if state.home: special_mask |= sp.DS4_SPECIAL_BUTTON_PS
                if state.capture: special_mask |= sp.DS4_SPECIAL_BUTTON_TOUCHPAD
                sub.bSpecial = special_mask

                # 5. Grip Paddle Mappings (GL & GR)
                if state.gl:
                    btn_mask, sub.bTriggerL, sub.bTriggerR = self._apply_paddle_action(
                        self.gl_mapping, btn_mask, sub.bTriggerL, sub.bTriggerR, active_layout, btn
                    )
                if state.gr:
                    btn_mask, sub.bTriggerL, sub.bTriggerR = self._apply_paddle_action(
                        self.gr_mapping, btn_mask, sub.bTriggerL, sub.bTriggerR, active_layout, btn
                    )

                # 6. D-Pad Directional Hat (lower 4 bits of wButtons)
                dpad_val = _get_dpad_direction(state.dpad_up, state.dpad_down, state.dpad_left, state.dpad_right)
                sub.wButtons = (btn_mask & ~0xF) | (dpad_val & 0xF)

                # 7. 6-Axis Motion Data
                sub.wGyroX = _gyro_to_int16(getattr(state, "raw_gyro_x", 0), getattr(state, "gyro_x", 0.0))
                sub.wGyroY = _gyro_to_int16(getattr(state, "raw_gyro_y", 0), getattr(state, "gyro_y", 0.0))
                sub.wGyroZ = _gyro_to_int16(getattr(state, "raw_gyro_z", 0), getattr(state, "gyro_z", 0.0))
                sub.wAccelX = _accel_to_int16(getattr(state, "raw_accel_x", 0), getattr(state, "accel_x", 0.0))
                sub.wAccelY = _accel_to_int16(getattr(state, "raw_accel_y", 0), getattr(state, "accel_y", 0.0))
                sub.wAccelZ = _accel_to_int16(getattr(state, "raw_accel_z", 0), getattr(state, "accel_z", 0.0))

                # 8. Microsecond Timestamp & Battery
                ts = getattr(state, "motion_timestamp", 0) or (getattr(state, "timestamp", 0) & 0xFFFF)
                sub.wTimestamp = int(ts) & 0xFFFF
                sub.bBatteryLvl = max(0, min(10, getattr(state, "battery_percentage", 100) // 10))

                # 9. Send Extended Report
                self._gamepad.update_extended_report(rep)

            except Exception as e:
                logger.debug("Error committing DS4 gamepad frame: %s", e)

    def _apply_paddle_action(
        self, mapping: str, btn_mask: int, trig_l: int, trig_r: int, layout: str, btn
    ) -> Tuple[int, int, int]:
        """Map GL/GR paddle to configured DS4 action."""
        if mapping == "L3":
            btn_mask |= btn.DS4_BUTTON_THUMB_LEFT
        elif mapping == "R3":
            btn_mask |= btn.DS4_BUTTON_THUMB_RIGHT
        elif mapping == "LB":
            btn_mask |= btn.DS4_BUTTON_SHOULDER_LEFT
        elif mapping == "RB":
            btn_mask |= btn.DS4_BUTTON_SHOULDER_RIGHT
        elif mapping == "A":
            btn_mask |= (btn.DS4_BUTTON_CIRCLE if layout == "nintendo" else btn.DS4_BUTTON_CROSS)
        elif mapping == "B":
            btn_mask |= (btn.DS4_BUTTON_CROSS if layout == "nintendo" else btn.DS4_BUTTON_CIRCLE)
        elif mapping == "X":
            btn_mask |= (btn.DS4_BUTTON_TRIANGLE if layout == "nintendo" else btn.DS4_BUTTON_SQUARE)
        elif mapping == "Y":
            btn_mask |= (btn.DS4_BUTTON_SQUARE if layout == "nintendo" else btn.DS4_BUTTON_TRIANGLE)
        elif mapping in ("LT", "ZL"):
            trig_l = 255
            btn_mask |= btn.DS4_BUTTON_TRIGGER_LEFT
        elif mapping in ("RT", "ZR"):
            trig_r = 255
            btn_mask |= btn.DS4_BUTTON_TRIGGER_RIGHT
        return btn_mask, trig_l, trig_r


__all__ = ["VirtualDS4Controller"]
