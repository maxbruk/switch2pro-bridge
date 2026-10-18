"""Parser for Nintendo Switch 2 and Switch Pro Controller input reports."""

import math
from dataclasses import dataclass
from typing import Tuple, Optional
from constants import (
    REPORT_ID_COMMON,
    REPORT_ID_PRO2,
    REPORT_ID_GAMECUBE,
    BTN_Y, BTN_X, BTN_B, BTN_A, BTN_R, BTN_ZR,
    BTN_MINUS, BTN_PLUS, BTN_R_THUMB, BTN_L_THUMB,
    BTN_HOME, BTN_CAPTURE, BTN_C,
    BTN_DPAD_DOWN, BTN_DPAD_UP, BTN_DPAD_RIGHT, BTN_DPAD_LEFT,
    BTN_L, BTN_ZL, BTN_GL, BTN_GR,
)



@dataclass
class ControllerState:
    timestamp: int
    buttons_raw: int
    
    # Standard digital buttons
    a: bool
    b: bool
    x: bool
    y: bool
    plus: bool
    minus: bool
    home: bool
    capture: bool
    
    # Bumpers and stick clicks
    l: bool
    r: bool
    l_thumb: bool
    r_thumb: bool
    
    # D-Pad
    dpad_up: bool
    dpad_down: bool
    dpad_left: bool
    dpad_right: bool
    
    # Triggers (normalized 0.0 to 1.0)
    zl_analog: float
    zr_analog: float
    
    # Analog Sticks (normalized -1.0 to 1.0)
    left_stick_x: float
    left_stick_y: float
    right_stick_x: float
    right_stick_y: float
    
    # Back Grip Buttons (Switch 2 Pro Controller)
    gl: bool = False
    gr: bool = False

    # Diagnostics
    battery_percentage: int = 100
    battery_voltage: float = 3.8



def decode_uint12_sticks(data: bytes) -> Tuple[int, int]:
    """Decode 3 bytes containing packed 12-bit (x, y) analog stick values."""
    if len(data) < 3:
        return 2048, 2048
    val = int.from_bytes(data[:3], byteorder="little")
    x = val & 0xFFF
    y = (val >> 12) & 0xFFF
    return x, y


def normalize_stick_axis(raw_val: int, center: int = 2048, max_delta: int = 1600) -> float:
    """Normalize 12-bit stick position (0..4095) to [-1.0, 1.0]."""
    diff = raw_val - center
    clamped = max(-max_delta, min(diff, max_delta))
    return clamped / max_delta


def apply_radial_deadzone(x: float, y: float, deadzone: float = 0.05) -> Tuple[float, float]:
    """Apply smooth radial deadzone to analog stick coordinates."""
    magnitude = math.sqrt(x * x + y * y)
    if magnitude <= deadzone:
        return 0.0, 0.0
    # Rescale from deadzone..1.0 to 0.0..1.0
    scale = (magnitude - deadzone) / (1.0 - deadzone)
    norm_x = (x / magnitude) * scale
    norm_y = (y / magnitude) * scale
    return max(-1.0, min(1.0, norm_x)), max(-1.0, min(1.0, norm_y))


def _parse_power_on_09_report(raw_bytes: bytes, deadzone: float) -> ControllerState:
    """Parse legacy USB power-on 0x09 report (12-24 bytes)."""
    b0 = raw_bytes[3]
    b1 = raw_bytes[4]
    b2 = raw_bytes[5]
    
    buttons = 0
    if b0 & 0x01: buttons |= BTN_Y
    if b0 & 0x02: buttons |= BTN_X
    if b0 & 0x04: buttons |= BTN_B
    if b0 & 0x08: buttons |= BTN_A
    if b0 & 0x40: buttons |= BTN_R
    if b0 & 0x80: buttons |= BTN_ZR
    if b1 & 0x01: buttons |= BTN_MINUS
    if b1 & 0x02: buttons |= BTN_PLUS
    if b1 & 0x04: buttons |= BTN_R_THUMB
    if b1 & 0x08: buttons |= BTN_L_THUMB
    if b1 & 0x10: buttons |= BTN_HOME
    if b1 & 0x20: buttons |= BTN_CAPTURE
    if b2 & 0x01: buttons |= BTN_DPAD_DOWN
    if b2 & 0x02: buttons |= BTN_DPAD_UP
    if b2 & 0x04: buttons |= BTN_DPAD_RIGHT
    if b2 & 0x08: buttons |= BTN_DPAD_LEFT
    if b2 & 0x40: buttons |= BTN_L
    if b2 & 0x80: buttons |= BTN_ZL

    lx, ly = decode_uint12_sticks(raw_bytes[6:9])
    rx, ry = decode_uint12_sticks(raw_bytes[9:12])
    
    norm_lx, norm_ly = apply_radial_deadzone(
        normalize_stick_axis(lx), normalize_stick_axis(ly), deadzone
    )
    norm_rx, norm_ry = apply_radial_deadzone(
        normalize_stick_axis(rx), normalize_stick_axis(ry), deadzone
    )
    
    return ControllerState(
        timestamp=raw_bytes[1],
        buttons_raw=buttons,
        a=bool(buttons & BTN_A),
        b=bool(buttons & BTN_B),
        x=bool(buttons & BTN_X),
        y=bool(buttons & BTN_Y),
        plus=bool(buttons & BTN_PLUS),
        minus=bool(buttons & BTN_MINUS),
        home=bool(buttons & BTN_HOME),
        capture=bool(buttons & BTN_CAPTURE),
        l=bool(buttons & BTN_L),
        r=bool(buttons & BTN_R),
        l_thumb=bool(buttons & BTN_L_THUMB),
        r_thumb=bool(buttons & BTN_R_THUMB),
        dpad_up=bool(buttons & BTN_DPAD_UP),
        dpad_down=bool(buttons & BTN_DPAD_DOWN),
        dpad_left=bool(buttons & BTN_DPAD_LEFT),
        dpad_right=bool(buttons & BTN_DPAD_RIGHT),
        zl_analog=1.0 if (buttons & BTN_ZL) else 0.0,
        zr_analog=1.0 if (buttons & BTN_ZR) else 0.0,
        left_stick_x=norm_lx,
        left_stick_y=norm_ly,
        right_stick_x=norm_rx,
        right_stick_y=norm_ry,
        gl=False,
        gr=False,
        battery_percentage=80,
        battery_voltage=3.8,
    )


def parse_input_report(
    raw_bytes: bytes,
    deadzone: float = 0.05,
    has_report_id: Optional[bool] = None,
) -> Optional[ControllerState]:
    """Parse raw bytes from USB HID or BLE into ControllerState.
    
    Supports:
    - Report 0x05 (Switch 2 common report, 64 bytes with leading 0x05 on USB)
    - Report 0x09 (Pro 2 power-on raw report, 12-24 bytes on USB)
    - Payload without leading report ID (60-64 byte BLE GATT notifications)
    """
    if not raw_bytes or len(raw_bytes) < 12:
        return None

    if has_report_id is False:
        # Pure BLE GATT payload: byte 0 is part of timestamp counter, NOT a report ID
        data = raw_bytes
    elif has_report_id is True:
        # Explicit USB HID report with leading report ID
        report_id = raw_bytes[0]
        if report_id == REPORT_ID_COMMON and len(raw_bytes) >= 64:
            data = raw_bytes[1:]
        elif report_id == REPORT_ID_PRO2 and 12 <= len(raw_bytes) <= 24:
            return _parse_power_on_09_report(raw_bytes, deadzone)
        else:
            data = raw_bytes[1:] if len(raw_bytes) >= 64 else raw_bytes
    else:
        # Auto-detect:
        # 0x09 report is strictly a small 12-24 byte USB report (never 60+ bytes)
        report_id = raw_bytes[0]
        if report_id == REPORT_ID_PRO2 and 12 <= len(raw_bytes) <= 24:
            return _parse_power_on_09_report(raw_bytes, deadzone)
        elif report_id == REPORT_ID_COMMON and len(raw_bytes) == 64:
            data = raw_bytes[1:]
        else:
            data = raw_bytes

    # Standard Report 0x05 / BLE input payload layout
    # data[0:4]: timestamp counter
    # data[4:8]: buttons (u32 little-endian)
    # data[10:13]: left stick (12-bit x, y)
    # data[13:16]: right stick (12-bit x, y)
    # data[31:33]: battery millivolts
    if len(data) < 16:
        return None

    timestamp = int.from_bytes(data[0:4], "little")
    buttons = int.from_bytes(data[4:8], "little")

    lx_raw, ly_raw = decode_uint12_sticks(data[10:13])
    rx_raw, ry_raw = decode_uint12_sticks(data[13:16])

    norm_lx, norm_ly = apply_radial_deadzone(
        normalize_stick_axis(lx_raw), normalize_stick_axis(ly_raw), deadzone
    )
    norm_rx, norm_ry = apply_radial_deadzone(
        normalize_stick_axis(rx_raw), normalize_stick_axis(ry_raw), deadzone
    )

    # Battery
    voltage = 3.8
    percentage = 85
    if len(data) >= 33:
        mv = int.from_bytes(data[31:33], "little")
        if 2800 <= mv <= 4300:
            voltage = mv / 1000.0
            # 3.4V ~ 0%, 4.2V ~ 100%
            percentage = max(0, min(100, int((voltage - 3.4) / (4.2 - 3.4) * 100)))

    zl_pressed = bool(buttons & BTN_ZL)
    zr_pressed = bool(buttons & BTN_ZR)

    return ControllerState(
        timestamp=timestamp,
        buttons_raw=buttons,
        a=bool(buttons & BTN_A),
        b=bool(buttons & BTN_B),
        x=bool(buttons & BTN_X),
        y=bool(buttons & BTN_Y),
        plus=bool(buttons & BTN_PLUS),
        minus=bool(buttons & BTN_MINUS),
        home=bool(buttons & BTN_HOME),
        capture=bool(buttons & BTN_CAPTURE),
        l=bool(buttons & BTN_L),
        r=bool(buttons & BTN_R),
        l_thumb=bool(buttons & BTN_L_THUMB),
        r_thumb=bool(buttons & BTN_R_THUMB),
        dpad_up=bool(buttons & BTN_DPAD_UP),
        dpad_down=bool(buttons & BTN_DPAD_DOWN),
        dpad_left=bool(buttons & BTN_DPAD_LEFT),
        dpad_right=bool(buttons & BTN_DPAD_RIGHT),
        zl_analog=1.0 if zl_pressed else 0.0,
        zr_analog=1.0 if zr_pressed else 0.0,
        left_stick_x=norm_lx,
        left_stick_y=norm_ly,
        right_stick_x=norm_rx,
        right_stick_y=norm_ry,
        gl=bool(buttons & BTN_GL),
        gr=bool(buttons & BTN_GR),
        battery_percentage=percentage,
        battery_voltage=voltage,
    )

