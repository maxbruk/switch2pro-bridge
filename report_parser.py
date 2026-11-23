"""Parser for Nintendo Switch 2 and Switch Pro Controller input reports."""

import math
import struct
from dataclasses import dataclass
from typing import Tuple, Optional
from constants import (
    REPORT_ID_COMMON,
    REPORT_ID_PRO2,
    REPORT_ID_GAMECUBE,
    OUTPUT_REPORT_ID_PRO2,
    RUMBLE_MAX_AMPLITUDE,
    RUMBLE_HIGH_FREQ,
    RUMBLE_LOW_FREQ,
    BTN_Y, BTN_X, BTN_B, BTN_A, BTN_R, BTN_ZR,
    BTN_MINUS, BTN_PLUS, BTN_R_THUMB, BTN_L_THUMB,
    BTN_HOME, BTN_CAPTURE, BTN_C,
    BTN_DPAD_DOWN, BTN_DPAD_UP, BTN_DPAD_RIGHT, BTN_DPAD_LEFT,
    BTN_L, BTN_ZL, BTN_GL, BTN_GR,
)

# IMU Motion Scale Factors (ICM-42670-P: +/-8G FSR, +/-2000 dps FSR)
ACCEL_SCALE_G: float = 8.0 / 32767.0
GYRO_SCALE_DPS: float = 2000.0 / 32767.0
IMU_TEMP_SCALE: float = 1.0 / 126.9
IMU_TEMP_OFFSET: float = 25.0



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

    # 6-Axis Motion / IMU Sensors (Switch 2 Pro Controller ICM-42670-P)
    accel_x: float = 0.0          # in g (1.0 = 1g = 9.80665 m/s^2)
    accel_y: float = 0.0          # in g
    accel_z: float = 0.0          # in g
    gyro_x: float = 0.0           # in deg/s (degrees per second)
    gyro_y: float = 0.0           # in deg/s
    gyro_z: float = 0.0           # in deg/s
    raw_accel_x: int = 0          # signed 16-bit integer (-32768..32767)
    raw_accel_y: int = 0
    raw_accel_z: int = 0
    raw_gyro_x: int = 0           # signed 16-bit integer (-32768..32767)
    raw_gyro_y: int = 0
    raw_gyro_z: int = 0
    motion_timestamp: int = 0     # uint32 microsecond counter
    imu_temperature: float = 25.0 # Celsius
    has_motion: bool = False      # True when packet >= 60 bytes



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
    if max_delta <= 0:
        return 0.0
    diff = raw_val - center
    clamped = max(-max_delta, min(diff, max_delta))
    return clamped / max_delta


def apply_radial_deadzone(x: float, y: float, deadzone: float = 0.05) -> Tuple[float, float]:
    """Apply smooth radial deadzone to analog stick coordinates."""
    if deadzone >= 1.0:
        return 0.0, 0.0
    deadzone = max(0.0, deadzone)
    magnitude = math.sqrt(x * x + y * y)
    if magnitude <= deadzone or magnitude != magnitude:
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
        accel_x=0.0,
        accel_y=0.0,
        accel_z=0.0,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        raw_accel_x=0,
        raw_accel_y=0,
        raw_accel_z=0,
        raw_gyro_x=0,
        raw_gyro_y=0,
        raw_gyro_z=0,
        motion_timestamp=0,
        imu_temperature=25.0,
        has_motion=False,
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
        if report_id == REPORT_ID_PRO2 and 12 <= len(raw_bytes) <= 24:
            return _parse_power_on_09_report(raw_bytes, deadzone)
        else:
            data = raw_bytes[1:]
    else:
        # Auto-detect:
        # 0x09 report is strictly a small 12-24 byte USB report (never 60+ bytes)
        report_id = raw_bytes[0]
        if report_id == REPORT_ID_PRO2 and 12 <= len(raw_bytes) <= 24:
            return _parse_power_on_09_report(raw_bytes, deadzone)
        elif report_id == REPORT_ID_COMMON and len(raw_bytes) >= 64:
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

    # 6-Axis IMU Decoding (18-byte motion block at offset 42 of data payload)
    if len(data) >= 60:
        has_motion = True
        (
            motion_timestamp,
            temp_raw,
            raw_ax,
            raw_ay,
            raw_az,
            raw_gx,
            raw_gy,
            raw_gz,
        ) = struct.unpack_from("<Ih6h", data, 42)
        imu_temperature = IMU_TEMP_OFFSET + (temp_raw * IMU_TEMP_SCALE)
        accel_x = max(-8.0, min(8.0, raw_ax * ACCEL_SCALE_G))
        accel_y = max(-8.0, min(8.0, raw_ay * ACCEL_SCALE_G))
        accel_z = max(-8.0, min(8.0, raw_az * ACCEL_SCALE_G))
        gyro_x = max(-2000.0, min(2000.0, raw_gx * GYRO_SCALE_DPS))
        gyro_y = max(-2000.0, min(2000.0, raw_gy * GYRO_SCALE_DPS))
        gyro_z = max(-2000.0, min(2000.0, raw_gz * GYRO_SCALE_DPS))
    else:
        has_motion = False
        motion_timestamp = 0
        imu_temperature = 25.0
        raw_ax = raw_ay = raw_az = 0
        raw_gx = raw_gy = raw_gz = 0
        accel_x = accel_y = accel_z = 0.0
        gyro_x = gyro_y = gyro_z = 0.0

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
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        raw_accel_x=raw_ax,
        raw_accel_y=raw_ay,
        raw_accel_z=raw_az,
        raw_gyro_x=raw_gx,
        raw_gyro_y=raw_gy,
        raw_gyro_z=raw_gz,
        motion_timestamp=motion_timestamp,
        imu_temperature=imu_temperature,
        has_motion=has_motion,
    )


def encode_hd_rumble(high_freq: int, high_amp: int, low_freq: int, low_amp: int) -> bytes:
    """Encode 5-byte Switch 2 LRA HD Rumble block."""
    hf = max(0, min(0x03FF, int(high_freq)))
    ha = max(0, min(0xFFFF, int(high_amp)))
    lf = max(0, min(0x03FF, int(low_freq)))
    la = max(0, min(0xFFFF, int(low_amp)))
    b0 = hf & 0xFF
    b1 = ((ha >> 4) & 0xFC) | ((hf >> 8) & 0x03)
    b2 = ((ha >> 12) & 0x0F) | ((lf << 4) & 0xF0)
    b3 = (la & 0xC0) | ((lf >> 4) & 0x3F)
    b4 = (la >> 8) & 0xFF
    return bytes([b0, b1, b2, b3, b4])


def build_switch2_rumble_packet(
    large_motor: int,
    small_motor: int,
    seq: int = 0,
    include_report_id: bool = True,
    intensity: float = 1.0,
) -> bytes:
    """Build a Switch 2 Pro Controller (0x2069) HD Rumble output report (Report 0x02)."""
    lm = max(0, min(255, int(large_motor))) if isinstance(large_motor, (int, float)) and math.isfinite(large_motor) else 0
    sm = max(0, min(255, int(small_motor))) if isinstance(small_motor, (int, float)) and math.isfinite(small_motor) else 0
    scale = max(0.0, min(1.0, float(intensity))) if isinstance(intensity, (int, float)) and math.isfinite(intensity) else 1.0

    low_amp = int(round((lm / 255.0) * RUMBLE_MAX_AMPLITUDE * scale))
    high_amp = int(round((sm / 255.0) * RUMBLE_MAX_AMPLITUDE * scale))

    buf = bytearray(64)
    buf[0x00] = OUTPUT_REPORT_ID_PRO2
    buf[0x01] = 0x50 | (int(seq) & 0x0F)
    buf[0x02:0x07] = encode_hd_rumble(RUMBLE_HIGH_FREQ, high_amp, RUMBLE_LOW_FREQ, low_amp)
    buf[0x11:0x17] = buf[0x01:0x07]

    if include_report_id:
        return bytes(buf)
    return bytes(buf[0x01:0x17])

