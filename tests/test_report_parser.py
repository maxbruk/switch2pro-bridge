"""Unit tests for report_parser and stick decoding."""

import pytest
from constants import (
    REPORT_ID_COMMON,
    REPORT_ID_PRO2,
    BTN_A, BTN_B, BTN_X, BTN_Y, BTN_ZL, BTN_ZR,
    BTN_PLUS, BTN_MINUS, BTN_L_THUMB, BTN_R_THUMB,
    BTN_GL, BTN_GR,
)
from report_parser import (
    decode_uint12_sticks,
    normalize_stick_axis,
    apply_radial_deadzone,
    parse_input_report,
)


def test_decode_uint12_sticks():
    # 2048 (0x800) in 12-bit x and y:
    # x = 0x800 -> 12 bits: 000 0000 1000
    # y = 0x800 -> 12 bits: 000 0000 1000
    # packed 24 bits: (0x800 << 12) | 0x800 = 0x800800
    # little endian bytes: [0x00, 0x08, 0x80]
    data = bytes([0x00, 0x08, 0x80])
    x, y = decode_uint12_sticks(data)
    assert x == 2048
    assert y == 2048


def test_normalize_stick_axis():
    assert normalize_stick_axis(2048, center=2048, max_delta=1600) == 0.0
    assert normalize_stick_axis(3648, center=2048, max_delta=1600) == 1.0
    assert normalize_stick_axis(448, center=2048, max_delta=1600) == -1.0


def test_apply_radial_deadzone():
    # Inside deadzone
    nx, ny = apply_radial_deadzone(0.02, 0.02, deadzone=0.05)
    assert nx == 0.0 and ny == 0.0

    # Outside deadzone
    nx, ny = apply_radial_deadzone(1.0, 0.0, deadzone=0.05)
    assert nx == 1.0 and ny == 0.0


def test_parse_report_05():
    # 64-byte buffer with leading REPORT_ID_COMMON (0x05)
    buf = bytearray(64)
    buf[0] = REPORT_ID_COMMON
    
    # Counter at data[0:4] -> buf[1:5]
    buf[1:5] = (1234).to_bytes(4, "little")
    
    # Buttons: A, B, ZL, Plus at data[4:8] -> buf[5:9]
    buttons_mask = BTN_A | BTN_B | BTN_ZL | BTN_PLUS
    buf[5:9] = buttons_mask.to_bytes(4, "little")
    
    # Left stick centered at data[10:13] -> buf[11:14]
    buf[11:14] = bytes([0x00, 0x08, 0x80])
    # Right stick centered at data[13:16] -> buf[14:17]
    buf[14:17] = bytes([0x00, 0x08, 0x80])
    
    # Battery voltage 3800 mV at data[31:33] -> buf[32:34]
    buf[32:34] = (3800).to_bytes(2, "little")

    state = parse_input_report(bytes(buf), deadzone=0.05)
    assert state is not None
    assert state.a is True
    assert state.b is True
    assert state.x is False
    assert state.y is False
    assert state.plus is True
    assert state.minus is False
    assert state.zl_analog == 1.0
    assert state.zr_analog == 0.0
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0
    assert state.battery_voltage == 3.8
    assert 40 <= state.battery_percentage <= 60


def test_parse_report_05_grip_buttons():
    # Test GL and GR (bits 25 and 24)
    buf = bytearray(64)
    buf[0] = REPORT_ID_COMMON
    
    # Set buttons with GL and GR pressed
    buttons_mask = BTN_GL | BTN_GR
    buf[5:9] = buttons_mask.to_bytes(4, "little")
    
    # Sticks centered
    buf[11:14] = bytes([0x00, 0x08, 0x80])
    buf[14:17] = bytes([0x00, 0x08, 0x80])

    state = parse_input_report(bytes(buf), deadzone=0.05)
    assert state is not None
    assert state.gl is True
    assert state.gr is True
    assert state.a is False
    assert state.b is False


def test_ble_packet_timestamp_09_no_flicker():
    """Ensure BLE packets whose timestamp LSB happens to be 0x09 do not trigger 0x09 power-on report parsing."""
    buf = bytearray(60)
    # Byte 0 is 0x09 (timestamp & 0xFF == 0x09)
    buf[0] = 0x09
    buf[1:4] = (1).to_bytes(3, "little")  # remaining timestamp bytes
    
    # Buttons: A pressed
    buf[4:8] = (BTN_A).to_bytes(4, "little")
    
    # Sticks centered at data[10:13] and data[13:16]
    buf[10:13] = bytes([0x00, 0x08, 0x80])
    buf[13:16] = bytes([0x00, 0x08, 0x80])
    
    # 1. With explicit has_report_id=False (BLE mode)
    state = parse_input_report(bytes(buf), deadzone=0.05, has_report_id=False)
    assert state is not None
    assert state.a is True
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0
    assert state.right_stick_x == 0.0
    assert state.right_stick_y == 0.0

    # 2. With auto-detect has_report_id=None (len is 60, not 12-24, so shouldn't parse as 0x09)
    state_auto = parse_input_report(bytes(buf), deadzone=0.05, has_report_id=None)
    assert state_auto is not None
    assert state_auto.a is True
    assert state_auto.left_stick_x == 0.0
    assert state_auto.left_stick_y == 0.0


def test_ble_packet_timestamp_05_no_flicker():
    """Ensure BLE 60-byte packet with timestamp LSB 0x05 parses correctly with has_report_id=False."""
    buf = bytearray(60)
    buf[0] = 0x05  # timestamp LSB is 0x05
    buf[1:4] = (0).to_bytes(3, "little")
    buf[4:8] = (BTN_B).to_bytes(4, "little")
    buf[10:13] = bytes([0x00, 0x08, 0x80])
    buf[13:16] = bytes([0x00, 0x08, 0x80])

    state = parse_input_report(bytes(buf), deadzone=0.05, has_report_id=False)
    assert state is not None
    assert state.b is True
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0


def test_usb_report_09_power_on():
    """Verify small 12-byte USB power-on 0x09 report is parsed correctly."""
    buf = bytearray(12)
    buf[0] = REPORT_ID_PRO2  # 0x09
    buf[1] = 42  # counter
    buf[3] = 0x08  # A button
    buf[6:9] = bytes([0x00, 0x08, 0x80])  # Left stick centered
    buf[9:12] = bytes([0x00, 0x08, 0x80])  # Right stick centered

    state = parse_input_report(bytes(buf), deadzone=0.05, has_report_id=True)
    assert state is not None
    assert state.a is True
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0

