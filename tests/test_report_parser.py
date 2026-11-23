import struct
import pytest
from constants import (
    REPORT_ID_COMMON,
    REPORT_ID_PRO2,
    OUTPUT_REPORT_ID_PRO2,
    RUMBLE_HIGH_FREQ,
    RUMBLE_LOW_FREQ,
    RUMBLE_MAX_AMPLITUDE,
    BTN_A, BTN_B, BTN_X, BTN_Y, BTN_ZL, BTN_ZR,
    BTN_PLUS, BTN_MINUS, BTN_L_THUMB, BTN_R_THUMB,
    BTN_GL, BTN_GR,
)
from report_parser import (
    ControllerState,
    decode_uint12_sticks,
    normalize_stick_axis,
    apply_radial_deadzone,
    parse_input_report,
    encode_hd_rumble,
    build_switch2_rumble_packet,
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


# ---------------------------------------------------------------------------
# Synthetic Payload Test Helpers for IMU Verification
# ---------------------------------------------------------------------------

def _pack_uint12_sticks(x: int, y: int) -> bytes:
    """Pack two 12-bit stick coordinates into 3 bytes little-endian."""
    val = (x & 0xFFF) | ((y & 0xFFF) << 12)
    return val.to_bytes(3, "little")


def _build_synthetic_report_05_usb(
    counter: int = 1234,
    buttons: int = 0,
    lx: int = 2048,
    ly: int = 2048,
    rx: int = 2048,
    ry: int = 2048,
    battery_mv: int = 3800,
    motion_timestamp: int = 0,
    temp_raw: int = 0,
    raw_ax: int = 0,
    raw_ay: int = 0,
    raw_az: int = 0,
    raw_gx: int = 0,
    raw_gy: int = 0,
    raw_gz: int = 0,
) -> bytes:
    """Build a synthetic 64-byte USB Report 0x05 payload with leading report ID."""
    buf = bytearray(64)
    buf[0] = REPORT_ID_COMMON  # 0x05
    buf[1:5] = counter.to_bytes(4, "little")
    buf[5:9] = buttons.to_bytes(4, "little")
    buf[11:14] = _pack_uint12_sticks(lx, ly)
    buf[14:17] = _pack_uint12_sticks(rx, ry)
    buf[32:34] = battery_mv.to_bytes(2, "little")
    # 18-byte motion block at buf[43:61] (corresponds to data[42:60])
    struct.pack_into(
        "<Ihhhhhhh",
        buf,
        43,
        motion_timestamp,
        temp_raw,
        raw_ax,
        raw_ay,
        raw_az,
        raw_gx,
        raw_gy,
        raw_gz,
    )
    return bytes(buf)


def _build_synthetic_report_05_ble(
    counter: int = 5678,
    buttons: int = 0,
    lx: int = 2048,
    ly: int = 2048,
    rx: int = 2048,
    ry: int = 2048,
    battery_mv: int = 3800,
    motion_timestamp: int = 0,
    temp_raw: int = 0,
    raw_ax: int = 0,
    raw_ay: int = 0,
    raw_az: int = 0,
    raw_gx: int = 0,
    raw_gy: int = 0,
    raw_gz: int = 0,
) -> bytes:
    """Build a synthetic 60-byte BLE Report 0x05 payload (omits leading report ID)."""
    buf = bytearray(60)
    buf[0:4] = counter.to_bytes(4, "little")
    buf[4:8] = buttons.to_bytes(4, "little")
    buf[10:13] = _pack_uint12_sticks(lx, ly)
    buf[13:16] = _pack_uint12_sticks(rx, ry)
    buf[31:33] = battery_mv.to_bytes(2, "little")
    # 18-byte motion block at buf[42:60] (corresponds to data[42:60])
    struct.pack_into(
        "<Ihhhhhhh",
        buf,
        42,
        motion_timestamp,
        temp_raw,
        raw_ax,
        raw_ay,
        raw_az,
        raw_gx,
        raw_gy,
        raw_gz,
    )
    return bytes(buf)


# ---------------------------------------------------------------------------
# Milestone 1: IMU Decoding & Robustness Unit Tests
# ---------------------------------------------------------------------------

def test_parse_report_05_imu_zero_motion():
    """Verify that zero motion registers decode to 0.0g and 0.0 deg/s with has_motion=True."""
    # Test USB 64-byte payload
    usb_payload = _build_synthetic_report_05_usb(
        motion_timestamp=1000000,
        temp_raw=0,
        raw_ax=0, raw_ay=0, raw_az=0,
        raw_gx=0, raw_gy=0, raw_gz=0,
    )
    state_usb = parse_input_report(usb_payload)
    assert state_usb is not None
    assert state_usb.has_motion is True
    assert state_usb.motion_timestamp == 1000000
    assert state_usb.imu_temperature == pytest.approx(25.0, abs=0.01)
    assert state_usb.accel_x == pytest.approx(0.0, abs=1e-5)
    assert state_usb.accel_y == pytest.approx(0.0, abs=1e-5)
    assert state_usb.accel_z == pytest.approx(0.0, abs=1e-5)
    assert state_usb.gyro_x == pytest.approx(0.0, abs=1e-5)
    assert state_usb.gyro_y == pytest.approx(0.0, abs=1e-5)
    assert state_usb.gyro_z == pytest.approx(0.0, abs=1e-5)
    assert state_usb.raw_accel_x == 0
    assert state_usb.raw_accel_y == 0
    assert state_usb.raw_accel_z == 0
    assert state_usb.raw_gyro_x == 0
    assert state_usb.raw_gyro_y == 0
    assert state_usb.raw_gyro_z == 0

    # Test BLE 60-byte payload
    ble_payload = _build_synthetic_report_05_ble(
        motion_timestamp=2000000,
        temp_raw=0,
        raw_ax=0, raw_ay=0, raw_az=0,
        raw_gx=0, raw_gy=0, raw_gz=0,
    )
    state_ble = parse_input_report(ble_payload, has_report_id=False)
    assert state_ble is not None
    assert state_ble.has_motion is True
    assert state_ble.motion_timestamp == 2000000
    assert state_ble.accel_z == pytest.approx(0.0, abs=1e-5)
    assert state_ble.gyro_z == pytest.approx(0.0, abs=1e-5)


def test_parse_report_05_imu_resting_tabletop():
    """Verify flat tabletop resting state: Accel Z = 1.0g (raw=4096), Gyro = 0 deg/s."""
    # 4096 * (8.0 / 32767.0) = 1.0000305... g
    # temp_raw = 254 -> 25.0 + 254 / 126.9 = 27.0 deg C
    usb_payload = _build_synthetic_report_05_usb(
        buttons=BTN_A,
        motion_timestamp=345678,
        temp_raw=254,
        raw_ax=0,
        raw_ay=0,
        raw_az=4096,
        raw_gx=0,
        raw_gy=0,
        raw_gz=0,
    )
    state = parse_input_report(usb_payload)
    assert state is not None
    assert state.has_motion is True
    assert state.a is True
    assert state.motion_timestamp == 345678
    assert state.accel_x == pytest.approx(0.0, abs=1e-5)
    assert state.accel_y == pytest.approx(0.0, abs=1e-5)
    assert state.accel_z == pytest.approx(1.0, abs=1e-3)
    assert state.raw_accel_z == 4096
    assert state.gyro_x == pytest.approx(0.0, abs=1e-5)
    assert state.gyro_y == pytest.approx(0.0, abs=1e-5)
    assert state.gyro_z == pytest.approx(0.0, abs=1e-5)
    assert state.imu_temperature == pytest.approx(27.0, abs=0.1)

    # Test BLE equivalent
    ble_payload = _build_synthetic_report_05_ble(
        buttons=BTN_B,
        motion_timestamp=654321,
        temp_raw=254,
        raw_ax=0,
        raw_ay=0,
        raw_az=4096,
    )
    state_ble = parse_input_report(ble_payload, has_report_id=False)
    assert state_ble is not None
    assert state_ble.has_motion is True
    assert state_ble.b is True
    assert state_ble.accel_z == pytest.approx(1.0, abs=1e-3)


def test_parse_report_05_imu_high_rate_rotation():
    """Verify high-rate angular velocity: Gyro X = 90 dps, Gyro Y = -45 dps, Gyro Z = 180 dps."""
    # raw_gx = round(90 * 32767 / 2000) = 1475 -> 90.0296 dps
    # raw_gy = round(-45 * 32767 / 2000) = -737 -> -44.984 dps
    # raw_gz = round(180 * 32767 / 2000) = 2949 -> 180.000 dps
    # raw_ax = round(0.5 * 32767 / 8) = 2048 -> 0.5g centripetal
    usb_payload = _build_synthetic_report_05_usb(
        motion_timestamp=999999,
        raw_ax=2048,
        raw_ay=0,
        raw_az=4096,
        raw_gx=1475,
        raw_gy=-737,
        raw_gz=2949,
    )
    state = parse_input_report(usb_payload)
    assert state is not None
    assert state.has_motion is True
    assert state.raw_gyro_x == 1475
    assert state.raw_gyro_y == -737
    assert state.raw_gyro_z == 2949
    assert state.gyro_x == pytest.approx(90.0, abs=0.1)
    assert state.gyro_y == pytest.approx(-45.0, abs=0.1)
    assert state.gyro_z == pytest.approx(180.0, abs=0.1)
    assert state.accel_x == pytest.approx(0.5, abs=1e-3)
    assert state.accel_z == pytest.approx(1.0, abs=1e-3)


def test_parse_report_05_imu_boundary_saturation_positive():
    """Verify maximum positive signed 16-bit saturation (+32767): Accel = +8.0g, Gyro = +2000.0 dps."""
    usb_payload = _build_synthetic_report_05_usb(
        raw_ax=32767, raw_ay=32767, raw_az=32767,
        raw_gx=32767, raw_gy=32767, raw_gz=32767,
    )
    state = parse_input_report(usb_payload)
    assert state is not None
    assert state.has_motion is True
    assert state.raw_accel_x == 32767
    assert state.raw_accel_y == 32767
    assert state.raw_accel_z == 32767
    assert state.raw_gyro_x == 32767
    assert state.raw_gyro_y == 32767
    assert state.raw_gyro_z == 32767
    assert state.accel_x == pytest.approx(8.0, abs=1e-4)
    assert state.accel_y == pytest.approx(8.0, abs=1e-4)
    assert state.accel_z == pytest.approx(8.0, abs=1e-4)
    assert state.gyro_x == pytest.approx(2000.0, abs=1e-2)
    assert state.gyro_y == pytest.approx(2000.0, abs=1e-2)
    assert state.gyro_z == pytest.approx(2000.0, abs=1e-2)


def test_parse_report_05_imu_boundary_saturation_negative():
    """Verify maximum negative signed 16-bit saturation (-32768): Accel = -8.0g, Gyro = -2000.0 dps."""
    ble_payload = _build_synthetic_report_05_ble(
        raw_ax=-32768, raw_ay=-32768, raw_az=-32768,
        raw_gx=-32768, raw_gy=-32768, raw_gz=-32768,
    )
    state = parse_input_report(ble_payload, has_report_id=False)
    assert state is not None
    assert state.has_motion is True
    # Clamped or unclamped raw range check
    assert state.raw_accel_x in (-32768, -32767)
    assert state.raw_gyro_x in (-32768, -32767)
    assert state.accel_x == pytest.approx(-8.0, abs=1e-3)
    assert state.accel_y == pytest.approx(-8.0, abs=1e-3)
    assert state.accel_z == pytest.approx(-8.0, abs=1e-3)
    assert state.gyro_x == pytest.approx(-2000.0, abs=0.1)
    assert state.gyro_y == pytest.approx(-2000.0, abs=0.1)
    assert state.gyro_z == pytest.approx(-2000.0, abs=0.1)


def test_parse_report_09_imu_fallback_defaults():
    """Verify small USB Report 0x09 power-on report defaults all IMU fields safely with has_motion=False."""
    # 12-byte Report 0x09 with Button A pressed
    buf = bytearray(12)
    buf[0] = REPORT_ID_PRO2  # 0x09
    buf[1] = 42              # Counter
    buf[3] = 0x08            # Button A
    buf[6:9] = bytes([0x00, 0x08, 0x80])   # Left stick centered
    buf[9:12] = bytes([0x00, 0x08, 0x80])  # Right stick centered

    state = parse_input_report(bytes(buf), has_report_id=True)
    assert state is not None
    assert state.a is True
    assert state.has_motion is False
    assert state.motion_timestamp == 0
    assert state.imu_temperature == 25.0
    assert state.accel_x == 0.0
    assert state.accel_y == 0.0
    assert state.accel_z == 0.0
    assert state.gyro_x == 0.0
    assert state.gyro_y == 0.0
    assert state.gyro_z == 0.0
    assert state.raw_accel_x == 0
    assert state.raw_accel_y == 0
    assert state.raw_accel_z == 0
    assert state.raw_gyro_x == 0
    assert state.raw_gyro_y == 0
    assert state.raw_gyro_z == 0


def test_parse_truncated_payload_graceful_handling():
    """Verify truncated packets (< 60 bytes) decode available fields without exception and set has_motion=False."""
    full_usb = _build_synthetic_report_05_usb(buttons=BTN_A | BTN_B, raw_az=4096)

    # Incomplete packet lengths: 17 (sticks only), 34 (battery only), 42 (pre-motion), 59 (cut off Gyro Z)
    for length in (17, 34, 42, 59):
        truncated = full_usb[:length]
        state = parse_input_report(truncated, has_report_id=True)
        assert state is not None, f"Failed on length {length}"
        assert state.has_motion is False
        assert state.accel_x == 0.0
        assert state.accel_y == 0.0
        assert state.accel_z == 0.0
        assert state.gyro_x == 0.0
        assert state.gyro_y == 0.0
        assert state.gyro_z == 0.0
        assert state.raw_accel_x == 0
        assert state.raw_gyro_x == 0
        assert state.motion_timestamp == 0
        assert state.imu_temperature == 25.0

    # Payload shorter than minimum header (len < 12) or sticks (len < 16 payload, 17 raw USB) returns None
    assert parse_input_report(full_usb[:16], has_report_id=True) is None
    assert parse_input_report(b"\x05\x01\x02\x03", has_report_id=True) is None
    assert parse_input_report(b"", has_report_id=True) is None


def test_ble_payload_auto_detect_motion():
    """Verify auto-detect mode (has_report_id=None) correctly recognizes and parses 60-byte BLE motion packets."""
    ble_payload = _build_synthetic_report_05_ble(
        counter=99,
        buttons=BTN_X,
        raw_az=4096,
        motion_timestamp=777777,
    )
    state = parse_input_report(ble_payload, has_report_id=None)
    assert state is not None
    assert state.x is True
    assert state.has_motion is True
    assert state.motion_timestamp == 777777
    assert state.accel_z == pytest.approx(1.0, abs=1e-3)


def test_controller_state_dataclass_defaults():
    """Verify ControllerState can be instantiated with only legacy arguments without passing motion kwargs."""
    state = ControllerState(
        timestamp=100,
        buttons_raw=0,
        a=False, b=False, x=False, y=False,
        plus=False, minus=False, home=False, capture=False,
        l=False, r=False, l_thumb=False, r_thumb=False,
        dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
        zl_analog=0.0, zr_analog=0.0,
        left_stick_x=0.0, left_stick_y=0.0,
        right_stick_x=0.0, right_stick_y=0.0,
    )
    assert state.has_motion is False
    assert state.accel_x == 0.0
    assert state.gyro_x == 0.0
    assert state.raw_accel_x == 0
    assert state.raw_gyro_x == 0
    assert state.motion_timestamp == 0
    assert state.imu_temperature == 25.0


# ---------------------------------------------------------------------------
# Milestone 1: USB Truncation & Deadzone Regression Tests
# ---------------------------------------------------------------------------

def test_parse_truncated_usb_60_bytes_has_motion_false():
    """Verify truncated 60-byte USB report with has_report_id=True yields has_motion=False and unshifted controls."""
    full_usb = _build_synthetic_report_05_usb(
        counter=1234,
        buttons=BTN_A | BTN_B,
        lx=2048,
        ly=2048,
        motion_timestamp=99999,
        raw_ax=1000,
        raw_ay=2000,
        raw_az=4096,
        raw_gx=500,
        raw_gy=600,
        raw_gz=700,
    )
    # 60 bytes = 1 byte report ID + 59 bytes payload (1 byte short of 18-byte motion block at offset 42..60)
    truncated_60 = full_usb[:60]
    assert len(truncated_60) == 60

    state = parse_input_report(truncated_60, has_report_id=True)
    assert state is not None
    assert state.has_motion is False

    # Motion fields must revert to safe defaults
    assert state.motion_timestamp == 0
    assert state.imu_temperature == 25.0
    assert state.accel_x == 0.0
    assert state.accel_y == 0.0
    assert state.accel_z == 0.0
    assert state.gyro_x == 0.0
    assert state.gyro_y == 0.0
    assert state.gyro_z == 0.0
    assert state.raw_accel_x == 0
    assert state.raw_accel_y == 0
    assert state.raw_accel_z == 0
    assert state.raw_gyro_x == 0
    assert state.raw_gyro_y == 0
    assert state.raw_gyro_z == 0

    # Core inputs must remain intact and aligned (not shifted by 1 byte)
    assert state.a is True
    assert state.b is True
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0


def test_parse_usb_61_to_63_bytes_motion():
    """Verify USB reports with lengths 61, 62, and 63 strip report ID cleanly and decode motion correctly."""
    full_usb = _build_synthetic_report_05_usb(
        counter=9876,
        buttons=BTN_X | BTN_Y,
        lx=2048,
        ly=2048,
        motion_timestamp=54321,
        temp_raw=254,
        raw_ax=1024,
        raw_ay=0,
        raw_az=4096,
        raw_gx=1475,
        raw_gy=-737,
        raw_gz=2949,
    )

    # Test packet lengths 61, 62, 63 (payload lengths 60, 61, 62)
    for length in (61, 62, 63):
        sliced = full_usb[:length]
        assert len(sliced) == length

        state = parse_input_report(sliced, has_report_id=True)
        assert state is not None, f"Failed parsing length {length}"
        assert state.has_motion is True, f"Expected has_motion=True for length {length}"

        # Buttons and timestamps
        assert state.x is True
        assert state.y is True
        assert state.motion_timestamp == 54321
        assert state.imu_temperature == pytest.approx(27.0, abs=0.1)

        # Raw IMU values
        assert state.raw_accel_x == 1024
        assert state.raw_accel_y == 0
        assert state.raw_accel_z == 4096
        assert state.raw_gyro_x == 1475
        assert state.raw_gyro_y == -737
        assert state.raw_gyro_z == 2949

        # Scaled physical values
        assert state.accel_x == pytest.approx(0.25, abs=1e-3)
        assert state.accel_y == pytest.approx(0.0, abs=1e-5)
        assert state.accel_z == pytest.approx(1.0, abs=1e-3)
        assert state.gyro_x == pytest.approx(90.0, abs=0.1)
        assert state.gyro_y == pytest.approx(-45.0, abs=0.1)
        assert state.gyro_z == pytest.approx(180.0, abs=0.1)


def test_apply_radial_deadzone_boundary_one():
    """Verify deadzone >= 1.0 returns (0.0, 0.0) without ZeroDivisionError across all coordinates."""
    # Sub-1.0 magnitude
    assert apply_radial_deadzone(0.0, 0.0, deadzone=1.0) == (0.0, 0.0)
    assert apply_radial_deadzone(0.5, 0.5, deadzone=1.0) == (0.0, 0.0)

    # Exactly 1.0 magnitude
    assert apply_radial_deadzone(1.0, 0.0, deadzone=1.0) == (0.0, 0.0)
    assert apply_radial_deadzone(0.0, 1.0, deadzone=1.0) == (0.0, 0.0)

    # Diagonal corner magnitude = sqrt(2) > 1.0
    assert apply_radial_deadzone(1.0, 1.0, deadzone=1.0) == (0.0, 0.0)

    # Deadzone > 1.0
    assert apply_radial_deadzone(1.0, 1.0, deadzone=1.5) == (0.0, 0.0)

    # Integration check via parse_input_report with deadzone=1.0
    full_usb = _build_synthetic_report_05_usb(lx=4095, ly=4095, rx=0, ry=4095)
    state = parse_input_report(full_usb, deadzone=1.0, has_report_id=True)
    assert state is not None
    assert state.left_stick_x == 0.0
    assert state.left_stick_y == 0.0
    assert state.right_stick_x == 0.0
    assert state.right_stick_y == 0.0


def test_apply_radial_deadzone_zero_and_negative():
    """Verify deadzone <= 0.0 handles center and deflections cleanly without ZeroDivisionError."""
    assert apply_radial_deadzone(0.0, 0.0, deadzone=0.0) == (0.0, 0.0)
    assert apply_radial_deadzone(0.5, 0.0, deadzone=0.0) == (0.5, 0.0)
    assert apply_radial_deadzone(0.0, 0.0, deadzone=-0.05) == (0.0, 0.0)
    assert apply_radial_deadzone(0.5, 0.0, deadzone=-0.05) == (0.5, 0.0)


def test_normalize_stick_axis_edge_cases():
    """Verify normalize_stick_axis handles non-positive max_delta without ZeroDivisionError."""
    assert normalize_stick_axis(2048, max_delta=0) == 0.0
    assert normalize_stick_axis(3000, max_delta=0) == 0.0
    assert normalize_stick_axis(3000, max_delta=-100) == 0.0


def test_encode_hd_rumble_zero_and_full_scale():
    """Verify Switch 2 LRA 5-byte HD rumble encoding at zero and full amplitude."""
    zero_block = encode_hd_rumble(RUMBLE_HIGH_FREQ, 0, RUMBLE_LOW_FREQ, 0)
    assert len(zero_block) == 5
    # High freq 0x0187 -> b0=0x87, b1 low 2 bits=0x01; Low freq 0x0112 -> b2=0x20, b3=0x11, b4=0x00
    assert zero_block == bytes([0x87, 0x01, 0x20, 0x11, 0x00])

    full_block = encode_hd_rumble(RUMBLE_HIGH_FREQ, RUMBLE_MAX_AMPLITUDE, RUMBLE_LOW_FREQ, RUMBLE_MAX_AMPLITUDE)
    assert len(full_block) == 5
    assert full_block != zero_block


def test_build_switch2_rumble_packet_usb_ble_and_edge_cases():
    """Verify USB 64-byte Report 0x02, BLE 22-byte payload, sequence nibble, intensity, and non-finite inputs."""
    usb_pkt = build_switch2_rumble_packet(255, 255, seq=5, include_report_id=True, intensity=1.0)
    assert len(usb_pkt) == 64
    assert usb_pkt[0] == OUTPUT_REPORT_ID_PRO2
    assert usb_pkt[1] == 0x55
    # Right motor block at offset 0x11..0x17 mirrors left motor block at 0x01..0x07
    assert usb_pkt[0x11:0x17] == usb_pkt[0x01:0x07]

    ble_pkt = build_switch2_rumble_packet(255, 255, seq=5, include_report_id=False, intensity=1.0)
    assert len(ble_pkt) == 22
    assert ble_pkt == usb_pkt[0x01:0x17]

    half_pkt = build_switch2_rumble_packet(255, 128, seq=0, include_report_id=True, intensity=0.5)
    zero_pkt = build_switch2_rumble_packet(0, 0, seq=0, include_report_id=True, intensity=1.0)
    assert half_pkt[0x02:0x07] != zero_pkt[0x02:0x07]
    assert half_pkt[0x02:0x07] != usb_pkt[0x02:0x07]

    nan_pkt = build_switch2_rumble_packet(float("nan"), float("inf"), seq=0x13, include_report_id=True, intensity=float("nan"))  # type: ignore
    assert len(nan_pkt) == 64
    assert nan_pkt[1] == 0x53
    assert nan_pkt[0x02:0x07] == zero_pkt[0x02:0x07]


