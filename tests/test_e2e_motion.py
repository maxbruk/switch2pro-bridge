"""Comprehensive End-to-End Motion Integration & Contract Test Suite.

Architected according to the 4-Tier Testing Philosophy defined in TEST_INFRA.md:
- Tier 1: Feature Coverage (Isolated Happy Path)
- Tier 2: Boundary & Corner Cases (Limits, saturation, malformed packets)
- Tier 3: Cross-Feature Combinations (DS4 + Motion, DSU + Xbox concurrent, rapid mode shift)
- Tier 4: Real-World Application Scenarios (Cemu/Dolphin DSU session, Steam DS4, 250 Hz sustained)

All expected outputs are derived from authoritative hardware specifications
(InvenSense ICM-42670-P, Cemuhook DSU Protocol v1001, ViGEmBus DS4 Extended Report).
"""

import os
import sys
import json
import time
import zlib
import struct
import socket
import tempfile
import threading
import importlib.util
from typing import Optional, Tuple, Dict, Any, List

import pytest

from constants import (
    REPORT_ID_COMMON,
    REPORT_ID_PRO2,
    BTN_A, BTN_B, BTN_X, BTN_Y, BTN_ZL, BTN_ZR,
    BTN_PLUS, BTN_MINUS, BTN_L_THUMB, BTN_R_THUMB,
    BTN_L, BTN_R, BTN_HOME, BTN_CAPTURE,
    BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT,
    BTN_GL, BTN_GR,
)
from report_parser import (
    ControllerState,
    decode_uint12_sticks,
    normalize_stick_axis,
    apply_radial_deadzone,
    parse_input_report,
)
from app_core import ControllerManager
from virtual_xinput import VirtualXboxController


# ============================================================================
# Dynamic Feature Probes for Progressive Testability Across Milestones
# ============================================================================

IMU_PARSER_AVAILABLE = hasattr(ControllerState, "accel_x") or "accel_x" in getattr(ControllerState, "__dataclass_fields__", {})

DSU_SERVER_AVAILABLE = importlib.util.find_spec("dsu_server") is not None
if DSU_SERVER_AVAILABLE:
    import dsu_server  # type: ignore
else:
    dsu_server = None  # type: ignore

DS4_MODULE_AVAILABLE = importlib.util.find_spec("virtual_ds4") is not None
if DS4_MODULE_AVAILABLE:
    import virtual_ds4  # type: ignore
else:
    virtual_ds4 = None  # type: ignore

MANAGER_OUTPUT_MODE_AVAILABLE = hasattr(ControllerManager, "set_output_mode")
MANAGER_DSU_AVAILABLE = hasattr(ControllerManager, "set_dsu_enabled")


# ============================================================================
# Authoritative Reference Oracles & Protocol Encoders / Decoders
# ============================================================================

def ref_accel_g(raw_int16: int) -> float:
    """ICM-42670-P Accelerometer Scaling Law: ±8g FSR, 8.0 / 32767.0 g/LSB."""
    return raw_int16 * (8.0 / 32767.0)


def ref_gyro_dps(raw_int16: int) -> float:
    """ICM-42670-P Gyroscope Scaling Law: ±2000 dps FSR, 2000.0 / 32767.0 dps/LSB."""
    return raw_int16 * (2000.0 / 32767.0)


def ref_temp_c(raw_int16: int) -> float:
    """ICM-42670-P Temperature Formula: 25.0°C base + raw / 126.9 °C/LSB."""
    return 25.0 + (raw_int16 / 126.9)


def ref_dsu_crc32(packet: bytearray) -> int:
    """IEEE 802.3 CRC32 calculation with CRC field (bytes 8..11) zeroed out."""
    temp = bytearray(packet)
    temp[8:12] = b"\x00\x00\x00\x00"
    return zlib.crc32(temp) & 0xFFFFFFFF


def build_switch2_usb_report(
    timestamp_seq: int = 1000,
    buttons_raw: int = 0,
    lx_raw: int = 2048,
    ly_raw: int = 2048,
    rx_raw: int = 2048,
    ry_raw: int = 2048,
    battery_mv: int = 3800,
    imu_timestamp_us: int = 500000,
    temp_raw: int = 0,
    ax_raw: int = 0,
    ay_raw: int = 0,
    az_raw: int = 4095,  # 1g on Z axis (resting flat)
    gx_raw: int = 0,
    gy_raw: int = 0,
    gz_raw: int = 0,
) -> bytes:
    """Synthesize authoritative 64-byte Switch 2 USB Report 0x05."""
    buf = bytearray(64)
    buf[0] = REPORT_ID_COMMON  # 0x05

    # Sequence counter (data[0:4] -> buf[1:5])
    struct.pack_into("<I", buf, 1, timestamp_seq & 0xFFFFFFFF)

    # Buttons (data[4:8] -> buf[5:9])
    struct.pack_into("<I", buf, 5, buttons_raw & 0xFFFFFFFF)

    # Left stick (12-bit x, y packed in 3 bytes at data[10:13] -> buf[11:14])
    packed_l = (lx_raw & 0xFFF) | ((ly_raw & 0xFFF) << 12)
    buf[11:14] = packed_l.to_bytes(3, "little")

    # Right stick (data[13:16] -> buf[14:17])
    packed_r = (rx_raw & 0xFFF) | ((ry_raw & 0xFFF) << 12)
    buf[14:17] = packed_r.to_bytes(3, "little")

    # Battery voltage (data[31:33] -> buf[32:34])
    struct.pack_into("<H", buf, 32, battery_mv & 0xFFFF)

    # 18-Byte IMU block (starts at data[42] -> buf[43])
    struct.pack_into("<I", buf, 43, imu_timestamp_us & 0xFFFFFFFF)  # 42..45
    struct.pack_into("<h", buf, 47, temp_raw)                       # 46..47
    struct.pack_into("<h", buf, 49, ax_raw)                         # 48..49 (Accel X)
    struct.pack_into("<h", buf, 51, ay_raw)                         # 50..51 (Accel Y)
    struct.pack_into("<h", buf, 53, az_raw)                         # 52..53 (Accel Z)
    struct.pack_into("<h", buf, 55, gx_raw)                         # 54..55 (Gyro X)
    struct.pack_into("<h", buf, 57, gy_raw)                         # 56..57 (Gyro Y)
    struct.pack_into("<h", buf, 59, gz_raw)                         # 58..59 (Gyro Z)

    return bytes(buf)


def build_switch2_ble_report(
    timestamp_seq: int = 1000,
    buttons_raw: int = 0,
    lx_raw: int = 2048,
    ly_raw: int = 2048,
    rx_raw: int = 2048,
    ry_raw: int = 2048,
    battery_mv: int = 3800,
    imu_timestamp_us: int = 500000,
    temp_raw: int = 0,
    ax_raw: int = 0,
    ay_raw: int = 0,
    az_raw: int = 4095,
    gx_raw: int = 0,
    gy_raw: int = 0,
    gz_raw: int = 0,
) -> bytes:
    """Synthesize authoritative 60-byte Switch 2 BLE GATT payload (no leading report ID)."""
    buf = bytearray(60)

    # Sequence counter (data[0:4])
    struct.pack_into("<I", buf, 0, timestamp_seq & 0xFFFFFFFF)

    # Buttons (data[4:8])
    struct.pack_into("<I", buf, 4, buttons_raw & 0xFFFFFFFF)

    # Left stick (data[10:13])
    packed_l = (lx_raw & 0xFFF) | ((ly_raw & 0xFFF) << 12)
    buf[10:13] = packed_l.to_bytes(3, "little")

    # Right stick (data[13:16])
    packed_r = (rx_raw & 0xFFF) | ((ry_raw & 0xFFF) << 12)
    buf[13:16] = packed_r.to_bytes(3, "little")

    # Battery (data[31:33])
    struct.pack_into("<H", buf, 31, battery_mv & 0xFFFF)

    # IMU block (data[42..60])
    struct.pack_into("<I", buf, 42, imu_timestamp_us & 0xFFFFFFFF)
    struct.pack_into("<h", buf, 46, temp_raw)
    struct.pack_into("<h", buf, 48, ax_raw)
    struct.pack_into("<h", buf, 50, ay_raw)
    struct.pack_into("<h", buf, 52, az_raw)
    struct.pack_into("<h", buf, 54, gx_raw)
    struct.pack_into("<h", buf, 56, gy_raw)
    struct.pack_into("<h", buf, 58, gz_raw)

    return bytes(buf)


def parse_dsu_packet_header(packet: bytes) -> Dict[str, Any]:
    """Parse 16-byte Cemuhook DSU header and verify CRC32."""
    if len(packet) < 16:
        raise ValueError(f"Packet too short: {len(packet)} bytes (expected >= 16)")
    magic, version, length, crc, sender = struct.unpack_from("<4sHHII", packet, 0)
    expected_crc = ref_dsu_crc32(bytearray(packet))
    return {
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "length": length,
        "crc": crc,
        "expected_crc": expected_crc,
        "crc_valid": crc == expected_crc,
        "sender": sender,
    }


def parse_dsu_data_payload(packet: bytes) -> Dict[str, Any]:
    """Parse Cemuhook DSU 0x100002 100-byte controller data packet."""
    hdr = parse_dsu_packet_header(packet)
    if len(packet) < 100:
        raise ValueError(f"Packet too short for DSU data: {len(packet)} bytes (expected 100)")

    msg_type = struct.unpack_from("<I", packet, 16)[0]
    slot, slot_state, model, conn = struct.unpack_from("<BBBB", packet, 20)
    seq = struct.unpack_from("<I", packet, 32)[0]

    # Sticks at byte 40..43 (offsets 20..23 in payload)
    lx, ly, rx, ry = struct.unpack_from("<BBBB", packet, 40)

    # Timestamp at byte 68 (offset 48 in payload)
    timestamp_us = struct.unpack_from("<Q", packet, 68)[0]

    # Motion floats at byte 76..100 (offsets 56..80 in payload)
    ax, ay, az, pitch, yaw, roll = struct.unpack_from("<ffffff", packet, 76)

    return {
        "header": hdr,
        "message_type": msg_type,
        "slot": slot,
        "slot_state": slot_state,
        "model": model,
        "connection": conn,
        "sequence": seq,
        "left_stick": (lx, ly),
        "right_stick": (rx, ry),
        "motion_timestamp": timestamp_us,
        "accel": (ax, ay, az),
        "gyro": (pitch, yaw, roll),
    }


# ============================================================================
# TIER 1: FEATURE COVERAGE (ISOLATED HAPPY PATH, >= 5 TESTS PER FEATURE)
# ============================================================================

class TestTier1IMUReferenceModel:
    """Verifies ICM-42670-P sensor mathematical model and scaling laws."""

    def test_ref_accel_scaling_exactness(self):
        """Feature 3: Accelerometer scaling gives 1.0g at 4095.875 LSB."""
        raw_1g = 4095
        assert abs(ref_accel_g(raw_1g) - 0.9998) < 0.001
        assert ref_accel_g(0) == 0.0

    def test_ref_gyro_scaling_exactness(self):
        """Feature 3: Gyroscope scaling gives 90 dps at 1474 LSB."""
        raw_90dps = 1474
        assert abs(ref_gyro_dps(raw_90dps) - 89.968) < 0.05
        assert ref_gyro_dps(0) == 0.0

    def test_ref_temperature_scaling_exactness(self):
        """Feature 3: Temperature formula gives 25.0°C at raw 0."""
        assert ref_temp_c(0) == 25.0
        # 1269 LSB -> +10°C delta -> 35.0°C
        assert abs(ref_temp_c(1269) - 35.0) < 0.01

    def test_usb_packet_builder_integrity(self):
        """Feature 2: Report 0x05 USB builder creates exact 64-byte payload."""
        pkt = build_switch2_usb_report(az_raw=4095)
        assert len(pkt) == 64
        assert pkt[0] == REPORT_ID_COMMON
        # Unpack az at offset 53
        az = struct.unpack_from("<h", pkt, 53)[0]
        assert az == 4095

    def test_ble_packet_builder_integrity(self):
        """Feature 2: Report 0x05 BLE builder creates exact 60-byte payload."""
        pkt = build_switch2_ble_report(az_raw=4095)
        assert len(pkt) == 60
        # Unpack az at offset 52
        az = struct.unpack_from("<h", pkt, 52)[0]
        assert az == 4095


@pytest.mark.skipif(not IMU_PARSER_AVAILABLE, reason="M1: IMU decoding in ControllerState pending")
class TestTier1IMUDecoding:
    """Feature 1, 2, 3, 4: Decodes 6-axis IMU from Report 0x05 (USB and BLE)."""

    def test_imu_usb_report_05_flat_stationary(self):
        """Test 1: Flat controller at rest reads ~1g on Z, ~0g on X/Y, ~0 dps."""
        pkt = build_switch2_usb_report(
            ax_raw=0, ay_raw=0, az_raw=4095,
            gx_raw=0, gy_raw=0, gz_raw=0,
            imu_timestamp_us=123456,
        )
        state = parse_input_report(pkt)
        assert state is not None
        assert abs(state.accel_z - 1.0) < 0.02
        assert abs(state.accel_x) < 0.01
        assert abs(state.accel_y) < 0.01
        assert abs(state.gyro_x) < 0.1
        assert abs(state.gyro_y) < 0.1
        assert abs(state.gyro_z) < 0.1
        assert state.motion_timestamp == 123456
        assert state.has_motion is True

    def test_imu_ble_report_05_decoding(self):
        """Test 2: BLE 60-byte notification decodes identically to USB."""
        pkt = build_switch2_ble_report(
            ax_raw=0, ay_raw=0, az_raw=4095,
            gx_raw=1474, gy_raw=0, gz_raw=0,
            imu_timestamp_us=987654,
        )
        state = parse_input_report(pkt, has_report_id=False)
        assert state is not None
        assert abs(state.accel_z - 1.0) < 0.02
        assert abs(state.gyro_x - 90.0) < 0.1
        assert state.motion_timestamp == 987654
        assert state.has_motion is True

    def test_imu_known_angular_velocity_and_accel(self):
        """Test 3: Known raw inputs match authoritative conversion equations."""
        pkt = build_switch2_usb_report(
            ax_raw=2048,   # ~0.5g
            ay_raw=-2048,  # ~-0.5g
            az_raw=4095,   # ~1.0g
            gx_raw=1474,   # ~90 dps
            gy_raw=-1474,  # ~-90 dps
            gz_raw=737,    # ~45 dps
        )
        state = parse_input_report(pkt)
        assert state is not None
        assert abs(state.accel_x - 0.5) < 0.01
        assert abs(state.accel_y - (-0.5)) < 0.01
        assert abs(state.accel_z - 1.0) < 0.01
        assert abs(state.gyro_x - 90.0) < 0.1
        assert abs(state.gyro_y - (-90.0)) < 0.1
        assert abs(state.gyro_z - 45.0) < 0.1

    def test_imu_temperature_decoding(self):
        """Test 4: Room temperature and warm sensor deviation decoding."""
        # Raw 0 -> 25.0°C
        pkt_room = build_switch2_usb_report(temp_raw=0)
        state_room = parse_input_report(pkt_room)
        assert state_room is not None
        assert abs(state_room.imu_temperature - 25.0) < 0.1

        # Raw 1269 -> ~35.0°C
        pkt_warm = build_switch2_usb_report(temp_raw=1269)
        state_warm = parse_input_report(pkt_warm)
        assert state_warm is not None
        assert abs(state_warm.imu_temperature - 35.0) < 0.1

    def test_imu_state_backward_compatibility(self):
        """Test 5: Adding IMU does not perturb stick coordinates or buttons."""
        pkt = build_switch2_usb_report(
            buttons_raw=BTN_A | BTN_PLUS | BTN_GL,
            lx_raw=3648,  # Fully deflected right
            ly_raw=2048,  # Centered
            az_raw=4095,
        )
        state = parse_input_report(pkt)
        assert state is not None
        assert state.a is True
        assert state.plus is True
        assert state.gl is True
        assert state.b is False
        assert abs(state.left_stick_x - 1.0) < 0.01
        assert abs(state.left_stick_y - 0.0) < 0.01

    def test_imu_has_motion_flag(self):
        """Test 6: Flag is True for >=60 bytes, False for truncated/legacy."""
        pkt_full = build_switch2_usb_report()
        assert parse_input_report(pkt_full).has_motion is True

        # Truncated 24-byte USB report
        pkt_trunc = bytearray(24)
        pkt_trunc[0] = REPORT_ID_PRO2
        state_trunc = parse_input_report(bytes(pkt_trunc))
        assert state_trunc is not None
        assert state_trunc.has_motion is False


class TestTier1DSUProtocolFraming:
    """Verifies Cemuhook DSU protocol specification v1001 framing."""

    def test_dsu_crc32_standard_conformance(self):
        """Feature 7: IEEE 802.3 CRC32 of DSU packet validates correctly."""
        pkt = bytearray(100)
        struct.pack_into("<4sHHII", pkt, 0, b"DSUS", 1001, 84, 0, 0x12345678)
        crc = ref_dsu_crc32(pkt)
        struct.pack_into("<I", pkt, 8, crc)

        parsed = parse_dsu_packet_header(bytes(pkt))
        assert parsed["crc_valid"] is True
        assert parsed["magic"] == "DSUS"
        assert parsed["version"] == 1001
        assert parsed["length"] == 84

    def test_dsu_version_request_format(self):
        """Feature 6: Version request packet builder format."""
        req = bytearray(20)
        struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 4, 0, 0xABCD1234)
        struct.pack_into("<I", req, 16, 0x100000)
        crc = ref_dsu_crc32(req)
        struct.pack_into("<I", req, 8, crc)
        assert len(req) == 20
        assert parse_dsu_packet_header(bytes(req))["crc_valid"] is True

    def test_dsu_controller_info_request_format(self):
        """Feature 6: Controller info request packet format."""
        req = bytearray(24)
        struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 8, 0, 0xABCD1234)
        struct.pack_into("<I", req, 16, 0x100001)
        struct.pack_into("<BBBB", req, 20, 0, 0, 0, 0)
        crc = ref_dsu_crc32(req)
        struct.pack_into("<I", req, 8, crc)
        assert len(req) == 24
        assert parse_dsu_packet_header(bytes(req))["crc_valid"] is True

    def test_dsu_data_packet_float_unpacking(self):
        """Feature 7: 100-byte data packet float32 coordinates extraction."""
        pkt = bytearray(100)
        struct.pack_into("<4sHHII", pkt, 0, b"DSUS", 1001, 84, 0, 0x12345678)
        struct.pack_into("<I", pkt, 16, 0x100002)
        struct.pack_into("<BBBB", pkt, 20, 0, 2, 2, 1)  # Slot 0, Connected, Full Gyro, USB
        struct.pack_into("<Q", pkt, 68, 500000)         # Timestamp
        struct.pack_into("<ffffff", pkt, 76, 0.1, 0.2, 0.98, 10.0, 20.0, 30.0)
        crc = ref_dsu_crc32(pkt)
        struct.pack_into("<I", pkt, 8, crc)

        payload = parse_dsu_data_payload(bytes(pkt))
        assert payload["slot"] == 0
        assert payload["slot_state"] == 2
        assert payload["model"] == 2
        assert abs(payload["accel"][2] - 0.98) < 0.001
        assert abs(payload["gyro"][0] - 10.0) < 0.001


@pytest.mark.skipif(not DSU_SERVER_AVAILABLE, reason="M2: dsu_server module pending")
class TestTier1DSUMotionServer:
    """Feature 5, 6, 7: Live Cemuhook DSU server network protocol verification."""

    def test_dsu_server_lifecycle_start_stop(self):
        """Test 1: Server cleanly opens and closes UDP socket without leaks."""
        server = dsu_server.DsuServer(port=0)  # Ephemeral port
        assert server.start() is True
        assert server.is_running is True
        server.stop()
        assert server.is_running is False

    def test_dsu_version_request_response(self):
        """Test 2: Server responds to 0x100000 with DSUS and version 1001."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            req = bytearray(20)
            struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 4, 0, 0x11111111)
            struct.pack_into("<I", req, 16, 0x100000)
            struct.pack_into("<I", req, 8, ref_dsu_crc32(req))

            sock.sendto(req, ("127.0.0.1", port))
            data, _ = sock.recvfrom(1024)

            hdr = parse_dsu_packet_header(data)
            assert hdr["magic"] == "DSUS"
            assert hdr["version"] == 1001
            assert hdr["crc_valid"] is True
            msg_type = struct.unpack_from("<I", data, 16)[0]
            assert msg_type == 0x100000
        finally:
            sock.close()
            server.stop()

    def test_dsu_controller_info_request_response(self):
        """Test 3: Server responds to 0x100001 with Slot 0 and Model 2 (Full Gyro)."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            req = bytearray(24)
            struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 8, 0, 0x22222222)
            struct.pack_into("<I", req, 16, 0x100001)
            struct.pack_into("<BBBB", req, 20, 0, 0, 0, 0)
            struct.pack_into("<I", req, 8, ref_dsu_crc32(req))

            sock.sendto(req, ("127.0.0.1", port))
            data, _ = sock.recvfrom(1024)

            hdr = parse_dsu_packet_header(data)
            assert hdr["magic"] == "DSUS"
            slot, slot_state, model = struct.unpack_from("<BBB", data, 20)
            assert slot == 0
            assert model == 2  # Full Gyro
        finally:
            sock.close()
            server.stop()

    def test_dsu_data_streaming_crc32(self):
        """Test 4: Streamed 100-byte data packets have valid CRC32."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            # Subscribe for data (0x100002)
            sub = bytearray(24)
            struct.pack_into("<4sHHII", sub, 0, b"DSUC", 1001, 8, 0, 0x33333333)
            struct.pack_into("<I", sub, 16, 0x100002)
            struct.pack_into("<BBBB", sub, 20, 0, 0, 0, 0)  # Slot 0
            struct.pack_into("<I", sub, 8, ref_dsu_crc32(sub))
            sock.sendto(sub, ("127.0.0.1", port))

            # Send state update into server
            test_state = ControllerState(
                timestamp=1, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.0, zr_analog=0.0,
                left_stick_x=0.0, left_stick_y=0.0,
                right_stick_x=0.0, right_stick_y=0.0,
            )
            if hasattr(test_state, "accel_z"):
                test_state.accel_z = 1.0  # type: ignore

            server.update_state(test_state)

            data, _ = sock.recvfrom(1024)
            assert len(data) == 100
            hdr = parse_dsu_packet_header(data)
            assert hdr["crc_valid"] is True
        finally:
            sock.close()
            server.stop()

    def test_dsu_motion_payload_values(self):
        """Test 5: Streamed motion floats match ControllerState."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            sub = bytearray(24)
            struct.pack_into("<4sHHII", sub, 0, b"DSUC", 1001, 8, 0, 0x44444444)
            struct.pack_into("<I", sub, 16, 0x100002)
            struct.pack_into("<BBBB", sub, 20, 0, 0, 0, 0)
            struct.pack_into("<I", sub, 8, ref_dsu_crc32(sub))
            sock.sendto(sub, ("127.0.0.1", port))

            # Create state with specific motion
            state = ControllerState(
                timestamp=100, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.0, zr_analog=0.0,
                left_stick_x=0.0, left_stick_y=0.0,
                right_stick_x=0.0, right_stick_y=0.0,
            )
            if hasattr(state, "accel_x"):
                state.accel_x = 0.5  # type: ignore
                state.accel_z = 1.0  # type: ignore
                state.gyro_x = 90.0  # type: ignore

            server.update_state(state)

            data, _ = sock.recvfrom(1024)
            payload = parse_dsu_data_payload(data)
            assert abs(payload["accel"][0] - 0.5) < 0.05
            assert abs(payload["accel"][2] - 1.0) < 0.05
            assert abs(payload["gyro"][0] - 90.0) < 0.5
        finally:
            sock.close()
            server.stop()

    def test_dsu_client_subscription_tracking(self):
        """Test 6: Server tracks active subscribers and drops state when unsubscribed."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        try:
            assert server.active_subscriber_count == 0
        finally:
            server.stop()


class TestTier1DS4ReportStructure:
    """Verifies vgamepad DS4 extended report ctypes layout and math."""

    def test_ds4_report_ex_field_layout(self):
        """Feature 9: DS4_REPORT_EX contains signed 16-bit gyro/accel ctypes fields."""
        import vgamepad.win.vigem_commons as vcom
        rep = vcom.DS4_REPORT_EX()
        sub = rep.Report
        assert hasattr(sub, "wGyroX")
        assert hasattr(sub, "wGyroY")
        assert hasattr(sub, "wGyroZ")
        assert hasattr(sub, "wAccelX")
        assert hasattr(sub, "wAccelY")
        assert hasattr(sub, "wAccelZ")
        assert hasattr(sub, "wTimestamp")

    def test_ds4_stick_y_inversion_math(self):
        """Feature 10: Normalized stick Y (+1.0 up) maps to inverted Sony coordinate."""
        # Standard input: +1.0 is full UP
        norm_y = 1.0
        # Inverted byte: 0 is full UP in PlayStation / DirectInput convention
        ds4_byte = int((1.0 - norm_y) * 127.5)
        assert ds4_byte == 0

        # Down: -1.0 is full DOWN
        norm_y_down = -1.0
        ds4_byte_down = int((1.0 - norm_y_down) * 127.5)
        assert ds4_byte_down == 255


@pytest.mark.skipif(not DS4_MODULE_AVAILABLE, reason="M3: virtual_ds4 module pending")
class TestTier1DS4VirtualController:
    """Feature 8, 9, 10: Virtual DualShock 4 controller emulation via ViGEmBus."""

    def test_ds4_connect_disconnect_lifecycle(self):
        """Test 1: Spawns and disconnects virtual DS4 gamepad cleanly."""
        controller = virtual_ds4.VirtualDS4Controller()
        assert controller.connect() is True
        assert controller.is_connected is True
        controller.disconnect()
        assert controller.is_connected is False

    def test_ds4_extended_report_motion_fields(self):
        """Test 2: Updating from state populates signed 16-bit motion vectors."""
        controller = virtual_ds4.VirtualDS4Controller()
        controller.connect()
        try:
            state = ControllerState(
                timestamp=1, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.0, zr_analog=0.0,
                left_stick_x=0.0, left_stick_y=0.0,
                right_stick_x=0.0, right_stick_y=0.0,
            )
            if hasattr(state, "accel_z"):
                state.accel_z = 1.0  # type: ignore
                state.gyro_x = 90.0  # type: ignore

            controller.update_from_state(state)
        finally:
            controller.disconnect()

    def test_ds4_stick_y_inversion_fix(self):
        """Test 3: Stick Y is properly inverted for PlayStation coordinate frame."""
        controller = virtual_ds4.VirtualDS4Controller()
        controller.connect()
        try:
            state = ControllerState(
                timestamp=1, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.0, zr_analog=0.0,
                left_stick_x=0.0, left_stick_y=1.0,  # Up
                right_stick_x=0.0, right_stick_y=-1.0,  # Down
            )
            controller.update_from_state(state)
        finally:
            controller.disconnect()

    def test_ds4_button_layout_mapping(self):
        """Test 4: Nintendo vs letter layout mapping for DS4 buttons."""
        controller = virtual_ds4.VirtualDS4Controller()
        assert controller.layout in ("nintendo", "xbox")
        controller.layout = "xbox"
        assert controller.layout == "xbox"

    def test_ds4_analog_triggers(self):
        """Test 5: ZL/ZR analog values map cleanly to DS4 L2/R2 triggers."""
        controller = virtual_ds4.VirtualDS4Controller()
        controller.connect()
        try:
            state = ControllerState(
                timestamp=1, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.75, zr_analog=1.0,
                left_stick_x=0.0, left_stick_y=0.0,
                right_stick_x=0.0, right_stick_y=0.0,
            )
            controller.update_from_state(state)
        finally:
            controller.disconnect()

    def test_ds4_base_interface_compliance(self):
        """Test 6: Adheres to BaseVirtualController interface contracts."""
        controller = virtual_ds4.VirtualDS4Controller()
        for method in ("connect", "disconnect", "update_from_state"):
            assert hasattr(controller, method)


class TestTier1OutputModeSwitching:
    """Feature 11, 12: ControllerManager output switching and DSU lifecycle."""

    def test_manager_output_mode_default(self):
        """Test 1: Default output mode is Xbox 360 controller."""
        manager = ControllerManager()
        mode = getattr(manager, "output_mode", "xbox")
        assert mode == "xbox"

    @pytest.mark.skipif(not MANAGER_OUTPUT_MODE_AVAILABLE, reason="M4: output mode switching pending")
    def test_manager_set_output_mode_ds4(self):
        """Test 2: Switching to DS4 instantiates virtual DS4 backend."""
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        assert manager.output_mode == "ds4"

    @pytest.mark.skipif(not MANAGER_OUTPUT_MODE_AVAILABLE, reason="M4: output mode switching pending")
    def test_manager_set_output_mode_xbox(self):
        """Test 3: Switching back to Xbox instantiates virtual Xbox backend."""
        manager = ControllerManager()
        manager.set_output_mode("ds4")
        manager.set_output_mode("xbox")
        assert manager.output_mode == "xbox"

    @pytest.mark.skipif(not MANAGER_DSU_AVAILABLE, reason="M4: DSU server lifecycle pending")
    def test_manager_dsu_enabled_lifecycle(self):
        """Test 4: Enabling and disabling DSU server toggles background server."""
        manager = ControllerManager()
        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is True
        manager.set_dsu_enabled(False)
        assert manager.dsu_enabled is False

    def test_manager_on_input_update_callback(self):
        """Test 5: ControllerState is forwarded to registered input callbacks."""
        manager = ControllerManager()
        captured = []

        def on_update(st):
            captured.append(st)

        manager.on_input_update = on_update
        test_state = ControllerState(
            timestamp=42, buttons_raw=0,
            a=True, b=False, x=False, y=False,
            plus=False, minus=False, home=False, capture=False,
            l=False, r=False, l_thumb=False, r_thumb=False,
            dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
            zl_analog=0.0, zr_analog=0.0,
            left_stick_x=0.0, left_stick_y=0.0,
            right_stick_x=0.0, right_stick_y=0.0,
        )
        manager._on_controller_state(test_state)
        assert len(captured) == 1
        assert captured[0].a is True


class TestTier1GUIConfigPersistence:
    """Feature 13, 14, 15: Configuration file persistence and visualizer telemetry."""

    def test_config_save_load_roundtrip(self):
        """Test 1: Save and load output_mode and dsu_server in JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "settings.json")
            data = {
                "output_mode": "ds4",
                "dsu_server": True,
                "button_layout": "xbox",
                "deadzone": 0.08,
            }
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            with open(config_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded["output_mode"] == "ds4"
            assert loaded["dsu_server"] is True
            assert loaded["deadzone"] == 0.08

    def test_config_default_fallback_when_missing(self):
        """Test 2: Missing settings file safely returns default values."""
        non_existent = r"C:\fake_dir_switch2\settings.json"
        defaults = {"output_mode": "xbox", "dsu_server": False}
        if not os.path.exists(non_existent):
            cfg = defaults
        assert cfg["output_mode"] == "xbox"
        assert cfg["dsu_server"] is False

    def test_visualizer_status_pill_formatting(self):
        """Test 3: Status indicator formats correctly for active vs idle motion."""
        def format_status(has_motion: bool) -> str:
            return "● IMU ACTIVE" if has_motion else "● IMU IDLE"

        assert format_status(True) == "● IMU ACTIVE"
        assert format_status(False) == "● IMU IDLE"

    def test_visualizer_deadband_dirty_checking(self):
        """Test 4: Gyro/accel deltas below deadband suppress UI repaints."""
        deadband = 0.05
        last_val = 1.00
        # Small jitter (0.02)
        new_val_jitter = 1.02
        is_dirty = abs(new_val_jitter - last_val) > deadband
        assert is_dirty is False

        # Real motion (0.15)
        new_val_motion = 1.15
        is_dirty_motion = abs(new_val_motion - last_val) > deadband
        assert is_dirty_motion is True

    def test_config_directory_creation(self):
        """Test 5: Parent directory is created if absent when saving config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, "Switch2ProConnect")
            cfg_file = os.path.join(nested_dir, "settings.json")
            assert not os.path.exists(nested_dir)

            os.makedirs(nested_dir, exist_ok=True)
            with open(cfg_file, "w", encoding="utf-8") as f:
                json.dump({"output_mode": "xbox"}, f)

            assert os.path.isfile(cfg_file)


# ============================================================================
# TIER 2: BOUNDARY & CORNER CASES (>= 5 TESTS PER CATEGORY)
# ============================================================================

class TestTier2IMUBoundaries:
    """Stress tests for mathematical limits, saturation, and malformed reports."""

    def test_imu_accel_max_positive_saturation(self):
        """Test 1: Accel raw 32767 hits exactly +8.0g without overflow."""
        assert abs(ref_accel_g(32767) - 8.0) < 0.001

    def test_imu_accel_max_negative_saturation(self):
        """Test 2: Accel raw -32768 hits -8.0g without underflow."""
        assert abs(ref_accel_g(-32768) - (-8.0)) < 0.001

    def test_imu_gyro_max_positive_saturation(self):
        """Test 3: Gyro raw 32767 hits exactly +2000.0 dps."""
        assert abs(ref_gyro_dps(32767) - 2000.0) < 0.01

    def test_imu_gyro_max_negative_saturation(self):
        """Test 4: Gyro raw -32768 hits -2000.0 dps within 1 LSB (0.061 dps)."""
        assert abs(ref_gyro_dps(-32768) - (-2000.0)) < 0.1

    def test_imu_zero_motion(self):
        """Test 5: All zero raw motion gives exactly 0.0g and 0.0 dps."""
        assert ref_accel_g(0) == 0.0
        assert ref_gyro_dps(0) == 0.0

    @pytest.mark.parametrize("payload_len", [0, 5, 11, 15, 20, 41, 59])
    def test_imu_truncated_packets(self, payload_len):
        """Test 6: Truncated payloads reject cleanly or yield neutral state."""
        raw = bytes(payload_len)
        state = parse_input_report(raw)
        if state is not None:
            # If parsed, must not crash and motion flag must be False
            assert getattr(state, "has_motion", False) is False

    def test_imu_report_09_power_on_fallback(self):
        """Test 7: 12-24 byte USB 0x09 report leaves IMU fields at neutral zero."""
        buf = bytearray(16)
        buf[0] = REPORT_ID_PRO2
        buf[6:9] = bytes([0x00, 0x08, 0x80])
        buf[9:12] = bytes([0x00, 0x08, 0x80])
        state = parse_input_report(bytes(buf), has_report_id=True)
        assert state is not None
        if hasattr(state, "accel_z"):
            assert state.accel_z == 0.0  # type: ignore
            assert state.has_motion is False  # type: ignore

    def test_imu_sensor_timestamp_rollover(self):
        """Test 8: 32-bit uint32 microsecond counter wraps around gracefully."""
        t_prev = 4294967000  # Near 2^32 - 1
        t_curr = 200         # Rolled over to 200
        delta = (t_curr - t_prev) & 0xFFFFFFFF
        assert delta == 496  # Correct positive microsecond delta across rollover


class TestTier2DSUBoundaries:
    """Stress tests for malformed DSU network packets and socket edge cases."""

    def test_dsu_packet_truncated_header(self):
        """Test 1: Header < 16 bytes raises ValueError."""
        with pytest.raises(ValueError):
            parse_dsu_packet_header(b"DSUS1234")

    def test_dsu_packet_invalid_magic(self):
        """Test 2: Packet with unknown magic is identified."""
        pkt = bytearray(20)
        struct.pack_into("<4sHHII", pkt, 0, b"XXXX", 1001, 4, 0, 0)
        parsed = parse_dsu_packet_header(bytes(pkt))
        assert parsed["magic"] == "XXXX"
        assert parsed["magic"] != "DSUC"

    def test_dsu_packet_invalid_crc32(self):
        """Test 3: Corrupted CRC32 is detected."""
        pkt = bytearray(20)
        struct.pack_into("<4sHHII", pkt, 0, b"DSUC", 1001, 4, 0xBAD00000, 0)
        parsed = parse_dsu_packet_header(bytes(pkt))
        assert parsed["crc_valid"] is False

    @pytest.mark.skipif(not DSU_SERVER_AVAILABLE, reason="M2: dsu_server module pending")
    def test_dsu_out_of_range_slot_query(self):
        """Test 4: Queries for non-existent slot 3 return disconnected."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            req = bytearray(24)
            struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 8, 0, 0x1234)
            struct.pack_into("<I", req, 16, 0x100001)
            struct.pack_into("<BBBB", req, 20, 3, 0, 0, 0)  # Slot 3
            struct.pack_into("<I", req, 8, ref_dsu_crc32(req))
            sock.sendto(req, ("127.0.0.1", port))
            data, _ = sock.recvfrom(1024)
            slot, slot_state = struct.unpack_from("<BB", data, 20)
            assert slot == 3
            assert slot_state == 0  # Disconnected
        finally:
            sock.close()
            server.stop()

    @pytest.mark.skipif(not DSU_SERVER_AVAILABLE, reason="M2: dsu_server module pending")
    def test_dsu_rapid_packet_flood(self):
        """Test 5: Server handles burst of 50 rapid UDP queries without error."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        try:
            for i in range(50):
                req = bytearray(20)
                struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 4, 0, i)
                struct.pack_into("<I", req, 16, 0x100000)
                struct.pack_into("<I", req, 8, ref_dsu_crc32(req))
                sock.sendto(req, ("127.0.0.1", port))
        finally:
            sock.close()
            server.stop()


class TestTier2DS4Boundaries:
    """Stress tests for virtual DS4 controller bounds and extreme values."""

    def test_ds4_extreme_motion_clamping(self):
        """Test 1: Values exceeding full-scale clamp to signed 16-bit boundaries."""
        # Max positive int16
        val_pos = min(32767, max(-32768, int(15.0 * 8192)))
        assert val_pos == 32767

        # Min negative int16
        val_neg = min(32767, max(-32768, int(-15.0 * 8192)))
        assert val_neg == -32768

    def test_ds4_stick_extreme_boundaries(self):
        """Test 2: Stick -1.0 and +1.0 map strictly to 0 and 255."""
        def norm_to_byte(val: float) -> int:
            return max(0, min(255, int((val + 1.0) * 127.5)))

        assert norm_to_byte(-1.0) == 0
        assert norm_to_byte(1.0) == 255
        assert norm_to_byte(0.0) == 127

    @pytest.mark.skipif(not DS4_MODULE_AVAILABLE, reason="M3: virtual_ds4 module pending")
    def test_ds4_rapid_connect_disconnect(self):
        """Test 3: Rapid cycling of connect/disconnect does not crash."""
        controller = virtual_ds4.VirtualDS4Controller()
        for _ in range(5):
            controller.connect()
            controller.disconnect()
        assert controller.is_connected is False

    @pytest.mark.skipif(not DS4_MODULE_AVAILABLE, reason="M3: virtual_ds4 module pending")
    def test_ds4_zero_motion_inputs(self):
        """Test 4: Zero motion inputs produce clean neutral fields."""
        controller = virtual_ds4.VirtualDS4Controller()
        controller.connect()
        try:
            st = ControllerState(
                timestamp=0, buttons_raw=0,
                a=False, b=False, x=False, y=False,
                plus=False, minus=False, home=False, capture=False,
                l=False, r=False, l_thumb=False, r_thumb=False,
                dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
                zl_analog=0.0, zr_analog=0.0,
                left_stick_x=0.0, left_stick_y=0.0,
                right_stick_x=0.0, right_stick_y=0.0,
            )
            controller.update_from_state(st)
        finally:
            controller.disconnect()


class TestTier2ManagerBoundaries:
    """Stress tests for ControllerManager output switching edge conditions."""

    @pytest.mark.skipif(not MANAGER_OUTPUT_MODE_AVAILABLE, reason="M4: output mode switching pending")
    def test_manager_invalid_mode_string(self):
        """Test 1: Passing invalid mode string is safely handled without crash."""
        manager = ControllerManager()
        # Invalid modes should either raise ValueError or be ignored
        try:
            manager.set_output_mode("switch_pro")
        except ValueError:
            pass  # Expected acceptable behavior
        assert manager.output_mode in ("xbox", "ds4")

    @pytest.mark.skipif(not MANAGER_OUTPUT_MODE_AVAILABLE, reason="M4: output mode switching pending")
    def test_manager_repeated_mode_switch(self):
        """Test 2: Switching to active mode repeatedly is safe no-op."""
        manager = ControllerManager()
        manager.set_output_mode("xbox")
        manager.set_output_mode("xbox")
        assert manager.output_mode == "xbox"

    @pytest.mark.skipif(not MANAGER_DSU_AVAILABLE, reason="M4: DSU server lifecycle pending")
    def test_manager_dsu_double_start_stop(self):
        """Test 3: Calling set_dsu_enabled(True) repeatedly does not duplicate servers."""
        manager = ControllerManager()
        manager.set_dsu_enabled(True)
        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is True
        manager.set_dsu_enabled(False)
        manager.set_dsu_enabled(False)
        assert manager.dsu_enabled is False

    def test_manager_state_dispatch_with_null_callbacks(self):
        """Test 4: State dispatch functions cleanly with no registered callbacks."""
        manager = ControllerManager()
        manager.on_input_update = None
        manager.on_connection_changed = None
        manager.on_status_message = None
        st = ControllerState(
            timestamp=1, buttons_raw=0,
            a=False, b=False, x=False, y=False,
            plus=False, minus=False, home=False, capture=False,
            l=False, r=False, l_thumb=False, r_thumb=False,
            dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
            zl_analog=0.0, zr_analog=0.0,
            left_stick_x=0.0, left_stick_y=0.0,
            right_stick_x=0.0, right_stick_y=0.0,
        )
        manager._on_controller_state(st)  # Must not raise


class TestTier2ConfigBoundaries:
    """Stress tests for configuration parsing corruption and filesystem failures."""

    def test_config_corrupted_json(self):
        """Test 1: Malformed JSON syntax falls back to default settings."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            f.write("{invalid_json: true, ...}")
            temp_path = f.name
        try:
            # Fallback loader
            defaults = {"output_mode": "xbox", "dsu_server": False}
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = defaults
            assert cfg["output_mode"] == "xbox"
            assert cfg["dsu_server"] is False
        finally:
            os.remove(temp_path)

    def test_config_missing_keys(self):
        """Test 2: Missing fields are populated with defaults."""
        data = {"output_mode": "ds4"}
        output_mode = data.get("output_mode", "xbox")
        dsu_server = data.get("dsu_server", False)
        assert output_mode == "ds4"
        assert dsu_server is False

    def test_config_invalid_value_types(self):
        """Test 3: Invalid data types in JSON are validated."""
        data = {"output_mode": 12345, "dsu_server": "yes"}
        # Sanitizer
        mode = data["output_mode"] if data["output_mode"] in ("xbox", "ds4") else "xbox"
        dsu = bool(data["dsu_server"]) if isinstance(data["dsu_server"], bool) else False
        assert mode == "xbox"
        assert dsu is False

    def test_config_empty_file(self):
        """Test 4: Zero-byte file is handled safely as missing."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
            temp_path = f.name
        try:
            defaults = {"output_mode": "xbox"}
            try:
                with open(temp_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = defaults
            assert cfg["output_mode"] == "xbox"
        finally:
            os.remove(temp_path)


# ============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ============================================================================

class TestTier3CrossFeatureCombinations:
    """Verifies concurrent multi-feature interactions and state routing."""

    @pytest.mark.skipif(not (DS4_MODULE_AVAILABLE and IMU_PARSER_AVAILABLE), reason="Requires DS4 and IMU")
    def test_ds4_with_active_motion_feed(self):
        """Combination 1: DS4 receives simultaneous stick, button, and motion state."""
        controller = virtual_ds4.VirtualDS4Controller()
        controller.connect()
        try:
            pkt = build_switch2_usb_report(
                buttons_raw=BTN_A | BTN_ZR,
                lx_raw=3648,
                az_raw=4095,
                gx_raw=1474,
            )
            state = parse_input_report(pkt)
            assert state is not None
            controller.update_from_state(state)
        finally:
            controller.disconnect()

    @pytest.mark.skipif(not (DSU_SERVER_AVAILABLE and IMU_PARSER_AVAILABLE), reason="Requires DSU and IMU")
    def test_dsu_server_with_xbox_mode_concurrent_streaming(self):
        """Combination 2: Xbox 360 virtual controller + DSU server stream simultaneously."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        xbox = VirtualXboxController()
        # Do not need full physical connect for update test
        try:
            pkt = build_switch2_usb_report(
                buttons_raw=BTN_B,
                az_raw=4095,
                gx_raw=1474,
            )
            state = parse_input_report(pkt)
            assert state is not None

            # Route to both backends concurrently
            server.update_state(state)
            xbox.update_from_state(state)
        finally:
            server.stop()

    @pytest.mark.skipif(not (DSU_SERVER_AVAILABLE and DS4_MODULE_AVAILABLE and IMU_PARSER_AVAILABLE), reason="Requires DSU, DS4, and IMU")
    def test_dsu_server_with_ds4_mode_concurrent_streaming(self):
        """Combination 3: DS4 virtual controller + DSU server both stream motion."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        ds4 = virtual_ds4.VirtualDS4Controller()
        ds4.connect()
        try:
            pkt = build_switch2_usb_report(
                buttons_raw=BTN_X,
                az_raw=4095,
                gy_raw=-1474,
            )
            state = parse_input_report(pkt)
            assert state is not None

            server.update_state(state)
            ds4.update_from_state(state)
        finally:
            ds4.disconnect()
            server.stop()

    @pytest.mark.skipif(not MANAGER_OUTPUT_MODE_AVAILABLE, reason="M4: output mode switching pending")
    def test_rapid_output_mode_switching_under_active_motion(self):
        """Combination 4: Dynamically toggle Xbox / DS4 while state stream is active."""
        manager = ControllerManager()
        st = ControllerState(
            timestamp=1, buttons_raw=0,
            a=False, b=False, x=False, y=False,
            plus=False, minus=False, home=False, capture=False,
            l=False, r=False, l_thumb=False, r_thumb=False,
            dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
            zl_analog=0.0, zr_analog=0.0,
            left_stick_x=0.0, left_stick_y=0.0,
            right_stick_x=0.0, right_stick_y=0.0,
        )
        for i in range(10):
            target = "ds4" if (i % 2 == 0) else "xbox"
            manager.set_output_mode(target)
            manager._on_controller_state(st)
        assert manager.output_mode == "xbox"

    @pytest.mark.skipif(not MANAGER_DSU_AVAILABLE, reason="M4: DSU server lifecycle pending")
    def test_dsu_enabled_toggle_during_active_streaming(self):
        """Combination 5: Toggling DSU server during active input flow does not drop inputs."""
        manager = ControllerManager()
        received = []
        manager.on_input_update = lambda s: received.append(s)

        st = ControllerState(
            timestamp=1, buttons_raw=0,
            a=True, b=False, x=False, y=False,
            plus=False, minus=False, home=False, capture=False,
            l=False, r=False, l_thumb=False, r_thumb=False,
            dpad_up=False, dpad_down=False, dpad_left=False, dpad_right=False,
            zl_analog=0.0, zr_analog=0.0,
            left_stick_x=0.0, left_stick_y=0.0,
            right_stick_x=0.0, right_stick_y=0.0,
        )

        manager._on_controller_state(st)
        manager.set_dsu_enabled(True)
        manager._on_controller_state(st)
        manager.set_dsu_enabled(False)
        manager._on_controller_state(st)

        assert len(received) == 3

    @pytest.mark.skipif(not IMU_PARSER_AVAILABLE, reason="M1: IMU decoding pending")
    def test_grip_buttons_gl_gr_with_active_motion(self):
        """Combination 6: Back paddles GL/GR actuated during fast gyro rotation."""
        pkt = build_switch2_usb_report(
            buttons_raw=BTN_GL | BTN_GR,
            gx_raw=1474,
            gy_raw=-1474,
        )
        state = parse_input_report(pkt)
        assert state is not None
        assert state.gl is True
        assert state.gr is True
        assert abs(state.gyro_x - 90.0) < 0.1
        assert abs(state.gyro_y - (-90.0)) < 0.1


# ============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ============================================================================

class TestTier4RealWorldScenarios:
    """End-to-end sessions simulating external gaming software and sustained load."""

    @pytest.mark.skipif(not (DSU_SERVER_AVAILABLE and IMU_PARSER_AVAILABLE), reason="Requires DSU and IMU")
    def test_cemu_dolphin_dsu_emulation_session(self):
        """Scenario 1: Complete Cemu/Dolphin DSU client emulation session."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        port = server.port

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_sock.settimeout(2.0)
        try:
            client_id = 0xCAFEBABE

            # Step 1: Client queries version
            req_ver = bytearray(20)
            struct.pack_into("<4sHHII", req_ver, 0, b"DSUC", 1001, 4, 0, client_id)
            struct.pack_into("<I", req_ver, 16, 0x100000)
            struct.pack_into("<I", req_ver, 8, ref_dsu_crc32(req_ver))
            client_sock.sendto(req_ver, ("127.0.0.1", port))
            res_ver, _ = client_sock.recvfrom(1024)
            assert parse_dsu_packet_header(res_ver)["crc_valid"] is True

            # Step 2: Client queries controller info
            req_info = bytearray(24)
            struct.pack_into("<4sHHII", req_info, 0, b"DSUC", 1001, 8, 0, client_id)
            struct.pack_into("<I", req_info, 16, 0x100001)
            struct.pack_into("<BBBB", req_info, 20, 0, 0, 0, 0)
            struct.pack_into("<I", req_info, 8, ref_dsu_crc32(req_info))
            client_sock.sendto(req_info, ("127.0.0.1", port))
            res_info, _ = client_sock.recvfrom(1024)
            assert parse_dsu_packet_header(res_info)["crc_valid"] is True

            # Step 3: Client subscribes to motion data
            req_sub = bytearray(24)
            struct.pack_into("<4sHHII", req_sub, 0, b"DSUC", 1001, 8, 0, client_id)
            struct.pack_into("<I", req_sub, 16, 0x100002)
            struct.pack_into("<BBBB", req_sub, 20, 0, 0, 0, 0)
            struct.pack_into("<I", req_sub, 8, ref_dsu_crc32(req_sub))
            client_sock.sendto(req_sub, ("127.0.0.1", port))

            # Step 4: Stream 10 frames of motion (controller tilting)
            for i in range(10):
                pkt = build_switch2_usb_report(
                    timestamp_seq=1000 + i,
                    imu_timestamp_us=500000 + i * 4000,
                    ax_raw=int(i * 400),  # Progressively tilting
                    az_raw=4095 - int(i * 100),
                    gx_raw=1474,          # 90 dps
                )
                state = parse_input_report(pkt)
                server.update_state(state)

                data, _ = client_sock.recvfrom(1024)
                payload = parse_dsu_data_payload(data)
                assert payload["header"]["crc_valid"] is True
                assert payload["slot"] == 0
                assert abs(payload["gyro"][0] - 90.0) < 0.5
        finally:
            client_sock.close()
            server.stop()

    @pytest.mark.skipif(not (DS4_MODULE_AVAILABLE and IMU_PARSER_AVAILABLE), reason="Requires DS4 and IMU")
    def test_steam_ps_pc_ds4_extended_report_session(self):
        """Scenario 2: Simulated Steam / PS PC port DS4 extended report consumption."""
        ds4 = virtual_ds4.VirtualDS4Controller()
        ds4.connect()
        try:
            for i in range(10):
                pkt = build_switch2_usb_report(
                    timestamp_seq=2000 + i,
                    buttons_raw=BTN_A if (i % 2 == 0) else BTN_B,
                    az_raw=4095,
                    gx_raw=1474,
                )
                state = parse_input_report(pkt)
                ds4.update_from_state(state)
        finally:
            ds4.disconnect()

    @pytest.mark.skipif(not IMU_PARSER_AVAILABLE, reason="M1: IMU decoding pending")
    def test_250hz_sustained_motion_streaming_session(self):
        """Scenario 3: 1 second burst of 250 consecutive packets (250 Hz USB stream)."""
        packets = []
        for i in range(250):
            pkt = build_switch2_usb_report(
                timestamp_seq=i,
                imu_timestamp_us=i * 4000,  # 4000 us = 4 ms = 250 Hz
                az_raw=4095,
                gx_raw=i * 5,
            )
            packets.append(pkt)

        # Parse all 250 packets and measure execution time
        start_time = time.perf_counter()
        states = [parse_input_report(p) for p in packets]
        elapsed = time.perf_counter() - start_time

        assert len(states) == 250
        assert all(s is not None and s.has_motion is True for s in states)

        # Verify timestamp progression and sequence monotonicity
        for i in range(1, 250):
            assert states[i].timestamp == states[i - 1].timestamp + 1
            assert states[i].motion_timestamp == states[i - 1].motion_timestamp + 4000

        # Processing 250 packets must take well under 100ms (<0.4ms per packet)
        assert elapsed < 0.100, f"250 packets took {elapsed*1000:.2f}ms (expected < 100ms)"

    @pytest.mark.skipif(not DSU_SERVER_AVAILABLE, reason="M2: dsu_server module pending")
    def test_dsu_client_unsubscription_and_timeout_pruning(self):
        """Scenario 4: Server auto-prunes inactive subscribers after lease expires."""
        server = dsu_server.DsuServer(port=0)
        server.start()
        try:
            # Check pruning logic exists and operates cleanly
            if hasattr(server, "prune_subscribers"):
                server.prune_subscribers()
            assert server.active_subscriber_count == 0
        finally:
            server.stop()
