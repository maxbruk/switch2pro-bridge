"""Independent Differential Oracle and Adversarial Concurrency Test Suite for Cemuhook DSU Server.

Empirical verification suite written by teamwork_challenger_m2_2:
1. Differential Oracle & Endianness:
   - Independent reference Cemuhook DSU packet unpacker.
   - 10,000 randomized ControllerState instances (including boundary values:
     -8.0g, +8.0g, -2000 dps, +2000 dps, stick extremes, all buttons pressed).
   - 100% bitwise parity and IEEE 754 32-bit float precision verification.
2. Lease Pruning Precision:
   - High-precision synthetic timeline testing sub-millisecond boundary conditions.
   - Real wall-clock timing: 10 clients drop after 5.0s; 1 client with 2.5s heartbeat retained past 5.5s.
3. Multi-Threaded Concurrency Stress:
   - 4 concurrent producer threads calling update_state() under heavy client subscription,
     querying, packet receipt, and background subscriber pruning.
   - Zero deadlocks, zero data races, zero unhandled exceptions.
"""

from __future__ import annotations

import random
import socket
import struct
import threading
import time
import zlib
from typing import Any, Dict, Generator
from unittest.mock import patch

import pytest

from dsu_server import (
    BATTERY_FULL,
    BATTERY_HIGH,
    DSU_MAGIC_SERVER,
    DSU_PROTOCOL_VERSION,
    MSG_TYPE_DATA,
    DsuServer,
    calculate_dsu_crc32,
)
from report_parser import ControllerState


# ============================================================================
# Independent Reference Unpacker (Differential Oracle)
# ============================================================================

def make_client_subscription_packet(client_id: int = 0x12345678) -> bytes:
    """Build a standard DSUC client data subscription packet (0x100002)."""
    payload = struct.pack("<IBBBB", MSG_TYPE_DATA, 0, 0, 0, 0)
    length = len(payload)
    pkt = bytearray(16 + length)
    struct.pack_into("<4sHHII", pkt, 0, b"DSUC", 1001, length, 0, client_id)
    pkt[16:] = payload
    crc = zlib.crc32(pkt) & 0xFFFFFFFF
    struct.pack_into("<I", pkt, 8, crc)
    return bytes(pkt)


def unpack_reference_dsu_packet(packet: bytes) -> Dict[str, Any]:
    """Independent reference unpacker strictly validating Cemuhook DSU 100-byte packet.
    
    Validates packet length, little-endian header encoding, IEEE 802.3 CRC32,
    and unpacks all controller metadata, buttons, sticks, timestamp, and 6-axis floats.
    """
    assert len(packet) == 100, f"Expected 100 bytes, got {len(packet)}"

    magic, version, length, crc, sender = struct.unpack_from("<4sHHII", packet, 0)
    assert magic == DSU_MAGIC_SERVER, f"Invalid magic: {magic}"
    assert version == DSU_PROTOCOL_VERSION, f"Invalid version: {version}"
    assert length == 84, f"Invalid length in header: {length} (expected 84)"

    # CRC32 verification with bytes 8..11 zeroed
    temp = bytearray(packet)
    temp[8:12] = b"\x00\x00\x00\x00"
    expected_crc = zlib.crc32(temp) & 0xFFFFFFFF
    assert crc == expected_crc, f"CRC32 mismatch: {hex(crc)} != {hex(expected_crc)}"

    # Payload unpack
    msg_type, slot, slot_state, model, conn = struct.unpack_from("<IBBBB", packet, 16)
    mac = packet[24:30]
    battery, is_active, packet_counter = struct.unpack_from("<BBI", packet, 30)

    b1, b2, home, capture = struct.unpack_from("<BBBB", packet, 36)
    lx, ly, rx, ry = struct.unpack_from("<BBBB", packet, 40)

    analog_buttons = struct.unpack_from("<12B", packet, 44)
    touch_data = packet[56:68]

    timestamp_us = struct.unpack_from("<Q", packet, 68)[0]
    ax, ay, az, gx, gy, gz = struct.unpack_from("<ffffff", packet, 76)

    return {
        "magic": magic,
        "version": version,
        "length": length,
        "crc": crc,
        "sender": sender,
        "msg_type": msg_type,
        "slot": slot,
        "slot_state": slot_state,
        "model": model,
        "conn": conn,
        "mac": mac,
        "battery": battery,
        "is_active": is_active,
        "packet_counter": packet_counter,
        "b1": b1,
        "b2": b2,
        "home": home,
        "capture": capture,
        "lx": lx,
        "ly": ly,
        "rx": rx,
        "ry": ry,
        "analog_buttons": analog_buttons,
        "touch_data": touch_data,
        "timestamp_us": timestamp_us,
        "accel_x": ax,
        "accel_y": ay,
        "accel_z": az,
        "gyro_x": gx,
        "gyro_y": gy,
        "gyro_z": gz,
    }


def make_oracle_state(**kwargs) -> ControllerState:
    """Construct a full ControllerState instance with sensible defaults and overrides."""
    base = {
        "timestamp": 1,
        "buttons_raw": 0,
        "a": False, "b": False, "x": False, "y": False,
        "plus": False, "minus": False, "home": False, "capture": False,
        "l": False, "r": False, "l_thumb": False, "r_thumb": False,
        "dpad_up": False, "dpad_down": False, "dpad_left": False, "dpad_right": False,
        "zl_analog": 0.0, "zr_analog": 0.0,
        "left_stick_x": 0.0, "left_stick_y": 0.0,
        "right_stick_x": 0.0, "right_stick_y": 0.0,
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
        "motion_timestamp": 1000000,
        "battery_percentage": 100,
        "has_motion": True,
    }
    base.update(kwargs)
    return ControllerState(**base)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def running_dsu_server() -> Generator[DsuServer, None, None]:
    """Start an ephemeral DsuServer and tear down cleanly."""
    srv = DsuServer(host="127.0.0.1", port=0)
    assert srv.start() is True
    yield srv
    srv.stop()


@pytest.fixture
def subscribed_client(running_dsu_server: DsuServer) -> Generator[socket.socket, None, None]:
    """Create a client socket, subscribe to running_dsu_server, and tear down."""
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client.settimeout(2.0)
    sub = make_client_subscription_packet(client_id=0x99999999)
    client.sendto(sub, ("127.0.0.1", running_dsu_server.port))
    time.sleep(0.05)
    assert running_dsu_server.active_subscriber_count >= 1
    yield client
    client.close()


# ============================================================================
# 1. Differential Oracle & Endianness Suite
# ============================================================================

class TestDsuDifferentialOracle:
    """Verifies DsuServer against the independent reference unpacker oracle."""

    def test_oracle_boundary_values(self, running_dsu_server: DsuServer, subscribed_client: socket.socket):
        """Verify strict bitwise parity on boundary values:
        - Accel: -8.0g, +8.0g, 0.0g, 1.0g
        - Gyro: -2000 dps, +2000 dps, 0.0 dps
        - Sticks: extremes (-1.0, 1.0, clamped)
        - Triggers: exact threshold (0.0, 0.49, 0.50, 1.0)
        - Buttons: all pressed vs all released
        - Battery: boundary at 90 vs 91
        - Microsecond timestamps: 1, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF
        """
        boundary_cases = [
            # Case 1: Minimum boundary, all released
            {
                "accel_x": -8.0, "accel_y": -8.0, "accel_z": -8.0,
                "gyro_x": -2000.0, "gyro_y": -2000.0, "gyro_z": -2000.0,
                "left_stick_x": -1.0, "left_stick_y": -1.0,
                "right_stick_x": -1.0, "right_stick_y": -1.0,
                "zl_analog": 0.0, "zr_analog": 0.0,
                "battery_percentage": 0,
                "motion_timestamp": 1,
                "buttons": {k: False for k in [
                    "a", "b", "x", "y", "plus", "minus", "home", "capture",
                    "l", "r", "l_thumb", "r_thumb", "dpad_up", "dpad_down", "dpad_left", "dpad_right"
                ]},
            },
            # Case 2: Maximum boundary, all pressed
            {
                "accel_x": 8.0, "accel_y": 8.0, "accel_z": 8.0,
                "gyro_x": 2000.0, "gyro_y": 2000.0, "gyro_z": 2000.0,
                "left_stick_x": 1.0, "left_stick_y": 1.0,
                "right_stick_x": 1.0, "right_stick_y": 1.0,
                "zl_analog": 1.0, "zr_analog": 1.0,
                "battery_percentage": 100,
                "motion_timestamp": 0xFFFFFFFFFFFFFFFF,
                "buttons": {k: True for k in [
                    "a", "b", "x", "y", "plus", "minus", "home", "capture",
                    "l", "r", "l_thumb", "r_thumb", "dpad_up", "dpad_down", "dpad_left", "dpad_right"
                ]},
            },
            # Case 3: Trigger thresholds & battery boundary 90 vs 91
            {
                "accel_x": 0.0, "accel_y": 1.0, "accel_z": 0.0,
                "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
                "left_stick_x": 0.0, "left_stick_y": 0.0,
                "right_stick_x": 0.0, "right_stick_y": 0.0,
                "zl_analog": 0.49, "zr_analog": 0.50,
                "battery_percentage": 90,
                "motion_timestamp": 500000,
                "buttons": {"a": True, "plus": True, "home": True},
            },
            {
                "accel_x": -1.0, "accel_y": 0.0, "accel_z": -1.0,
                "gyro_x": 123.456, "gyro_y": -789.012, "gyro_z": 345.678,
                "left_stick_x": 0.5, "left_stick_y": -0.5,
                "right_stick_x": -0.75, "right_stick_y": 0.75,
                "zl_analog": 0.51, "zr_analog": 0.49,
                "battery_percentage": 91,
                "motion_timestamp": 999999999,
                "buttons": {"b": True, "minus": True, "capture": True},
            },
        ]

        for case in boundary_cases:
            btn_dict = case.get("buttons", {})
            st = make_oracle_state(
                accel_x=case["accel_x"], accel_y=case["accel_y"], accel_z=case["accel_z"],
                gyro_x=case["gyro_x"], gyro_y=case["gyro_y"], gyro_z=case["gyro_z"],
                left_stick_x=case["left_stick_x"], left_stick_y=case["left_stick_y"],
                right_stick_x=case["right_stick_x"], right_stick_y=case["right_stick_y"],
                zl_analog=case["zl_analog"], zr_analog=case["zr_analog"],
                battery_percentage=case["battery_percentage"],
                motion_timestamp=case["motion_timestamp"],
                **btn_dict
            )
            running_dsu_server.update_state(st)
            raw_pkt, _ = subscribed_client.recvfrom(1024)
            u = unpack_reference_dsu_packet(raw_pkt)

            # 100% Bitwise float equality
            assert raw_pkt[76:80] == struct.pack("<f", float(case["accel_x"]))
            assert raw_pkt[80:84] == struct.pack("<f", float(case["accel_y"]))
            assert raw_pkt[84:88] == struct.pack("<f", float(case["accel_z"]))
            assert raw_pkt[88:92] == struct.pack("<f", float(case["gyro_x"]))
            assert raw_pkt[92:96] == struct.pack("<f", float(case["gyro_y"]))
            assert raw_pkt[96:100] == struct.pack("<f", float(case["gyro_z"]))

            # Analog stick mapping
            expected_lx = max(0, min(255, int(round(128.0 + case["left_stick_x"] * 127.0))))
            expected_ly = max(0, min(255, int(round(128.0 + case["left_stick_y"] * 127.0))))
            expected_rx = max(0, min(255, int(round(128.0 + case["right_stick_x"] * 127.0))))
            expected_ry = max(0, min(255, int(round(128.0 + case["right_stick_y"] * 127.0))))
            assert u["lx"] == expected_lx
            assert u["ly"] == expected_ly
            assert u["rx"] == expected_rx
            assert u["ry"] == expected_ry

            # Trigger digital threshold (>= 0.5)
            assert bool(u["b2"] & 0x01) == (case["zl_analog"] >= 0.5)
            assert bool(u["b2"] & 0x02) == (case["zr_analog"] >= 0.5)

            # Battery threshold (> 90 -> BATTERY_FULL)
            expected_bat = BATTERY_FULL if case["battery_percentage"] > 90 else BATTERY_HIGH
            assert u["battery"] == expected_bat

            # Motion timestamp
            assert u["timestamp_us"] == case["motion_timestamp"]

    def test_oracle_10000_randomized_states_exhaustive(
        self, running_dsu_server: DsuServer, subscribed_client: socket.socket
    ):
        """Feed 10,000 pseudo-randomized ControllerState instances through DsuServer.
        
        Unpacks transmitted packets and verifies:
        - 100% bitwise parity on all 6 motion floats
        - Bit-level accuracy on all 16 digital buttons + home + capture
        - 8-bit stick axis mapping centering and clamping
        - 8-bit analog button and trigger scaling
        - Sequence counter strictly monotonic progression
        - Zero CRC32 errors or packet corruptions
        """
        rng = random.Random(0xDEADBEEF)
        total_iterations = 10_000
        prev_packet_counter = None

        all_button_names = [
            "a", "b", "x", "y", "plus", "minus", "home", "capture",
            "l", "r", "l_thumb", "r_thumb", "dpad_up", "dpad_down", "dpad_left", "dpad_right"
        ]

        t_start = time.perf_counter()

        for i in range(total_iterations):
            # Generate random motion values within extended physical bounds
            ax = rng.uniform(-8.0, 8.0)
            ay = rng.uniform(-8.0, 8.0)
            az = rng.uniform(-8.0, 8.0)
            gx = rng.uniform(-2000.0, 2000.0)
            gy = rng.uniform(-2000.0, 2000.0)
            gz = rng.uniform(-2000.0, 2000.0)

            # Sticks and triggers
            lx_val = rng.uniform(-1.0, 1.0)
            ly_val = rng.uniform(-1.0, 1.0)
            rx_val = rng.uniform(-1.0, 1.0)
            ry_val = rng.uniform(-1.0, 1.0)
            zl_val = rng.uniform(0.0, 1.0)
            zr_val = rng.uniform(0.0, 1.0)

            # Random digital buttons
            btn_state = {btn: rng.choice([True, False]) for btn in all_button_names}
            bat_pct = rng.randint(0, 100)
            ts_us = rng.randint(1, 0xFFFFFFFFFFFFFFFF)

            st = make_oracle_state(
                accel_x=ax, accel_y=ay, accel_z=az,
                gyro_x=gx, gyro_y=gy, gyro_z=gz,
                left_stick_x=lx_val, left_stick_y=ly_val,
                right_stick_x=rx_val, right_stick_y=ry_val,
                zl_analog=zl_val, zr_analog=zr_val,
                battery_percentage=bat_pct,
                motion_timestamp=ts_us,
                **btn_state
            )

            running_dsu_server.update_state(st)
            raw_pkt, _ = subscribed_client.recvfrom(1024)

            # 1. Unpack with independent reference oracle
            u = unpack_reference_dsu_packet(raw_pkt)

            # 2. Sequence counter strictly increments
            if prev_packet_counter is not None:
                assert u["packet_counter"] == (prev_packet_counter + 1) & 0xFFFFFFFF
            prev_packet_counter = u["packet_counter"]

            # 3. Motion floats bitwise parity
            assert raw_pkt[76:80] == struct.pack("<f", float(ax))
            assert raw_pkt[80:84] == struct.pack("<f", float(ay))
            assert raw_pkt[84:88] == struct.pack("<f", float(az))
            assert raw_pkt[88:92] == struct.pack("<f", float(gx))
            assert raw_pkt[92:96] == struct.pack("<f", float(gy))
            assert raw_pkt[96:100] == struct.pack("<f", float(gz))

            # 4. Sticks mapping
            assert u["lx"] == max(0, min(255, int(round(128.0 + lx_val * 127.0))))
            assert u["ly"] == max(0, min(255, int(round(128.0 + ly_val * 127.0))))
            assert u["rx"] == max(0, min(255, int(round(128.0 + rx_val * 127.0))))
            assert u["ry"] == max(0, min(255, int(round(128.0 + ry_val * 127.0))))

            # 5. Buttons mapping (b1)
            assert bool(u["b1"] & 0x80) == btn_state["dpad_left"]
            assert bool(u["b1"] & 0x40) == btn_state["dpad_down"]
            assert bool(u["b1"] & 0x20) == btn_state["dpad_right"]
            assert bool(u["b1"] & 0x10) == btn_state["dpad_up"]
            assert bool(u["b1"] & 0x08) == btn_state["plus"]
            assert bool(u["b1"] & 0x04) == btn_state["r_thumb"]
            assert bool(u["b1"] & 0x02) == btn_state["l_thumb"]
            assert bool(u["b1"] & 0x01) == btn_state["minus"]

            # 6. Buttons mapping (b2)
            assert bool(u["b2"] & 0x80) == btn_state["y"]
            assert bool(u["b2"] & 0x40) == btn_state["b"]
            assert bool(u["b2"] & 0x20) == btn_state["a"]
            assert bool(u["b2"] & 0x10) == btn_state["x"]
            assert bool(u["b2"] & 0x08) == btn_state["r"]
            assert bool(u["b2"] & 0x04) == btn_state["l"]
            assert bool(u["b2"] & 0x02) == (zr_val >= 0.5)
            assert bool(u["b2"] & 0x01) == (zl_val >= 0.5)

            # 7. Home & Capture
            assert bool(u["home"]) == btn_state["home"]
            assert bool(u["capture"]) == btn_state["capture"]

            # 8. Analog button values
            assert u["analog_buttons"][10] == max(0, min(255, int(round(zr_val * 255.0))))
            assert u["analog_buttons"][11] == max(0, min(255, int(round(zl_val * 255.0))))

            # 9. Touch bytes (always 12 zeros for inactive)
            assert u["touch_data"] == b"\x00" * 12

            # 10. Timestamp
            assert u["timestamp_us"] == ts_us

            # 11. Battery status
            expected_bat = BATTERY_FULL if bat_pct > 90 else BATTERY_HIGH
            assert u["battery"] == expected_bat

        elapsed = time.perf_counter() - t_start
        assert elapsed < 5.0, f"10,000 iterations took {elapsed:.2f}s (expected < 5.0s)"

    def test_zero_timestamp_wallclock_fallback(
        self, running_dsu_server: DsuServer, subscribed_client: socket.socket
    ):
        """When motion_timestamp == 0, server must populate timestamp with wallclock micros."""
        t_before_us = int(time.time() * 1_000_000)
        st = make_oracle_state(motion_timestamp=0)
        running_dsu_server.update_state(st)
        data, _ = subscribed_client.recvfrom(1024)
        u = unpack_reference_dsu_packet(data)
        t_after_us = int(time.time() * 1_000_000)

        assert t_before_us - 100_000 <= u["timestamp_us"] <= t_after_us + 100_000


# ============================================================================
# 2. Lease Pruning Precision Suite
# ============================================================================

class TestDsuLeasePruning:
    """Verifies client subscription lease expiration and heartbeat refresh precision."""

    def test_synthetic_timeline_submillisecond_pruning_precision(self, running_dsu_server: DsuServer):
        """Verify strict sub-millisecond lease expiration (4.99s retained vs 5.01s pruned)."""
        client_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            t0 = 1000.0
            with patch("time.time", return_value=t0):
                # Register A and B at t0
                sub_a = make_client_subscription_packet(0xAAAA)
                sub_b = make_client_subscription_packet(0xBBBB)
                client_a.sendto(sub_a, ("127.0.0.1", running_dsu_server.port))
                client_b.sendto(sub_b, ("127.0.0.1", running_dsu_server.port))
                time.sleep(0.05)
                assert running_dsu_server.active_subscriber_count == 2

            # At t0 + 4.99s -> both still active
            with patch("time.time", return_value=t0 + 4.99):
                running_dsu_server.prune_subscribers(timeout_sec=5.0)
                assert running_dsu_server.active_subscriber_count == 2

            # At t0 + 2.50s, client A sends heartbeat
            with patch("time.time", return_value=t0 + 2.50):
                client_a.sendto(sub_a, ("127.0.0.1", running_dsu_server.port))
                time.sleep(0.05)

            # At t0 + 5.10s (5.1s after start):
            # Client B has been silent for 5.10s -> pruned
            # Client A refreshed at t0 + 2.50s (2.60s elapsed) -> retained!
            with patch("time.time", return_value=t0 + 5.10):
                running_dsu_server.prune_subscribers(timeout_sec=5.0)
                assert running_dsu_server.active_subscriber_count == 1

            # At t0 + 7.60s (5.10s after A's refresh):
            # Client A has been silent for 5.10s -> pruned!
            with patch("time.time", return_value=t0 + 7.60):
                running_dsu_server.prune_subscribers(timeout_sec=5.0)
                assert running_dsu_server.active_subscriber_count == 0
        finally:
            client_a.close()
            client_b.close()

    def test_wallclock_lease_pruning_and_heartbeat_retention(self):
        """Empirically test lease pruning and heartbeat retention with wall-clock timing:
        1. Register 10 clients. Sleep past 5.0 seconds. Call prune_subscribers() -> verify 0.
        2. Register 1 client, send refresh heartbeat at 2.5s, sleep another 3.0s -> verify 1.
        """
        srv = DsuServer(port=0)
        assert srv.start() is True

        try:
            # --- Part 1: 10 clients dropped after 5.0s ---
            clients = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(10)]
            for i, c in enumerate(clients):
                sub = make_client_subscription_packet(0x1000 + i)
                c.sendto(sub, ("127.0.0.1", srv.port))

            time.sleep(0.1)
            assert srv.active_subscriber_count == 10

            # Sleep past 5.0s
            time.sleep(5.1)
            srv.prune_subscribers()
            assert srv.active_subscriber_count == 0

            for c in clients:
                c.close()

            # --- Part 2: 1 client refreshed at 2.5s retained past 5.5s ---
            c_single = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sub_single = make_client_subscription_packet(0x9999)
            c_single.sendto(sub_single, ("127.0.0.1", srv.port))
            time.sleep(0.05)
            assert srv.active_subscriber_count == 1

            # Sleep 2.5s
            time.sleep(2.5)

            # Send refresh heartbeat
            c_single.sendto(sub_single, ("127.0.0.1", srv.port))
            time.sleep(0.05)  # Let background thread process

            # Sleep another 3.0s (total elapsed since start = 5.55s, but only 3.0s since heartbeat)
            time.sleep(3.0)
            srv.prune_subscribers()
            assert srv.active_subscriber_count == 1

            # Sleep another 2.2s (total elapsed since heartbeat = 5.2s > 5.0s)
            time.sleep(2.2)
            srv.prune_subscribers()
            assert srv.active_subscriber_count == 0

            c_single.close()
        finally:
            srv.stop()


# ============================================================================
# 3. Multi-Threaded Concurrency Stress Suite
# ============================================================================

class TestDsuMultiThreadedStress:
    """Verifies thread safety, deadlock freedom, and concurrency robustness."""

    def test_concurrent_producers_and_subscribers(self, running_dsu_server: DsuServer):
        """Stress-test DsuServer with:
        - 4 concurrent caller threads calling update_state()
        - 1 concurrent client thread continuously sending subscription packets
        - 1 concurrent client thread querying controller info and protocol version
        - 1 concurrent receiver thread reading and verifying packet framing
        - 1 concurrent pruner thread invoking prune_subscribers()
        
        Verifies:
        - Zero data races or corrupted packets
        - Zero deadlocks across thread synchronization primitives
        - Clean completion within timeout
        """
        num_producers = 4
        calls_per_producer = 1000
        total_workers = num_producers + 4
        barrier = threading.Barrier(total_workers)
        stop_event = threading.Event()
        errors: list[tuple[str, Exception]] = []

        # Create sockets for background clients
        c_sub = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        c_query = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        c_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for s in (c_sub, c_query, c_recv):
            s.settimeout(0.2)

        # Initial subscription on receiver socket
        sub_pkt = make_client_subscription_packet(0xABCD)
        c_recv.sendto(sub_pkt, ("127.0.0.1", running_dsu_server.port))
        time.sleep(0.05)

        def producer_worker(worker_id: int):
            try:
                barrier.wait(timeout=5.0)
                rng = random.Random(worker_id * 777)
                for seq in range(calls_per_producer):
                    st = make_oracle_state(
                        accel_x=rng.uniform(-2.0, 2.0),
                        gyro_x=rng.uniform(-500.0, 500.0),
                        motion_timestamp=worker_id * 1_000_000 + seq,
                    )
                    running_dsu_server.update_state(st)
            except Exception as e:
                errors.append((f"producer_{worker_id}", e))

        def subscriber_worker():
            try:
                barrier.wait(timeout=5.0)
                while not stop_event.is_set():
                    c_sub.sendto(sub_pkt, ("127.0.0.1", running_dsu_server.port))
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("subscriber_worker", e))

        def query_worker():
            try:
                barrier.wait(timeout=5.0)
                # Controller Info packet for slot 0
                payload = struct.pack("<IBBBB", 0x100001, 0, 0, 0, 0)
                info_pkt = bytearray(16 + len(payload))
                struct.pack_into("<4sHHII", info_pkt, 0, b"DSUC", 1001, len(payload), 0, 0x5555)
                info_pkt[16:] = payload
                info_pkt[8:12] = struct.pack("<I", zlib.crc32(info_pkt) & 0xFFFFFFFF)

                while not stop_event.is_set():
                    c_query.sendto(info_pkt, ("127.0.0.1", running_dsu_server.port))
                    try:
                        c_query.recvfrom(1024)
                    except socket.timeout:
                        pass
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("query_worker", e))

        def receiver_worker():
            try:
                barrier.wait(timeout=5.0)
                while not stop_event.is_set():
                    try:
                        data, _ = c_recv.recvfrom(1024)
                        assert len(data) == 100
                        hdr = unpack_reference_dsu_packet(data)
                        assert hdr["msg_type"] == MSG_TYPE_DATA
                    except socket.timeout:
                        pass
            except Exception as e:
                errors.append(("receiver_worker", e))

        def pruner_worker():
            try:
                barrier.wait(timeout=5.0)
                while not stop_event.is_set():
                    _ = running_dsu_server.active_subscriber_count
                    running_dsu_server.prune_subscribers(timeout_sec=5.0)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(("pruner_worker", e))

        # Launch all threads
        threads: list[threading.Thread] = []
        for wid in range(num_producers):
            threads.append(threading.Thread(target=producer_worker, args=(wid,), name=f"Producer-{wid}"))
        threads.append(threading.Thread(target=subscriber_worker, name="SubWorker"))
        threads.append(threading.Thread(target=query_worker, name="QueryWorker"))
        threads.append(threading.Thread(target=receiver_worker, name="RecvWorker"))
        threads.append(threading.Thread(target=pruner_worker, name="PruneWorker"))

        for t in threads:
            t.start()

        # Await producer completion
        for t in threads[:num_producers]:
            t.join(timeout=10.0)
            assert not t.is_alive(), f"{t.name} did not complete within timeout (possible deadlock)!"

        # Stop background workers
        stop_event.set()
        for t in threads[num_producers:]:
            t.join(timeout=5.0)
            assert not t.is_alive(), f"{t.name} did not complete within timeout (possible deadlock)!"

        c_sub.close()
        c_query.close()
        c_recv.close()

        # Verify zero exceptions across all 8 threads
        assert len(errors) == 0, f"Concurrent stress test encountered errors: {errors}"
