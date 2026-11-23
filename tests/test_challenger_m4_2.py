"""Empirical Adversarial Verification Suite for Milestone 4 Iteration 2.

Authored by: M4 Adversarial Challenger 2 (teamwork_challenger_m4_2)
Purpose: Independent empirical verification of Milestone 4:
1. Multi-Threaded Concurrency Hammer:
   - 4 threads generating rapid ControllerState events (250 Hz simulation).
   - 2 threads toggling manager.output_mode between 'xbox' and 'ds4'.
   - 2 threads toggling manager.dsu_enabled between True and False.
   - 1 thread toggling button layout and GL/GR mappings.
   - Zero exceptions, zero deadlocks, verified state consistency at completion.
2. End-to-End Real UDP Motion Streaming Verification:
   - Real UDP client connected to dynamic ephemeral port on ControllerManager DsuServer.
   - Cemuhook DSU handshake/version (0x100000), list_ports/controller info (0x100001), and motion subscription (0x100002).
   - Validation of exact 100-byte motion packet format, header magic, IEEE 802.3 CRC32,
     monotonically incrementing sequence counters, and IEEE 754 float32 motion vectors.
3. Teardown Stress:
   - manager.shutdown() invoked while threads actively pump states and network requests.
   - Clean shutdown with all background listener threads terminated and sockets closed.
4. Edge Case & Boundary Fuzzing:
   - Malformed UDP payloads (truncated, corrupt CRC32, unknown magic).
   - Non-finite float handling (NaN, Inf) through dual dispatch.
   - Atomicity of invalid mode rejections during state streaming.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time
import unittest.mock as mock
import zlib
from typing import List, Optional, Tuple

import pytest

from app_core import ControllerManager
from dsu_server import (
    CONN_USB,
    DSU_MAGIC_CLIENT,
    DSU_MAGIC_SERVER,
    DSU_PROTOCOL_VERSION,
    MODEL_FULL_GYRO,
    MSG_TYPE_CONTROLLER_INFO,
    MSG_TYPE_DATA,
    MSG_TYPE_VERSION,
    SLOT_CONNECTED,
    SLOT_DISCONNECTED,
    DsuServer,
    calculate_dsu_crc32,
)
from report_parser import ControllerState
from tests.test_virtual_controllers import MockVDS4Gamepad, MockVX360Gamepad, make_state
from virtual_ds4 import VirtualDS4Controller
from virtual_xinput import VirtualXboxController


# ============================================================================
# Helpers for UDP Protocol Communication
# ============================================================================

def build_dsu_client_packet(msg_type: int, payload: bytes = b"", client_id: int = 0x12345678) -> bytes:
    """Build a compliant DSUC client packet with correct header, length, and CRC32."""
    full_payload = struct.pack("<I", msg_type) + payload
    length = len(full_payload)
    pkt = bytearray(16 + length)
    struct.pack_into("<4sHHII", pkt, 0, DSU_MAGIC_CLIENT, DSU_PROTOCOL_VERSION, length, 0, client_id)
    pkt[16:] = full_payload
    crc = calculate_dsu_crc32(pkt)
    struct.pack_into("<I", pkt, 8, crc)
    return bytes(pkt)


# ============================================================================
# Suite 1: Multi-Threaded Concurrency Hammer
# ============================================================================

class TestMultiThreadedConcurrencyHammer:
    """9-thread concurrency hammer stress-testing ControllerManager thread safety."""

    def test_9_thread_concurrency_hammer_stress(self):
        """Stress-test ControllerManager with 9 simultaneous threads for 3.0 seconds:
        - 4 state producer threads generating 250 Hz ControllerState events.
        - 2 threads toggling output_mode between 'xbox' and 'ds4'.
        - 2 threads toggling dsu_enabled between True and False.
        - 1 thread toggling button layout and grip paddle mappings.
        Verifies:
        - Zero unhandled exceptions in any thread.
        - Zero deadlocks (all threads terminate within join timeout).
        - Consistent state at termination (matching layout, paddle mappings, output_mode).
        """
        manager = ControllerManager()
        # Use port 0 (ephemeral UDP port) to avoid port 26760 collisions
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)

        stop_event = threading.Event()
        thread_exceptions: List[Tuple[str, Exception]] = []
        lock_exceptions = threading.Lock()

        def record_exception(thread_name: str, exc: Exception):
            with lock_exceptions:
                thread_exceptions.append((thread_name, exc))

        # 4 State Producer Threads
        def state_producer(producer_id: int):
            try:
                iteration = 0
                while not stop_event.is_set():
                    iteration += 1
                    ax = 0.05 * producer_id + (iteration % 10) * 0.01
                    gx = 10.0 * producer_id + (iteration % 5) * 1.5
                    st = make_state(
                        accel_x=ax,
                        accel_y=-0.5,
                        accel_z=0.98,
                        gyro_x=gx,
                        gyro_y=0.0,
                        gyro_z=-12.5,
                        motion_timestamp=iteration * 4000,
                        a=(iteration % 2 == 0),
                        b=(iteration % 3 == 0),
                    )
                    manager._on_controller_state(st)
                    # ~250 Hz pacing
                    time.sleep(0.004)
            except Exception as e:
                record_exception(f"Producer_{producer_id}", e)

        # 2 Output Mode Toggler Threads
        def mode_toggler(toggler_id: int):
            try:
                toggle_count = 0
                while not stop_event.is_set():
                    target = "ds4" if (toggle_count % 2 == toggler_id % 2) else "xbox"
                    manager.set_output_mode(target)
                    toggle_count += 1
                    time.sleep(0.01)
            except Exception as e:
                record_exception(f"ModeToggler_{toggler_id}", e)

        # 2 DSU Server Lifecycle Toggler Threads
        def dsu_toggler(toggler_id: int):
            try:
                dsu_count = 0
                while not stop_event.is_set():
                    enable = (dsu_count % 2 == toggler_id % 2)
                    manager.set_dsu_enabled(enable)
                    dsu_count += 1
                    time.sleep(0.015)
            except Exception as e:
                record_exception(f"DsuToggler_{toggler_id}", e)

        # 1 Button Layout & Paddle Mapping Toggler Thread
        def mapping_toggler():
            layouts = ["nintendo", "xbox"]
            paddles = ["L3", "R3", "LB", "RB", "LT", "RT", "Disabled"]
            try:
                map_count = 0
                while not stop_event.is_set():
                    map_count += 1
                    manager.set_button_layout(layouts[map_count % len(layouts)])
                    manager.set_gl_mapping(paddles[map_count % len(paddles)])
                    manager.set_gr_mapping(paddles[(map_count + 1) % len(paddles)])
                    manager.set_deadzone(0.01 + (map_count % 5) * 0.02)
                    time.sleep(0.012)
            except Exception as e:
                record_exception("MappingToggler", e)

        threads: List[threading.Thread] = []
        for i in range(4):
            threads.append(threading.Thread(target=state_producer, args=(i,), name=f"Producer_{i}"))
        for i in range(2):
            threads.append(threading.Thread(target=mode_toggler, args=(i,), name=f"ModeToggler_{i}"))
        for i in range(2):
            threads.append(threading.Thread(target=dsu_toggler, args=(i,), name=f"DsuToggler_{i}"))
        threads.append(threading.Thread(target=mapping_toggler, name="MappingToggler"))

        # Launch all 9 threads simultaneously
        for t in threads:
            t.start()

        # Run under full contention for 3.0 seconds
        time.sleep(3.0)
        stop_event.set()

        # Join all threads with strict timeout (deadlock check)
        for t in threads:
            t.join(timeout=3.0)
            assert not t.is_alive(), f"Thread {t.name} failed to join — deadlock detected!"

        # Invariant 1: Zero exceptions across all threads
        assert len(thread_exceptions) == 0, f"Exceptions occurred during concurrency hammer: {thread_exceptions}"

        # Invariant 2: Manager state consistency at completion
        assert manager.output_mode in ("xbox", "ds4")
        if manager.output_mode == "ds4":
            assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
        else:
            assert isinstance(manager.virtual_gamepad, VirtualXboxController)

        assert manager.virtual_gamepad.layout == manager.button_layout
        assert manager.virtual_gamepad.gl_mapping == manager.gl_mapping
        assert manager.virtual_gamepad.gr_mapping == manager.gr_mapping

        # Invariant 3: DSU server state matches dsu_enabled flag
        assert isinstance(manager.dsu_enabled, bool)
        assert manager.dsu_server.is_running == manager.dsu_enabled

        # Cleanup
        manager.shutdown()

    def test_concurrency_hammer_with_connected_controller(self):
        """Concurrency hammer executed while is_connected=True with mock virtual gamepads.
        Verifies hot-swapping virtual controller backends under active state streaming.
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.is_connected = True

        stop_event = threading.Event()
        thread_exceptions: List[Tuple[str, Exception]] = []

        def state_producer(producer_id: int):
            try:
                while not stop_event.is_set():
                    st = make_state(accel_x=0.2 * producer_id, gyro_z=5.0)
                    manager._on_controller_state(st)
                    time.sleep(0.003)
            except Exception as e:
                thread_exceptions.append((f"Producer_{producer_id}", e))

        def mode_toggler(toggler_id: int):
            try:
                cnt = 0
                while not stop_event.is_set():
                    cnt += 1
                    target = "ds4" if (cnt % 2 == 0) else "xbox"
                    manager.set_output_mode(target)
                    time.sleep(0.01)
            except Exception as e:
                thread_exceptions.append((f"ModeToggler_{toggler_id}", e))

        def dsu_toggler():
            try:
                cnt = 0
                while not stop_event.is_set():
                    cnt += 1
                    manager.set_dsu_enabled(cnt % 2 == 0)
                    time.sleep(0.02)
            except Exception as e:
                thread_exceptions.append(("DsuToggler", e))

        with mock.patch("vgamepad.VX360Gamepad", side_effect=MockVX360Gamepad), \
             mock.patch("vgamepad.VDS4Gamepad", side_effect=MockVDS4Gamepad):

            threads = [
                threading.Thread(target=state_producer, args=(0,)),
                threading.Thread(target=state_producer, args=(1,)),
                threading.Thread(target=mode_toggler, args=(0,)),
                threading.Thread(target=mode_toggler, args=(1,)),
                threading.Thread(target=dsu_toggler),
            ]

            for t in threads:
                t.start()

            time.sleep(2.0)
            stop_event.set()

            for t in threads:
                t.join(timeout=3.0)
                assert not t.is_alive()

        assert len(thread_exceptions) == 0, f"Exceptions in connected hammer: {thread_exceptions}"
        assert manager.virtual_gamepad.is_connected is True
        manager.shutdown()


# ============================================================================
# Suite 2: End-to-End Real UDP Motion Streaming Verification
# ============================================================================

class TestRealUdpMotionStreaming:
    """Verifies end-to-end network protocol conformance on real OS UDP sockets."""

    def test_real_udp_cemuhook_protocol_handshake_and_info(self):
        """Verifies full Cemuhook DSU protocol handshake:
        1. Protocol version request (0x100000) -> 24-byte response, magic 'DSUS', version 1001, CRC32.
        2. Controller info request (0x100001) for slot 0 -> 32-byte response, model=2, conn=1, status=2.
        3. Controller info request for invalid slot 1 -> slot_state=0 (disconnected).
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        assert manager.dsu_enabled is True
        bound_port = manager.dsu_server.port
        assert bound_port > 0

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2.0)

        try:
            # 1. Version Handshake Request (0x100000)
            ver_pkt = build_dsu_client_packet(MSG_TYPE_VERSION)
            client.sendto(ver_pkt, ("127.0.0.1", bound_port))
            resp, _ = client.recvfrom(1024)

            assert len(resp) == 24
            magic, ver, length, crc_recv, sender = struct.unpack_from("<4sHHII", resp, 0)
            assert magic == DSU_MAGIC_SERVER
            assert ver == DSU_PROTOCOL_VERSION
            assert length == 8
            assert calculate_dsu_crc32(resp) == crc_recv
            resp_type, resp_ver = struct.unpack_from("<IH", resp, 16)
            assert resp_type == MSG_TYPE_VERSION
            assert resp_ver == DSU_PROTOCOL_VERSION

            # 2. Controller Info Request for Slot 0 (0x100001)
            info_pkt = build_dsu_client_packet(MSG_TYPE_CONTROLLER_INFO, struct.pack("<IB", 1, 0))
            client.sendto(info_pkt, ("127.0.0.1", bound_port))
            resp, _ = client.recvfrom(1024)

            assert len(resp) == 32
            magic, ver, length, crc_recv, sender = struct.unpack_from("<4sHHII", resp, 0)
            assert magic == DSU_MAGIC_SERVER
            assert ver == DSU_PROTOCOL_VERSION
            assert length == 16
            assert calculate_dsu_crc32(resp) == crc_recv

            resp_type, slot, slot_state, model, conn = struct.unpack_from("<IBBBB", resp, 16)
            assert resp_type == MSG_TYPE_CONTROLLER_INFO
            assert slot == 0
            assert slot_state == SLOT_CONNECTED
            assert model == MODEL_FULL_GYRO
            assert conn == CONN_USB

            # 3. Controller Info Request for Slot 1 (disconnected slot)
            info_pkt_slot1 = build_dsu_client_packet(MSG_TYPE_CONTROLLER_INFO, struct.pack("<IB", 1, 1))
            client.sendto(info_pkt_slot1, ("127.0.0.1", bound_port))
            resp, _ = client.recvfrom(1024)

            assert len(resp) == 32
            resp_type, slot, slot_state, model, conn = struct.unpack_from("<IBBBB", resp, 16)
            assert slot == 1
            assert slot_state == SLOT_DISCONNECTED

        finally:
            client.close()
            manager.shutdown()

    def test_real_udp_motion_data_streaming_byte_layout_and_sequence(self):
        """Verifies exact 100-byte motion packet layout, IEEE 754 float32 precision,
        and monotonic sequence counter incrementing across consecutive frames.
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        bound_port = manager.dsu_server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(2.0)

        try:
            # Send subscription request (0x100002) using client magic 'DSUC'
            sub_pkt = build_dsu_client_packet(MSG_TYPE_DATA, struct.pack("<BBBB", 0, 0, 0, 0))
            client.sendto(sub_pkt, ("127.0.0.1", bound_port))
            time.sleep(0.04)  # Allow server listener thread to register subscriber

            test_vectors = [
                # (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, timestamp)
                (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 100000),
                (0.5, -0.25, 0.85, 45.2, -30.1, 180.0, 104000),
                (-1.5, 2.0, -0.75, -500.0, 750.0, -1200.0, 108000),
                (7.99, -7.99, 0.0, 1999.0, -1999.0, 0.0, 112000),
                (-0.0125, 0.0345, 0.9999, -0.123, 0.456, -0.789, 116000),
            ]

            previous_sequence: Optional[int] = None

            for ax, ay, az, gx, gy, gz, ts in test_vectors:
                st = make_state(
                    accel_x=ax,
                    accel_y=ay,
                    accel_z=az,
                    gyro_x=gx,
                    gyro_y=gy,
                    gyro_z=gz,
                    motion_timestamp=ts,
                    a=True,
                    b=False,
                    left_stick_x=0.5,
                    left_stick_y=-0.75,
                )
                manager._on_controller_state(st)

                packet, _ = client.recvfrom(1024)

                # Byte layout verification
                assert len(packet) == 100, f"Expected 100 bytes, got {len(packet)}"

                # Header (bytes 0..15)
                magic, ver, payload_len, crc_recv, server_id = struct.unpack_from("<4sHHII", packet, 0)
                assert magic == DSU_MAGIC_SERVER
                assert ver == DSU_PROTOCOL_VERSION
                assert payload_len == 84

                # CRC32 verification
                expected_crc = calculate_dsu_crc32(packet)
                assert crc_recv == expected_crc, f"CRC mismatch: expected {expected_crc:#x}, got {crc_recv:#x}"

                # Message type (bytes 16..19)
                msg_type = struct.unpack_from("<I", packet, 16)[0]
                assert msg_type == MSG_TYPE_DATA

                # Sequence counter (bytes 32..35)
                seq = struct.unpack_from("<I", packet, 32)[0]
                if previous_sequence is not None:
                    assert seq == previous_sequence + 1, f"Sequence not incrementing: prev={previous_sequence}, curr={seq}"
                previous_sequence = seq

                # Buttons (bytes 36..37)
                b2 = packet[37]
                assert (b2 & 0x20) != 0  # Button 'A' pressed

                # Timestamp (bytes 68..75, uint64 in microseconds)
                ts_recv = struct.unpack_from("<Q", packet, 68)[0]
                assert ts_recv == ts

                # 6-Axis Motion Floats (bytes 76..99, 6 x float32 little-endian)
                ax_r, ay_r, az_r, gx_r, gy_r, gz_r = struct.unpack_from("<ffffff", packet, 76)
                assert math.isclose(ax_r, ax, rel_tol=1e-5, abs_tol=1e-5)
                assert math.isclose(ay_r, ay, rel_tol=1e-5, abs_tol=1e-5)
                assert math.isclose(az_r, az, rel_tol=1e-5, abs_tol=1e-5)
                assert math.isclose(gx_r, gx, rel_tol=1e-4, abs_tol=1e-4)
                assert math.isclose(gy_r, gy, rel_tol=1e-4, abs_tol=1e-4)
                assert math.isclose(gz_r, gz, rel_tol=1e-4, abs_tol=1e-4)

        finally:
            client.close()
            manager.shutdown()

    def test_multi_client_concurrent_udp_streaming(self):
        """Verifies that multiple concurrent UDP clients (e.g. Cemu + Dolphin running simultaneously)
        both receive independent motion stream packets without dropped frames or starvation.
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        bound_port = manager.dsu_server.port

        client1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client1.settimeout(2.0)
        client2.settimeout(2.0)

        try:
            # Subscribe both clients
            sub1 = build_dsu_client_packet(MSG_TYPE_DATA, client_id=0x11111111)
            sub2 = build_dsu_client_packet(MSG_TYPE_DATA, client_id=0x22222222)
            client1.sendto(sub1, ("127.0.0.1", bound_port))
            client2.sendto(sub2, ("127.0.0.1", bound_port))
            time.sleep(0.04)

            # Send 5 frames
            for i in range(5):
                st = make_state(accel_z=1.0, gyro_x=float(i * 10))
                manager._on_controller_state(st)

                pkt1, _ = client1.recvfrom(1024)
                pkt2, _ = client2.recvfrom(1024)

                assert len(pkt1) == 100
                assert len(pkt2) == 100
                assert calculate_dsu_crc32(pkt1) == struct.unpack_from("<I", pkt1, 8)[0]
                assert calculate_dsu_crc32(pkt2) == struct.unpack_from("<I", pkt2, 8)[0]

        finally:
            client1.close()
            client2.close()
            manager.shutdown()


# ============================================================================
# Suite 3: Teardown Stress
# ============================================================================

class TestTeardownStress:
    """Verifies graceful termination under high load without orphaned threads or sockets."""

    def test_manager_shutdown_during_active_streaming_hammer(self):
        """Invokes manager.shutdown() while 4 threads actively pump states and 1 client reads UDP.
        Verifies:
        - All background threads terminate cleanly.
        - Sockets are closed.
        - Zero deadlocks or uncaught exceptions during shutdown.
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        manager.is_connected = True
        bound_port = manager.dsu_server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(0.5)
        sub_pkt = build_dsu_client_packet(MSG_TYPE_DATA)
        client.sendto(sub_pkt, ("127.0.0.1", bound_port))
        time.sleep(0.03)

        stop_event = threading.Event()
        exceptions: List[Tuple[str, Exception]] = []

        def producer(pid: int):
            try:
                while not stop_event.is_set():
                    st = make_state(accel_x=0.5, gyro_z=20.0)
                    manager._on_controller_state(st)
                    time.sleep(0.002)
            except Exception as e:
                exceptions.append((f"Producer_{pid}", e))

        def client_reader():
            try:
                while not stop_event.is_set():
                    try:
                        client.recvfrom(1024)
                    except (socket.timeout, OSError):
                        pass
            except Exception as e:
                exceptions.append(("ClientReader", e))

        producer_threads = [
            threading.Thread(target=producer, args=(i,), name=f"TeardownProducer_{i}")
            for i in range(4)
        ]
        reader_thread = threading.Thread(target=client_reader, name="TeardownReader")

        for t in producer_threads:
            t.start()
        reader_thread.start()

        # Let the system run under heavy load for 0.4 seconds
        time.sleep(0.4)

        # Abruptly call shutdown() from another thread
        manager.shutdown()

        # Check manager state immediately post-shutdown
        assert manager.is_connected is False
        assert manager.dsu_enabled is False
        assert manager.dsu_server.is_running is False
        if manager.dsu_server._thread:
            assert not manager.dsu_server._thread.is_alive()

        # Stop worker threads
        stop_event.set()
        client.close()

        for t in producer_threads:
            t.join(timeout=2.0)
            assert not t.is_alive()
        reader_thread.join(timeout=2.0)
        assert not reader_thread.is_alive()

        assert len(exceptions) == 0, f"Exceptions during shutdown stress: {exceptions}"

    def test_trailing_state_dispatch_after_shutdown(self):
        """Dispatches controller state frames after manager.shutdown() has completed.
        Simulates trailing packets arriving from USB/BLE driver threads during exit.
        """
        manager = ControllerManager()
        manager.set_dsu_enabled(True)
        manager.shutdown()

        st = make_state(accel_z=1.0)
        # Should execute safely as a no-op without raising any exception
        manager._on_controller_state(st)
        manager._on_controller_state(None)

    def test_shutdown_idempotency(self):
        """Repeated shutdown() calls must be safe and idempotent."""
        manager = ControllerManager()
        manager.shutdown()
        manager.shutdown()
        manager.shutdown()
        assert manager.is_connected is False
        assert manager.dsu_enabled is False


# ============================================================================
# Suite 4: Edge Case & Boundary Fuzzing
# ============================================================================

class TestEdgeCaseAndBoundaryFuzzing:
    """Adversarial boundary stress tests."""

    def test_malformed_udp_packets_do_not_crash_dsu_listener(self):
        """Sends corrupt UDP packets to DSU listener:
        - Truncated packets (< 20 bytes).
        - Packets with invalid magic (e.g. b'CORR').
        - Packets with bad CRC32 checksums.
        - Oversized packets (> 1024 bytes).
        Verifies listener survives and continues answering valid requests.
        """
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        bound_port = manager.dsu_server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(1.5)

        try:
            # 1. Truncated packet (4 bytes)
            client.sendto(b"DSUC", ("127.0.0.1", bound_port))

            # 2. Invalid magic (20 bytes)
            client.sendto(b"NOPE" + b"\x00" * 16, ("127.0.0.1", bound_port))

            # 3. Bad CRC packet
            bad_crc_pkt = bytearray(20)
            struct.pack_into("<4sHHII", bad_crc_pkt, 0, DSU_MAGIC_CLIENT, 1001, 4, 0xDEADBEEF, 0x1234)
            struct.pack_into("<I", bad_crc_pkt, 16, MSG_TYPE_VERSION)
            client.sendto(bad_crc_pkt, ("127.0.0.1", bound_port))

            # 4. Oversized garbage packet
            client.sendto(b"X" * 2048, ("127.0.0.1", bound_port))

            time.sleep(0.05)
            # Verify listener is still fully operational by sending valid version request
            valid_pkt = build_dsu_client_packet(MSG_TYPE_VERSION)
            client.sendto(valid_pkt, ("127.0.0.1", bound_port))
            resp, _ = client.recvfrom(1024)
            assert len(resp) == 24
            assert resp[0:4] == DSU_MAGIC_SERVER

        finally:
            client.close()
            manager.shutdown()

    def test_non_finite_floats_in_state_dispatch(self):
        """Verifies dual-dispatch with non-finite floats (NaN, Inf) does not raise exceptions."""
        manager = ControllerManager()
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)
        manager.set_dsu_enabled(True)
        bound_port = manager.dsu_server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(1.5)

        try:
            sub_pkt = build_dsu_client_packet(MSG_TYPE_DATA)
            client.sendto(sub_pkt, ("127.0.0.1", bound_port))
            time.sleep(0.03)

            nan_state = make_state(
                accel_x=float("nan"),
                accel_y=float("inf"),
                accel_z=float("-inf"),
                gyro_x=float("nan"),
            )
            # Must not raise
            manager._on_controller_state(nan_state)

            pkt, _ = client.recvfrom(1024)
            assert len(pkt) == 100
            # Packet CRC must still be validly calculated
            crc_recv = struct.unpack_from("<I", pkt, 8)[0]
            assert calculate_dsu_crc32(pkt) == crc_recv

        finally:
            client.close()
            manager.shutdown()

    def test_invalid_output_mode_rejected_without_state_corruption(self):
        """Verifies that passing invalid output modes raises ValueError and leaves manager intact."""
        manager = ControllerManager()
        assert manager.output_mode == "xbox"

        for bad in ["playstation", "switch", "", None, 123, []]:
            with pytest.raises(ValueError):
                manager.set_output_mode(bad)  # type: ignore

        assert manager.output_mode == "xbox"
        assert isinstance(manager.virtual_gamepad, VirtualXboxController)
