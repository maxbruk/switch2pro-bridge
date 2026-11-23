"""Adversarial stress testing and network fuzzing harness for Cemuhook DSU Server (dsu_server.py).

Covers:
1. Network Packet Fuzzing (truncated 0..15 bytes, corrupted magic/version/CRC, oversized >1024 bytes).
2. Resilient Client Disconnect Handling (ICMP Port Unreachable / WSAECONNRESET).
3. Concurrent Client Flooding (100 simultaneous UDP clients, 2,000 updates at >250 Hz, IEEE 802.3 CRC32 verification).
4. Rapid Lifecycle Re-binding (20 start/stop cycles on identical port with clean socket release).
"""

import socket
import struct
import time
import zlib
import pytest

from dsu_server import (
    DsuServer,
    calculate_dsu_crc32,
    DSU_MAGIC_CLIENT,
    DSU_MAGIC_SERVER,
    DSU_PROTOCOL_VERSION,
    MSG_TYPE_VERSION,
    MSG_TYPE_CONTROLLER_INFO,
    MSG_TYPE_DATA,
)
from report_parser import ControllerState


def make_client_packet(msg_type: int, payload_extra: bytes = b"", client_id: int = 0x12345678) -> bytes:
    """Build a standard DSUC client packet with valid CRC32."""
    payload = struct.pack("<I", msg_type) + payload_extra
    length = len(payload)
    pkt = bytearray(16 + length)
    struct.pack_into("<4sHHII", pkt, 0, DSU_MAGIC_CLIENT, DSU_PROTOCOL_VERSION, length, 0, client_id)
    pkt[16:] = payload
    crc = calculate_dsu_crc32(pkt)
    struct.pack_into("<I", pkt, 8, crc)
    return bytes(pkt)


def make_dummy_state(**kwargs) -> ControllerState:
    """Create a fully populated ControllerState for testing."""
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
        "motion_timestamp": 500000,
        "has_motion": True,
    }
    base.update(kwargs)
    return ControllerState(**base)


class TestDsuAdversarialStress:
    """Empirical adversarial stress test suite for DsuServer."""

    def test_fuzz_truncated_packets(self):
        """Fuzz server with truncated packets (0 to 15 bytes) and confirm 0 crashes."""
        server = DsuServer(port=0)
        assert server.start() is True
        port = server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for size in range(16):
            client.sendto(b"\xAA" * size, ("127.0.0.1", port))
        time.sleep(0.05)

        # Confirm listener thread is healthy
        assert server._thread is not None and server._thread.is_alive()

        # Confirm legitimate request still gets a response
        client.settimeout(1.0)
        client.sendto(make_client_packet(MSG_TYPE_VERSION), ("127.0.0.1", port))
        resp, _ = client.recvfrom(1024)
        assert len(resp) == 24
        assert resp[:4] == DSU_MAGIC_SERVER

        client.close()
        server.stop()

    def test_fuzz_corrupted_magic_version_crc(self):
        """Fuzz server with corrupted magic, protocol versions, and incorrect CRC32."""
        server = DsuServer(port=0)
        assert server.start() is True
        port = server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # 1. Corrupted magic
        client.sendto(b"XXXX" + b"\x00" * 24, ("127.0.0.1", port))

        # 2. Invalid protocol version with valid CRC
        pkt_bad_version = bytearray(20)
        struct.pack_into("<4sHHII", pkt_bad_version, 0, DSU_MAGIC_CLIENT, 9999, 4, 0, 0x1234)
        struct.pack_into("<I", pkt_bad_version, 16, MSG_TYPE_VERSION)
        struct.pack_into("<I", pkt_bad_version, 8, calculate_dsu_crc32(pkt_bad_version))
        client.sendto(bytes(pkt_bad_version), ("127.0.0.1", port))

        # 3. Corrupted CRC32
        pkt_bad_crc = bytearray(20)
        struct.pack_into("<4sHHII", pkt_bad_crc, 0, DSU_MAGIC_CLIENT, DSU_PROTOCOL_VERSION, 4, 0xBAD0C0DE, 0x1234)
        struct.pack_into("<I", pkt_bad_crc, 16, MSG_TYPE_VERSION)
        client.sendto(bytes(pkt_bad_crc), ("127.0.0.1", port))

        time.sleep(0.05)
        assert server._thread is not None and server._thread.is_alive()

        # Verify legitimate request succeeds
        client.settimeout(1.0)
        client.sendto(make_client_packet(MSG_TYPE_VERSION), ("127.0.0.1", port))
        resp, _ = client.recvfrom(1024)
        assert len(resp) == 24
        assert resp[:4] == DSU_MAGIC_SERVER

        client.close()
        server.stop()

    def test_fuzz_oversized_packet_dos(self):
        """Fuzz server with oversized datagrams (>1024 bytes) and confirm server does not crash or terminate listener.
        
        Adversarial attack: Sends 2048-byte datagram to server UDP port.
        On Windows, recvfrom(1024) raises WinError 10040 (WSAEMSGSIZE).
        Server must safely absorb this without terminating its background listener thread.
        """
        server = DsuServer(port=0)
        assert server.start() is True
        port = server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Send 2048-byte oversized packet
        client.sendto(b"O" * 2048, ("127.0.0.1", port))
        time.sleep(0.05)

        # The listener thread MUST survive
        assert server._thread is not None and server._thread.is_alive(), (
            "VULNERABILITY: DsuServer listener thread crashed on oversized UDP datagram (WinError 10040 / WSAEMSGSIZE)!"
        )

        # Server must still be able to process legitimate requests
        client.settimeout(1.0)
        client.sendto(make_client_packet(MSG_TYPE_VERSION), ("127.0.0.1", port))
        resp, _ = client.recvfrom(1024)
        assert len(resp) == 24
        assert resp[:4] == DSU_MAGIC_SERVER

        client.close()
        server.stop()

    def test_client_disconnect_icmp_resilience(self):
        """Confirm server listener survives client disconnect and ICMP Port Unreachable (WinError 10054 / WSAECONNRESET).
        
        Adversarial scenario:
        1. Client registers for data stream (0x100002).
        2. Client abruptly terminates / closes socket.
        3. Server sends update_state(), triggering ICMP unreachable on loopback.
        4. Windows delivers WinError 10054 to server UDP socket on subsequent recvfrom().
        Server must not terminate its listener thread on connection reset.
        """
        server = DsuServer(port=0)
        assert server.start() is True
        port = server.port

        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.sendto(make_client_packet(MSG_TYPE_DATA), ("127.0.0.1", port))
        time.sleep(0.05)
        # Client abruptly closes
        client.close()

        # Server streams state to closed client
        state = make_dummy_state()
        for _ in range(5):
            server.update_state(state)
            time.sleep(0.005)

        time.sleep(0.05)
        assert server._thread is not None and server._thread.is_alive(), (
            "VULNERABILITY: DsuServer listener thread crashed when client disconnected (WinError 10054 / WSAECONNRESET)!"
        )

        # Ensure a new client can still connect and get responses
        new_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        new_client.settimeout(1.0)
        new_client.sendto(make_client_packet(MSG_TYPE_VERSION), ("127.0.0.1", port))
        resp, _ = new_client.recvfrom(1024)
        assert len(resp) == 24
        assert resp[:4] == DSU_MAGIC_SERVER

        new_client.close()
        server.stop()

    def test_concurrent_client_flooding_250hz(self):
        """Stress test with 100 concurrent UDP clients streaming 2,000 updates at 250 Hz.
        
        Validates:
        - 100 simultaneous subscribers registered
        - 2,000 updates broadcast successfully
        - All received packets maintain valid IEEE 802.3 CRC32 and correct 6-axis motion floats.
        """
        server = DsuServer(port=0)
        assert server.start() is True
        port = server.port

        clients = []
        reg_pkt = make_client_packet(MSG_TYPE_DATA)
        for _ in range(100):
            c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            c.bind(("127.0.0.1", 0))
            c.sendto(reg_pkt, ("127.0.0.1", port))
            clients.append(c)

        time.sleep(0.2)
        assert server.active_subscriber_count == 100

        # Stream 2,000 updates with distinct sensor payload
        test_state = make_dummy_state(
            accel_x=0.25, accel_y=-0.5, accel_z=0.82,
            gyro_x=15.0, gyro_y=-30.0, gyro_z=45.0,
        )

        t_start = time.perf_counter()
        for i in range(2000):
            test_state.motion_timestamp = 100_000 + i
            server.update_state(test_state)
        t_elapsed = time.perf_counter() - t_start

        # Confirm throughput exceeds 250 Hz
        rate = 2000.0 / t_elapsed
        assert rate >= 250.0, f"Streaming rate {rate:.1f} Hz was below 250 Hz requirement"

        # Verify received datagrams across clients
        verified_packets = 0
        for c in clients:
            c.settimeout(0.01)
            while True:
                try:
                    pkt, _ = c.recvfrom(1024)
                    assert len(pkt) == 100
                    assert pkt[:4] == DSU_MAGIC_SERVER
                    # Validate CRC32
                    calculated_crc = calculate_dsu_crc32(pkt)
                    pkt_crc = struct.unpack_from("<I", pkt, 8)[0]
                    assert pkt_crc == calculated_crc

                    # Validate 6-axis motion floats
                    ax, ay, az, gx, gy, gz = struct.unpack_from("<ffffff", pkt, 76)
                    assert abs(ax - 0.25) < 1e-4
                    assert abs(ay - (-0.5)) < 1e-4
                    assert abs(az - 0.82) < 1e-4
                    assert abs(gx - 15.0) < 1e-4
                    assert abs(gy - (-30.0)) < 1e-4
                    assert abs(gz - 45.0) < 1e-4
                    verified_packets += 1
                except (socket.timeout, BlockingIOError):
                    break

        assert verified_packets > 0, "No packets received by clients!"

        for c in clients:
            c.close()
        server.stop()

    def test_rapid_lifecycle_same_port(self):
        """Verify rapid start and stop lifecycle (20 iterations) on the same port releases sockets cleanly."""
        test_port = 26768
        for i in range(20):
            srv = DsuServer(host="127.0.0.1", port=test_port)
            assert srv.start() is True, f"Failed to bind on iteration {i}"
            assert srv.is_running is True
            srv.stop()
            assert srv.is_running is False

            # Verify port is free
            probe_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe_sock.bind(("127.0.0.1", test_port))
            except OSError as e:
                pytest.fail(f"Port {test_port} not released on iteration {i}: {e}")
            finally:
                probe_sock.close()
