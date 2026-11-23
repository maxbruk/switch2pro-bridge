"""Unit test suite for Cemuhook DSU Motion Server (dsu_server.py).

Verifies DSU Protocol v1001 compliance, IEEE 802.3 CRC32 calculation,
handshake messages (0x100000, 0x100001), 100-byte motion streaming (0x100002),
subscriber lifecycle, port collision resilience, clean thread teardown,
oversized UDP datagram resilience, and ICMP client disconnect resilience.
"""

import socket
import struct
import time
import zlib
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from dsu_server import DsuServer, calculate_dsu_crc32
from report_parser import ControllerState


# ============================================================================
# Protocol Helpers & Fixtures
# ============================================================================

def make_client_packet(msg_type: int, payload_extra: bytes = b"", client_id: int = 0x12345678) -> bytes:
    """Build a standard DSUC client packet with valid CRC32."""
    payload = struct.pack("<I", msg_type) + payload_extra
    length = len(payload)
    pkt = bytearray(16 + length)
    struct.pack_into("<4sHHII", pkt, 0, b"DSUC", 1001, length, 0, client_id)
    pkt[16:] = payload
    crc = zlib.crc32(pkt) & 0xFFFFFFFF
    struct.pack_into("<I", pkt, 8, crc)
    return bytes(pkt)


def unpack_server_header(packet: bytes) -> dict:
    """Unpack 16-byte server response header and verify CRC32."""
    assert len(packet) >= 16
    magic, version, length, crc, sender = struct.unpack_from("<4sHHII", packet, 0)
    temp = bytearray(packet)
    temp[8:12] = b"\x00\x00\x00\x00"
    expected_crc = zlib.crc32(temp) & 0xFFFFFFFF
    return {
        "magic": magic.decode("ascii", errors="replace"),
        "version": version,
        "length": length,
        "crc": crc,
        "expected_crc": expected_crc,
        "crc_valid": (crc == expected_crc),
        "sender": sender,
    }


@pytest.fixture
def test_server() -> Generator[DsuServer, None, None]:
    """Provides a running DsuServer on an ephemeral port, stopping on teardown."""
    server = DsuServer(host="127.0.0.1", port=0)
    assert server.start() is True
    yield server
    server.stop()


@pytest.fixture
def udp_client() -> Generator[socket.socket, None, None]:
    """Provides a UDP client socket with 1.5s timeout, closing on teardown."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(1.5)
    yield sock
    sock.close()


def make_dummy_state(**kwargs) -> ControllerState:
    """Helper to create ControllerState with defaults and optional overrides."""
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


# ============================================================================
# Unit Tests (1 through 10 + Ephemeral Port Resolution)
# ============================================================================

class TestDsuServerUnit:
    """Rigorous unit test suite for Milestone 2 DSU Server."""

    def test_crc32_standard_conformance(self):
        """Test 1: CRC32 calculation matches IEEE 802.3 and zeroes bytes 8..11."""
        pkt = bytearray(100)
        struct.pack_into("<4sHHII", pkt, 0, b"DSUS", 1001, 84, 0xAAAAAAAA, 0x11223344)
        calculated = calculate_dsu_crc32(pkt)

        # Verify bytes 8..11 zeroed reference
        temp = bytearray(pkt)
        temp[8:12] = b"\x00\x00\x00\x00"
        ref_crc = zlib.crc32(temp) & 0xFFFFFFFF
        assert calculated == ref_crc

        # Single byte change alters CRC
        pkt[20] ^= 0xFF
        assert calculate_dsu_crc32(pkt) != calculated

    def test_protocol_version_request_response(self, test_server, udp_client):
        """Test 2: Protocol version 0x100000 exchange returns 24-byte response."""
        req = make_client_packet(0x100000)
        udp_client.sendto(req, ("127.0.0.1", test_server.port))
        res, _ = udp_client.recvfrom(1024)

        assert len(res) == 24
        hdr = unpack_server_header(res)
        assert hdr["magic"] == "DSUS"
        assert hdr["version"] == 1001
        assert hdr["length"] == 8
        assert hdr["crc_valid"] is True
        msg_type, proto_ver = struct.unpack_from("<IH", res, 16)
        assert msg_type == 0x100000
        assert proto_ver == 1001

    def test_controller_info_request_response(self, test_server, udp_client):
        """Test 3: Controller info 0x100001 exchange for Slot 0 and non-existent slots."""
        # Query Slot 0
        req_slot0 = make_client_packet(0x100001, struct.pack("<BBBB", 0, 0, 0, 0))
        udp_client.sendto(req_slot0, ("127.0.0.1", test_server.port))
        res_0, _ = udp_client.recvfrom(1024)

        assert len(res_0) == 32
        hdr_0 = unpack_server_header(res_0)
        assert hdr_0["magic"] == "DSUS"
        assert hdr_0["length"] == 16
        assert hdr_0["crc_valid"] is True
        msg_type, slot, slot_state, model, conn = struct.unpack_from("<IBBBB", res_0, 16)
        assert msg_type == 0x100001
        assert slot == 0
        assert slot_state == 2  # Connected
        assert model == 2       # Full Gyro
        assert conn in (1, 2)   # USB or BT

        # Query Slot 3 (Out of range / disconnected)
        req_slot3 = make_client_packet(0x100001, struct.pack("<BBBB", 3, 0, 0, 0))
        udp_client.sendto(req_slot3, ("127.0.0.1", test_server.port))
        res_3, _ = udp_client.recvfrom(1024)
        assert len(res_3) == 32
        s3, s3_state = struct.unpack_from("<BB", res_3, 20)
        assert s3 == 3
        assert s3_state == 0    # Disconnected

    def test_data_streaming_unpacking_and_mapping(self, test_server, udp_client):
        """Test 4: Data message 0x100002 unpacks correct buttons, sticks, motion & timestamp."""
        # Subscribe
        sub = make_client_packet(0x100002, struct.pack("<BBBB", 0, 0, 0, 0))
        udp_client.sendto(sub, ("127.0.0.1", test_server.port))
        time.sleep(0.05)

        st = make_dummy_state(
            a=True, b=False, x=True, plus=True, dpad_up=True,
            left_stick_x=1.0, left_stick_y=0.0,
            zl_analog=0.5, zr_analog=1.0,
            accel_x=0.25, accel_y=-0.5, accel_z=0.98,
            gyro_x=15.0, gyro_y=-30.0, gyro_z=45.0,
            motion_timestamp=1234567,
        )
        test_server.update_state(st)
        res, _ = udp_client.recvfrom(1024)

        assert len(res) == 100
        hdr = unpack_server_header(res)
        assert hdr["length"] == 84
        assert hdr["crc_valid"] is True
        msg_type = struct.unpack_from("<I", res, 16)[0]
        assert msg_type == 0x100002

        # Verify sticks at byte 40
        lx, ly = struct.unpack_from("<BB", res, 40)
        assert abs(lx - 255) <= 1
        assert abs(ly - 128) <= 1

        # Verify timestamp at byte 68
        ts = struct.unpack_from("<Q", res, 68)[0]
        assert ts == 1234567

        # Verify floats at byte 76
        ax, ay, az, pitch, yaw, roll = struct.unpack_from("<ffffff", res, 76)
        assert abs(ax - 0.25) < 0.01
        assert abs(ay - (-0.5)) < 0.01
        assert abs(az - 0.98) < 0.01
        assert abs(pitch - 15.0) < 0.1
        assert abs(yaw - (-30.0)) < 0.1
        assert abs(roll - 45.0) < 0.1

    def test_client_subscription_registration_and_broadcasting(self, test_server):
        """Test 5: Multiple client subscription and simultaneous broadcast."""
        client_a = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_b = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_c = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for s in (client_a, client_b, client_c):
            s.settimeout(1.0)

        try:
            assert test_server.active_subscriber_count == 0

            # Subscribe A and B
            sub = make_client_packet(0x100002, struct.pack("<BBBB", 0, 0, 0, 0))
            client_a.sendto(sub, ("127.0.0.1", test_server.port))
            client_b.sendto(sub, ("127.0.0.1", test_server.port))
            time.sleep(0.05)
            assert test_server.active_subscriber_count == 2

            # Dispatch state
            test_server.update_state(make_dummy_state(gyro_x=55.0))
            data_a, _ = client_a.recvfrom(1024)
            data_b, _ = client_b.recvfrom(1024)
            assert len(data_a) == 100
            assert len(data_b) == 100

            # C receives nothing
            with pytest.raises(socket.timeout):
                client_c.recvfrom(1024)
        finally:
            client_a.close()
            client_b.close()
            client_c.close()

    def test_inactive_client_timeout_pruning(self, test_server, udp_client):
        """Test 6: Inactive client timeout (>5s) auto-pruning."""
        sub = make_client_packet(0x100002, struct.pack("<BBBB", 0, 0, 0, 0))
        udp_client.sendto(sub, ("127.0.0.1", test_server.port))
        time.sleep(0.05)
        assert test_server.active_subscriber_count == 1

        # Advance time by 5.5s
        now = time.time()
        with patch("time.time", return_value=now + 5.5):
            test_server.prune_subscribers()
            assert test_server.active_subscriber_count == 0

    def test_zero_clients_update_state_noop(self, test_server):
        """Test 7: Zero clients connected does not call sendto or raise."""
        assert test_server.active_subscriber_count == 0
        orig_sock = test_server._socket
        mock_sock = MagicMock()
        test_server._socket = mock_sock
        try:
            t0 = time.perf_counter()
            for _ in range(50):
                test_server.update_state(make_dummy_state())
            elapsed = time.perf_counter() - t0
            mock_sock.sendto.assert_not_called()
            assert elapsed < 0.010  # Negligible overhead (< 10 ms for 50 calls)
        finally:
            test_server._socket = orig_sock

    def test_socket_port_conflict_graceful_failure(self):
        """Test 8: Port collision returns False cleanly without unhandled exception."""
        dummy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dummy.bind(("127.0.0.1", 0))
        occupied_port = dummy.getsockname()[1]

        try:
            srv = DsuServer(host="127.0.0.1", port=occupied_port)
            started = srv.start()
            assert started is False
            assert srv.is_running is False
        finally:
            dummy.close()

        # Port is now free
        assert srv.start() is True
        assert srv.is_running is True
        srv.stop()

    def test_server_clean_lifecycle(self):
        """Test 9: start() and stop() clean daemon thread lifecycle."""
        srv = DsuServer(port=0)
        assert srv.is_running is False
        assert srv.start() is True
        assert srv.is_running is True
        assert srv._thread.is_alive() is True
        assert srv._thread.daemon is True

        # Re-entrant start
        assert srv.start() is True

        # Teardown
        srv.stop()
        assert srv.is_running is False
        srv._thread.join(timeout=1.0)
        assert srv._thread.is_alive() is False

        # Re-entrant stop
        srv.stop()

    def test_malformed_packets_dropped_silently(self, test_server, udp_client):
        """Test 10: Truncated headers and invalid magic do not crash server."""
        # Truncated
        udp_client.sendto(b"SHORT", ("127.0.0.1", test_server.port))
        # Bad magic
        bad = bytearray(20)
        struct.pack_into("<4sHHII", bad, 0, b"XXXX", 1001, 4, 0, 0)
        udp_client.sendto(bad, ("127.0.0.1", test_server.port))

        # Server still alive and responds to valid request
        req = make_client_packet(0x100000)
        udp_client.sendto(req, ("127.0.0.1", test_server.port))
        res, _ = udp_client.recvfrom(1024)
        assert unpack_server_header(res)["crc_valid"] is True

    def test_ephemeral_port_resolution(self):
        """Test 11: Binding port 0 resolves actual OS-assigned ephemeral port."""
        srv = DsuServer(port=0)
        assert srv.port == 0
        assert srv.start() is True
        try:
            assert srv.port > 0
            assert isinstance(srv.port, int)
        finally:
            srv.stop()

    def test_client_abrupt_disconnect_does_not_crash_listener(self, test_server):
        """Test 12: Confirm server listener survives client disconnect and ICMP Port Unreachable (WinError 10054)."""
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sub = make_client_packet(0x100002, struct.pack("<BBBB", 0, 0, 0, 0))
        client.sendto(sub, ("127.0.0.1", test_server.port))
        time.sleep(0.05)
        # Abruptly close client socket
        client.close()

        # Server sends multiple states to closed endpoint
        dummy_st = make_dummy_state()
        for _ in range(5):
            test_server.update_state(dummy_st)
            time.sleep(0.005)

        time.sleep(0.05)
        # Verify listener thread is still running
        assert test_server._thread is not None and test_server._thread.is_alive()

        # Verify a new client can still interact with the server
        new_client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        new_client.settimeout(1.5)
        try:
            new_client.sendto(make_client_packet(0x100000), ("127.0.0.1", test_server.port))
            res, _ = new_client.recvfrom(1024)
            assert len(res) == 24
            assert res[:4] == b"DSUS"
            assert unpack_server_header(res)["crc_valid"] is True
        finally:
            new_client.close()

    def test_oversized_packet_does_not_crash_listener(self, test_server, udp_client):
        """Test 13: Confirm oversized datagrams (>1024 bytes) do not crash listener thread on Windows (WinError 10040)."""
        udp_client.sendto(b"X" * 2048, ("127.0.0.1", test_server.port))
        time.sleep(0.05)
        assert test_server._thread is not None and test_server._thread.is_alive()

        # Legitimate request works
        udp_client.sendto(make_client_packet(0x100000), ("127.0.0.1", test_server.port))
        res, _ = udp_client.recvfrom(1024)
        assert len(res) == 24
        assert res[:4] == b"DSUS"
        assert unpack_server_header(res)["crc_valid"] is True
