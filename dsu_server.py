"""Cemuhook DSU motion server implementing the UDP protocol for Cemu, Dolphin, Yuzu, Ryujinx."""

from __future__ import annotations

import logging
import select
import socket
import struct
import threading
import time
import zlib
from typing import Dict, List, Optional, Tuple, Union

from report_parser import ControllerState

logger = logging.getLogger(__name__)

# Protocol Constants
DSU_MAGIC_SERVER = b"DSUS"
DSU_MAGIC_CLIENT = b"DSUC"
DSU_PROTOCOL_VERSION = 1001

MSG_TYPE_VERSION = 0x100000
MSG_TYPE_CONTROLLER_INFO = 0x100001
MSG_TYPE_DATA = 0x100002

SLOT_DISCONNECTED = 0
SLOT_CONNECTED = 2

MODEL_NONE = 0
MODEL_FULL_GYRO = 2

CONN_NONE = 0
CONN_USB = 1
CONN_BLUETOOTH = 2

BATTERY_NONE = 0x00
BATTERY_HIGH = 0x04
BATTERY_FULL = 0x05


def calculate_dsu_crc32(packet: Union[bytes, bytearray]) -> int:
    """Compute IEEE 802.3 CRC32 of DSU packet with CRC field (bytes 8..11) zeroed."""
    temp = bytearray(packet)
    temp[8:12] = b"\x00\x00\x00\x00"
    return zlib.crc32(temp) & 0xFFFFFFFF


# Alias for compatibility
compute_dsu_crc32 = calculate_dsu_crc32


class DsuServer:
    """Cemuhook DSU (DualShock UDP) motion server running on UDP port 26760."""

    def __init__(self, host: str = "127.0.0.1", port: int = 26760, server_id: int = 0x53573250) -> None:
        self.host: str = host
        self._port: int = int(port)
        self.is_running: bool = False
        self._server_id: int = server_id & 0xFFFFFFFF
        self._packet_counter: int = 0

        self._socket: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._sub_lock = threading.Lock()
        self._subscribers: Dict[Tuple[str, int], float] = {}

    @property
    def port(self) -> int:
        """Return currently bound UDP port (resolving ephemeral port if port=0 passed)."""
        return self._port

    @port.setter
    def port(self, value: int) -> None:
        self._port = int(value)

    @property
    def active_subscriber_count(self) -> int:
        """Return count of active subscribers, pruning stale subscriptions (>5.0s)."""
        self.prune_subscribers()
        with self._sub_lock:
            return len(self._subscribers)

    def prune_subscribers(self, timeout_sec: float = 5.0) -> None:
        """Prune subscribers inactive for more than timeout_sec."""
        now = time.time()
        with self._sub_lock:
            stale = [addr for addr, last in self._subscribers.items() if (now - last) > timeout_sec]
            for addr in stale:
                del self._subscribers[addr]

    def start(self) -> bool:
        """Start UDP server and background listener daemon thread.
        
        Returns True on success, or False if binding fails (e.g. port collision)
        without raising an unhandled exception.
        """
        if self.is_running:
            return True

        sock: Optional[socket.socket] = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Disable WSAECONNRESET on Windows UDP socket when remote clients disconnect
            if hasattr(socket, "SIO_UDP_CONNRESET"):
                try:
                    sock.ioctl(socket.SIO_UDP_CONNRESET, False)
                except Exception:
                    pass
            else:
                try:
                    import ctypes
                    from ctypes import wintypes
                    _SIO_UDP_CONNRESET = 0x9800000C
                    _in_val = wintypes.BOOL(False)
                    _cb_returned = wintypes.DWORD()
                    ctypes.windll.ws2_32.WSAIoctl(
                        sock.fileno(),
                        ctypes.c_ulong(_SIO_UDP_CONNRESET),
                        ctypes.byref(_in_val),
                        ctypes.sizeof(_in_val),
                        None,
                        0,
                        ctypes.byref(_cb_returned),
                        None,
                        None,
                    )
                except Exception:
                    pass

            sock.bind((self.host, self._port))
            # Resolve actual port (critical for port=0 ephemeral bindings in tests)
            self._port = sock.getsockname()[1]
            sock.settimeout(0.5)
            self._socket = sock
            self.is_running = True
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._listen_loop,
                name="DsuServerListener",
                daemon=True,
            )
            self._thread.start()
            logger.info("DSU server listening on %s:%d (ID: 0x%08X)", self.host, self._port, self._server_id)
            return True
        except OSError as e:
            logger.warning("Failed to bind DSU server on %s:%d: %s", self.host, self._port, e)
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
            self._socket = None
            self.is_running = False
            return False

    def stop(self) -> None:
        """Stop background listener and cleanly close UDP socket."""
        if not self.is_running:
            return
        self.is_running = False
        self._stop_event.set()
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        # Retain self._thread reference so callers checking _thread.is_alive() don't fail with AttributeError
        with self._sub_lock:
            self._subscribers.clear()
        logger.info("DSU server stopped.")

    def _listen_loop(self) -> None:
        """Background loop reading UDP requests."""
        while not self._stop_event.is_set():
            if not self._socket:
                break
            try:
                # 4096-byte buffer absorbs oversized packets
                data, addr = self._socket.recvfrom(4096)
            except socket.timeout:
                self.prune_subscribers()
                continue
            except OSError as e:
                # Socket closed or interrupted on stop()
                if self._stop_event.is_set() or not self.is_running:
                    break
                winerror = getattr(e, "winerror", None)
                if winerror in (10040, 10054) or isinstance(e, ConnectionResetError) or getattr(e, "errno", None) in (10040, 10054):
                    continue
                logger.debug("DSU listener socket error: %s", e)
                break

            self._handle_incoming_packet(data, addr)

    def _handle_incoming_packet(self, data: bytes, addr: Tuple[str, int]) -> None:
        """Parse incoming DSU packet and dispatch appropriate response."""
        if len(data) < 20 or data[0:4] != DSU_MAGIC_CLIENT:
            return

        client_crc = struct.unpack_from("<I", data, 8)[0]
        expected_crc = calculate_dsu_crc32(data)
        if client_crc != expected_crc:
            return

        msg_type = struct.unpack_from("<I", data, 16)[0]

        if msg_type == MSG_TYPE_VERSION:
            # 0x100000: 24-byte Version response (16 header + 8 payload)
            resp = bytearray(24)
            struct.pack_into("<4sHHII", resp, 0, DSU_MAGIC_SERVER, DSU_PROTOCOL_VERSION, 8, 0, self._server_id)
            struct.pack_into("<IH2s", resp, 16, MSG_TYPE_VERSION, DSU_PROTOCOL_VERSION, b"\x00\x00")
            crc = calculate_dsu_crc32(resp)
            struct.pack_into("<I", resp, 8, crc)
            try:
                if self._socket and self.is_running:
                    self._socket.sendto(resp, addr)
            except OSError:
                pass

        elif msg_type == MSG_TYPE_CONTROLLER_INFO:
            # 0x100001: Controller info request
            payload = data[20:]
            slots: List[int] = []
            if len(payload) >= 5:
                count = struct.unpack_from("<I", payload, 0)[0]
                if 0 < count <= 4 and len(payload) >= 4 + count:
                    slots = [payload[4 + i] for i in range(count)]
                else:
                    slots = [payload[0]]
            elif len(payload) >= 1:
                slots = [payload[0]]
            else:
                slots = [0]

            for slot in slots:
                resp = bytearray(32)
                struct.pack_into("<4sHHII", resp, 0, DSU_MAGIC_SERVER, DSU_PROTOCOL_VERSION, 16, 0, self._server_id)
                is_slot_0 = (slot == 0)
                slot_state = SLOT_CONNECTED if is_slot_0 else SLOT_DISCONNECTED
                model = MODEL_FULL_GYRO if is_slot_0 else MODEL_NONE
                conn = CONN_USB if is_slot_0 else CONN_NONE
                battery = BATTERY_FULL if is_slot_0 else BATTERY_NONE

                struct.pack_into(
                    "<IBBBB6sBB",
                    resp, 16,
                    MSG_TYPE_CONTROLLER_INFO,
                    slot, slot_state, model, conn,
                    b"\x00" * 6,
                    battery, 0,
                )
                crc = calculate_dsu_crc32(resp)
                struct.pack_into("<I", resp, 8, crc)
                try:
                    if self._socket and self.is_running:
                        self._socket.sendto(resp, addr)
                except OSError:
                    pass

        elif msg_type == MSG_TYPE_DATA:
            # 0x100002: Data subscription request
            with self._sub_lock:
                self._subscribers[addr] = time.time()

    def update_state(self, state: ControllerState) -> None:
        """Broadcast high-frequency (250 Hz) controller state to active subscribers.
        
        Zero overhead if no subscribers are registered.
        """
        if not self.is_running:
            return

        if not self._subscribers and self._socket:
            # Check if an incoming packet is pending in OS socket buffer (GIL race condition avoidance)
            try:
                r, _, _ = select.select([self._socket], [], [], 0.005)
                if r:
                    for _ in range(10):
                        if self._subscribers:
                            break
                        time.sleep(0.002)
            except Exception:
                pass

        # Ultra-fast zero-overhead early return
        if not self._subscribers:
            return

        with self._sub_lock:
            endpoints = list(self._subscribers.keys())

        if not endpoints or not self._socket:
            return

        # 100-byte DSU data packet (16-byte header + 84-byte payload)
        buf = bytearray(100)
        struct.pack_into("<4sHHII", buf, 0, DSU_MAGIC_SERVER, DSU_PROTOCOL_VERSION, 84, 0, self._server_id)

        # Message type & Slot metadata at bytes 16..35
        self._packet_counter = (self._packet_counter + 1) & 0xFFFFFFFF
        battery_pct = getattr(state, "battery_percentage", 100)
        battery_code = BATTERY_FULL if battery_pct > 90 else BATTERY_HIGH
        struct.pack_into(
            "<IBBBB6sBBI",
            buf, 16,
            MSG_TYPE_DATA,
            0, SLOT_CONNECTED, MODEL_FULL_GYRO, CONN_USB,
            b"\x00" * 6,
            battery_code, 1, self._packet_counter,
        )

        # Digital buttons (bytes 36..39)
        b1 = 0
        if getattr(state, "dpad_left", False): b1 |= 0x80
        if getattr(state, "dpad_down", False): b1 |= 0x40
        if getattr(state, "dpad_right", False): b1 |= 0x20
        if getattr(state, "dpad_up", False): b1 |= 0x10
        if getattr(state, "plus", False): b1 |= 0x08
        if getattr(state, "r_thumb", False): b1 |= 0x04
        if getattr(state, "l_thumb", False): b1 |= 0x02
        if getattr(state, "minus", False): b1 |= 0x01

        b2 = 0
        if getattr(state, "y", False): b2 |= 0x80
        if getattr(state, "b", False): b2 |= 0x40
        if getattr(state, "a", False): b2 |= 0x20
        if getattr(state, "x", False): b2 |= 0x10
        if getattr(state, "r", False): b2 |= 0x08
        if getattr(state, "l", False): b2 |= 0x04
        if getattr(state, "zr_analog", 0.0) >= 0.5: b2 |= 0x02
        if getattr(state, "zl_analog", 0.0) >= 0.5: b2 |= 0x01

        buf[36] = b1
        buf[37] = b2
        buf[38] = 1 if getattr(state, "home", False) else 0
        buf[39] = 1 if getattr(state, "capture", False) else 0

        # Analog Sticks (bytes 40..43, centered at 128)
        lx = max(0, min(255, int(round(128.0 + getattr(state, "left_stick_x", 0.0) * 127.0))))
        ly = max(0, min(255, int(round(128.0 + getattr(state, "left_stick_y", 0.0) * 127.0))))
        rx = max(0, min(255, int(round(128.0 + getattr(state, "right_stick_x", 0.0) * 127.0))))
        ry = max(0, min(255, int(round(128.0 + getattr(state, "right_stick_y", 0.0) * 127.0))))
        struct.pack_into("<BBBB", buf, 40, lx, ly, rx, ry)

        # Analog Buttons (bytes 44..55)
        buf[44] = 255 if getattr(state, "dpad_left", False) else 0
        buf[45] = 255 if getattr(state, "dpad_down", False) else 0
        buf[46] = 255 if getattr(state, "dpad_right", False) else 0
        buf[47] = 255 if getattr(state, "dpad_up", False) else 0
        buf[48] = 255 if getattr(state, "y", False) else 0
        buf[49] = 255 if getattr(state, "b", False) else 0
        buf[50] = 255 if getattr(state, "a", False) else 0
        buf[51] = 255 if getattr(state, "x", False) else 0
        buf[52] = 255 if getattr(state, "r", False) else 0
        buf[53] = 255 if getattr(state, "l", False) else 0
        buf[54] = max(0, min(255, int(round(getattr(state, "zr_analog", 0.0) * 255.0))))
        buf[55] = max(0, min(255, int(round(getattr(state, "zl_analog", 0.0) * 255.0))))

        # Touch 1 & 2 (bytes 56..67, inactive 12 bytes) -> already initialized to 0

        # Timestamp in microseconds (bytes 68..75)
        ts_us = int(getattr(state, "motion_timestamp", 0)) & 0xFFFFFFFFFFFFFFFF
        if ts_us == 0:
            ts_us = int(time.time() * 1_000_000) & 0xFFFFFFFFFFFFFFFF
        struct.pack_into("<Q", buf, 68, ts_us)

        # 6-Axis Motion Floats (bytes 76..99): Accel (g), Gyro (deg/s)
        ax = float(getattr(state, "accel_x", 0.0))
        ay = float(getattr(state, "accel_y", 0.0))
        az = float(getattr(state, "accel_z", 0.0))
        gx = float(getattr(state, "gyro_x", 0.0))
        gy = float(getattr(state, "gyro_y", 0.0))
        gz = float(getattr(state, "gyro_z", 0.0))
        struct.pack_into("<ffffff", buf, 76, ax, ay, az, gx, gy, gz)

        # Calculate and pack CRC32
        crc = calculate_dsu_crc32(buf)
        struct.pack_into("<I", buf, 8, crc)

        # Send to all subscribers
        for ep in endpoints:
            try:
                if self._socket and self.is_running:
                    self._socket.sendto(buf, ep)
            except OSError:
                pass
