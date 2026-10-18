"""Bluetooth Low Energy (BLE) connector and pairing protocol for Nintendo Switch 2 Pro Controller."""

import asyncio
import uuid
import logging
import threading
from typing import Optional, Callable, Dict, Any
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

from constants import (
    NINTENDO_VENDOR_ID,
    SWITCH2_PRO_PID,
    SWITCH1_PRO_PID,
    NSO_GAMECUBE_PID,
    SUPPORTED_PIDS,
    DEVICE_NAMES,
    NINTENDO_BLE_MANUFACTURER_ID,
    SW2_SERVICE_UUID,
    INPUT_REPORT_UUID,
    COMMAND_WRITE_UUID,
    COMMAND_PAIR,
    SUBCOMMAND_PAIR_SET_MAC,
    SUBCOMMAND_PAIR_STEP1,
    SUBCOMMAND_PAIR_STEP2,
    SUBCOMMAND_PAIR_FINISH,
    PAIRING_STEP1_BYTES,
    PAIRING_STEP2_BYTES,
)
from report_parser import parse_input_report, ControllerState

logger = logging.getLogger(__name__)


class BleProConnector:
    """Manages Bluetooth Low Energy scanning, GATT pairing handshake, and input reports."""

    def __init__(self,
                 on_state_change: Optional[Callable[[ControllerState], None]] = None,
                 on_disconnect: Optional[Callable[[], None]] = None,
                 on_status: Optional[Callable[[str], None]] = None):
        self.on_state_change = on_state_change
        self.on_disconnect = on_disconnect
        self.on_status = on_status

        self.client: Optional[BleakClient] = None
        self.device: Optional[BLEDevice] = None
        self.device_info: Optional[Dict[str, Any]] = None
        self.is_connected = False
        self.deadzone = 0.05

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._scanner: Optional[BleakScanner] = None
        self._lock = threading.Lock()

    def start_connection(self):
        """Start background asyncio worker thread to scan and connect via BLE."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_event_loop, name="BleProWorker", daemon=True)
            self._thread.start()

    def stop_connection(self):
        """Stop BLE scanner and disconnect cleanly."""
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._async_disconnect(), self._loop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run_event_loop(self):
        """Dedicated asyncio event loop thread for Bleak."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._scan_and_connect_routine())
        except Exception as e:
            logger.debug("BLE event loop ended: %s", e)
        finally:
            try:
                self._loop.run_until_complete(self._async_disconnect())
            except Exception:
                pass
            self._loop.close()
            self._loop = None

    async def _scan_and_connect_routine(self):
        """Scan for Switch 2 Pro Controller advertisements and connect."""
        if self.on_status:
            self.on_status("Scanning for Switch 2 Pro Controller (Bluetooth)...")
        logger.info("Starting BLE scan for Switch 2 Pro Controller...")

        # Detect local Bluetooth adapter MAC
        try:
            pc_mac = uuid.getnode()
        except Exception:
            pc_mac = 0

        found_device = None
        is_pairing_mode = False
        product_id = SWITCH2_PRO_PID

        def detection_callback(device: BLEDevice, adv: AdvertisementData):
            nonlocal found_device, is_pairing_mode, product_id
            mdata = adv.manufacturer_data.get(NINTENDO_BLE_MANUFACTURER_ID)
            if not mdata or len(mdata) < 7:
                return

            vid = int.from_bytes(mdata[3:5], "little")
            pid = int.from_bytes(mdata[5:7], "little")

            if vid == NINTENDO_VENDOR_ID and pid in SUPPORTED_PIDS:
                reconnect_mac = 0
                if len(mdata) >= 16:
                    reconnect_mac = int.from_bytes(mdata[10:16], "little")

                is_paired_to_this_pc = (reconnect_mac != 0 and reconnect_mac == pc_mac)
                is_sync_pairing = (reconnect_mac == 0)

                if not is_sync_pairing and not is_paired_to_this_pc:
                    # Controller is waking up in Console-Reconnect mode targeting another device (e.g. Switch console).
                    # In this state the controller ignores connections from new hosts.
                    # DO NOT abort scanning: prompt user to hold Sync button to enter pairing mode!
                    logger.info(
                        "Controller detected in Console-Reconnect mode (Target MAC=0x%012X). Waiting for Sync button...",
                        reconnect_mac,
                    )
                    if self.on_status:
                        self.on_status("Controller detected! Hold Sync button for 3s until LEDs sweep back & forth...")
                    return

                logger.info(
                    "Detected ready Switch controller: PID=0x%04X, PairingMode=%s, ReconnectMAC=0x%012X",
                    pid, is_sync_pairing, reconnect_mac,
                )
                found_device = device
                product_id = pid
                is_pairing_mode = is_sync_pairing

        self._scanner = BleakScanner(detection_callback)
        await self._scanner.start()

        # Scan until found or stopped (allow up to 30 seconds for user to hold Sync)
        timeout_seconds = 30.0
        elapsed = 0.0
        while not self._stop_event.is_set() and found_device is None:
            await asyncio.sleep(0.5)
            elapsed += 0.5
            if elapsed >= timeout_seconds:
                break

        await self._scanner.stop()
        self._scanner = None

        if found_device is None:
            if self.on_status:
                self.on_status("No controller in pairing mode found. Hold Sync button until LEDs sweep to pair.")
            logger.warning("BLE scan finished without detecting a pairing Switch Pro Controller.")
            return

        self.device = found_device
        self.device_info = {
            "name": DEVICE_NAMES.get(product_id, "Nintendo Switch Pro Controller"),
            "product_id": product_id,
            "address": found_device.address,
            "mode": "Bluetooth (BLE)",
        }

        await self._connect_client(found_device, is_pairing_mode)

    async def _connect_client(self, device: BLEDevice, is_pairing_mode: bool):
        """Establish GATT connection, pair if needed, and subscribe to input reports."""
        if self.on_status:
            self.on_status(f"Connecting to {device.name or device.address}...")

        def on_disconnected(client: BleakClient):
            was_connected = self.is_connected
            self.is_connected = False
            logger.info("BLE controller disconnected from host (was_connected=%s).", was_connected)
            if was_connected and self.on_disconnect:
                self.on_disconnect()

        try:
            # Pass BLEDevice directly for optimal WinRT backend handling on Windows
            self.client = BleakClient(device, disconnected_callback=on_disconnected)
            await self.client.connect(timeout=15.0)
            if not self.client.is_connected:
                if self.on_status:
                    self.on_status("BLE connection timed out. Ensure Sync was held until LEDs flash.")
                return

            logger.info("Connected to BLE controller GATT profile.")

            # Optimize connection latency on Windows
            try:
                from bleak.backends.winrt.client import BleakClientWinRT
                from winrt.windows.devices.bluetooth import BluetoothLEPreferredConnectionParameters
                backend = self.client._backend
                if isinstance(backend, BleakClientWinRT):
                    backend._requester.request_preferred_connection_parameters(
                        BluetoothLEPreferredConnectionParameters.throughput_optimized
                    )
            except Exception:
                pass

            # Listen for command responses on COMMAND_RESPONSE_UUID
            self._cmd_future: Optional[asyncio.Future] = None
            def on_cmd_response(sender, data: bytearray):
                if self._cmd_future and not self._cmd_future.done():
                    self._cmd_future.set_result(bytes(data))

            try:
                await self.client.start_notify(COMMAND_RESPONSE_UUID, on_cmd_response)
            except Exception as e:
                logger.debug("Command response notify subscription notice: %s", e)

            # Perform host pairing handshake if newly synced
            if is_pairing_mode:
                if self.on_status:
                    self.on_status("Pairing with controller (Bluetooth LE handshake)...")
                await self._perform_pairing_handshake()

            # Set solid Player 1 indicator LED (stops searching/flashing pattern)
            try:
                await self._set_player_led(1)
            except Exception as e:
                logger.debug("Set Player LED notice: %s", e)

            # Subscribe to input report characteristic notifications
            if self.on_status:
                self.on_status("Subscribing to input reports...")

            def input_report_handler(sender, data: bytearray):
                state = parse_input_report(bytes(data), deadzone=self.deadzone, has_report_id=False)
                if state and self.on_state_change:
                    self.on_state_change(state)

            await self.client.start_notify(INPUT_REPORT_UUID, input_report_handler)
            self.is_connected = True

            if self.on_status:
                self.on_status("Connected via Bluetooth!")
            logger.info("BLE input report streaming active.")

            # Keep loop alive while connected
            while not self._stop_event.is_set() and self.client.is_connected:
                await asyncio.sleep(0.5)

        except Exception as e:
            err_type = type(e).__name__
            err_str = str(e).strip()
            if not err_str or "Timeout" in err_type:
                msg = "Connection timed out. Ensure the Sync button was held until LEDs cycle back & forth."
            else:
                msg = f"{err_type}: {err_str}"
            logger.error("Failed in BLE connection pipeline: %s (%s)", err_type, err_str)
            if self.on_status:
                self.on_status(f"Bluetooth Error: {msg}")
        finally:
            await self._async_disconnect()

    async def _write_command(self, cmd_id: int, subcmd_id: int, data: bytes = b""):
        """Build and transmit framed Switch 2 BLE command: [cmd_id, 0x91, 0x01, subcmd_id, 0x00, len, 0x00, 0x00] + data."""
        buf = bytes([cmd_id, 0x91, 0x01, subcmd_id, 0x00, len(data), 0x00, 0x00]) + data
        logger.debug("Writing Switch 2 BLE command 0x%02X sub 0x%02X (%d bytes)", cmd_id, subcmd_id, len(buf))
        self._cmd_future = asyncio.get_running_loop().create_future()
        await self.client.write_gatt_char(COMMAND_WRITE_UUID, buf, response=False)
        try:
            return await asyncio.wait_for(asyncio.shield(self._cmd_future), timeout=1.0)
        except asyncio.TimeoutError:
            return None

    async def _perform_pairing_handshake(self):
        """Send the host-bonding sequence with correct 0x91 0x01 framing."""
        if not self.client:
            return
        try:
            # 1. SET_MAC: Send our PC's Bluetooth adapter MAC
            pc_mac = uuid.getnode()
            pc_mac_bytes = pc_mac.to_bytes(6, "little")
            mac_payload = b"\x00\x02" + pc_mac_bytes + pc_mac_bytes
            await self._write_command(COMMAND_PAIR, SUBCOMMAND_PAIR_SET_MAC, mac_payload)
            await asyncio.sleep(0.04)

            # 2. Handshake step 1 (17 bytes)
            await self._write_command(COMMAND_PAIR, SUBCOMMAND_PAIR_STEP1, PAIRING_STEP1_BYTES)
            await asyncio.sleep(0.04)

            # 3. Handshake step 2 (17 bytes)
            await self._write_command(COMMAND_PAIR, SUBCOMMAND_PAIR_STEP2, PAIRING_STEP2_BYTES)
            await asyncio.sleep(0.04)

            # 4. FINISH (1 byte: 0x00)
            await self._write_command(COMMAND_PAIR, SUBCOMMAND_PAIR_FINISH, b"\x00")
            logger.info("Bluetooth LE pairing handshake completed successfully.")
        except Exception as e:
            logger.warning("Error during BLE pairing handshake: %s", e)

    async def _set_player_led(self, player_number: int = 1):
        """Set solid player indicator LED pattern (e.g. Player 1 solid green light)."""
        pattern = {1: 0x01, 2: 0x03, 3: 0x07, 4: 0x0F}.get(player_number, 0x01)
        data = bytes([pattern, 0x00, 0x00, 0x00])
        COMMAND_LEDS = 0x09
        SUBCOMMAND_LEDS_SET_PLAYER = 0x07
        await self._write_command(COMMAND_LEDS, SUBCOMMAND_LEDS_SET_PLAYER, data)
        logger.info("Set controller Player %d LED solid.", player_number)

    async def _async_disconnect(self):
        """Asynchronously disconnect GATT client."""
        self.is_connected = False
        if self.client and self.client.is_connected:
            try:
                await self.client.stop_notify(INPUT_REPORT_UUID)
            except Exception:
                pass
            try:
                await self.client.stop_notify(COMMAND_RESPONSE_UUID)
            except Exception:
                pass
            try:
                await self.client.disconnect()
            except Exception:
                pass
        self.client = None
