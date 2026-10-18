"""Direct USB-C communication and WinUSB handshake for Nintendo Switch 2 Pro Controller."""

import os
import time
import uuid
import ctypes
from ctypes import wintypes
import logging
import threading
from typing import Optional, Callable, Dict, Any, List
import hid

from constants import (
    NINTENDO_VENDOR_ID,
    SWITCH2_PRO_PID,
    SWITCH1_PRO_PID,
    NSO_GAMECUBE_PID,
    SUPPORTED_PIDS,
    DEVICE_NAMES,
    OUTPUT_REPORT_ID_PRO2,
    PRO2_OUTPUT_REPORT_BODY_SIZE,
    USB_STARTUP_COMMANDS,
    USB_INPUT_REPORT_SIZE,
)
from report_parser import parse_input_report, ControllerState

# Enable 1ms high-resolution timer for Windows scheduler
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

logger = logging.getLogger(__name__)

# Default Device Interface GUID for Switch 2 Pro Controller WinUSB interface
DEFAULT_WINUSB_GUID = "{6F13725E-EF0E-4FD3-AE5F-B2DE989EC825}"


def _guid_from_string(value: str):
    """Convert GUID string into ctypes GUID structure."""
    guid_val = uuid.UUID(value.strip("{}"))
    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]
    return GUID(
        guid_val.time_low,
        guid_val.time_mid,
        guid_val.time_hi_version,
        (ctypes.c_ubyte * 8)(*guid_val.bytes[8:]),
    )


def get_winusb_interface_paths(product_id: int = SWITCH2_PRO_PID) -> List[str]:
    """Discover WinUSB device interface paths for the controller's MI_01 interface."""
    guids = [DEFAULT_WINUSB_GUID]

    # Look up registry for registered DeviceInterfaceGUIDs
    try:
        import winreg
        base_path = rf"SYSTEM\CurrentControlSet\Enum\USB\VID_{NINTENDO_VENDOR_ID:04X}&PID_{product_id:04X}&MI_01"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base_path) as base_key:
            idx = 0
            while True:
                try:
                    instance = winreg.EnumKey(base_key, idx)
                    idx += 1
                except OSError:
                    break
                try:
                    with winreg.OpenKey(base_key, instance + r"\Device Parameters") as params_key:
                        val = winreg.QueryValueEx(params_key, "DeviceInterfaceGUID")[0]
                        if isinstance(val, str):
                            if val not in guids:
                                guids.append(val)
                        else:
                            for g in val:
                                if g not in guids:
                                    guids.append(g)
                except OSError:
                    pass
    except Exception as e:
        logger.debug("Registry query for WinUSB GUIDs notice: %s", e)

    # Use SetupAPI to enumerate device interface paths
    paths = []
    try:
        setupapi = ctypes.WinDLL("setupapi", use_last_error=True)
        DIGCF_PRESENT = 0x00000002
        DIGCF_DEVICEINTERFACE = 0x00000010
        INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

        for guid_str in guids:
            guid = _guid_from_string(guid_str)

            class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("InterfaceClassGuid", type(guid)),
                    ("Flags", wintypes.DWORD),
                    ("Reserved", ctypes.c_void_p),
                ]

            class SP_DEVICE_INTERFACE_DETAIL_DATA_W(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("DevicePath", wintypes.WCHAR * 1024),
                ]

            setupapi.SetupDiGetClassDevsW.argtypes = [
                ctypes.POINTER(type(guid)), wintypes.LPCWSTR, wintypes.HWND, wintypes.DWORD
            ]
            setupapi.SetupDiGetClassDevsW.restype = wintypes.HANDLE

            setupapi.SetupDiEnumDeviceInterfaces.argtypes = [
                wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(type(guid)), wintypes.DWORD,
                ctypes.POINTER(SP_DEVICE_INTERFACE_DATA)
            ]
            setupapi.SetupDiEnumDeviceInterfaces.restype = wintypes.BOOL

            setupapi.SetupDiGetDeviceInterfaceDetailW.argtypes = [
                wintypes.HANDLE, ctypes.POINTER(SP_DEVICE_INTERFACE_DATA),
                ctypes.POINTER(SP_DEVICE_INTERFACE_DETAIL_DATA_W), wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p
            ]
            setupapi.SetupDiGetDeviceInterfaceDetailW.restype = wintypes.BOOL
            setupapi.SetupDiDestroyDeviceInfoList.argtypes = [wintypes.HANDLE]

            info_set = setupapi.SetupDiGetClassDevsW(ctypes.byref(guid), None, None, DIGCF_PRESENT | DIGCF_DEVICEINTERFACE)
            if info_set == INVALID_HANDLE_VALUE:
                continue

            try:
                index = 0
                while True:
                    iface_data = SP_DEVICE_INTERFACE_DATA()
                    iface_data.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
                    if not setupapi.SetupDiEnumDeviceInterfaces(info_set, None, ctypes.byref(guid), index, ctypes.byref(iface_data)):
                        break
                    index += 1

                    detail = SP_DEVICE_INTERFACE_DETAIL_DATA_W()
                    detail.cbSize = 8 if ctypes.sizeof(ctypes.c_void_p) == 8 else 6
                    req = wintypes.DWORD()
                    if setupapi.SetupDiGetDeviceInterfaceDetailW(
                        info_set, ctypes.byref(iface_data), ctypes.byref(detail), ctypes.sizeof(detail),
                        ctypes.byref(req), None
                    ):
                        p = detail.DevicePath
                        needle = f"vid_{NINTENDO_VENDOR_ID:04x}&pid_{product_id:04x}&mi_01"
                        if needle in p.lower() and p not in paths:
                            paths.append(p)
            finally:
                setupapi.SetupDiDestroyDeviceInfoList(info_set)
    except Exception as e:
        logger.error("SetupAPI WinUSB search failed: %s", e)

    return paths


def send_winusb_startup_commands(product_id: int = SWITCH2_PRO_PID) -> bool:
    """Send the 5 critical startup commands to the controller's WinUSB vendor endpoint (MI_01)."""
    paths = get_winusb_interface_paths(product_id)
    if not paths:
        logger.debug("No WinUSB MI_01 paths found for product 0x%04X", product_id)
        return False

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    winusb = ctypes.WinDLL("winusb", use_last_error=True)

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080
    FILE_FLAG_OVERLAPPED = 0x40000000
    INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    winusb.WinUsb_Initialize.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.HANDLE)]
    winusb.WinUsb_Initialize.restype = wintypes.BOOL

    class USB_INTERFACE_DESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("bLength", ctypes.c_ubyte),
            ("bDescriptorType", ctypes.c_ubyte),
            ("bInterfaceNumber", ctypes.c_ubyte),
            ("bAlternateSetting", ctypes.c_ubyte),
            ("bNumEndpoints", ctypes.c_ubyte),
            ("bInterfaceClass", ctypes.c_ubyte),
            ("bInterfaceSubClass", ctypes.c_ubyte),
            ("bInterfaceProtocol", ctypes.c_ubyte),
            ("iInterface", ctypes.c_ubyte),
        ]

    class WINUSB_PIPE_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PipeType", ctypes.c_int),
            ("PipeId", ctypes.c_ubyte),
            ("MaximumPacketSize", ctypes.c_ushort),
            ("Interval", ctypes.c_ubyte),
        ]

    winusb.WinUsb_QueryInterfaceSettings.argtypes = [wintypes.HANDLE, ctypes.c_ubyte, ctypes.POINTER(USB_INTERFACE_DESCRIPTOR)]
    winusb.WinUsb_QueryInterfaceSettings.restype = wintypes.BOOL
    winusb.WinUsb_QueryPipe.argtypes = [wintypes.HANDLE, ctypes.c_ubyte, ctypes.c_ubyte, ctypes.POINTER(WINWIN_PIPE := WINUSB_PIPE_INFORMATION)]
    winusb.WinUsb_QueryPipe.restype = wintypes.BOOL

    winusb.WinUsb_WritePipe.argtypes = [
        wintypes.HANDLE, ctypes.c_ubyte, ctypes.POINTER(ctypes.c_ubyte), wintypes.ULONG,
        ctypes.POINTER(wintypes.ULONG), ctypes.c_void_p
    ]
    winusb.WinUsb_WritePipe.restype = wintypes.BOOL
    winusb.WinUsb_Free.argtypes = [wintypes.HANDLE]

    for p in paths:
        fh = kernel32.CreateFileW(
            p, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, None
        )
        if fh == INVALID_HANDLE_VALUE:
            logger.debug("WinUSB CreateFileW failed for %s", p)
            continue

        uh = wintypes.HANDLE()
        try:
            if not winusb.WinUsb_Initialize(fh, ctypes.byref(uh)):
                continue

            desc = USB_INTERFACE_DESCRIPTOR()
            winusb.WinUsb_QueryInterfaceSettings(uh, 0, ctypes.byref(desc))
            ep_out = 0x02  # Default Switch 2 bulk command out endpoint
            for i in range(desc.bNumEndpoints):
                pipe = WINUSB_PIPE_INFORMATION()
                if winusb.WinUsb_QueryPipe(uh, 0, i, ctypes.byref(pipe)):
                    if pipe.PipeType == 2 and (pipe.PipeId & 0x80) == 0:
                        ep_out = pipe.PipeId
                        break

            for cmd in USB_STARTUP_COMMANDS:
                buf = (ctypes.c_ubyte * len(cmd)).from_buffer_copy(cmd)
                w = wintypes.ULONG()
                winusb.WinUsb_WritePipe(uh, ep_out, buf, len(cmd), ctypes.byref(w), None)
                time.sleep(0.02)

            logger.info("WinUSB startup handshake delivered to %s successfully.", p)
            return True
        except Exception as e:
            logger.warning("WinUSB write failed: %s", e)
        finally:
            if uh:
                try:
                    winusb.WinUsb_Free(uh)
                except Exception:
                    pass
            kernel32.CloseHandle(fh)

    return False


def enumerate_usb_controllers() -> List[Dict[str, Any]]:
    """Enumerate connected Nintendo Switch 2 and Pro Controllers via USB HID."""
    found = []
    try:
        all_devs = hid.enumerate(NINTENDO_VENDOR_ID, 0)
        for dev in all_devs:
            pid = dev.get("product_id")
            if pid in SUPPORTED_PIDS:
                path = dev.get("path")
                path_str = path.decode("utf-8", errors="ignore") if isinstance(path, bytes) else str(path)
                interface_num = dev.get("interface_number", -1)

                found.append({
                    "path": path,
                    "path_str": path_str,
                    "vendor_id": NINTENDO_VENDOR_ID,
                    "product_id": pid,
                    "name": DEVICE_NAMES.get(pid, f"Nintendo Controller (0x{pid:04X})"),
                    "serial": dev.get("serial_number", ""),
                    "interface": interface_num,
                })
    except Exception as e:
        logger.error("Error enumerating USB HID controllers: %s", e)
    return found


class UsbProConnector:
    """Manages wired USB-C connection, handshake, and report streaming for Switch 2 Pro Controller."""

    def __init__(self,
                 on_state_change: Optional[Callable[[ControllerState], None]] = None,
                 on_disconnect: Optional[Callable[[], None]] = None):
        self.on_state_change = on_state_change
        self.on_disconnect = on_disconnect
        self.dev = None
        self.device_info: Optional[Dict[str, Any]] = None
        self.is_connected = False
        self.deadzone = 0.05
        self._stop_event = threading.Event()
        self._read_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def connect(self, device_info: Optional[Dict[str, Any]] = None) -> bool:
        """Connect to controller, perform the USB handshake, and start reading inputs."""
        with self._lock:
            if self.is_connected:
                return True

            if device_info is None:
                controllers = enumerate_usb_controllers()
                if not controllers:
                    logger.warning("No wired Switch 2 Pro Controller found on USB.")
                    return False
                device_info = controllers[0]

            self.device_info = device_info
            path = device_info["path"]
            product_id = device_info.get("product_id", SWITCH2_PRO_PID)

            # 1. Send the essential Switch 2 WinUSB startup commands (MI_01)
            winusb_ok = send_winusb_startup_commands(product_id)
            if not winusb_ok:
                logger.debug("WinUSB route not available; will attempt HID output fallback.")

            # 2. Open the HID device handle for reading input reports (MI_00)
            try:
                self.dev = hid.device()
                self.dev.open_path(path)
                logger.info("Opened USB device: %s at %s", device_info["name"], device_info["path_str"])
            except Exception as e:
                logger.error("Failed to open USB device handle: %s", e)
                self.dev = None
                return False

            # If WinUSB was not available, try writing commands via HID output reports as fallback
            if not winusb_ok:
                self._send_hid_handshake_fallback()

            self._stop_event.clear()
            self.is_connected = True
            self._read_thread = threading.Thread(target=self._read_loop, name="UsbProReadLoop", daemon=True)
            self._read_thread.start()
            logger.info("USB Pro Controller read loop started successfully.")
            return True

    def _send_hid_handshake_fallback(self) -> bool:
        """Fallback handshake over HID interface 0."""
        if not self.dev:
            return False
        for cmd in USB_STARTUP_COMMANDS:
            try:
                report = bytes([OUTPUT_REPORT_ID_PRO2]) + bytes(cmd).ljust(PRO2_OUTPUT_REPORT_BODY_SIZE, b"\x00")
                self.dev.write(report)
                time.sleep(0.02)
            except Exception:
                pass
        return True

    def _read_loop(self):
        """Continuous high-frequency input reading thread."""
        logger.info("USB read loop active.")
        consecutive_errors = 0

        # Boost thread priority on Windows for lowest possible scheduling latency
        try:
            THREAD_PRIORITY_HIGHEST = 2
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(),
                THREAD_PRIORITY_HIGHEST,
            )
        except Exception:
            pass

        while not self._stop_event.is_set() and self.dev is not None:
            try:
                data = self.dev.read(USB_INPUT_REPORT_SIZE, timeout_ms=20)
                if not data:
                    continue

                consecutive_errors = 0
                state = parse_input_report(bytes(data), deadzone=self.deadzone, has_report_id=True)
                if state and self.on_state_change:
                    self.on_state_change(state)

            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    logger.info("USB controller disconnected or read failed: %s", e)
                    break
                time.sleep(0.01)

        self._cleanup()
        if self.on_disconnect:
            try:
                self.on_disconnect()
            except Exception:
                pass

    def _cleanup(self):
        """Internal handle cleanup."""
        self.is_connected = False
        if self.dev is not None:
            try:
                self.dev.close()
            except Exception:
                pass
            self.dev = None

    def disconnect(self):
        """Gracefully disconnect and close the USB handle."""
        self._stop_event.set()
        with self._lock:
            if self._read_thread and self._read_thread.is_alive():
                self._read_thread.join(timeout=1.0)
            self._cleanup()
            logger.info("USB Pro Controller disconnected cleanly.")
