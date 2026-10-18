"""Virtual Xbox 360 controller emulation via ViGEmBus and vgamepad."""

import os
import sys
import ctypes
import logging
import threading
from typing import Optional
from report_parser import ControllerState

# Enable 1ms high-resolution timer in Windows kernel for ultra-low latency
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

logger = logging.getLogger(__name__)

# Track ViGEmBus driver availability
_VIGEM_INITIALIZED = False
_vg_module = None



def check_vigem_driver_available() -> bool:
    """Check if ViGEmBus driver is installed and operational."""
    global _vg_module, _VIGEM_INITIALIZED
    try:
        import vgamepad as vg
        # Attempt to create and immediately destroy a dummy controller
        dummy = vg.VX360Gamepad()
        del dummy
        _vg_module = vg
        _VIGEM_INITIALIZED = True
        return True
    except Exception as e:
        logger.debug("ViGEmBus driver not ready: %s", e)
        _VIGEM_INITIALIZED = False
        return False


def install_vigem_driver() -> bool:
    """Trigger the elevated installer for ViGEmBus driver."""
    candidates = [
        os.path.join(getattr(sys, "_MEIPASS", ""), "drivers", "ViGEmBusSetup_x64.msi"),
        os.path.join(os.path.dirname(sys.executable), "drivers", "ViGEmBusSetup_x64.msi"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "drivers", "ViGEmBusSetup_x64.msi"),
    ]
    msi_path = next((p for p in candidates if os.path.isfile(p)), None)
    
    if not msi_path:
        logger.error("Driver installer not found in candidates: %s", candidates)
        return False
        
    try:
        # Launch elevated msiexec installer with UAC prompt
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "msiexec.exe",
            f'/i "{msi_path}" /passive /norestart',
            None,
            1,
        )
        return result > 32
    except Exception as e:
        logger.error("Failed to run elevated driver installer: %s", e)
        return False


class VirtualXboxController:
    """Manages an emulated Xbox 360 controller for seamless Windows game compatibility."""

    def __init__(self):
        self._gamepad = None
        self._lock = threading.Lock()
        self.is_connected = False
        self.layout = "nintendo"  # "nintendo" (physical position) or "xbox" (letter match)
        self.gl_mapping = "L3"   # Grip Left: "L3", "LB", "A", "X", "LT"
        self.gr_mapping = "R3"   # Grip Right: "R3", "RB", "B", "Y", "RT"


    def connect(self) -> bool:
        """Create virtual Xbox 360 gamepad."""
        with self._lock:
            if self._gamepad is not None:
                return True
            try:
                import vgamepad as vg
                self._gamepad = vg.VX360Gamepad()
                self.is_connected = True
                logger.info("Virtual Xbox 360 controller spawned successfully.")
                return True
            except Exception as e:
                logger.error("Failed to spawn virtual Xbox 360 controller: %s", e)
                self.is_connected = False
                return False

    def disconnect(self):
        """Destroy virtual Xbox 360 gamepad and reset inputs."""
        with self._lock:
            if self._gamepad is not None:
                try:
                    self._gamepad.reset()
                    self._gamepad.update()
                except Exception:
                    pass
                self._gamepad = None
            self.is_connected = False
            logger.info("Virtual Xbox 360 controller disconnected.")

    def update_from_state(self, state: ControllerState, layout: Optional[str] = None):
        """Update virtual Xbox 360 inputs from ControllerState."""
        with self._lock:
            if self._gamepad is None:
                return

            vg = _vg_module or sys.modules.get("vgamepad")
            if vg is None:
                return

            active_layout = layout or self.layout
            gp = self._gamepad
            gp.reset()

            btn = vg.XUSB_BUTTON

            # Button Layout Mapping
            if active_layout == "nintendo":
                # Nintendo Physical Position matching Xbox:
                # Switch B (Bottom) -> Xbox A
                # Switch A (Right)  -> Xbox B
                # Switch Y (Left)   -> Xbox X
                # Switch X (Top)    -> Xbox Y
                if state.b: gp.press_button(btn.XUSB_GAMEPAD_A)
                if state.a: gp.press_button(btn.XUSB_GAMEPAD_B)
                if state.y: gp.press_button(btn.XUSB_GAMEPAD_X)
                if state.x: gp.press_button(btn.XUSB_GAMEPAD_Y)
            else:
                # Literal Letter Match (A->A, B->B, X->X, Y->Y)
                if state.a: gp.press_button(btn.XUSB_GAMEPAD_A)
                if state.b: gp.press_button(btn.XUSB_GAMEPAD_B)
                if state.x: gp.press_button(btn.XUSB_GAMEPAD_X)
                if state.y: gp.press_button(btn.XUSB_GAMEPAD_Y)

            # Bumpers and Center Buttons
            if state.l: gp.press_button(btn.XUSB_GAMEPAD_LEFT_SHOULDER)
            if state.r: gp.press_button(btn.XUSB_GAMEPAD_RIGHT_SHOULDER)
            if state.plus: gp.press_button(btn.XUSB_GAMEPAD_START)
            if state.minus: gp.press_button(btn.XUSB_GAMEPAD_BACK)
            if state.home: gp.press_button(btn.XUSB_GAMEPAD_GUIDE)

            # Stick Clicks
            if state.l_thumb: gp.press_button(btn.XUSB_GAMEPAD_LEFT_THUMB)
            if state.r_thumb: gp.press_button(btn.XUSB_GAMEPAD_RIGHT_THUMB)

            # D-Pad
            if state.dpad_up: gp.press_button(btn.XUSB_GAMEPAD_DPAD_UP)
            if state.dpad_down: gp.press_button(btn.XUSB_GAMEPAD_DPAD_DOWN)
            if state.dpad_left: gp.press_button(btn.XUSB_GAMEPAD_DPAD_LEFT)
            if state.dpad_right: gp.press_button(btn.XUSB_GAMEPAD_DPAD_RIGHT)

            # Analog Triggers (0.0 to 1.0)
            gp.left_trigger_float(value_float=state.zl_analog)
            gp.right_trigger_float(value_float=state.zr_analog)

            # Back Grip Buttons (GL / GR)
            if state.gl:
                self._apply_mapped_action(gp, self.gl_mapping, btn)
            if state.gr:
                self._apply_mapped_action(gp, self.gr_mapping, btn)

            # Analog Sticks (-1.0 to 1.0)
            gp.left_joystick_float(x_value_float=state.left_stick_x, y_value_float=state.left_stick_y)
            gp.right_joystick_float(x_value_float=state.right_stick_x, y_value_float=state.right_stick_y)

            # Commit the frame to ViGEmBus
            gp.update()

    def _apply_mapped_action(self, gp, mapping: str, btn):
        """Map GL/GR paddle to configured Xbox 360 action."""
        if mapping == "L3":
            gp.press_button(btn.XUSB_GAMEPAD_LEFT_THUMB)
        elif mapping == "R3":
            gp.press_button(btn.XUSB_GAMEPAD_RIGHT_THUMB)
        elif mapping == "LB":
            gp.press_button(btn.XUSB_GAMEPAD_LEFT_SHOULDER)
        elif mapping == "RB":
            gp.press_button(btn.XUSB_GAMEPAD_RIGHT_SHOULDER)
        elif mapping == "A":
            gp.press_button(btn.XUSB_GAMEPAD_A)
        elif mapping == "B":
            gp.press_button(btn.XUSB_GAMEPAD_B)
        elif mapping == "X":
            gp.press_button(btn.XUSB_GAMEPAD_X)
        elif mapping == "Y":
            gp.press_button(btn.XUSB_GAMEPAD_Y)
        elif mapping == "LT":
            gp.left_trigger_float(value_float=1.0)
        elif mapping == "RT":
            gp.right_trigger_float(value_float=1.0)

