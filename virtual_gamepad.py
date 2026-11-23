"""Polymorphic virtual gamepad abstraction for Switch 2 Pro Controller Windows Bridge."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

from report_parser import ControllerState

logger = logging.getLogger(__name__)


class BaseVirtualController(ABC):
    """Abstract base class defining the contract for virtual controller backends."""

    def __init__(self):
        self._lock = threading.Lock()
        self.is_connected: bool = False
        self.layout: str = "nintendo"  # "nintendo" (physical position) or "xbox" (letter match)
        self.gl_mapping: str = "L3"    # Grip Left paddle mapping
        self.gr_mapping: str = "R3"    # Grip Right paddle mapping
        self._on_rumble: Optional[Callable[[int, int], None]] = None
        self._notif_cb = None
        self._notif_registered: bool = False

    @property
    def on_rumble(self) -> Optional[Callable[[int, int], None]]:
        """Optional force-feedback callback receiving (large_motor, small_motor)."""
        return self._on_rumble

    @on_rumble.setter
    def on_rumble(self, cb: Optional[Callable[[int, int], None]]) -> None:
        self._on_rumble = cb
        if cb is not None and not self._notif_registered:
            gp = getattr(self, "_gamepad", None)
            if gp is not None:
                self._register_rumble_notification(gp)

    def _register_rumble_notification(self, gamepad) -> None:
        """Attach ViGEmBus force-feedback notification callback if on_rumble is configured."""
        if self._on_rumble is None or gamepad is None:
            return
        try:
            def _on_vigem_notif(client, target, large_motor, small_motor, led_number, user_data):
                cb = self._on_rumble
                if cb is not None:
                    try:
                        cb(int(large_motor), int(small_motor))
                    except Exception:
                        pass

            self._notif_cb = _on_vigem_notif
            if hasattr(gamepad, "register_notification"):
                gamepad.register_notification(self._notif_cb)
                self._notif_registered = True
        except Exception as notif_err:
            logger.debug("ViGEm rumble notification registration notice: %s", notif_err)

    def _unregister_rumble_notification(self, gamepad) -> None:
        """Detach ViGEmBus force-feedback notification callback cleanly."""
        if self._notif_registered and gamepad is not None:
            try:
                if hasattr(gamepad, "unregister_notification"):
                    gamepad.unregister_notification()
            except Exception:
                pass
        self._notif_registered = False
        self._notif_cb = None

    @abstractmethod
    def connect(self) -> bool:
        """Create and initialize the virtual controller backend.

        Returns:
            True if connected / spawned successfully, False otherwise.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Destroy the virtual controller backend and release resources."""
        pass

    @abstractmethod
    def update_from_state(self, state: ControllerState, layout: Optional[str] = None) -> None:
        """Update virtual gamepad inputs from ControllerState.

        Args:
            state: The ControllerState containing buttons, sticks, triggers, and IMU data.
            layout: Optional layout override ('nintendo' or 'xbox'). If None, uses self.layout.
        """
        pass

    def update(self, state: ControllerState) -> None:
        """Convenience alias for update_from_state."""
        self.update_from_state(state)


# Lazy module-level imports via PEP 562 to prevent circular dependencies
__all__ = ["BaseVirtualController", "VirtualXboxController", "VirtualDS4Controller"]


def __getattr__(name: str):
    if name == "VirtualXboxController":
        from virtual_xinput import VirtualXboxController
        return VirtualXboxController
    if name == "VirtualDS4Controller":
        from virtual_ds4 import VirtualDS4Controller
        return VirtualDS4Controller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
