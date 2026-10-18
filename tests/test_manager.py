"""Unit tests for ControllerManager and layout mapping."""

import pytest
from app_core import ControllerManager
from report_parser import ControllerState
from constants import (
    NINTENDO_VENDOR_ID,
    SWITCH2_PRO_PID,
    SWITCH1_PRO_PID,
    NSO_GAMECUBE_PID,
    USB_STARTUP_COMMANDS,
)


def test_constants_definitions():
    assert NINTENDO_VENDOR_ID == 0x057E
    assert SWITCH2_PRO_PID == 0x2069
    assert SWITCH1_PRO_PID == 0x2009
    assert NSO_GAMECUBE_PID == 0x2073
    assert len(USB_STARTUP_COMMANDS) == 5


def test_manager_defaults():
    manager = ControllerManager()
    assert manager.is_connected is False
    assert manager.auto_connect_enabled is True
    assert manager.button_layout == "nintendo"
    assert manager.deadzone == 0.05


def test_manager_layout_switch():
    manager = ControllerManager()
    manager.set_button_layout("xbox")
    assert manager.button_layout == "xbox"
    assert manager.virtual_gamepad.layout == "xbox"

    manager.set_button_layout("nintendo")
    assert manager.button_layout == "nintendo"
    assert manager.virtual_gamepad.layout == "nintendo"


def test_manager_deadzone():
    manager = ControllerManager()
    manager.set_deadzone(0.12)
    assert manager.deadzone == 0.12
    assert manager.usb_connector.deadzone == 0.12
    assert manager.ble_connector.deadzone == 0.12


def test_manager_callbacks():
    manager = ControllerManager()
    captured = {}

    def on_conn(connected, transport, name, batt):
        captured["connected"] = connected
        captured["transport"] = transport
        captured["name"] = name
        captured["batt"] = batt

    manager.on_connection_changed = on_conn
    manager._notify_connection(True, "USB-C", "Switch 2 Pro Controller", 100)

    assert captured["connected"] is True
    assert captured["transport"] == "USB-C"
    assert captured["name"] == "Switch 2 Pro Controller"
    assert captured["batt"] == 100


def test_manager_gl_gr_mapping():
    manager = ControllerManager()
    assert manager.virtual_gamepad.gl_mapping == "L3"
    assert manager.virtual_gamepad.gr_mapping == "R3"

    manager.set_gl_mapping("LB")
    manager.set_gr_mapping("RB")
    assert manager.virtual_gamepad.gl_mapping == "LB"
    assert manager.virtual_gamepad.gr_mapping == "RB"

    manager.set_gl_mapping("Disabled")
    manager.set_gr_mapping("RT")
    assert manager.virtual_gamepad.gl_mapping == "Disabled"
    assert manager.virtual_gamepad.gr_mapping == "RT"


class MockBtn:
    XUSB_GAMEPAD_LEFT_THUMB = 1
    XUSB_GAMEPAD_RIGHT_THUMB = 2
    XUSB_GAMEPAD_LEFT_SHOULDER = 4
    XUSB_GAMEPAD_RIGHT_SHOULDER = 8
    XUSB_GAMEPAD_A = 16
    XUSB_GAMEPAD_B = 32
    XUSB_GAMEPAD_X = 64
    XUSB_GAMEPAD_Y = 128


class MockGamepad:
    def __init__(self):
        self.pressed = []
        self.left_trigger = 0.0
        self.right_trigger = 0.0

    def press_button(self, btn):
        self.pressed.append(btn)

    def left_trigger_float(self, value_float):
        self.left_trigger = value_float

    def right_trigger_float(self, value_float):
        self.right_trigger = value_float


def test_virtual_gamepad_apply_mapped_actions():
    manager = ControllerManager()
    vg = manager.virtual_gamepad
    mock_btn = MockBtn()

    # Test L3 & R3
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "L3", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_LEFT_THUMB]

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "R3", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_RIGHT_THUMB]

    # Test LB & RB
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "LB", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_LEFT_SHOULDER]

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "RB", mock_btn)
    assert gp.pressed == [MockBtn.XUSB_GAMEPAD_RIGHT_SHOULDER]

    # Test Face buttons
    for code, attr in [("A", "XUSB_GAMEPAD_A"), ("B", "XUSB_GAMEPAD_B"), ("X", "XUSB_GAMEPAD_X"), ("Y", "XUSB_GAMEPAD_Y")]:
        gp = MockGamepad()
        vg._apply_mapped_action(gp, code, mock_btn)
        assert gp.pressed == [getattr(MockBtn, attr)]

    # Test LT & RT triggers
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "LT", mock_btn)
    assert gp.left_trigger == 1.0

    gp = MockGamepad()
    vg._apply_mapped_action(gp, "RT", mock_btn)
    assert gp.right_trigger == 1.0

    # Test Disabled
    gp = MockGamepad()
    vg._apply_mapped_action(gp, "Disabled", mock_btn)
    assert gp.pressed == []
    assert gp.left_trigger == 0.0
    assert gp.right_trigger == 0.0


