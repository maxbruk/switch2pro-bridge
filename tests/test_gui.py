"""Comprehensive Automated Test Suite for Switch 2 Pro Controller GUI & Settings (Milestone 5).

Covers:
- Suite 1: Settings Persistence (Schema, file I/O, fallback, error recovery, AppData) — 16 Tests
- Suite 2: GUI Controls & Wiring (Mode segmented button, DSU switch, persistence hooks) — 10 Tests
- Suite 3: Connection State & Lifecycle (USB/BLE state, status pills, shutdown cleanup) — 6 Tests
- Suite 4: Visualizer Telemetry & Rate Limiting (250Hz atomic store, deadband dirty-checking, status logic) — 10 Tests
- Suite 5: Robustness, Fault Tolerance & Concurrency (Port conflicts, corrupted state, rapid clicks) — 6 Tests
Total: 48 Tests
"""

import os
import sys
import json
import time
import threading
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from report_parser import ControllerState
from app_core import ControllerManager
from gui import (
    DEFAULT_SETTINGS,
    Switch2ProApp,
    _ensure_tcl_tk_env,
    get_settings_path,
    load_settings,
    save_settings,
    determine_imu_status,
    format_imu_telemetry,
)


# ============================================================================
# Check Tkinter Availability for Headless Execution
# ============================================================================

_ensure_tcl_tk_env()
TK_AVAILABLE = False
for _attempt in range(2):
    try:
        import tkinter as tk
        _probe = tk.Tk()
        _probe.withdraw()
        _probe.destroy()
        TK_AVAILABLE = True
        break
    except Exception:
        time.sleep(0.05)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(autouse=True, scope="module")
def configure_tcl_tk_env():
    """Ensure TCL_LIBRARY and TK_LIBRARY resolve cleanly from base Python install."""
    base_dir = getattr(sys, "base_prefix", sys.prefix)
    tcl_dir = os.path.join(base_dir, "tcl", "tcl8.6")
    tk_dir = os.path.join(base_dir, "tcl", "tk8.6")
    if not os.path.isdir(tcl_dir):
        tcl_dir = os.path.join(sys.prefix, "tcl", "tcl8.6")
    if not os.path.isdir(tk_dir):
        tk_dir = os.path.join(sys.prefix, "tcl", "tk8.6")

    old_tcl = os.environ.get("TCL_LIBRARY")
    old_tk = os.environ.get("TK_LIBRARY")
    if os.path.isdir(tcl_dir):
        os.environ["TCL_LIBRARY"] = tcl_dir
    if os.path.isdir(tk_dir):
        os.environ["TK_LIBRARY"] = tk_dir
    yield
    if old_tcl is not None:
        os.environ["TCL_LIBRARY"] = old_tcl
    else:
        os.environ.pop("TCL_LIBRARY", None)
    if old_tk is not None:
        os.environ["TK_LIBRARY"] = old_tk
    else:
        os.environ.pop("TK_LIBRARY", None)


@pytest.fixture
def headless_app(tmp_path, monkeypatch):
    """Provides an isolated headless Switch2ProApp instance for UI testing."""
    if not TK_AVAILABLE:
        pytest.skip("Tkinter display context not available")

    # Ensure Tcl/Tk can locate initialization scripts in python installation
    for base in (sys.base_prefix, sys.prefix):
        tcl_dir = os.path.join(base, "tcl", "tcl8.6")
        tk_dir = os.path.join(base, "tcl", "tk8.6")
        if os.path.isdir(tcl_dir):
            monkeypatch.setenv("TCL_LIBRARY", tcl_dir)
        if os.path.isdir(tk_dir):
            monkeypatch.setenv("TK_LIBRARY", tk_dir)

    settings_file = str(tmp_path / "settings.json")
    monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
    monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
    monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

    app = Switch2ProApp(settings_path=settings_file, auto_start_detector=False)
    app.withdraw()
    app.update_idletasks()

    try:
        yield app
    finally:
        try:
            app.destroy()
        except Exception:
            pass


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
# SUITE 1: Settings Persistence (Pure Python, 16 Tests)
# ============================================================================

class TestSettingsPersistence:
    """Suite 1: Pure logic tests for configuration schema, file I/O, and error recovery."""

    def test_default_settings_schema(self):
        """Test 1: Verify exact schema keys and default values in DEFAULT_SETTINGS."""
        expected_keys = {
            "layout", "deadzone", "auto_connect", "gl_mapping",
            "gr_mapping", "output_mode", "dsu_enabled"
        }
        assert set(DEFAULT_SETTINGS.keys()) == expected_keys
        assert DEFAULT_SETTINGS["output_mode"] == "xbox"
        assert DEFAULT_SETTINGS["dsu_enabled"] is False
        assert DEFAULT_SETTINGS["deadzone"] == 0.05
        assert DEFAULT_SETTINGS["layout"] == "nintendo"
        assert DEFAULT_SETTINGS["auto_connect"] is True
        assert DEFAULT_SETTINGS["gl_mapping"] == "L3"
        assert DEFAULT_SETTINGS["gr_mapping"] == "R3"

    def test_get_settings_path_respects_appdata(self, tmp_path):
        """Test 2: get_settings_path constructs path inside provided appdata directory."""
        path = get_settings_path(appdata_dir=str(tmp_path))
        expected = os.path.join(str(tmp_path), "Switch2ProConnect", "settings.json")
        assert path == expected

    def test_get_settings_path_creates_missing_directory(self, tmp_path):
        """Test 3: get_settings_path automatically creates parent directory if absent."""
        custom_dir = str(tmp_path / "custom_appdata")
        path = get_settings_path(appdata_dir=custom_dir)
        parent_dir = os.path.dirname(path)
        assert os.path.isdir(parent_dir)

    def test_get_settings_path_fallback_when_appdata_none(self, monkeypatch):
        """Test 4: Falls back to script directory if APPDATA is unset."""
        monkeypatch.delenv("APPDATA", raising=False)
        path = get_settings_path(appdata_dir=None)
        assert os.path.basename(path) == "settings.json"
        assert os.path.isabs(path)

    def test_load_settings_file_not_found_returns_defaults(self, tmp_path):
        """Test 5: Loading from non-existent file returns default settings."""
        missing_file = str(tmp_path / "does_not_exist.json")
        loaded = load_settings(missing_file)
        assert loaded == DEFAULT_SETTINGS

    def test_save_and_load_roundtrip(self, tmp_path):
        """Test 6: Saving custom settings and reloading them returns exact values."""
        cfg_file = str(tmp_path / "custom_settings.json")
        custom = {
            "output_mode": "ds4",
            "dsu_enabled": True,
            "layout": "xbox",
            "deadzone": 0.12,
            "auto_connect": False,
            "gl_mapping": "LB",
            "gr_mapping": "RB",
        }
        saved = save_settings(custom, cfg_file)
        assert saved is True
        assert os.path.isfile(cfg_file)

        loaded = load_settings(cfg_file)
        for key, val in custom.items():
            assert loaded[key] == val

    def test_save_creates_parent_directory_if_missing(self, tmp_path):
        """Test 7: save_settings creates intermediate parent directories."""
        nested_file = str(tmp_path / "dir1" / "dir2" / "settings.json")
        saved = save_settings({"output_mode": "ds4"}, nested_file)
        assert saved is True
        assert os.path.isfile(nested_file)

    def test_corrupted_json_syntax_returns_defaults(self, tmp_path):
        """Test 8: Syntax errors in settings JSON recover cleanly with defaults."""
        corrupt_file = str(tmp_path / "corrupt.json")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write("{ invalid_json: [ missing_bracket")

        loaded = load_settings(corrupt_file)
        assert loaded == DEFAULT_SETTINGS

    def test_corrupted_json_binary_garbage_returns_defaults(self, tmp_path):
        """Test 9: Binary garbage bytes in settings JSON recover cleanly."""
        garbage_file = str(tmp_path / "garbage.json")
        with open(garbage_file, "wb") as f:
            f.write(b"\x00\xff\xfe\x12\x34\xaa\xbb\xcc")

        loaded = load_settings(garbage_file)
        assert loaded == DEFAULT_SETTINGS

    def test_empty_settings_file_returns_defaults(self, tmp_path):
        """Test 10: Zero-byte empty file recovers cleanly with defaults."""
        empty_file = str(tmp_path / "empty.json")
        with open(empty_file, "w", encoding="utf-8") as f:
            f.write("")

        loaded = load_settings(empty_file)
        assert loaded == DEFAULT_SETTINGS

    def test_partial_json_missing_keys_uses_defaults(self, tmp_path):
        """Test 11: JSON containing only partial keys populates missing ones with defaults."""
        partial_file = str(tmp_path / "partial.json")
        with open(partial_file, "w", encoding="utf-8") as f:
            json.dump({"output_mode": "ds4"}, f)

        loaded = load_settings(partial_file)
        assert loaded["output_mode"] == "ds4"
        assert loaded["dsu_enabled"] is False
        assert loaded["deadzone"] == 0.05
        assert loaded["layout"] == "nintendo"

    def test_backward_compatible_aliases(self, tmp_path):
        """Test 12: Backward-compatible aliases (dsu_server, button_layout) are mapped."""
        alias_file = str(tmp_path / "alias.json")
        with open(alias_file, "w", encoding="utf-8") as f:
            json.dump({"dsu_server": True, "button_layout": "xbox"}, f)

        loaded = load_settings(alias_file)
        assert loaded["dsu_enabled"] is True
        assert loaded["layout"] == "xbox"

    def test_invalid_output_mode_recovers_to_xbox(self, tmp_path):
        """Test 13: Invalid output_mode strings or types normalize to xbox."""
        for invalid_val in ["playstation", 1234, None, "", {}]:
            bad_file = str(tmp_path / "bad_mode.json")
            with open(bad_file, "w", encoding="utf-8") as f:
                json.dump({"output_mode": invalid_val}, f)

            loaded = load_settings(bad_file)
            assert loaded["output_mode"] == "xbox"

    def test_invalid_dsu_enabled_type_recovers_to_false(self, tmp_path):
        """Test 14: Non-boolean dsu_enabled values normalize to False."""
        for invalid_val in ["yes", [1, 2], None, 100]:
            bad_file = str(tmp_path / "bad_dsu.json")
            with open(bad_file, "w", encoding="utf-8") as f:
                json.dump({"dsu_enabled": invalid_val}, f)

            loaded = load_settings(bad_file)
            assert loaded["dsu_enabled"] is False

    def test_invalid_deadzone_type_and_range_recovers(self, tmp_path):
        """Test 15: Out-of-bounds or non-float deadzone values recover to 0.05."""
        for invalid_val in ["string", -0.5, 1.5, None, []]:
            bad_file = str(tmp_path / "bad_dz.json")
            with open(bad_file, "w", encoding="utf-8") as f:
                json.dump({"deadzone": invalid_val}, f)

            loaded = load_settings(bad_file)
            assert loaded["deadzone"] == 0.05

    def test_save_settings_handles_os_error_gracefully(self, tmp_path, monkeypatch):
        """Test 16: OSError during file writing returns False without raising."""
        cfg_file = str(tmp_path / "readonly.json")
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=OSError("Disk full")))
        result = save_settings({"output_mode": "xbox"}, cfg_file)
        assert result is False


# ============================================================================
# SUITE 2: GUI Controls & Wiring (10 Tests)
# ============================================================================

class TestGuiWiring:
    """Suite 2: Headless integration tests for widget creation, callbacks, and manager syncing."""

    def test_app_init_headless_creates_m5_widgets(self, headless_app):
        """Test 17: All Milestone 5 widgets and controls are instantiated."""
        app = headless_app
        assert hasattr(app, "output_mode_selector")
        assert hasattr(app, "mode_selector")
        assert hasattr(app, "switch_dsu")
        assert hasattr(app, "lbl_dsu_status")
        assert hasattr(app, "imu_status_pill")
        assert hasattr(app, "lbl_imu_status")
        assert hasattr(app, "lbl_accel")
        assert hasattr(app, "lbl_imu_accel")
        assert hasattr(app, "lbl_gyro")
        assert hasattr(app, "lbl_imu_gyro")
        assert hasattr(app, "lbl_emulation")

    def test_app_loads_saved_output_mode_ds4(self, tmp_path, monkeypatch):
        """Test 18: App initializes with output_mode ds4 when configured in settings."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")

        settings_file = str(tmp_path / "settings_ds4.json")
        save_settings({"output_mode": "ds4"}, settings_file)

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=settings_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "ds4"
            assert app.output_mode_selector.get() == "DualShock 4 (Motion)"
            assert app.lbl_emulation.cget("text") == "DualShock 4"
        finally:
            app.destroy()

    def test_app_loads_saved_dsu_enabled_true(self, tmp_path, monkeypatch):
        """Test 19: App initializes with DSU server enabled when saved as True."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")

        settings_file = str(tmp_path / "settings_dsu.json")
        save_settings({"dsu_enabled": True}, settings_file)

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=settings_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.dsu_enabled is True
            assert app.switch_dsu.get() == 1
            assert "Active" in app.lbl_dsu_status.cget("text")
        finally:
            app.destroy()

    def test_mode_selector_callback_switches_to_ds4_and_saves(self, headless_app):
        """Test 20: Selecting DualShock 4 switches manager mode and writes to disk."""
        app = headless_app
        app._on_output_mode_changed("DualShock 4 (Motion)")
        assert app.manager.output_mode == "ds4"
        assert app.lbl_emulation.cget("text") == "DualShock 4"

        loaded = load_settings(app._get_settings_path())
        assert loaded["output_mode"] == "ds4"

    def test_mode_selector_callback_switches_to_xbox_and_saves(self, headless_app):
        """Test 21: Selecting Xbox 360 switches manager mode and writes to disk."""
        app = headless_app
        app._on_output_mode_changed("DualShock 4 (Motion)")
        assert app.manager.output_mode == "ds4"

        app._on_output_mode_changed("Xbox 360")
        assert app.manager.output_mode == "xbox"
        assert app.lbl_emulation.cget("text") == "Xbox 360"

        loaded = load_settings(app._get_settings_path())
        assert loaded["output_mode"] == "xbox"

    def test_mode_selector_fuzzy_normalization(self, headless_app):
        """Test 22: Fuzzy input string normalization for mode selector."""
        app = headless_app
        for ds4_val in ["ds4", "DS4", "DualShock 4", "DualShock 4 (Motion)"]:
            app._on_output_mode_changed(ds4_val)
            assert app.manager.output_mode == "ds4"

        for xbox_val in ["xbox", "XBOX", "Xbox 360", "xinput"]:
            app._on_output_mode_changed(xbox_val)
            assert app.manager.output_mode == "xbox"

    def test_dsu_switch_toggle_on_starts_dsu_and_saves(self, headless_app):
        """Test 23: Toggling DSU switch ON enables manager DSU and saves settings."""
        app = headless_app
        app.switch_dsu.select()
        app._on_dsu_switch_changed()
        assert app.manager.dsu_enabled is True
        assert "Active" in app.lbl_dsu_status.cget("text")

        loaded = load_settings(app._get_settings_path())
        assert loaded["dsu_enabled"] is True

    def test_dsu_switch_toggle_off_stops_dsu_and_saves(self, headless_app):
        """Test 24: Toggling DSU switch OFF disables manager DSU and saves settings."""
        app = headless_app
        app.switch_dsu.select()
        app._on_dsu_switch_changed()
        assert app.manager.dsu_enabled is True

        app.switch_dsu.deselect()
        app._on_dsu_switch_changed()
        assert app.manager.dsu_enabled is False
        assert "Inactive" in app.lbl_dsu_status.cget("text")

        loaded = load_settings(app._get_settings_path())
        assert loaded["dsu_enabled"] is False

    def test_existing_controls_save_settings_intact(self, headless_app):
        """Test 25: Existing layout/deadzone/paddle changes persist with M5 keys."""
        app = headless_app
        app._on_layout_changed("Xbox (Letters)")
        app._on_deadzone_changed(0.15)
        app._on_gl_mapping_changed("LB (Left Bumper)")

        loaded = load_settings(app._get_settings_path())
        assert loaded["layout"] == "xbox"
        assert abs(loaded["deadzone"] - 0.15) < 1e-4
        assert loaded["gl_mapping"] == "LB"
        assert "output_mode" in loaded
        assert "dsu_enabled" in loaded

    def test_hot_switching_mode_while_connected(self, headless_app):
        """Test 26: Hot-switching output mode while connected swaps controllers safely."""
        app = headless_app
        app.is_connected = True
        app._on_output_mode_changed("DualShock 4 (Motion)")
        assert app.manager.output_mode == "ds4"

        app._on_output_mode_changed("Xbox 360")
        assert app.manager.output_mode == "xbox"


# ============================================================================
# SUITE 3: Connection State & Lifecycle (6 Tests)
# ============================================================================

class TestConnectionUIAndStatus:
    """Suite 3: Headless verification of transport updates, status pills, and shutdown."""

    def test_connection_changed_connected_usb(self, headless_app):
        """Test 27: Connecting via USB-C updates UI elements and badges."""
        app = headless_app
        app._update_connection_ui(True, "USB-C", "Switch 2 Pro Controller", 88)
        assert app.is_connected is True
        assert "DISCONNECT" in app.btn_main.cget("text")
        assert "CONNECTED (USB-C)" in app.status_pill.cget("text")
        assert app.lbl_battery.cget("text") == "88%"
        assert app.lbl_device.cget("text") == "Switch 2 Pro Controller"

    def test_connection_changed_connected_bluetooth(self, headless_app):
        """Test 28: Connecting via Bluetooth updates UI with Bluetooth tag."""
        app = headless_app
        app._update_connection_ui(True, "Bluetooth LE", "Switch 2 Pro Controller", 72)
        assert app.is_connected is True
        assert "Bluetooth" in app.status_pill.cget("text")
        assert app.lbl_battery.cget("text") == "72%"

    def test_connection_changed_disconnected_resets_ui(self, headless_app):
        """Test 29: Disconnecting restores default UI and clears badges."""
        app = headless_app
        app._update_connection_ui(True, "USB-C", "Switch 2 Pro Controller", 100)
        assert app.is_connected is True

        app._update_connection_ui(False, "None", "Not Connected", 0)
        assert app.is_connected is False
        assert "CONNECT CONTROLLER" in app.btn_main.cget("text")
        assert "DISCONNECTED" in app.status_pill.cget("text")
        assert app.lbl_battery.cget("text") == "-- %"
        assert "IMU N/A" in app.imu_status_pill.cget("text")

    def test_thread_safe_callback_marshaling(self, headless_app):
        """Test 30: Thread-safe callback correctly marshals to Tkinter event loop."""
        app = headless_app
        after_mock = MagicMock()
        app.after = after_mock

        t = threading.Thread(
            target=app._on_connection_changed,
            args=(True, "USB-C", "Switch 2 Pro Controller", 95)
        )
        t.start()
        t.join()

        after_mock.assert_called_with(0, app._update_connection_ui, True, "USB-C", "Switch 2 Pro Controller", 95)

    def test_status_message_marshaling(self, headless_app):
        """Test 31: Status messages from background threads marshal via app.after."""
        app = headless_app
        after_mock = MagicMock()
        app.after = after_mock

        app._on_status_message("Connection established.")
        assert after_mock.called
        assert after_mock.call_args[0][0] == 0

    def test_app_destroy_shuts_down_manager(self, headless_app):
        """Test 32: Calling destroy invokes manager.shutdown cleanly."""
        app = headless_app
        shutdown_mock = MagicMock()
        app.manager.shutdown = shutdown_mock
        app.destroy()
        assert shutdown_mock.called


# ============================================================================
# SUITE 4: Visualizer Telemetry & Rate Limiting (10 Tests)
# ============================================================================

class TestVisualizerAndRateLimiting:
    """Suite 4: Tests covering 250 Hz decoupled store, deadband dirty-checking, and telemetry."""

    def test_on_input_update_atomic_store_rate(self, headless_app):
        """Test 33: High-frequency input updates perform atomic reference storage without lag."""
        app = headless_app
        state = make_dummy_state(has_motion=True, gyro_x=10.0)

        start = time.perf_counter()
        for _ in range(1000):
            app._on_input_update(state)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50.0  # Under 50ms for 1,000 updates (< 50us per call)
        assert app._latest_state is state

    def test_visualizer_tick_skips_rendering_when_disconnected(self, headless_app):
        """Test 34: _visualizer_tick does not call render logic when disconnected."""
        app = headless_app
        app.is_connected = False
        app._latest_state = make_dummy_state()

        render_mock = MagicMock()
        app._render_live_inputs_optimized = render_mock
        app._visualizer_tick()
        assert not render_mock.called

    def test_visualizer_tick_skips_when_latest_state_none(self, headless_app):
        """Test 35: _visualizer_tick does not call render logic if state is None."""
        app = headless_app
        app.is_connected = True
        app._latest_state = None

        render_mock = MagicMock()
        app._render_live_inputs_optimized = render_mock
        app._visualizer_tick()
        assert not render_mock.called

    def test_imu_status_logic_active(self):
        """Test 36: Active motion transitions badge text to ● IMU ACTIVE."""
        state = make_dummy_state(has_motion=True, gyro_x=25.0, accel_z=1.0)
        badge_text, text_color, fg_color = determine_imu_status(state, is_connected=True)
        assert "IMU ACTIVE" in badge_text
        assert text_color == "#10B981"

    def test_imu_status_logic_idle(self):
        """Test 37: Stationary controller transitions badge text to ● IMU IDLE."""
        state = make_dummy_state(has_motion=True, gyro_x=0.1, gyro_y=0.0, gyro_z=0.2, accel_z=1.0)
        badge_text, text_color, fg_color = determine_imu_status(state, is_connected=True)
        assert "IMU IDLE" in badge_text
        assert text_color == "#9CA3AF"

    def test_imu_status_logic_na_when_no_motion(self):
        """Test 38: Packets without motion block (Report 0x09) set badge to ○ IMU N/A."""
        state = make_dummy_state(has_motion=False)
        badge_text, text_color, fg_color = determine_imu_status(state, is_connected=True)
        assert "IMU N/A" in badge_text
        assert text_color == "#6B7280"

    def test_imu_status_logic_na_when_disconnected(self):
        """Test 39: Disconnected controller always reports ○ IMU N/A."""
        state = make_dummy_state(has_motion=True, gyro_x=20.0)
        badge_text, text_color, fg_color = determine_imu_status(state, is_connected=False)
        assert "IMU N/A" in badge_text

    def test_imu_telemetry_formatting(self):
        """Test 40: format_imu_telemetry accurately formats acceleration and angular rates."""
        state = make_dummy_state(
            accel_x=0.012, accel_y=-0.984, accel_z=0.051,
            gyro_x=12.34, gyro_y=-5.67, gyro_z=0.12
        )
        accel_str, gyro_str = format_imu_telemetry(state)
        assert "+0.01g" in accel_str
        assert "-0.98g" in accel_str
        assert "+12.3" in gyro_str
        assert "-5.7" in gyro_str

    def test_deadband_dirty_checking_suppresses_small_noise(self, headless_app):
        """Test 41: Sensor noise below deadband suppresses label updates."""
        app = headless_app
        app.is_connected = True

        state1 = make_dummy_state(has_motion=True, gyro_x=10.0, accel_x=0.0, accel_z=1.0)
        app._render_live_inputs_optimized(state1)
        assert app._prev_gx == 10.0
        assert app._prev_ax == 0.0

        # Small jitter (gyro delta 0.4 deg/s < 1.0 deg/s, accel delta 0.01g < 0.02g)
        state2 = make_dummy_state(has_motion=True, gyro_x=10.4, accel_x=0.01, accel_z=1.0)
        app._render_live_inputs_optimized(state2)
        assert app._prev_gx == 10.0
        assert app._prev_ax == 0.0

    def test_deadband_dirty_checking_triggers_on_significant_motion(self, headless_app):
        """Test 42: Motion exceeding deadband triggers cache and label updates."""
        app = headless_app
        app.is_connected = True

        state1 = make_dummy_state(has_motion=True, gyro_x=10.0, accel_x=0.0, accel_z=1.0)
        app._render_live_inputs_optimized(state1)

        # Significant motion (gyro delta 5.0 deg/s > 1.0 deg/s, accel delta 0.10g > 0.02g)
        state3 = make_dummy_state(has_motion=True, gyro_x=15.0, accel_x=0.10, accel_z=1.0)
        app._render_live_inputs_optimized(state3)
        assert app._prev_gx == 15.0
        assert app._prev_ax == 0.10


# ============================================================================
# SUITE 5: Robustness, Fault Tolerance & Concurrency (6 Tests)
# ============================================================================

class TestGuiRobustnessAndEdgeCases:
    """Suite 5: Edge cases covering rapid toggling, port conflicts, and stream teardown."""

    def test_rapid_mode_switching_concurrency(self, headless_app):
        """Test 43: Rapid alternating between Xbox and DS4 modes does not corrupt state."""
        app = headless_app
        for i in range(20):
            mode = "DualShock 4 (Motion)" if i % 2 == 0 else "Xbox 360"
            app._on_output_mode_changed(mode)

        expected_final = "Xbox 360"
        assert app.output_mode_selector.get() == expected_final
        assert app.manager.output_mode == "xbox"
        loaded = load_settings(app._get_settings_path())
        assert loaded["output_mode"] == "xbox"

    def test_rapid_dsu_switching_concurrency(self, headless_app):
        """Test 44: Rapid toggling of Cemuhook DSU switch maintains coherent state."""
        app = headless_app
        for i in range(20):
            if i % 2 == 0:
                app.switch_dsu.select()
            else:
                app.switch_dsu.deselect()
            app._on_dsu_switch_changed()

        assert app.manager.dsu_enabled is False
        loaded = load_settings(app._get_settings_path())
        assert loaded["dsu_enabled"] is False

    def test_dsu_bind_failure_ui_graceful_recovery(self, headless_app, monkeypatch):
        """Test 45: If DSU server fails to bind (port in use), UI deselects switch."""
        app = headless_app
        # Simulate manager bind failure where manager.dsu_enabled stays False
        monkeypatch.setattr(ControllerManager, "set_dsu_enabled", lambda self, en: None)
        app.switch_dsu.select()
        app._on_dsu_switch_changed()

        assert app.switch_dsu.get() == 0
        assert "Inactive" in app.lbl_dsu_status.cget("text")

    def test_malformed_state_exception_resilience(self, headless_app):
        """Test 46: Malformed state object in _visualizer_tick does not crash app."""
        app = headless_app
        app.is_connected = True
        app._latest_state = object()  # Lacks state attributes

        # Should catch exception internally and continue
        app._visualizer_tick()

    def test_settings_save_permission_denied(self, headless_app, monkeypatch):
        """Test 47: Permission error during _save_settings does not crash UI."""
        app = headless_app
        monkeypatch.setattr("gui.save_settings", MagicMock(side_effect=PermissionError("Denied")))
        # Should execute silently without exception
        app._save_settings()

    def test_app_quit_during_active_stream(self, headless_app):
        """Test 48: Destroying window while background thread pushes states does not deadlock."""
        app = headless_app
        app.is_connected = True
        stop_event = threading.Event()

        def push_loop():
            state = make_dummy_state(has_motion=True, gyro_x=5.0)
            while not stop_event.is_set():
                app._on_input_update(state)
                time.sleep(0.001)

        t = threading.Thread(target=push_loop, daemon=True)
        t.start()
        time.sleep(0.01)

        app.destroy()
        stop_event.set()
        t.join(timeout=1.0)
        assert not t.is_alive()
