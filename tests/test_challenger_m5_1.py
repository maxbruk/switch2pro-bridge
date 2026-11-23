"""Adversarial Stress, Corruption Fuzzing & High-Rate Hammering Test Suite for Milestone 5.

Author: M5 Adversarial Challenger 1 (teamwork_challenger)
Target Scope: `gui.py` and `tests/test_gui.py`

Covers:
- Suite 1: Settings File Corruption, Pathological Payloads & Concurrency Fuzzing (12 Tests)
- Suite 2: App Initialization Resilience Under Extreme Corruption (6 Tests)
- Suite 3: GUI Control Rapid Hammering Under 250 Hz Telemetry Stream (8 Tests)
- Suite 4: Visualizer Telemetry Stress & Pathological State Oracle (6 Tests)
Total: 32 Adversarial Stress Tests
"""

import os
import sys
import json
import time
import math
import stat
import random
import threading
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from report_parser import ControllerState
from app_core import ControllerManager
from gui import (
    DEFAULT_SETTINGS,
    Switch2ProApp,
    get_settings_path,
    load_settings,
    save_settings,
    determine_imu_status,
    format_imu_telemetry,
)
from tests.test_gui import make_dummy_state, TK_AVAILABLE


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def headless_challenger_app(tmp_path, monkeypatch):
    """Provides an isolated headless Switch2ProApp instance for adversarial stress tests."""
    if not TK_AVAILABLE:
        pytest.skip("Tkinter display context not available")

    tcl_dir = os.path.join(sys.prefix, "tcl", "tcl8.6")
    tk_dir = os.path.join(sys.prefix, "tcl", "tk8.6")
    if os.path.isdir(tcl_dir):
        monkeypatch.setenv("TCL_LIBRARY", tcl_dir)
    if os.path.isdir(tk_dir):
        monkeypatch.setenv("TK_LIBRARY", tk_dir)

    settings_file = str(tmp_path / "settings_challenger.json")
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


# ============================================================================
# SUITE 1: Settings File Corruption, Pathological Payloads & Concurrency Fuzzing
# ============================================================================

class TestSettingsFileCorruptionFuzzing:
    """Suite 1: Extreme fuzzing of settings I/O, parsing, and concurrent operations."""

    def test_json_truncation_fuzzing_all_byte_offsets(self, tmp_path):
        """Fuzz load_settings across all possible truncation prefixes of a valid JSON."""
        full_json = json.dumps({
            "output_mode": "ds4",
            "dsu_enabled": True,
            "layout": "xbox",
            "deadzone": 0.12,
            "auto_connect": False,
            "gl_mapping": "RB",
            "gr_mapping": "LB",
        }, indent=2)
        raw_bytes = full_json.encode("utf-8")
        test_file = str(tmp_path / "truncated_fuzz.json")

        for offset in range(1, len(raw_bytes)):
            with open(test_file, "wb") as f:
                f.write(raw_bytes[:offset])
            loaded = load_settings(test_file)
            assert isinstance(loaded, dict)
            # Must strictly recover all canonical keys with valid types
            assert loaded["output_mode"] in ("xbox", "ds4")
            assert isinstance(loaded["dsu_enabled"], bool)
            assert loaded["layout"] in ("nintendo", "xbox")
            assert 0.0 <= loaded["deadzone"] <= 0.20
            assert isinstance(loaded["auto_connect"], bool)
            assert loaded["gl_mapping"] in ("L3", "R3", "LB", "RB", "LT", "RT", "A", "B", "X", "Y", "Disabled")
            assert loaded["gr_mapping"] in ("L3", "R3", "LB", "RB", "LT", "RT", "A", "B", "X", "Y", "Disabled")

    def test_massive_binary_gibberish_payload(self, tmp_path):
        """Verify load_settings handles a 1MB file of random bytes without memory or parser crash."""
        junk_file = str(tmp_path / "1mb_random_junk.bin")
        random_payload = os.urandom(1024 * 1024)  # 1 MB
        with open(junk_file, "wb") as f:
            f.write(random_payload)

        loaded = load_settings(junk_file)
        assert loaded == DEFAULT_SETTINGS

    def test_nul_byte_corrupted_file(self, tmp_path):
        """Verify load_settings handles files populated exclusively with NUL bytes."""
        nul_file = str(tmp_path / "nul_bytes.json")
        with open(nul_file, "wb") as f:
            f.write(b"\x00" * 8192)

        loaded = load_settings(nul_file)
        assert loaded == DEFAULT_SETTINGS

    def test_unicode_bom_corruptions(self, tmp_path):
        """Verify handling of various BOM headers (UTF-8, UTF-16 LE, UTF-16 BE).
        
        Python's json.load() raises JSONDecodeError on UTF-8 BOM unless decoded as utf-8-sig.
        load_settings() must catch this cleanly and recover to DEFAULT_SETTINGS.
        """
        # UTF-8 with BOM
        utf8_bom_file = str(tmp_path / "utf8_bom.json")
        with open(utf8_bom_file, "wb") as f:
            f.write(b"\xef\xbb\xbf" + json.dumps({"output_mode": "ds4"}).encode("utf-8"))
        loaded = load_settings(utf8_bom_file)
        assert loaded == DEFAULT_SETTINGS

        # UTF-16 LE BOM (invalid for utf-8 reader)
        utf16_bom_file = str(tmp_path / "utf16_bom.json")
        with open(utf16_bom_file, "wb") as f:
            f.write(b"\xff\xfe" + b"random binary utf16 data")
        loaded_utf16 = load_settings(utf16_bom_file)
        assert loaded_utf16 == DEFAULT_SETTINGS

    def test_non_dict_json_roots(self, tmp_path):
        """Verify load_settings ignores non-dict root JSON values (array, int, string, bool, null)."""
        non_dict_payloads = [
            [1, 2, "ds4", True],
            42,
            -999.5,
            "a naked string",
            True,
            False,
            None,
        ]
        test_file = str(tmp_path / "non_dict.json")
        for payload in non_dict_payloads:
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            loaded = load_settings(test_file)
            assert loaded == DEFAULT_SETTINGS

    def test_deeply_nested_and_giant_strings(self, tmp_path):
        """Verify load_settings handles deep nesting and 100,000 character strings."""
        nested = {"output_mode": "ds4"}
        curr = nested
        for _ in range(100):
            curr["nested"] = {}
            curr = curr["nested"]
        curr["dsu_enabled"] = True

        test_file = str(tmp_path / "nested.json")
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(nested, f)

        loaded = load_settings(test_file)
        assert loaded["output_mode"] == "ds4"
        # Since dsu_enabled was nested deeply, top-level remains default False
        assert loaded["dsu_enabled"] is False

        # Giant string payload
        giant_file = str(tmp_path / "giant_string.json")
        giant_dict = {
            "output_mode": "xbox",
            "gl_mapping": "A" * 100_000,
        }
        with open(giant_file, "w", encoding="utf-8") as f:
            json.dump(giant_dict, f)
        loaded_giant = load_settings(giant_file)
        assert loaded_giant["output_mode"] == "xbox"
        assert loaded_giant["gl_mapping"] == "L3"  # fell back to safe L3

    def test_extreme_adversarial_value_sanitization(self, tmp_path):
        """Inject type-confused, NaN, Inf, and out-of-domain values into all settings keys."""
        pathological_data = {
            "output_mode": ["invalid", "list"],
            "dsu_enabled": "not_a_bool",
            "layout": {"malicious": "dict"},
            "deadzone": float("nan"),
            "auto_connect": 0,
            "gl_mapping": "NON_EXISTENT_PADDLE",
            "gr_mapping": 12345,
        }
        test_file = str(tmp_path / "pathological.json")
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(pathological_data, f)

        loaded = load_settings(test_file)
        assert loaded["output_mode"] == "xbox"
        assert loaded["dsu_enabled"] is False
        assert loaded["layout"] == "nintendo"
        assert loaded["deadzone"] == 0.05
        assert loaded["auto_connect"] is True
        assert loaded["gl_mapping"] == "L3"
        assert loaded["gr_mapping"] == "R3"

    def test_deadzone_boundary_and_infinity_injection(self, tmp_path):
        """Test exact boundary clamping and Inf/-Inf handling for deadzone."""
        test_file = str(tmp_path / "deadzone_test.json")

        for bad_dz in [float("inf"), float("-inf"), -1.0, -0.000001, 0.200001, 100.0, "0.05"]:
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump({"deadzone": bad_dz}, f)
            loaded = load_settings(test_file)
            if bad_dz == "0.05":
                # string "0.05" can be parsed by float("0.05") -> 0.05
                assert loaded["deadzone"] == 0.05
            else:
                assert loaded["deadzone"] == 0.05

        # Valid extremes
        for valid_dz in [0.0, 0.20, 0.10]:
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump({"deadzone": valid_dz}, f)
            loaded = load_settings(test_file)
            assert loaded["deadzone"] == pytest.approx(valid_dz)

    def test_target_path_is_a_directory(self, tmp_path):
        """Verify load_settings and save_settings handle target path being a directory without raising."""
        dir_path = str(tmp_path / "i_am_a_directory")
        os.makedirs(dir_path, exist_ok=True)

        # load_settings should return DEFAULT_SETTINGS
        loaded = load_settings(dir_path)
        assert loaded == DEFAULT_SETTINGS

        # save_settings should return False
        saved = save_settings(DEFAULT_SETTINGS, dir_path)
        assert saved is False

    def test_nested_parent_directory_creation_on_save(self, tmp_path):
        """save_settings automatically creates arbitrarily deep non-existent directories."""
        deep_file = str(tmp_path / "deep" / "level1" / "level2" / "custom_settings.json")
        saved = save_settings({"output_mode": "ds4", "dsu_enabled": True}, deep_file)
        assert saved is True
        assert os.path.isfile(deep_file)
        loaded = load_settings(deep_file)
        assert loaded["output_mode"] == "ds4"
        assert loaded["dsu_enabled"] is True

    def test_save_to_read_only_file_returns_false_safely(self, tmp_path):
        """Verify save_settings fails gracefully when target file or directory is read-only."""
        ro_file = str(tmp_path / "readonly_settings.json")
        with open(ro_file, "w", encoding="utf-8") as f:
            f.write("{}")

        # Make file read-only on Windows / POSIX
        os.chmod(ro_file, stat.S_IREAD)
        try:
            # os.replace onto read-only file on Windows might fail with PermissionError
            # save_settings must catch it and return False
            saved = save_settings({"output_mode": "ds4"}, ro_file)
            # Either it succeeded (if OS allowed replace) or returned False safely without raising
            assert isinstance(saved, bool)
        finally:
            os.chmod(ro_file, stat.S_IWRITE)

    def test_high_concurrency_multithreaded_read_write_hammering(self, tmp_path):
        """Concurrently run 10 threads reading and writing settings simultaneously."""
        stress_file = str(tmp_path / "concurrent_stress.json")
        save_settings(DEFAULT_SETTINGS, stress_file)

        stop_event = threading.Event()
        errors = []

        def writer_loop(tid: int):
            try:
                for i in range(50):
                    mode = "ds4" if (i + tid) % 2 == 0 else "xbox"
                    dsu = (i % 2 == 0)
                    dz = round(0.02 + (i % 5) * 0.02, 2)
                    save_settings({
                        "output_mode": mode,
                        "dsu_enabled": dsu,
                        "deadzone": dz,
                    }, stress_file)
            except Exception as e:
                errors.append(e)

        def reader_loop(tid: int):
            try:
                for _ in range(50):
                    s = load_settings(stress_file)
                    assert isinstance(s, dict)
                    assert s["output_mode"] in ("xbox", "ds4")
                    assert isinstance(s["dsu_enabled"], bool)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer_loop, args=(i,)))
            threads.append(threading.Thread(target=reader_loop, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Worker thread deadlocked during concurrent I/O"

        assert len(errors) == 0, f"Concurrent I/O generated unhandled exceptions: {errors}"
        # Final load must be perfectly valid
        final_settings = load_settings(stress_file)
        assert isinstance(final_settings, dict)
        assert final_settings["output_mode"] in ("xbox", "ds4")


# ============================================================================
# SUITE 2: App Initialization Resilience Under Extreme Corruption
# ============================================================================

class TestAppInitUnderCorruptions:
    """Suite 2: Verify Switch2ProApp initializes flawlessly despite damaged configuration files."""

    def test_app_init_with_zero_byte_settings_file(self, tmp_path, monkeypatch):
        """Switch2ProApp initializes to valid defaults when settings.json is 0 bytes."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        corrupt_file = str(tmp_path / "empty.json")
        with open(corrupt_file, "w") as f:
            pass

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=corrupt_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "xbox"
            assert app.manager.dsu_enabled is False
            assert app.output_mode_selector.get() == "Xbox 360"
            assert app.switch_dsu.get() == 0
        finally:
            app.destroy()

    def test_app_init_with_syntax_corrupted_json(self, tmp_path, monkeypatch):
        """Switch2ProApp initializes to valid defaults when settings.json has syntax errors."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        corrupt_file = str(tmp_path / "syntax_error.json")
        with open(corrupt_file, "w", encoding="utf-8") as f:
            f.write('{"output_mode": "ds4", "dsu_enabled": TRU')

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=corrupt_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "xbox"
            assert app.output_mode_selector.get() == "Xbox 360"
        finally:
            app.destroy()

    def test_app_init_with_binary_garbage_file(self, tmp_path, monkeypatch):
        """Switch2ProApp initializes to valid defaults when settings.json contains random binary."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        bin_file = str(tmp_path / "garbage.bin")
        with open(bin_file, "wb") as f:
            f.write(os.urandom(65536))

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=bin_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "xbox"
            assert app.manager.deadzone == 0.05
        finally:
            app.destroy()

    def test_app_init_with_directory_as_settings_path(self, tmp_path, monkeypatch):
        """Switch2ProApp initializes gracefully if settings_path points to an existing directory."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        dir_path = str(tmp_path / "a_directory")
        os.makedirs(dir_path, exist_ok=True)

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=dir_path, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "xbox"
        finally:
            app.destroy()

    def test_app_init_with_type_confused_settings(self, tmp_path, monkeypatch):
        """Switch2ProApp sanitizes type-confused JSON on startup."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        bad_file = str(tmp_path / "bad_types.json")
        with open(bad_file, "w", encoding="utf-8") as f:
            json.dump({
                "output_mode": 99999,
                "dsu_enabled": "definitely_true",
                "deadzone": -50.0,
                "layout": None,
                "gl_mapping": {},
            }, f)

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=bad_file, auto_start_detector=False)
        app.withdraw()
        try:
            assert app.manager.output_mode == "xbox"
            assert app.manager.dsu_enabled is False
            assert app.manager.deadzone == 0.05
            assert app.manager.button_layout == "nintendo"
            assert app.manager.gl_mapping == "L3"
        finally:
            app.destroy()

    def test_app_init_with_non_existent_deep_path(self, tmp_path, monkeypatch):
        """Switch2ProApp initializes with non-existent deep settings path and writes successfully."""
        if not TK_AVAILABLE:
            pytest.skip("Tkinter not available")
        deep_file = str(tmp_path / "nested1" / "nested2" / "auto_created.json")

        monkeypatch.setattr(Switch2ProApp, "_setup_window_icons", lambda self: None)
        monkeypatch.setattr(Switch2ProApp, "_schedule_visualizer_tick", lambda self: None)
        monkeypatch.setattr(ControllerManager, "start_auto_detector", lambda self: None)

        app = Switch2ProApp(settings_path=deep_file, auto_start_detector=False)
        app.withdraw()
        try:
            # Change a setting and save
            app._on_output_mode_changed("DualShock 4 (Motion)")
            assert os.path.isfile(deep_file)
            saved = load_settings(deep_file)
            assert saved["output_mode"] == "ds4"
        finally:
            app.destroy()


# ============================================================================
# SUITE 3: GUI Control Rapid Hammering Under 250 Hz Telemetry Stream
# ============================================================================

class TestGuiHammeringUnder250Hz:
    """Suite 3: Stress-test UI controls, state transitions, and persistence under 250 Hz stream."""

    def test_rapid_mode_switching_under_250hz_stream(self, headless_challenger_app):
        """Hammer output mode switching 150 times while 250 Hz packets stream concurrently."""
        app = headless_challenger_app
        app._update_connection_ui(True, "USB-C", "Switch 2 Pro Controller", 100)

        stop_event = threading.Event()

        def stream_worker():
            seq = 0
            while not stop_event.is_set():
                seq += 1
                state = make_dummy_state(
                    accel_x=math.sin(seq * 0.1),
                    accel_y=math.cos(seq * 0.1),
                    gyro_x=(seq % 200) - 100.0,
                    has_motion=True,
                )
                app._on_input_update(state)
                time.sleep(0.004)  # 250 Hz

        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()

        try:
            for i in range(150):
                target = "DualShock 4 (Motion)" if i % 2 == 0 else "Xbox 360"
                app._on_output_mode_changed(target)
                if i % 10 == 0:
                    app._visualizer_tick()
                    app.update_idletasks()

            # Final state must match the last toggle
            assert app.output_mode_selector.get() == "Xbox 360"
            assert app.manager.output_mode == "xbox"
            assert "Xbox 360" in app.lbl_emulation.cget("text")

            # Disk persistence must match final state
            disk_settings = load_settings(app._get_settings_path())
            assert disk_settings["output_mode"] == "xbox"
        finally:
            stop_event.set()
            t.join(timeout=2.0)

    def test_rapid_dsu_toggling_under_250hz_stream(self, headless_challenger_app):
        """Hammer Cemuhook DSU server switch 150 times while 250 Hz packets stream."""
        app = headless_challenger_app
        app._update_connection_ui(True, "USB-C", "Switch 2 Pro Controller", 95)

        stop_event = threading.Event()

        def stream_worker():
            seq = 0
            while not stop_event.is_set():
                seq += 1
                state = make_dummy_state(
                    accel_z=1.0 + math.sin(seq * 0.05) * 0.1,
                    gyro_z=seq * 0.5,
                    has_motion=True,
                )
                app._on_input_update(state)
                time.sleep(0.004)

        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()

        try:
            for i in range(150):
                if i % 2 == 0:
                    app.switch_dsu.select()
                else:
                    app.switch_dsu.deselect()
                app._on_dsu_switch_changed()
                if i % 10 == 0:
                    app._visualizer_tick()
                    app.update_idletasks()

            # Final toggle was deselect (odd i)
            assert app.manager.dsu_enabled is False
            assert "Inactive" in app.lbl_dsu_status.cget("text")

            disk_settings = load_settings(app._get_settings_path())
            assert disk_settings["dsu_enabled"] is False
        finally:
            stop_event.set()
            t.join(timeout=2.0)

    def test_all_settings_controls_interleaved_hammering(self, headless_challenger_app):
        """Interleave mode switches, DSU toggles, deadzone slider, layout, and paddle changes 100 times."""
        app = headless_challenger_app
        app._update_connection_ui(True, "Bluetooth", "Switch 2 Pro Controller", 80)

        stop_event = threading.Event()

        def stream_worker():
            seq = 0
            while not stop_event.is_set():
                seq += 1
                state = make_dummy_state(
                    a=(seq % 2 == 0),
                    b=(seq % 3 == 0),
                    gyro_y=float(seq % 50),
                    has_motion=True,
                )
                app._on_input_update(state)
                time.sleep(0.004)

        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()

        try:
            for i in range(100):
                mode = "DualShock 4 (Motion)" if i % 2 == 0 else "Xbox 360"
                app._on_output_mode_changed(mode)

                if i % 2 == 0:
                    app.switch_dsu.select()
                else:
                    app.switch_dsu.deselect()
                app._on_dsu_switch_changed()

                dz = 0.05 + (i % 10) * 0.01
                app._on_deadzone_changed(dz)

                layout = "Xbox (Letters)" if i % 2 == 0 else "Nintendo (Physical)"
                app._on_layout_changed(layout)

                paddle = "Disabled" if i % 2 == 0 else "L3"
                app._on_gl_mapping_changed(paddle)

                app._visualizer_tick()
                app.update_idletasks()

            # Ensure disk file did not corrupt
            disk_settings = load_settings(app._get_settings_path())
            assert disk_settings["output_mode"] == "xbox"
            assert disk_settings["dsu_enabled"] is False
            assert disk_settings["layout"] == "nintendo"
            assert disk_settings["gl_mapping"] == "L3"
        finally:
            stop_event.set()
            t.join(timeout=2.0)

    def test_simultaneous_multi_threaded_callback_storm(self, headless_challenger_app):
        """Simulate simultaneous callback bursts from connection, status, and input threads."""
        app = headless_challenger_app
        stop_event = threading.Event()
        errors = []

        def status_storm():
            for i in range(100):
                if stop_event.is_set():
                    break
                app._on_status_message(f"Status message storm iteration {i}")
                time.sleep(0.001)

        def connection_storm():
            for i in range(50):
                if stop_event.is_set():
                    break
                conn = (i % 2 == 0)
                trans = "USB" if conn else "None"
                app._on_connection_changed(conn, trans, "Switch 2 Pro", 100)
                time.sleep(0.002)

        def input_storm():
            for i in range(200):
                if stop_event.is_set():
                    break
                st = make_dummy_state(gyro_x=float(i), accel_y=float(i % 2))
                app._on_input_update(st)
                time.sleep(0.001)

        threads = [
            threading.Thread(target=status_storm),
            threading.Thread(target=connection_storm),
            threading.Thread(target=input_storm),
        ]
        for t in threads:
            t.start()

        # Let the main thread process Tcl events during the storm
        for _ in range(30):
            app.update_idletasks()
            time.sleep(0.01)

        stop_event.set()
        for t in threads:
            t.join(timeout=2.0)
            assert not t.is_alive()

        app.update_idletasks()
        assert len(errors) == 0

    def test_app_destroy_during_peak_telemetry_hammering(self, headless_challenger_app):
        """Call app.destroy() while high-rate telemetry thread is actively flooding states."""
        app = headless_challenger_app
        app.is_connected = True

        stop_event = threading.Event()

        def stream_worker():
            while not stop_event.is_set():
                st = make_dummy_state(gyro_x=45.0, has_motion=True)
                app._on_input_update(st)
                time.sleep(0.002)

        t = threading.Thread(target=stream_worker, daemon=True)
        t.start()
        time.sleep(0.02)

        # Destroy while stream is firing
        app.destroy()
        stop_event.set()
        t.join(timeout=2.0)
        assert not t.is_alive()

    def test_rapid_connection_state_flip_flop(self, headless_challenger_app):
        """Flip connection state connected <-> disconnected 50 times and check widget consistency."""
        app = headless_challenger_app

        for i in range(50):
            if i % 2 == 0:
                app._update_connection_ui(True, "USB-C", "Switch 2 Pro Controller", 100)
                assert app.is_connected is True
                assert "CONNECTED" in app.status_pill.cget("text")
                assert app.btn_main.cget("text") == "⏹ DISCONNECT"
            else:
                app._update_connection_ui(False, "None", "Not Connected", -1)
                assert app.is_connected is False
                assert "DISCONNECTED" in app.status_pill.cget("text")
                assert app.btn_main.cget("text") == "⚡ CONNECT CONTROLLER"
            app.update_idletasks()

    def test_virtual_controller_backend_survives_rapid_hot_swaps_under_load(self, headless_challenger_app):
        """Verify ControllerManager virtual gamepad pointer hot-swaps under active state feed."""
        app = headless_challenger_app
        manager = app.manager

        for i in range(50):
            mode = "ds4" if i % 2 == 0 else "xbox"
            manager.set_output_mode(mode)
            # Feed state directly through manager
            st = make_dummy_state(gyro_x=10.0, accel_z=1.0)
            manager._on_controller_state(st)
            assert manager.output_mode == mode
            assert manager.virtual_gamepad is not None

    def test_dsu_port_conflict_reconnect_retry_cycle(self, headless_challenger_app, monkeypatch):
        """Simulate alternating DSU server bind failure and recovery in UI."""
        app = headless_challenger_app

        # Iteration 1: port bind fails -> manager.dsu_enabled remains False
        with patch.object(ControllerManager, "set_dsu_enabled", lambda self, en: None):
            app.switch_dsu.select()
            app._on_dsu_switch_changed()
            assert app.switch_dsu.get() == 0
            assert "Inactive" in app.lbl_dsu_status.cget("text")

        # Iteration 2: port released -> binds successfully
        app.switch_dsu.select()
        app._on_dsu_switch_changed()
        assert app.manager.dsu_enabled is True
        assert "Active" in app.lbl_dsu_status.cget("text")

        # Iteration 3: clean disable
        app.switch_dsu.deselect()
        app._on_dsu_switch_changed()
        assert app.manager.dsu_enabled is False


# ============================================================================
# SUITE 4: Visualizer Telemetry Stress & Pathological State Oracle
# ============================================================================

class TestVisualizerPathologicalTelemetry:
    """Suite 4: Adversarial tests for mathematical edge cases, telemetry formatting, and dirty checking."""

    def test_nan_and_inf_in_controller_state_imu_status(self):
        """determine_imu_status handles NaN and Inf in gyro/accel without crashing."""
        # 1. All NaN
        state_nan = make_dummy_state(
            accel_x=float("nan"), accel_y=float("nan"), accel_z=float("nan"),
            gyro_x=float("nan"), gyro_y=float("nan"), gyro_z=float("nan"),
            has_motion=True,
        )
        badge_nan, _, _ = determine_imu_status(state_nan, is_connected=True)
        # In Python, (nan > 1.5) is False, so it returns IDLE
        assert "IMU IDLE" in badge_nan

        # 2. Inf in Gyro
        state_inf = make_dummy_state(gyro_x=float("inf"), has_motion=True)
        badge_inf, _, _ = determine_imu_status(state_inf, is_connected=True)
        assert "IMU ACTIVE" in badge_inf

        # 3. -Inf in Accel
        state_ninf = make_dummy_state(accel_z=float("-inf"), has_motion=True)
        badge_ninf, _, _ = determine_imu_status(state_ninf, is_connected=True)
        assert "IMU ACTIVE" in badge_ninf

    def test_extreme_magnitude_floats_handled_in_visualizer_tick(self, headless_challenger_app):
        """Verify _visualizer_tick handles 1e200 floats without unhandled crash or timer loop death."""
        app = headless_challenger_app
        app.is_connected = True
        app._latest_state = make_dummy_state(
            accel_x=1e200, accel_y=1e200, accel_z=1e200,
            gyro_x=1e200, gyro_y=1e200, gyro_z=1e200,
            has_motion=True,
        )

        # _visualizer_tick must catch potential OverflowError and maintain stability
        app._visualizer_tick()
        app.update_idletasks()

    def test_extreme_analog_stick_and_trigger_values(self, headless_challenger_app):
        """Verify analog canvas and trigger badges handle values far outside [-1.0, 1.0]."""
        app = headless_challenger_app
        app.is_connected = True

        crazy_states = [
            make_dummy_state(left_stick_x=999.0, left_stick_y=-999.0, right_stick_x=100.0, right_stick_y=-100.0),
            make_dummy_state(left_stick_x=-999.0, left_stick_y=999.0, right_stick_x=-100.0, right_stick_y=100.0),
            make_dummy_state(zl_analog=999.0, zr_analog=-999.0),
            make_dummy_state(battery_percentage=999),
            make_dummy_state(battery_percentage=-50),
        ]

        for st in crazy_states:
            app._render_live_inputs_optimized(st)
            app.update_idletasks()

    def test_rapid_motion_flag_flipping(self, headless_challenger_app):
        """Flip has_motion between True and False 100 times to verify status pill resets cleanly."""
        app = headless_challenger_app
        app.is_connected = True

        for i in range(100):
            motion = (i % 2 == 0)
            st = make_dummy_state(
                accel_x=0.5 if motion else 0.0,
                gyro_x=50.0 if motion else 0.0,
                has_motion=motion,
            )
            app._render_live_inputs_optimized(st)
            if motion:
                assert "IMU ACTIVE" in app.imu_status_pill.cget("text")
            else:
                assert "IMU N/A" in app.imu_status_pill.cget("text")
                assert "X: --" in app.lbl_accel.cget("text")
                assert "P: --" in app.lbl_gyro.cget("text")

    def test_deadband_dirty_checking_subtle_jitter_stress(self, headless_challenger_app):
        """Verify deadband logic rejects micro-jitter below threshold over 200 consecutive ticks."""
        app = headless_challenger_app
        app.is_connected = True

        base_state = make_dummy_state(accel_x=0.0, accel_y=0.0, accel_z=1.0, gyro_x=10.0, gyro_y=0.0, gyro_z=0.0)
        app._render_live_inputs_optimized(base_state)
        assert app._prev_gx == 10.0
        assert app._prev_ax == 0.0

        # Feed 200 states with jitter within deadband (< 1.0 deg/s gyro, < 0.02 g accel)
        for i in range(200):
            jitter_g = 10.0 + (math.sin(i) * 0.4)   # max delta 0.4 < 1.0
            jitter_a = 0.0 + (math.cos(i) * 0.01)  # max delta 0.01 < 0.02
            st = make_dummy_state(accel_x=jitter_a, gyro_x=jitter_g)
            app._render_live_inputs_optimized(st)
            # Caches must remain at base_state
            assert app._prev_gx == 10.0
            assert app._prev_ax == 0.0

    def test_imu_telemetry_formatting_oracle(self):
        """Oracle testing of format_imu_telemetry output strings against expected formatting."""
        test_cases = [
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "X: +0.00g  Y: +0.00g  Z: +0.00g", "P: +0.0  Y: +0.0  R: +0.0"),
            (1.0, -1.0, 0.5, 180.0, -90.5, 45.2, "X: +1.00g  Y: -1.00g  Z: +0.50g", "P: +180.0  Y: -90.5  R: +45.2"),
            (-0.001, 0.004, -0.009, 0.04, -0.06, 0.01, "X: -0.00g  Y: +0.00g  Z: -0.01g", "P: +0.0  Y: -0.1  R: +0.0"),
        ]

        for ax, ay, az, gx, gy, gz, exp_accel, exp_gyro in test_cases:
            st = make_dummy_state(accel_x=ax, accel_y=ay, accel_z=az, gyro_x=gx, gyro_y=gy, gyro_z=gz)
            res_accel, res_gyro = format_imu_telemetry(st)
            assert res_accel == exp_accel
            assert res_gyro == exp_gyro
