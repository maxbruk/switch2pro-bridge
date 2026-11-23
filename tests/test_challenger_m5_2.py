"""Adversarial Empirical Test Suite for Milestone 5: 6-Axis IMU Live Visualizer, Rate Limiting & Telemetry.

Challenger: M5 Adversarial Challenger 2 (teamwork_challenger)
Mission:
1. 250 Hz Input Rate Stress & Decoupled Rendering Verification:
   - Pump synthetic ControllerState frames at 250 Hz for thousands of iterations.
   - Verify that _on_input_update executes in sub-microsecond time without blocking the producer thread.
   - Verify that the 60 FPS tick loop (_visualizer_tick) runs without frame dropping or memory leaks.
2. Deadband Dirty-Checking Rigor:
   - Test sub-threshold jitter on accelerometer (|Δa| < 0.02g) and gyroscope (|Δω| < 1.0°/s).
   - Verify that label .configure() is suppressed to save CPU.
   - Test above-threshold motion (|Δa| >= 0.02g or |Δω| >= 1.0°/s).
   - Verify that label .configure() triggers with exact string formatting.
3. Motion Status Pill State Machine Verification:
   - Verify transitions between "○ IMU N/A", "● IMU IDLE", and "● IMU ACTIVE".
   - Test with non-finite floats (NaN, +Inf, -Inf), zero-vectors, and gravity-only orientation vectors.
"""

import math
import os
import sys
import time
import threading
import gc
from typing import Optional
from unittest.mock import MagicMock, call

import pytest

from report_parser import ControllerState
from app_core import ControllerManager
from gui import (
    DEFAULT_SETTINGS,
    Switch2ProApp,
    determine_imu_status,
    format_imu_telemetry,
    load_settings,
    save_settings,
)

# ============================================================================
# Tkinter Availability Probe for Headless Verification
# ============================================================================

try:
    import tkinter as tk
    _probe = tk.Tk()
    _probe.withdraw()
    _probe.destroy()
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


# ============================================================================
# Test Fixtures & Helpers
# ============================================================================

@pytest.fixture
def headless_app(tmp_path, monkeypatch):
    """Provides an isolated headless Switch2ProApp instance for UI testing."""
    if not TK_AVAILABLE:
        pytest.skip("Tkinter display context not available")

    # Ensure Tcl/Tk can locate initialization scripts in base python installation
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


def make_test_state(**kwargs) -> ControllerState:
    """Construct a full ControllerState with sane baseline defaults."""
    base = {
        "timestamp": 1000,
        "buttons_raw": 0,
        "a": False, "b": False, "x": False, "y": False,
        "plus": False, "minus": False, "home": False, "capture": False,
        "l": False, "r": False, "l_thumb": False, "r_thumb": False,
        "dpad_up": False, "dpad_down": False, "dpad_left": False, "dpad_right": False,
        "zl_analog": 0.0, "zr_analog": 0.0,
        "left_stick_x": 0.0, "left_stick_y": 0.0,
        "right_stick_x": 0.0, "right_stick_y": 0.0,
        "battery_percentage": 90,
        "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0,
        "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
        "raw_accel_x": 0, "raw_accel_y": 0, "raw_accel_z": 4096,
        "raw_gyro_x": 0, "raw_gyro_y": 0, "raw_gyro_z": 0,
        "motion_timestamp": 500000,
        "imu_temperature": 25.0,
        "has_motion": True,
    }
    base.update(kwargs)
    return ControllerState(**base)


# ============================================================================
# SUITE 1: 250 Hz Input Rate Stress & Decoupled Rendering Verification
# ============================================================================

class TestDecoupledRenderingAndInputRateStress:
    """Empirical verification of 250 Hz producer latency, thread isolation, and decoupled 60 FPS loop."""

    def test_producer_input_update_sub_microsecond_throughput(self, headless_app):
        """Verify that _on_input_update executes in sub-microsecond time over thousands of iterations.
        
        At 250 Hz, a frame arrives every 4,000 microseconds (4 ms). The producer thread must never
        be delayed by GUI rendering. Target: < 1.0 microsecond per update call (< 1000 ns).
        """
        app = headless_app
        iterations = 5000
        states = [make_test_state(accel_x=i * 0.001, gyro_z=i * 0.01) for i in range(iterations)]

        start_ns = time.perf_counter_ns()
        for st in states:
            app._on_input_update(st)
        elapsed_ns = time.perf_counter_ns() - start_ns

        avg_ns_per_call = elapsed_ns / iterations
        avg_us_per_call = avg_ns_per_call / 1000.0

        # Sub-microsecond execution guarantee (< 1.0 us = < 1000 ns)
        assert avg_us_per_call < 1.0, f"Average execution time {avg_us_per_call:.3f} µs exceeded 1.0 µs threshold"
        # Verify latest state reference is atomically retained
        assert app._latest_state is states[-1]

    def test_250hz_concurrent_producer_thread_stability(self, headless_app):
        """Simulate high-speed concurrent producer thread pumping 2,500 frames while main thread renders."""
        app = headless_app
        app.is_connected = True

        stop_event = threading.Event()
        producer_errors = []
        frame_count = 2500
        producer_durations = []

        def producer_worker():
            try:
                for i in range(frame_count):
                    st = make_test_state(
                        accel_x=(i % 100) * 0.01,
                        gyro_y=(i % 50) * 0.2,
                        motion_timestamp=i * 4000,
                    )
                    t0 = time.perf_counter_ns()
                    app._on_input_update(st)
                    t1 = time.perf_counter_ns()
                    producer_durations.append(t1 - t0)
            except Exception as e:
                producer_errors.append(e)

        thread = threading.Thread(target=producer_worker, name="HID_250Hz_Producer")
        thread.start()

        # Simultaneously simulate 60 FPS consumer visualizer ticks on main thread
        rendered_ticks = 0
        while thread.is_alive():
            if app._latest_state is not None:
                app._render_live_inputs_optimized(app._latest_state)
                rendered_ticks += 1
            time.sleep(0.001)

        thread.join(timeout=5.0)

        assert not thread.is_alive(), "Producer thread failed to complete within timeout"
        assert not producer_errors, f"Producer thread encountered exceptions: {producer_errors}"
        assert len(producer_durations) == frame_count

        avg_producer_ns = sum(producer_durations) / len(producer_durations)
        assert avg_producer_ns < 2000.0, f"Producer call average {avg_producer_ns:.2f} ns exceeded 2.0 µs under concurrency"
        assert rendered_ticks > 0, "Consumer thread did not render any frames"

    def test_visualizer_tick_exception_resilience_and_reschedule(self, headless_app):
        """Verify that _visualizer_tick guarantees re-scheduling via finally block even if render fails."""
        app = headless_app
        app.is_connected = True
        app._latest_state = make_test_state()

        # Mock render to raise an unexpected runtime error
        faulty_render = MagicMock(side_effect=RuntimeError("Transient Tcl GUI rendering fault"))
        app._render_live_inputs_optimized = faulty_render

        # Mock app.after to verify rescheduled timer
        after_mock = MagicMock(return_value="timer_id_999")
        app.after = after_mock

        # Run tick — must not propagate exception and must schedule next tick
        app._visualizer_tick()

        assert faulty_render.called
        after_mock.assert_called_with(16, app._visualizer_tick)
        assert app._visualizer_timer_id == "timer_id_999"

    def test_visualizer_tick_memory_leak_verification(self, headless_app):
        """Verify that thousands of visualizer rendering iterations execute with zero unbounded memory growth."""
        app = headless_app
        app.is_connected = True

        # Pre-warm JIT / caches
        for i in range(100):
            app._render_live_inputs_optimized(make_test_state(accel_x=0.01 * (i % 5)))

        gc.collect()

        # Track object count baseline
        initial_objects = len(gc.get_objects())

        # Run 1,000 active ticks with varying motion states
        for i in range(1000):
            st = make_test_state(
                accel_x=(i % 10) * 0.05,
                accel_y=(i % 8) * 0.03,
                accel_z=1.0 + (i % 6) * 0.04,
                gyro_x=(i % 12) * 1.5,
                gyro_y=(i % 15) * 2.0,
                gyro_z=(i % 7) * 0.8,
            )
            app._render_live_inputs_optimized(st)

        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory leak assertion: object growth should be bounded (< 100 net objects for internal closures)
        net_object_growth = final_objects - initial_objects
        assert net_object_growth < 150, f"Potential memory leak detected: net object growth was {net_object_growth}"

    def test_visualizer_tick_skips_when_disconnected_or_no_state(self, headless_app):
        """Verify that _visualizer_tick cleanly skips rendering when disconnected or state is None."""
        app = headless_app
        render_mock = MagicMock()
        app._render_live_inputs_optimized = render_mock

        # Case 1: is_connected = False, latest_state = valid
        app.is_connected = False
        app._latest_state = make_test_state()
        app._visualizer_tick()
        assert not render_mock.called

        # Case 2: is_connected = True, latest_state = None
        app.is_connected = True
        app._latest_state = None
        app._visualizer_tick()
        assert not render_mock.called


# ============================================================================
# SUITE 2: Deadband Dirty-Checking Rigor
# ============================================================================

class TestDeadbandDirtyCheckingRigor:
    """Stress-test sensor noise deadbands (|Δa| < 0.02g and |Δω| < 1.0°/s) and formatting accuracy."""

    def test_sub_threshold_jitter_suppressed_accel_below_0_02g(self, headless_app):
        """Verify that accelerometer sensor jitter below 0.02g deadband suppresses .configure() calls."""
        app = headless_app
        app.is_connected = True

        # Initialize baseline state at (0.0, 0.0, 1.0g)
        st_base = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.0, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        app._render_live_inputs_optimized(st_base)
        assert app._prev_ax == 0.0
        assert app._prev_ay == 0.0
        assert app._prev_az == 1.0

        # Spy on label configure
        spy_accel_configure = MagicMock(wraps=app.lbl_accel.configure)
        app.lbl_accel.configure = spy_accel_configure

        # Feed 30 sub-threshold jitter states (|Δa| < 0.02g on all axes)
        sub_threshold_deltas = [
            (0.015, -0.010, 0.018),
            (-0.019, 0.012, -0.015),
            (0.008, -0.018, 0.005),
            (-0.010, 0.010, -0.010),
            (0.019, 0.019, 0.019),
            (0.001, -0.002, 0.003),
        ] * 5

        for dx, dy, dz in sub_threshold_deltas:
            st = make_test_state(
                accel_x=0.0 + dx,
                accel_y=0.0 + dy,
                accel_z=1.0 + dz,
                gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
            )
            app._render_live_inputs_optimized(st)

        # Assertion: Label .configure() must NEVER have been called during sub-threshold jitter
        assert spy_accel_configure.call_count == 0, (
            f"Expected 0 configure calls for sub-threshold accel jitter, got {spy_accel_configure.call_count}"
        )
        # Verify internal caches were not updated
        assert app._prev_ax == 0.0
        assert app._prev_ay == 0.0
        assert app._prev_az == 1.0

    def test_sub_threshold_jitter_suppressed_gyro_below_1_0_deg_s(self, headless_app):
        """Verify that gyroscope sensor jitter below 1.0°/s deadband suppresses .configure() calls."""
        app = headless_app
        app.is_connected = True

        st_base = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.0, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        app._render_live_inputs_optimized(st_base)
        assert app._prev_gx == 0.0
        assert app._prev_gy == 0.0
        assert app._prev_gz == 0.0

        spy_gyro_configure = MagicMock(wraps=app.lbl_gyro.configure)
        app.lbl_gyro.configure = spy_gyro_configure

        # Feed 30 sub-threshold jitter states (|Δω| < 1.0°/s on all axes)
        sub_threshold_gyro = [
            (0.8, -0.5, 0.3),
            (-0.95, 0.7, -0.4),
            (0.2, -0.85, 0.9),
            (-0.6, 0.4, -0.7),
            (0.99, -0.99, 0.99),
        ] * 6

        for gx, gy, gz in sub_threshold_gyro:
            st = make_test_state(
                accel_x=0.0, accel_y=0.0, accel_z=1.0,
                gyro_x=gx, gyro_y=gy, gyro_z=gz,
            )
            app._render_live_inputs_optimized(st)

        assert spy_gyro_configure.call_count == 0, (
            f"Expected 0 configure calls for sub-threshold gyro jitter, got {spy_gyro_configure.call_count}"
        )
        assert app._prev_gx == 0.0
        assert app._prev_gy == 0.0
        assert app._prev_gz == 0.0

    def test_above_threshold_motion_accel_triggers_exact_formatting(self, headless_app):
        """Verify that accelerometer motion exceeding deadband (|Δa| > 0.02g) triggers configure with exact text."""
        app = headless_app
        app.is_connected = True

        # Baseline
        app._render_live_inputs_optimized(make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.0))

        spy_accel = MagicMock(wraps=app.lbl_accel.configure)
        spy_gyro = MagicMock(wraps=app.lbl_gyro.configure)
        app.lbl_accel.configure = spy_accel
        app.lbl_gyro.configure = spy_gyro

        # 1. Significant Accel X motion (0.05g > 0.02g), Gyro unchanged
        st_x = make_test_state(accel_x=0.05, accel_y=0.0, accel_z=1.0)
        app._render_live_inputs_optimized(st_x)

        assert spy_accel.call_count == 1
        assert spy_gyro.call_count == 0
        spy_accel.assert_called_with(text="X: +0.05g  Y: +0.00g  Z: +1.00g")
        assert app._prev_ax == 0.05

        # 2. Significant Accel Y motion (-0.12g > 0.02g)
        spy_accel.reset_mock()
        st_y = make_test_state(accel_x=0.05, accel_y=-0.12, accel_z=1.0)
        app._render_live_inputs_optimized(st_y)

        assert spy_accel.call_count == 1
        spy_accel.assert_called_with(text="X: +0.05g  Y: -0.12g  Z: +1.00g")
        assert app._prev_ay == -0.12

        # 3. Significant Accel Z motion (+0.85g)
        spy_accel.reset_mock()
        st_z = make_test_state(accel_x=0.05, accel_y=-0.12, accel_z=0.85)
        app._render_live_inputs_optimized(st_z)

        assert spy_accel.call_count == 1
        spy_accel.assert_called_with(text="X: +0.05g  Y: -0.12g  Z: +0.85g")
        assert app._prev_az == 0.85

    def test_above_threshold_motion_gyro_triggers_exact_formatting(self, headless_app):
        """Verify that gyroscope motion exceeding deadband (|Δω| > 1.0°/s) triggers configure with exact text."""
        app = headless_app
        app.is_connected = True

        # Baseline
        app._render_live_inputs_optimized(make_test_state(gyro_x=0.0, gyro_y=0.0, gyro_z=0.0))

        spy_accel = MagicMock(wraps=app.lbl_accel.configure)
        spy_gyro = MagicMock(wraps=app.lbl_gyro.configure)
        app.lbl_accel.configure = spy_accel
        app.lbl_gyro.configure = spy_gyro

        # 1. Significant Pitch motion (+14.5°/s > 1.0°/s)
        st_p = make_test_state(gyro_x=14.5, gyro_y=0.0, gyro_z=0.0)
        app._render_live_inputs_optimized(st_p)

        assert spy_gyro.call_count == 1
        assert spy_accel.call_count == 0
        spy_gyro.assert_called_with(text="P: +14.5  Y: +0.0  R: +0.0")
        assert app._prev_gx == 14.5

        # 2. Significant Yaw motion (-28.3°/s)
        spy_gyro.reset_mock()
        st_y = make_test_state(gyro_x=14.5, gyro_y=-28.3, gyro_z=0.0)
        app._render_live_inputs_optimized(st_y)

        assert spy_gyro.call_count == 1
        spy_gyro.assert_called_with(text="P: +14.5  Y: -28.3  R: +0.0")
        assert app._prev_gy == -28.3

        # 3. Significant Roll motion (+180.2°/s)
        spy_gyro.reset_mock()
        st_r = make_test_state(gyro_x=14.5, gyro_y=-28.3, gyro_z=180.2)
        app._render_live_inputs_optimized(st_r)

        assert spy_gyro.call_count == 1
        spy_gyro.assert_called_with(text="P: +14.5  Y: -28.3  R: +180.2")
        assert app._prev_gz == 180.2

    def test_deadband_boundary_conditions(self, headless_app):
        """Test exact boundary transitions around 0.02g and 1.0°/s."""
        app = headless_app
        app.is_connected = True

        app._render_live_inputs_optimized(make_test_state(accel_x=0.0, gyro_x=0.0))

        spy_accel = MagicMock(wraps=app.lbl_accel.configure)
        spy_gyro = MagicMock(wraps=app.lbl_gyro.configure)
        app.lbl_accel.configure = spy_accel
        app.lbl_gyro.configure = spy_gyro

        # Boundary: 0.019g delta (< 0.02) -> Suppressed
        app._render_live_inputs_optimized(make_test_state(accel_x=0.019))
        assert spy_accel.call_count == 0

        # Boundary: 0.025g delta (> 0.02) -> Triggered
        app._render_live_inputs_optimized(make_test_state(accel_x=0.025))
        assert spy_accel.call_count == 1
        assert app._prev_ax == 0.025

        # Boundary: 0.95°/s delta (< 1.0) -> Suppressed
        app._render_live_inputs_optimized(make_test_state(accel_x=0.025, gyro_x=0.95))
        assert spy_gyro.call_count == 0

        # Boundary: 1.05°/s delta (> 1.0) -> Triggered
        app._render_live_inputs_optimized(make_test_state(accel_x=0.025, gyro_x=1.05))
        assert spy_gyro.call_count == 1
        assert app._prev_gx == 1.05

    def test_has_motion_false_report_0x09_clears_labels_and_preserves_dirty_checking(self, headless_app):
        """Verify that Report 0x09 (has_motion=False) resets telemetry to '--' and doesn't re-trigger."""
        app = headless_app
        app.is_connected = True

        # First, display active motion state
        st_active = make_test_state(accel_x=0.5, gyro_x=20.0, has_motion=True)
        app._render_live_inputs_optimized(st_active)
        assert app._prev_ax == 0.5
        assert app._prev_gx == 20.0

        spy_accel = MagicMock(wraps=app.lbl_accel.configure)
        spy_gyro = MagicMock(wraps=app.lbl_gyro.configure)
        app.lbl_accel.configure = spy_accel
        app.lbl_gyro.configure = spy_gyro

        # Receive Report 0x09 without motion block
        st_no_motion = make_test_state(has_motion=False)
        app._render_live_inputs_optimized(st_no_motion)

        # Labels must be reset to placeholder
        assert spy_accel.call_count == 1
        assert spy_gyro.call_count == 1
        spy_accel.assert_called_with(text="X: --       Y: --       Z: --")
        spy_gyro.assert_called_with(text="P: --       Y: --       R: --")
        assert app._prev_ax == 0.0
        assert app._prev_gx == 0.0

        # Subsequent Report 0x09 packets should NOT re-trigger configure (dirty-check holds)
        spy_accel.reset_mock()
        spy_gyro.reset_mock()
        for _ in range(10):
            app._render_live_inputs_optimized(st_no_motion)

        assert spy_accel.call_count == 0
        assert spy_gyro.call_count == 0


# ============================================================================
# SUITE 3: Motion Status Pill State Machine Verification
# ============================================================================

class TestMotionStatusPillStateMachine:
    """Rigorous state machine testing for "○ IMU N/A", "● IMU IDLE", and "● IMU ACTIVE"."""

    def test_disconnected_and_no_motion_badge_state(self):
        """Verify that disconnected controller or packets without motion always yield ○ IMU N/A."""
        # Disconnected with high motion values
        state_high = make_test_state(gyro_x=50.0, accel_z=2.5, has_motion=True)
        badge, text_col, fg_col = determine_imu_status(state_high, is_connected=False)
        assert badge == "○ IMU N/A"
        assert text_col == "#6B7280"
        assert fg_col == "#262626"

        # Connected but None state
        badge, _, _ = determine_imu_status(None, is_connected=True)
        assert badge == "○ IMU N/A"

        # Connected but has_motion is False (Report 0x09)
        state_nomotion = make_test_state(has_motion=False)
        badge, _, _ = determine_imu_status(state_nomotion, is_connected=True)
        assert badge == "○ IMU N/A"

    def test_stationary_controller_gravity_only_orientations(self):
        """Verify that resting controller in any 3D gravity orientation yields ● IMU IDLE."""
        # 1. Standard flat upright (+Z)
        s1 = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.0, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        assert determine_imu_status(s1, True) == ("● IMU IDLE", "#9CA3AF", "#2C2C2E")

        # 2. Flat upside down (-Z)
        s2 = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=-1.0)
        assert determine_imu_status(s2, True)[0] == "● IMU IDLE"

        # 3. Standing upright on edge (+Y)
        s3 = make_test_state(accel_x=0.0, accel_y=1.0, accel_z=0.0)
        assert determine_imu_status(s3, True)[0] == "● IMU IDLE"

        # 4. Standing upside down (-Y)
        s4 = make_test_state(accel_x=0.0, accel_y=-1.0, accel_z=0.0)
        assert determine_imu_status(s4, True)[0] == "● IMU IDLE"

        # 5. Resting on left edge (+X)
        s5 = make_test_state(accel_x=1.0, accel_y=0.0, accel_z=0.0)
        assert determine_imu_status(s5, True)[0] == "● IMU IDLE"

        # 6. Resting on right edge (-X)
        s6 = make_test_state(accel_x=-1.0, accel_y=0.0, accel_z=0.0)
        assert determine_imu_status(s6, True)[0] == "● IMU IDLE"

        # 7. Tilted 45 degrees in pitch (Y & Z: 1/√2)
        inv_sqrt2 = 1.0 / math.sqrt(2)
        s7 = make_test_state(accel_x=0.0, accel_y=inv_sqrt2, accel_z=inv_sqrt2)
        assert determine_imu_status(s7, True)[0] == "● IMU IDLE"

        # 8. Compound 3D tilt (X, Y, Z: 1/√3)
        inv_sqrt3 = 1.0 / math.sqrt(3)
        s8 = make_test_state(accel_x=inv_sqrt3, accel_y=inv_sqrt3, accel_z=inv_sqrt3)
        assert determine_imu_status(s8, True)[0] == "● IMU IDLE"

        # 9. Stationary with real-world sensor calibration noise (0.985g total)
        s9 = make_test_state(accel_x=0.015, accel_y=-0.012, accel_z=0.985, gyro_x=0.3, gyro_y=-0.2, gyro_z=0.1)
        assert determine_imu_status(s9, True)[0] == "● IMU IDLE"

    def test_zero_vector_and_freefall_triggers_imu_active(self):
        """Verify that zero-gravity vector (freefall: 0g) triggers ● IMU ACTIVE (|Δa| = 1.0 > 0.08g)."""
        s_zero = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=0.0, gyro_x=0.0, gyro_y=0.0, gyro_z=0.0)
        badge, text_col, fg_col = determine_imu_status(s_zero, is_connected=True)
        assert badge == "● IMU ACTIVE"
        assert text_col == "#10B981"
        assert fg_col == "#064E3B"

    def test_gyro_threshold_transition_idle_to_active(self):
        """Verify gyro threshold boundary: magnitude <= 1.5°/s -> IDLE, > 1.5°/s -> ACTIVE."""
        # Sub-threshold: 1.45°/s magnitude on Pitch
        s_sub = make_test_state(gyro_x=1.45, gyro_y=0.0, gyro_z=0.0, accel_z=1.0)
        assert determine_imu_status(s_sub, True)[0] == "● IMU IDLE"

        # Above threshold: 1.55°/s magnitude on Pitch
        s_above = make_test_state(gyro_x=1.55, gyro_y=0.0, gyro_z=0.0, accel_z=1.0)
        assert determine_imu_status(s_above, True)[0] == "● IMU ACTIVE"

        # Multi-axis combination: 0.9°/s on each axis -> magnitude = sqrt(3 * 0.81) = 1.558 > 1.5
        s_multi = make_test_state(gyro_x=0.9, gyro_y=0.9, gyro_z=0.9, accel_z=1.0)
        assert determine_imu_status(s_multi, True)[0] == "● IMU ACTIVE"

    def test_accel_threshold_transition_idle_to_active(self):
        """Verify linear accel deviation threshold boundary: | |a| - 1.0 | <= 0.08g -> IDLE, > 0.08g -> ACTIVE."""
        # Sub-threshold: 1.07g magnitude (delta 0.07g <= 0.08g)
        s_sub = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.07, gyro_x=0.0)
        assert determine_imu_status(s_sub, True)[0] == "● IMU IDLE"

        # Above threshold: 1.09g magnitude (delta 0.09g > 0.08g)
        s_above = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=1.09, gyro_x=0.0)
        assert determine_imu_status(s_above, True)[0] == "● IMU ACTIVE"

        # Downward deviation: 0.91g magnitude (delta 0.09g > 0.08g)
        s_down = make_test_state(accel_x=0.0, accel_y=0.0, accel_z=0.91, gyro_x=0.0)
        assert determine_imu_status(s_down, True)[0] == "● IMU ACTIVE"

    def test_non_finite_floats_infinity_handling(self):
        """Verify that positive and negative infinity in motion fields trigger ACTIVE without unhandled errors."""
        # Gyro +Inf
        s_ginf = make_test_state(gyro_x=float("inf"), accel_z=1.0)
        assert determine_imu_status(s_ginf, True)[0] == "● IMU ACTIVE"

        # Gyro -Inf
        s_gneg = make_test_state(gyro_y=float("-inf"), accel_z=1.0)
        assert determine_imu_status(s_gneg, True)[0] == "● IMU ACTIVE"

        # Accel +Inf
        s_ainf = make_test_state(accel_z=float("inf"))
        assert determine_imu_status(s_ainf, True)[0] == "● IMU ACTIVE"

        # Accel -Inf
        s_aneg = make_test_state(accel_x=float("-inf"))
        assert determine_imu_status(s_aneg, True)[0] == "● IMU ACTIVE"

    def test_non_finite_floats_nan_handling(self):
        """Verify that NaN floats in motion fields evaluate cleanly without crashing."""
        # Gyro NaN
        s_gnan = make_test_state(gyro_x=float("nan"), accel_z=1.0)
        badge, text_col, fg_col = determine_imu_status(s_gnan, True)
        assert badge in ("● IMU IDLE", "● IMU ACTIVE")
        assert text_col.startswith("#")
        assert fg_col.startswith("#")

        # Accel NaN
        s_anan = make_test_state(accel_z=float("nan"))
        badge, text_col, fg_col = determine_imu_status(s_anan, True)
        assert badge in ("● IMU IDLE", "● IMU ACTIVE")

        # All NaN
        s_all_nan = make_test_state(
            accel_x=float("nan"), accel_y=float("nan"), accel_z=float("nan"),
            gyro_x=float("nan"), gyro_y=float("nan"), gyro_z=float("nan"),
        )
        badge, text_col, fg_col = determine_imu_status(s_all_nan, True)
        assert isinstance(badge, str)

    def test_motion_pill_ui_dirty_checking_suppresses_redundant_updates(self, headless_app):
        """Verify that imu_status_pill.configure() is ONLY called when the status text actually changes."""
        app = headless_app
        app.is_connected = True

        spy_pill = MagicMock(wraps=app.imu_status_pill.configure)
        app.imu_status_pill.configure = spy_pill

        # 1. First state: IDLE
        st_idle = make_test_state(accel_z=1.0, gyro_x=0.0)
        app._render_live_inputs_optimized(st_idle)
        assert spy_pill.call_count == 1
        spy_pill.assert_called_with(text="● IMU IDLE", text_color="#9CA3AF", fg_color="#2C2C2E")

        # 2. 50 more IDLE states: pill configure must NOT be called again
        spy_pill.reset_mock()
        for i in range(50):
            st_idle_jitter = make_test_state(accel_z=1.0 + (i % 3) * 0.01, gyro_x=(i % 4) * 0.1)
            app._render_live_inputs_optimized(st_idle_jitter)
        assert spy_pill.call_count == 0

        # 3. Transition to ACTIVE: pill configure called exactly once
        st_active = make_test_state(accel_z=1.0, gyro_x=10.0)
        app._render_live_inputs_optimized(st_active)
        assert spy_pill.call_count == 1
        spy_pill.assert_called_with(text="● IMU ACTIVE", text_color="#10B981", fg_color="#064E3B")

        # 4. 50 more ACTIVE states: pill configure must NOT be called again
        spy_pill.reset_mock()
        for i in range(50):
            st_active_jitter = make_test_state(accel_z=1.0, gyro_x=10.0 + i)
            app._render_live_inputs_optimized(st_active_jitter)
        assert spy_pill.call_count == 0

        # 5. Transition back to IDLE: pill configure called exactly once
        app._render_live_inputs_optimized(st_idle)
        assert spy_pill.call_count == 1
        spy_pill.assert_called_with(text="● IMU IDLE", text_color="#9CA3AF", fg_color="#2C2C2E")


# ============================================================================
# SUITE 4: Telemetry Formatting Consistency & Stress
# ============================================================================

class TestTelemetryFormattingConsistency:
    """Rigorous verification of Consolas telemetry text formatting and boundary numbers."""

    def test_exact_spacing_and_symbols(self):
        """Verify exact character layout, spacing, and decimal digits in telemetry strings."""
        st = make_test_state(
            accel_x=0.0, accel_y=-1.0, accel_z=0.05,
            gyro_x=0.0, gyro_y=-5.4, gyro_z=120.7,
        )
        accel_str, gyro_str = format_imu_telemetry(st)

        assert accel_str == "X: +0.00g  Y: -1.00g  Z: +0.05g"
        assert gyro_str == "P: +0.0  Y: -5.4  R: +120.7"

    def test_extreme_and_negative_telemetry_values(self):
        """Verify format handles full range of signed 16-bit physical sensor extremes."""
        st_max = make_test_state(
            accel_x=8.0, accel_y=-8.0, accel_z=8.0,
            gyro_x=2000.0, gyro_y=-2000.0, gyro_z=2000.0,
        )
        accel_str, gyro_str = format_imu_telemetry(st_max)
        assert "X: +8.00g" in accel_str
        assert "Y: -8.00g" in accel_str
        assert "P: +2000.0" in gyro_str
        assert "Y: -2000.0" in gyro_str
