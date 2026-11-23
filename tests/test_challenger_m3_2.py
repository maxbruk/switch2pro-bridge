"""Empirical Adversarial Verification Suite for Milestone 3 Iteration 2.

Authored by: M3 Adversarial Challenger 2.2 (teamwork_challenger_m3_2_2)
Purpose: Independent empirical verification of:
1. Differential Stick Oracle (Xbox vs DS4 stick conversion, monotonicity, neutral centering, boundary clamping)
2. Non-finite Fuzzing (NaN, Inf, subnormals, out-of-range floats across state space)
3. Concurrency Stress (multi-threaded update_from_state, concurrent connect/disconnect, paddle remapping)
"""

import math
import random
import sys
import threading
import time
from typing import List
import pytest

from report_parser import ControllerState
from virtual_ds4 import (
    VirtualDS4Controller,
    _norm_stick_to_byte,
    _norm_stick_y_inverted_to_byte,
    _norm_trigger_to_byte,
    _clamp_int16,
    _accel_to_int16,
    _gyro_to_int16,
    _get_dpad_direction,
)
from virtual_xinput import VirtualXboxController, _sanitize_float
from tests.test_virtual_controllers import MockVDS4Gamepad, MockVX360Gamepad, make_state


# ============================================================================
# 1. Differential Stick Oracle
# ============================================================================

class TestDifferentialStickOracle:
    """Rigorous differential oracle comparing Xbox 360 and DS4 stick transformations."""

    def test_dense_monotonicity_and_neutral_oracle(self):
        """Evaluate 100,001 points across [-2.5, 2.5] proving monotonicity, neutral, and zero clipping."""
        num_points = 100001
        step = 5.0 / (num_points - 1)
        inputs = [-2.5 + i * step for i in range(num_points)]

        prev_xbox_x = -999999
        prev_ds4_x = -1
        prev_ds4_y_inv = 999

        for x in inputs:
            # 1. Xbox conversion
            sanitized_xbox = _sanitize_float(x, min_val=-1.0, max_val=1.0, default=0.0)
            xbox_counts = int(round(sanitized_xbox * 32767))

            # 2. DS4 conversions
            ds4_x_byte = _norm_stick_to_byte(x)
            ds4_y_inv_byte = _norm_stick_y_inverted_to_byte(x)

            # Invariant 1: Bounds
            assert -32767 <= xbox_counts <= 32767
            assert 0 <= ds4_x_byte <= 255
            assert 0 <= ds4_y_inv_byte <= 255

            # Invariant 2: Monotonicity
            assert xbox_counts >= prev_xbox_x, f"Xbox monotonicity broken at {x}: {xbox_counts} < {prev_xbox_x}"
            assert ds4_x_byte >= prev_ds4_x, f"DS4 X monotonicity broken at {x}: {ds4_x_byte} < {prev_ds4_x}"
            assert ds4_y_inv_byte <= prev_ds4_y_inv, f"DS4 Y inversion monotonicity broken at {x}: {ds4_y_inv_byte} > {prev_ds4_y_inv}"

            # Invariant 3: Zero clipping outside [-1.0, 1.0]
            if x <= -1.0:
                assert xbox_counts == -32767, f"Xbox failed to clip negative extreme at {x}: got {xbox_counts}"
                assert ds4_x_byte == 0, f"DS4 X failed to clip negative extreme at {x}: got {ds4_x_byte}"
                assert ds4_y_inv_byte == 255, f"DS4 Y failed to clip negative extreme at {x}: got {ds4_y_inv_byte}"
            elif x >= 1.0:
                assert xbox_counts == 32767, f"Xbox failed to clip positive extreme at {x}: got {xbox_counts}"
                assert ds4_x_byte == 255, f"DS4 X failed to clip positive extreme at {x}: got {ds4_x_byte}"
                assert ds4_y_inv_byte == 0, f"DS4 Y failed to clip positive extreme at {x}: got {ds4_y_inv_byte}"

            prev_xbox_x = xbox_counts
            prev_ds4_x = ds4_x_byte
            prev_ds4_y_inv = ds4_y_inv_byte

    def test_exact_neutral_alignment(self):
        """Exact neutral alignment for 0.0, -0.0, and machine-epsilon values."""
        for zero_val in (0.0, -0.0, 1e-320, -1e-320):
            # Xbox
            xbox_val = _sanitize_float(zero_val, -1.0, 1.0, default=0.0)
            assert xbox_val == 0.0 or abs(xbox_val) < 1e-15
            assert int(round(xbox_val * 32767)) == 0

            # DS4
            ds4_x = _norm_stick_to_byte(zero_val)
            ds4_y = _norm_stick_y_inverted_to_byte(zero_val)
            assert ds4_x == 128, f"DS4 X neutral center was {ds4_x}, expected 128"
            assert ds4_y == 128, f"DS4 Y neutral center was {ds4_y}, expected 128"

    def test_differential_directional_agreement(self):
        """Differential directional agreement between Xbox and DS4 controllers."""
        test_points = [-1.0, -0.75, -0.5, -0.25, -0.01, 0.01, 0.25, 0.5, 0.75, 1.0]
        for val in test_points:
            xbox_cnt = int(round(_sanitize_float(val) * 32767))
            ds4_x = _norm_stick_to_byte(val)
            ds4_y = _norm_stick_y_inverted_to_byte(val)

            if val > 0:
                assert xbox_cnt > 0
                assert ds4_x > 128
                assert ds4_y < 128  # Sony inverted: up is toward 0
            else:
                assert xbox_cnt < 0
                assert ds4_x < 128
                assert ds4_y > 128  # Sony inverted: down is toward 255

    def test_stick_adc_jitter_neutral_stability(self):
        """Simulate micro-jitter (+/- 0.001 to +/- 0.003) around neutral center."""
        rng = random.Random(999)
        for _ in range(1000):
            jitter = rng.uniform(-0.003, 0.003)
            # Xbox
            xbox_cnt = int(round(_sanitize_float(jitter) * 32767))
            assert abs(xbox_cnt) <= 99  # Micro-deviation within deadzone tolerance

            # DS4
            ds4_x = _norm_stick_to_byte(jitter)
            ds4_y = _norm_stick_y_inverted_to_byte(jitter)
            assert ds4_x in (127, 128, 129)
            assert ds4_y in (127, 128, 129)


# ============================================================================
# 2. Non-Finite Fuzzing Harness
# ============================================================================

class TestNonFiniteFuzzingHarness:
    """Injects high volumes of non-finite, subnormal, and adversarial float inputs."""

    ADVERSARIAL_FLOATS = [
        float("nan"),
        float("-nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -0.0,
        1e-308,
        -1e-308,
        1e-320,
        -1e-320,
        sys.float_info.min,
        sys.float_info.max,
        -sys.float_info.max,
        1e300,
        -1e300,
        1.0,
        -1.0,
        1.5,
        -1.5,
        100.0,
        -100.0,
    ]

    ADVERSARIAL_INTS = [
        0,
        1,
        -1,
        32767,
        -32768,
        32768,
        -32769,
        65535,
        -65536,
        2147483647,
        -2147483648,
    ]

    def test_fuzz_virtual_ds4_controller(self):
        """Inject 5,000 randomized adversarial states into VirtualDS4Controller."""
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        rng = random.Random(42)

        for _ in range(5000):
            st = make_state(
                left_stick_x=rng.choice(self.ADVERSARIAL_FLOATS),
                left_stick_y=rng.choice(self.ADVERSARIAL_FLOATS),
                right_stick_x=rng.choice(self.ADVERSARIAL_FLOATS),
                right_stick_y=rng.choice(self.ADVERSARIAL_FLOATS),
                zl_analog=rng.choice(self.ADVERSARIAL_FLOATS),
                zr_analog=rng.choice(self.ADVERSARIAL_FLOATS),
                accel_x=rng.choice(self.ADVERSARIAL_FLOATS),
                accel_y=rng.choice(self.ADVERSARIAL_FLOATS),
                accel_z=rng.choice(self.ADVERSARIAL_FLOATS),
                gyro_x=rng.choice(self.ADVERSARIAL_FLOATS),
                gyro_y=rng.choice(self.ADVERSARIAL_FLOATS),
                gyro_z=rng.choice(self.ADVERSARIAL_FLOATS),
                raw_accel_x=rng.choice(self.ADVERSARIAL_INTS),
                raw_accel_y=rng.choice(self.ADVERSARIAL_INTS),
                raw_accel_z=rng.choice(self.ADVERSARIAL_INTS),
                raw_gyro_x=rng.choice(self.ADVERSARIAL_INTS),
                raw_gyro_y=rng.choice(self.ADVERSARIAL_INTS),
                raw_gyro_z=rng.choice(self.ADVERSARIAL_INTS),
                timestamp=rng.choice(self.ADVERSARIAL_INTS),
                motion_timestamp=rng.choice(self.ADVERSARIAL_INTS),
                battery_percentage=rng.choice([-10, 0, 50, 100, 200]),
            )
            c.update_from_state(st)
            snap = mock_gp.last_report_snapshot

            assert snap is not None
            # Validate output types and ranges
            assert 0 <= snap.bThumbLX <= 255
            assert 0 <= snap.bThumbLY <= 255
            assert 0 <= snap.bThumbRX <= 255
            assert 0 <= snap.bThumbRY <= 255
            assert 0 <= snap.bTriggerL <= 255
            assert 0 <= snap.bTriggerR <= 255
            assert -32768 <= snap.wAccelX <= 32767
            assert -32768 <= snap.wAccelY <= 32767
            assert -32768 <= snap.wAccelZ <= 32767
            assert -32768 <= snap.wGyroX <= 32767
            assert -32768 <= snap.wGyroY <= 32767
            assert -32768 <= snap.wGyroZ <= 32767
            assert 0 <= snap.bBatteryLvl <= 10
            assert 0 <= snap.wTimestamp <= 65535

    def test_fuzz_virtual_xbox_controller(self):
        """Inject 5,000 randomized adversarial states into VirtualXboxController."""
        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        rng = random.Random(1337)

        for _ in range(5000):
            st = make_state(
                left_stick_x=rng.choice(self.ADVERSARIAL_FLOATS),
                left_stick_y=rng.choice(self.ADVERSARIAL_FLOATS),
                right_stick_x=rng.choice(self.ADVERSARIAL_FLOATS),
                right_stick_y=rng.choice(self.ADVERSARIAL_FLOATS),
                zl_analog=rng.choice(self.ADVERSARIAL_FLOATS),
                zr_analog=rng.choice(self.ADVERSARIAL_FLOATS),
                a=rng.choice([True, False]),
                b=rng.choice([True, False]),
                gl=rng.choice([True, False]),
                gr=rng.choice([True, False]),
            )
            c.update_from_state(st)

        assert mock_gp.update_count == 5000


# ============================================================================
# 3. Concurrency Stress Harness
# ============================================================================

class TestConcurrencyStressHarness:
    """Stress tests thread safety under concurrent updates and dynamic lifecycle changes."""

    def test_concurrent_multithreaded_updates_ds4(self):
        """10 threads concurrently pumping 1,000 updates each into DS4 controller."""
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        num_threads = 10
        updates_per_thread = 1000
        errors: List[Exception] = []

        def worker(tid: int):
            for i in range(updates_per_thread):
                st = make_state(
                    left_stick_x=(i % 100) / 100.0,
                    accel_z=1.0,
                    a=(i % 2 == 0),
                )
                try:
                    c.update_from_state(st)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Concurrent updates failed: {errors}"
        assert mock_gp.update_ex_count == num_threads * updates_per_thread

    def test_concurrent_lifecycle_thrashing_ds4(self):
        """Thrash connect/disconnect while 4 updater threads push states simultaneously."""
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        stop_event = threading.Event()
        errors: List[Exception] = []

        def updater():
            st = make_state(a=True, accel_z=1.0, left_stick_x=0.5)
            while not stop_event.is_set():
                try:
                    c.update_from_state(st)
                except Exception as e:
                    errors.append(e)

        def lifecycle_cycler():
            while not stop_event.is_set():
                try:
                    c.connect()
                    c._gamepad = mock_gp
                    time.sleep(0.001)
                    c.disconnect()
                except Exception as e:
                    errors.append(e)

        def remapper():
            paddles = ["L3", "R3", "LB", "RB", "A", "B", "X", "Y", "LT", "RT"]
            while not stop_event.is_set():
                try:
                    c.gl_mapping = random.choice(paddles)
                    c.gr_mapping = random.choice(paddles)
                    c.layout = random.choice(["nintendo", "xbox"])
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=updater),
            threading.Thread(target=lifecycle_cycler),
            threading.Thread(target=remapper),
        ]

        for t in threads:
            t.start()

        time.sleep(0.6)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        assert len(errors) == 0, f"Lifecycle concurrency thrashing produced errors: {errors}"

    def test_concurrent_lifecycle_thrashing_xbox(self):
        """Thrash connect/disconnect on Xbox controller while 4 threads push updates."""
        c = VirtualXboxController()
        mock_gp = MockVX360Gamepad()
        stop_event = threading.Event()
        errors: List[Exception] = []

        def updater():
            st = make_state(a=True, left_stick_x=0.5)
            while not stop_event.is_set():
                try:
                    c.update_from_state(st)
                except Exception as e:
                    errors.append(e)

        def lifecycle_cycler():
            while not stop_event.is_set():
                try:
                    c.connect()
                    c._gamepad = mock_gp
                    time.sleep(0.001)
                    c.disconnect()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=updater),
            threading.Thread(target=updater),
            threading.Thread(target=lifecycle_cycler),
        ]

        for t in threads:
            t.start()

        time.sleep(0.6)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        assert len(errors) == 0, f"Xbox lifecycle concurrency produced errors: {errors}"

    def test_full_lifecycle_state_recovery_invariant(self):
        """Verify that after disconnect and reconnect, state updates restore exact controller mappings."""
        c = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        c._gamepad = mock_gp
        c.is_connected = True

        # Initial update
        st1 = make_state(a=True, left_stick_x=1.0, accel_z=1.0)
        c.update_from_state(st1)
        snap1 = mock_gp.last_report_snapshot
        assert snap1.bThumbLX == 255
        assert snap1.wAccelZ == 8192

        # Disconnect
        c.disconnect()
        assert c.is_connected is False
        assert c._gamepad is None

        # Updates while disconnected must be clean no-ops
        c.update_from_state(st1)

        # Reconnect
        mock_gp2 = MockVDS4Gamepad()
        c._gamepad = mock_gp2
        c.is_connected = True

        st2 = make_state(b=True, left_stick_x=-1.0, accel_z=-1.0)
        c.update_from_state(st2)
        snap2 = mock_gp2.last_report_snapshot
        assert snap2.bThumbLX == 0
        assert snap2.wAccelZ == -8192
