"""Adversarial and Differential Oracle Test Harness for Virtual DS4 & Gamepad Abstraction.

Empirical verification of:
1. Bitwise button invariance: exhaustive 2^14 truth table search, zero bit collisions.
2. D-Pad hat encoding: all 16 combinations, PlayStation standard & cancellation.
3. Analog trigger resolution: 256 discrete values, 0..255 linearity, > 0.5 click bit threshold.
4. Raw vs normalized IMU preference matrix: raw-only, norm-only, both, neither.
5. Fuzzing & invariant tests: stick Y monotonicity, non-finite floats, grip paddles.
6. Thread safety stress test on concurrent updates.
"""

import math
import random
import threading
from typing import List, Optional, Tuple

import pytest

from report_parser import ControllerState
import virtual_ds4
from virtual_ds4 import VirtualDS4Controller, _get_dpad_direction, _norm_trigger_to_byte, _norm_stick_y_inverted_to_byte
from tests.test_virtual_controllers import MockVDS4Gamepad, make_state


# ============================================================================
# 1. Exhaustive Bitwise Button Invariance & Collision Oracle (2^14 Combinations)
# ============================================================================

class TestBitwiseButtonInvarianceOracle:
    """Exhaustive truth table test verifying zero bit collisions across all button combinations."""

    # Reference DS4 bit masks from vigem / vgamepad
    DS4_BTN_SQUARE = 0x0010
    DS4_BTN_CROSS = 0x0020
    DS4_BTN_CIRCLE = 0x0040
    DS4_BTN_TRIANGLE = 0x0080
    DS4_BTN_SHOULDER_LEFT = 0x0100
    DS4_BTN_SHOULDER_RIGHT = 0x0200
    DS4_BTN_TRIGGER_LEFT = 0x0400
    DS4_BTN_TRIGGER_RIGHT = 0x0800
    DS4_BTN_SHARE = 0x1000
    DS4_BTN_OPTIONS = 0x2000
    DS4_BTN_THUMB_LEFT = 0x4000
    DS4_BTN_THUMB_RIGHT = 0x8000

    def test_2_pow_14_combinations_buttons_and_dpad_nintendo_layout(self):
        """Test all 2^14 (16,384) combinations of 10 digital buttons + 4 D-pad flags in Nintendo layout.

        Ensures zero bit collisions in wButtons and exact bitwise matching against the reference oracle.
        """
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True
        controller.layout = "nintendo"

        # 10 digital buttons (bits 0..9) + 4 dpad flags (bits 10..13) = 14 bits -> 16384 states
        button_names = ["b", "a", "y", "x", "l", "r", "plus", "minus", "l_thumb", "r_thumb"]
        nintendo_expected_bits = [
            self.DS4_BTN_CROSS,         # B -> Cross
            self.DS4_BTN_CIRCLE,        # A -> Circle
            self.DS4_BTN_SQUARE,        # Y -> Square
            self.DS4_BTN_TRIANGLE,      # X -> Triangle
            self.DS4_BTN_SHOULDER_LEFT, # L -> L1
            self.DS4_BTN_SHOULDER_RIGHT,# R -> R1
            self.DS4_BTN_OPTIONS,       # Plus -> Options
            self.DS4_BTN_SHARE,         # Minus -> Share
            self.DS4_BTN_THUMB_LEFT,    # L_Thumb -> L3
            self.DS4_BTN_THUMB_RIGHT,   # R_Thumb -> R3
        ]

        seen_wbuttons = set()

        for idx in range(16384):
            # Extract 10 button flags
            btn_flags = [(idx & (1 << b)) != 0 for b in range(10)]
            # Extract 4 dpad flags (up, down, left, right)
            dpad_up = bool(idx & (1 << 10))
            dpad_down = bool(idx & (1 << 11))
            dpad_left = bool(idx & (1 << 12))
            dpad_right = bool(idx & (1 << 13))

            kwargs = {name: flag for name, flag in zip(button_names, btn_flags)}
            kwargs.update({
                "dpad_up": dpad_up,
                "dpad_down": dpad_down,
                "dpad_left": dpad_left,
                "dpad_right": dpad_right,
            })
            state = make_state(**kwargs)

            controller.update_from_state(state, layout="nintendo")
            actual_wbuttons = mock_gp.last_report_snapshot.wButtons

            # Reference oracle calculation
            expected_mask = 0
            for flag, bit in zip(btn_flags, nintendo_expected_bits):
                if flag:
                    expected_mask |= bit

            expected_dpad = _get_dpad_direction(dpad_up, dpad_down, dpad_left, dpad_right)
            expected_wbuttons = (expected_mask & ~0xF) | (expected_dpad & 0xF)

            assert actual_wbuttons == expected_wbuttons, (
                f"Mismatch at index {idx:014b}: expected {hex(expected_wbuttons)}, got {hex(actual_wbuttons)}"
            )

            # Check that upper 12 bits and lower 4 bits are strictly partitioned
            assert (actual_wbuttons & 0xF) == (expected_dpad & 0xF)
            assert (actual_wbuttons & ~0xF) == expected_mask

    def test_2_pow_14_combinations_buttons_and_dpad_xbox_layout(self):
        """Test all 2^14 (16,384) combinations of 10 digital buttons + 4 D-pad flags in Xbox layout."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True
        controller.layout = "xbox"

        button_names = ["a", "b", "x", "y", "l", "r", "plus", "minus", "l_thumb", "r_thumb"]
        xbox_expected_bits = [
            self.DS4_BTN_CROSS,         # A -> Cross (letter match)
            self.DS4_BTN_CIRCLE,        # B -> Circle
            self.DS4_BTN_SQUARE,        # X -> Square
            self.DS4_BTN_TRIANGLE,      # Y -> Triangle
            self.DS4_BTN_SHOULDER_LEFT, # L -> L1
            self.DS4_BTN_SHOULDER_RIGHT,# R -> R1
            self.DS4_BTN_OPTIONS,       # Plus -> Options
            self.DS4_BTN_SHARE,         # Minus -> Share
            self.DS4_BTN_THUMB_LEFT,    # L_Thumb -> L3
            self.DS4_BTN_THUMB_RIGHT,   # R_Thumb -> R3
        ]

        for idx in range(16384):
            btn_flags = [(idx & (1 << b)) != 0 for b in range(10)]
            dpad_up = bool(idx & (1 << 10))
            dpad_down = bool(idx & (1 << 11))
            dpad_left = bool(idx & (1 << 12))
            dpad_right = bool(idx & (1 << 13))

            kwargs = {name: flag for name, flag in zip(button_names, btn_flags)}
            kwargs.update({
                "dpad_up": dpad_up,
                "dpad_down": dpad_down,
                "dpad_left": dpad_left,
                "dpad_right": dpad_right,
            })
            state = make_state(**kwargs)

            controller.update_from_state(state, layout="xbox")
            actual_wbuttons = mock_gp.last_report_snapshot.wButtons

            expected_mask = 0
            for flag, bit in zip(btn_flags, xbox_expected_bits):
                if flag:
                    expected_mask |= bit

            expected_dpad = _get_dpad_direction(dpad_up, dpad_down, dpad_left, dpad_right)
            expected_wbuttons = (expected_mask & ~0xF) | (expected_dpad & 0xF)

            assert actual_wbuttons == expected_wbuttons

    def test_2_pow_12_digital_buttons_and_triggers_bijection(self):
        """Test all 2^12 (4,096) upper digital button combinations (face + shoulders + clicks + center + triggers).

        Proves that the mapping is a strict bijection with ZERO bit collisions.
        """
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        observed_masks = set()

        for idx in range(4096):
            # 12 bits:
            # 0: B, 1: A, 2: Y, 3: X
            # 4: L, 5: R
            # 6: Plus, 7: Minus
            # 8: L_Thumb, 9: R_Thumb
            # 10: ZL (>0.5), 11: ZR (>0.5)
            b = bool(idx & (1 << 0))
            a = bool(idx & (1 << 1))
            y = bool(idx & (1 << 2))
            x = bool(idx & (1 << 3))
            l = bool(idx & (1 << 4))
            r = bool(idx & (1 << 5))
            plus = bool(idx & (1 << 6))
            minus = bool(idx & (1 << 7))
            l_thumb = bool(idx & (1 << 8))
            r_thumb = bool(idx & (1 << 9))
            zl_val = 1.0 if (idx & (1 << 10)) else 0.0
            zr_val = 1.0 if (idx & (1 << 11)) else 0.0

            state = make_state(
                b=b, a=a, y=y, x=x,
                l=l, r=r, plus=plus, minus=minus,
                l_thumb=l_thumb, r_thumb=r_thumb,
                zl_analog=zl_val, zr_analog=zr_val,
            )
            controller.update_from_state(state, layout="nintendo")
            snap = mock_gp.last_report_snapshot

            upper_bits = snap.wButtons & ~0xF
            assert upper_bits not in observed_masks, f"Collision detected for upper bits: {hex(upper_bits)}"
            observed_masks.add(upper_bits)

        # 4096 distinct masks observed
        assert len(observed_masks) == 4096

    def test_special_buttons_isolation_from_wbuttons(self):
        """Verify Home and Capture buttons map to bSpecial and never pollute wButtons."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        # Test all 4 combinations of (home, capture)
        for home in (False, True):
            for capture in (False, True):
                state = make_state(home=home, capture=capture)
                controller.update_from_state(state)
                snap = mock_gp.last_report_snapshot

                # wButtons should only have DPad neutral (0x8) and 0 digital buttons
                assert snap.wButtons == 0x8
                expected_special = (0x01 if home else 0) | (0x02 if capture else 0)
                assert snap.bSpecial == expected_special


# ============================================================================
# 2. D-Pad Hat Encoding: All 16 Combinations & PlayStation Standard
# ============================================================================

class TestDpadHatEncodingOracle:
    """Verifies all 16 combinations of (up, down, left, right) against the PlayStation standard."""

    def test_all_16_dpad_combinations_against_oracle(self):
        """Test exhaustive truth table of all 16 D-Pad input combinations."""
        # PlayStation D-Pad standard truth table:
        # 0x0: North (Up)
        # 0x1: NorthEast (Up + Right)
        # 0x2: East (Right)
        # 0x3: SouthEast (Down + Right)
        # 0x4: South (Down)
        # 0x5: SouthWest (Down + Left)
        # 0x6: West (Left)
        # 0x7: NorthWest (Up + Left)
        # 0x8: None (Neutral or Canceled)
        oracle_table = {
            # (up, down, left, right): expected_dpad
            (False, False, False, False): 0x8,  # None
            (True,  False, False, False): 0x0,  # North
            (False, True,  False, False): 0x4,  # South
            (False, False, True,  False): 0x6,  # West
            (False, False, False, True ): 0x2,  # East
            (True,  False, False, True ): 0x1,  # NorthEast
            (False, True,  False, True ): 0x3,  # SouthEast
            (False, True,  True,  False): 0x5,  # SouthWest
            (True,  False, True,  False): 0x7,  # NorthWest
            # Opposing cancellation cases:
            (True,  True,  False, False): 0x8,  # Up + Down -> None
            (False, False, True,  True ): 0x8,  # Left + Right -> None
            (True,  True,  False, True ): 0x2,  # Up + Down cancel -> East
            (True,  True,  True,  False): 0x6,  # Up + Down cancel -> West
            (True,  False, True,  True ): 0x0,  # Left + Right cancel -> North
            (False, True,  True,  True ): 0x4,  # Left + Right cancel -> South
            (True,  True,  True,  True ): 0x8,  # All cancel -> None
        }

        assert len(oracle_table) == 16, "Must test exactly 16 combinations."

        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        for (up, down, left, right), expected in oracle_table.items():
            # 1. Test helper function directly
            direct_val = _get_dpad_direction(up, down, left, right)
            assert direct_val == expected, (
                f"_get_dpad_direction({up}, {down}, {left}, {right}) returned {direct_val}, expected {expected}"
            )

            # 2. Test full controller report update
            state = make_state(dpad_up=up, dpad_down=down, dpad_left=left, dpad_right=right)
            controller.update_from_state(state)
            actual_hat = mock_gp.last_report_snapshot.wButtons & 0xF
            assert actual_hat == expected, (
                f"Controller D-Pad mismatch for ({up}, {down}, {left}, {right}): got {actual_hat}, expected {expected}"
            )


# ============================================================================
# 3. Analog Trigger Resolution & Click Threshold (> 0.5)
# ============================================================================

class TestAnalogTriggerResolutionAndThresholdOracle:
    """Verifies 256 discrete trigger values, linear mapping to 0..255, and > 0.5 click bit."""

    def test_256_discrete_trigger_values_linear_resolution(self):
        """Test all 256 discrete values i/255 for i in range(256).

        Verifies:
        1. bTriggerL and bTriggerR match i exactly for all i in [0..255].
        2. Threshold click bit is strictly 0 for val <= 0.5, and 1 for val > 0.5.
        """
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        DS4_BTN_TRIGGER_LEFT = 0x0400
        DS4_BTN_TRIGGER_RIGHT = 0x0800

        for i in range(256):
            norm_val = i / 255.0

            # Direct mapping check
            byte_val = _norm_trigger_to_byte(norm_val)
            assert byte_val == i, f"Resolution error at {i}: got {byte_val}"

            # Controller update check
            state = make_state(zl_analog=norm_val, zr_analog=norm_val)
            controller.update_from_state(state)
            snap = mock_gp.last_report_snapshot

            assert snap.bTriggerL == i, f"ZL mismatch at {i}: got {snap.bTriggerL}"
            assert snap.bTriggerR == i, f"ZR mismatch at {i}: got {snap.bTriggerR}"

            # Threshold click bit check (> 0.5)
            click_l = bool(snap.wButtons & DS4_BTN_TRIGGER_LEFT)
            click_r = bool(snap.wButtons & DS4_BTN_TRIGGER_RIGHT)
            expected_click = norm_val > 0.5

            assert click_l == expected_click, f"ZL click bit error at i={i} (val={norm_val}): got {click_l}, expected {expected_click}"
            assert click_r == expected_click, f"ZR click bit error at i={i} (val={norm_val}): got {click_r}, expected {expected_click}"

    def test_trigger_boundary_click_threshold_epsilon(self):
        """Test values around the exact 0.5 boundary."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        DS4_BTN_TRIGGER_LEFT = 0x0400

        # Exact 0.5: should NOT set click bit (strictly > 0.5)
        st = make_state(zl_analog=0.5)
        controller.update_from_state(st)
        assert not (mock_gp.last_report_snapshot.wButtons & DS4_BTN_TRIGGER_LEFT)
        assert mock_gp.last_report_snapshot.bTriggerL == 127

        # 0.5 - 1e-7: should NOT set click bit
        st = make_state(zl_analog=0.5 - 1e-7)
        controller.update_from_state(st)
        assert not (mock_gp.last_report_snapshot.wButtons & DS4_BTN_TRIGGER_LEFT)

        # 0.5 + 1e-7: MUST set click bit
        st = make_state(zl_analog=0.5 + 1e-7)
        controller.update_from_state(st)
        assert bool(mock_gp.last_report_snapshot.wButtons & DS4_BTN_TRIGGER_LEFT)

    def test_trigger_out_of_bounds_clamping(self):
        """Test negative and oversaturated trigger values."""
        assert _norm_trigger_to_byte(-0.5) == 0
        assert _norm_trigger_to_byte(-100.0) == 0
        assert _norm_trigger_to_byte(1.5) == 255
        assert _norm_trigger_to_byte(100.0) == 255


# ============================================================================
# 4. Raw vs Normalized IMU Preference Matrix
# ============================================================================

class TestRawVsNormalizedIMUPreferenceOracle:
    """Verifies the 4 states of IMU precedence: raw-only, norm-only, both, neither."""

    def test_imu_raw_only_state(self):
        """State 1: Only raw_* is populated; normalized is 0.0."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            raw_accel_x=4096, raw_accel_y=-2048, raw_accel_z=8192,
            raw_gyro_x=720, raw_gyro_y=-1440, raw_gyro_z=2880,
            accel_x=0.0, accel_y=0.0, accel_z=0.0,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 4096 * 2
        assert snap.wAccelY == -2048 * 2
        assert snap.wAccelZ == 8192 * 2
        assert snap.wGyroX == int(round(720 * (16.0 / 16.384)))
        assert snap.wGyroY == int(round(-1440 * (16.0 / 16.384)))
        assert snap.wGyroZ == int(round(2880 * (16.0 / 16.384)))

    def test_imu_normalized_only_state(self):
        """State 2: Only normalized accel_*/gyro_* is populated; raw is 0."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            raw_accel_x=0, raw_accel_y=0, raw_accel_z=0,
            raw_gyro_x=0, raw_gyro_y=0, raw_gyro_z=0,
            accel_x=0.5, accel_y=-0.25, accel_z=1.0,
            gyro_x=45.0, gyro_y=-90.0, gyro_z=180.0,
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        # 0.5 * 8192 = 4096
        assert snap.wAccelX == 4096
        # -0.25 * 8192 = -2048
        assert snap.wAccelY == -2048
        # 1.0 * 8192 = 8192
        assert snap.wAccelZ == 8192

        # 45.0 * 16.0 = 720
        assert snap.wGyroX == 720
        # -90.0 * 16.0 = -1440
        assert snap.wGyroY == -1440
        # 180.0 * 16.0 = 2880
        assert snap.wGyroZ == 2880

    def test_imu_both_raw_and_normalized_state_raw_precedence(self):
        """State 3: Both raw and normalized are populated with conflicting values.

        Normalized physical values MUST take precedence over raw counts.
        """
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            raw_accel_x=1000, raw_accel_y=2000, raw_accel_z=3000,
            raw_gyro_x=100, raw_gyro_y=200, raw_gyro_z=300,
            # Conflicting normalized values
            accel_x=0.99, accel_y=0.99, accel_z=0.99,
            gyro_x=99.0, gyro_y=99.0, gyro_z=99.0,
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 8110
        assert snap.wAccelY == 8110
        assert snap.wAccelZ == 8110
        assert snap.wGyroX == 1584
        assert snap.wGyroY == 1584
        assert snap.wGyroZ == 1584

    def test_imu_neither_state_zero_defaults(self):
        """State 4: Neither raw nor normalized is populated (all 0 / 0.0)."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            raw_accel_x=0, raw_accel_y=0, raw_accel_z=0,
            raw_gyro_x=0, raw_gyro_y=0, raw_gyro_z=0,
            accel_x=0.0, accel_y=0.0, accel_z=0.0,
            gyro_x=0.0, gyro_y=0.0, gyro_z=0.0,
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 0
        assert snap.wAccelY == 0
        assert snap.wAccelZ == 0
        assert snap.wGyroX == 0
        assert snap.wGyroY == 0
        assert snap.wGyroZ == 0

    def test_imu_extreme_raw_clamping_int16(self):
        """Verify raw integer overflow outside [-32768, 32767] is clamped safely."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            raw_accel_x=99999, raw_accel_y=-99999,
            raw_gyro_x=50000, raw_gyro_y=-50000,
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 32767
        assert snap.wAccelY == -32768
        assert snap.wGyroX == 32767
        assert snap.wGyroY == -32768

    def test_imu_non_finite_float_clamping(self):
        """Verify NaN, +Inf, -Inf in normalized values do not crash and clamp properly."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        st = make_state(
            accel_x=float("inf"),
            accel_y=float("-inf"),
            accel_z=float("nan"),
            gyro_x=float("inf"),
            gyro_y=float("-inf"),
            gyro_z=float("nan"),
        )
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        assert snap.wAccelX == 32767
        assert snap.wAccelY == -32768
        assert snap.wAccelZ == 0  # nan clamped to 0
        assert snap.wGyroX == 32767
        assert snap.wGyroY == -32768
        assert snap.wGyroZ == 0


# ============================================================================
# 5. Stick Inversion Monotonicity & Differential Fuzzing
# ============================================================================

class TestStickInversionFuzzingOracle:
    """Verifies stick Y monotonicity, boundaries, and fuzzes random inputs."""

    def test_stick_y_inversion_monotonicity(self):
        """As norm_y increases monotonically from -1.0 to +1.0,

        the Sony byte value MUST be monotonically non-increasing.
        """
        prev_byte = 256
        steps = 1000
        for step in range(steps + 1):
            norm_y = -1.0 + (step / steps) * 2.0
            byte_val = _norm_stick_y_inverted_to_byte(norm_y)
            assert 0 <= byte_val <= 255
            assert byte_val <= prev_byte, f"Monotonicity violation at norm_y={norm_y}: {byte_val} > {prev_byte}"
            prev_byte = byte_val

    def test_stick_random_fuzzing_no_out_of_bounds(self):
        """Fuzz sticks with 10,000 random floating point numbers including extreme ranges."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        rng = random.Random(42)
        for _ in range(1000):
            lx = rng.uniform(-5.0, 5.0)
            ly = rng.uniform(-5.0, 5.0)
            rx = rng.uniform(-5.0, 5.0)
            ry = rng.uniform(-5.0, 5.0)

            st = make_state(left_stick_x=lx, left_stick_y=ly, right_stick_x=rx, right_stick_y=ry)
            controller.update_from_state(st)
            snap = mock_gp.last_report_snapshot

            assert 0 <= snap.bThumbLX <= 255
            assert 0 <= snap.bThumbLY <= 255
            assert 0 <= snap.bThumbRX <= 255
            assert 0 <= snap.bThumbRY <= 255


# ============================================================================
# 6. Grip Paddle Remapping Differential Oracle
# ============================================================================

class TestGripPaddleRemappingOracle:
    """Verifies all paddle mapping actions under both Nintendo and Xbox layouts."""

    @pytest.mark.parametrize("mapping,expected_btn_nintendo,expected_btn_xbox", [
        ("L3", 0x4000, 0x4000),
        ("R3", 0x8000, 0x8000),
        ("LB", 0x0100, 0x0100),
        ("RB", 0x0200, 0x0200),
        ("A",  0x0040, 0x0020),  # Circle on Nintendo, Cross on Xbox
        ("B",  0x0020, 0x0040),  # Cross on Nintendo, Circle on Xbox
        ("X",  0x0080, 0x0010),  # Triangle on Nintendo, Square on Xbox
        ("Y",  0x0010, 0x0080),  # Square on Nintendo, Triangle on Xbox
    ])
    def test_digital_paddle_mappings(self, mapping, expected_btn_nintendo, expected_btn_xbox):
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        # Test GL in Nintendo layout
        controller.gl_mapping = mapping
        st = make_state(gl=True)
        controller.update_from_state(st, layout="nintendo")
        assert (mock_gp.last_report_snapshot.wButtons & ~0xF) == expected_btn_nintendo

        # Test GL in Xbox layout
        controller.update_from_state(st, layout="xbox")
        assert (mock_gp.last_report_snapshot.wButtons & ~0xF) == expected_btn_xbox

    def test_gl_mapped_to_lt_or_zl_correctly_actuates_left_trigger(self):
        """GL mapped to LT or ZL actuates Left Trigger."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        for m in ("LT", "ZL"):
            controller.gl_mapping = m
            st = make_state(gl=True, zl_analog=0.0)
            controller.update_from_state(st)
            snap = mock_gp.last_report_snapshot
            assert snap.bTriggerL == 255
            assert snap.bTriggerR == 0
            assert snap.wButtons & 0x0400

    def test_gr_mapped_to_rt_or_zr_correctly_actuates_right_trigger(self):
        """GR mapped to RT or ZR actuates Right Trigger."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        for m in ("RT", "ZR"):
            controller.gr_mapping = m
            st = make_state(gr=True, zr_analog=0.0)
            controller.update_from_state(st)
            snap = mock_gp.last_report_snapshot
            assert snap.bTriggerR == 255
            assert snap.bTriggerL == 0
            assert snap.wButtons & 0x0800

    def test_gl_mapped_to_rt_defect_cross_trigger_contamination(self):
        """ADVERSARIAL DEFECT: GL mapped to RT actuates bTriggerR (right trigger), NOT bTriggerL!"""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        controller.gl_mapping = "RT"
        st = make_state(gl=True, zl_analog=0.0, zr_analog=0.0)
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        # In a correct implementation:
        # - bTriggerR should be 255
        # - bTriggerL should be 0 (untouched)
        # - wButtons should have 0x0800 (DS4_BUTTON_TRIGGER_RIGHT)
        assert snap.bTriggerR == 255, f"bTriggerR was {snap.bTriggerR}, expected 255"
        assert snap.bTriggerL == 0, f"bTriggerL was corrupted to {snap.bTriggerL}, expected 0"
        assert snap.wButtons & 0x0800

    def test_gr_mapped_to_lt_defect_cross_trigger_contamination(self):
        """ADVERSARIAL DEFECT: GR mapped to LT actuates bTriggerL (left trigger), NOT bTriggerR!"""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        controller.gr_mapping = "LT"
        st = make_state(gr=True, zl_analog=0.0, zr_analog=0.0)
        controller.update_from_state(st)
        snap = mock_gp.last_report_snapshot

        # In a correct implementation:
        # - bTriggerL should be 255
        # - bTriggerR should be 0 (untouched)
        # - wButtons should have 0x0400 (DS4_BUTTON_TRIGGER_LEFT)
        assert snap.bTriggerL == 255, f"bTriggerL was {snap.bTriggerL}, expected 255"
        assert snap.bTriggerR == 0, f"bTriggerR was corrupted to {snap.bTriggerR}, expected 0"
        assert snap.wButtons & 0x0400


# ============================================================================
# 7. Real Parsed Report IMU Sensitivity Defect
# ============================================================================

class TestRealParsedReportEndToEndSensitivity:
    """Verifies that an authoritative Switch 2 parsed report produces 8192 LSB/g in DS4 report."""

    def test_switch2_parsed_report_accel_sensitivity_in_ds4(self):
        """ADVERSARIAL DEFECT: A real parsed Switch 2 report with 1.0g gravity must produce ~8192 in DS4 wAccelZ.

        Currently, raw != 0 precedence causes raw_accel_z (4095 LSB/g) to be forwarded directly
        without scaling to DS4 sensitivity (8192 LSB/g), causing a 50% sensitivity loss.
        """
        from report_parser import parse_input_report
        from tests.test_e2e_motion import build_switch2_usb_report

        # 1g resting gravity on Z (ax=0, ay=0, az=4095)
        pkt = build_switch2_usb_report(ax_raw=0, ay_raw=0, az_raw=4095)
        state = parse_input_report(pkt)
        assert state is not None
        assert abs(state.accel_z - 1.0) < 0.02
        assert state.raw_accel_z == 4095

        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        controller.update_from_state(state)
        snap = mock_gp.last_report_snapshot

        # DualShock 4 protocol expects 8192 LSB/g for 1.0g!
        # If wAccelZ is ~4095, games see 0.5g gravity instead of 1.0g!
        assert abs(snap.wAccelZ - 8192) <= 100, (
            f"DS4 wAccelZ was {snap.wAccelZ}, but DS4 protocol requires 8192 LSB/g for 1.0g! "
            f"Sensors are halved in sensitivity because raw_accel_z (4095 LSB/g) bypassed normalized accel_z (1.0g)."
        )



# ============================================================================
# 7. Concurrency & Thread-Safety Stress Harness
# ============================================================================

class TestVirtualDS4ConcurrencyStress:
    """Stress tests concurrent state updates from multiple threads."""

    def test_multi_threaded_concurrent_updates(self):
        """Spin up 4 threads blasting 500 state updates each to ensure zero deadlocks or crashes."""
        controller = VirtualDS4Controller()
        mock_gp = MockVDS4Gamepad()
        controller._gamepad = mock_gp
        controller.is_connected = True

        errors = []

        def worker_loop(thread_id: int):
            try:
                for i in range(500):
                    st = make_state(
                        a=(i % 2 == 0),
                        b=(i % 3 == 0),
                        zl_analog=(i % 256) / 255.0,
                        left_stick_x=((i % 100) - 50) / 50.0,
                        raw_accel_z=8192,
                    )
                    controller.update_from_state(st)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=worker_loop, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert not errors, f"Concurrent worker thread errors: {errors}"
        assert mock_gp.update_ex_count == 2000
