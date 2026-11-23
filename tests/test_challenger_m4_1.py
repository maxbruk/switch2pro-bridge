"""Adversarial stress and boundary tests for Milestone 4 (Core Output Mode & State Integration).

Covers:
1. Rapid Output Mode Swapping Stress (high frequency 250 Hz dispatch concurrent with mode switching).
2. Boundary & Malicious Input Fuzzing (invalid types, empty strings, whitespace, extreme payloads, homoglyphs).
3. DSU Bind Collision & Lifecycle Stress (blocked port handling, retry release, rapid toggles, recreation).
4. Fault Tolerance & Exception Isolation (crashing UI callbacks, corrupt motion floats, None states).
5. Multi-Threaded Concurrency Swarm (5-thread simultaneous stress test).
6. Clean Shutdown & Lifecycle Idempotency.
"""

import socket
import struct
import threading
import time
import unittest.mock as mock
import zlib
from typing import List, Optional, Set, Tuple

import pytest

from app_core import ControllerManager
from report_parser import ControllerState
from dsu_server import DsuServer, calculate_dsu_crc32
from virtual_xinput import VirtualXboxController
from virtual_ds4 import VirtualDS4Controller
from tests.test_virtual_controllers import MockVX360Gamepad, MockVDS4Gamepad, make_state


# ============================================================================
# Suite 1: Rapid Output Mode Swapping Under High-Frequency Dispatch Stress
# ============================================================================

class TestAdversarialOutputModeRapidSwapping:
    """Simulates high-frequency input dispatch concurrent with aggressive mode changes."""

    def test_rapid_swapping_high_frequency_state_dispatch(self):
        """1000 input states dispatched at 250 Hz rate while switching modes 100 times."""
        manager = ControllerManager()
        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):

            manager.is_connected = True
            manager.virtual_gamepad.connect()

            exceptions = []
            stop_flag = threading.Event()

            def producer():
                try:
                    for i in range(800):
                        st = make_state(
                            a=(i % 2 == 0),
                            b=(i % 3 == 0),
                            left_stick_x=((i % 100) - 50) / 50.0,
                            accel_z=1.0,
                            gyro_x=45.0,
                            motion_timestamp=1000000 + i * 4000,
                        )
                        manager._on_controller_state(st)
                        time.sleep(0.001)
                except Exception as e:
                    exceptions.append(("producer", e))

            def switcher():
                try:
                    for i in range(100):
                        target = "ds4" if (i % 2 == 0) else "xbox"
                        manager.set_output_mode(target)
                        time.sleep(0.005)
                except Exception as e:
                    exceptions.append(("switcher", e))

            def inspector():
                try:
                    while not stop_flag.is_set():
                        mode = manager.output_mode
                        gp = manager.virtual_gamepad
                        assert mode in ("xbox", "ds4")
                        assert gp is not None
                        time.sleep(0.002)
                except Exception as e:
                    exceptions.append(("inspector", e))

            t_prod = threading.Thread(target=producer, name="AdvProducer")
            t_switch = threading.Thread(target=switcher, name="AdvSwitcher")
            t_insp = threading.Thread(target=inspector, name="AdvInspector")

            t_insp.start()
            t_prod.start()
            t_switch.start()

            t_prod.join(timeout=8.0)
            t_switch.join(timeout=8.0)
            stop_flag.set()
            t_insp.join(timeout=3.0)

            assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
            assert not t_prod.is_alive(), "Producer thread did not terminate"
            assert not t_switch.is_alive(), "Switcher thread did not terminate"
            assert not t_insp.is_alive(), "Inspector thread did not terminate"

            # Final state verification
            assert manager.output_mode in ("xbox", "ds4")
            if manager.output_mode == "ds4":
                assert isinstance(manager.virtual_gamepad, VirtualDS4Controller)
            else:
                assert isinstance(manager.virtual_gamepad, VirtualXboxController)
            assert manager.virtual_gamepad.is_connected is True

    def test_rapid_swapping_preserves_custom_layout_and_paddles_under_race(self):
        """Ensure layout and grip paddle mappings are never lost during rapid concurrent mode swaps."""
        manager = ControllerManager()
        manager.set_button_layout("xbox")
        manager.set_gl_mapping("LB")
        manager.set_gr_mapping("RT")

        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):

            manager.is_connected = True
            manager.virtual_gamepad.connect()

            for i in range(50):
                target = "ds4" if (i % 2 == 0) else "xbox"
                manager.set_output_mode(target)
                gp = manager.virtual_gamepad
                assert gp.layout == "xbox", f"Layout lost on iteration {i}"
                assert gp.gl_mapping == "LB", f"GL mapping lost on iteration {i}"
                assert gp.gr_mapping == "RT", f"GR mapping lost on iteration {i}"

    def test_rapid_swapping_with_simulated_connect_delay(self):
        """Simulate ViGEmBus driver delay (5ms) during connect() without deadlocks."""
        manager = ControllerManager()
        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()

        def delayed_connect(gp_mock):
            time.sleep(0.005)
            return True

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):

            manager.is_connected = True

            # Patch BaseVirtualController connect to add simulated driver delay
            orig_ds4_connect = VirtualDS4Controller.connect
            orig_xbox_connect = VirtualXboxController.connect

            def slow_ds4(self_gp):
                time.sleep(0.004)
                return orig_ds4_connect(self_gp)

            def slow_xbox(self_gp):
                time.sleep(0.004)
                return orig_xbox_connect(self_gp)

            with mock.patch.object(VirtualDS4Controller, "connect", slow_ds4), \
                 mock.patch.object(VirtualXboxController, "connect", slow_xbox):

                exceptions = []

                def switcher():
                    try:
                        for i in range(30):
                            manager.set_output_mode("ds4" if (i % 2 == 0) else "xbox")
                    except Exception as e:
                        exceptions.append(e)

                def producer():
                    try:
                        st = make_state(accel_z=1.0)
                        for _ in range(300):
                            manager._on_controller_state(st)
                            time.sleep(0.001)
                    except Exception as e:
                        exceptions.append(e)

                t1 = threading.Thread(target=switcher)
                t2 = threading.Thread(target=producer)
                t1.start()
                t2.start()
                t1.join(timeout=6.0)
                t2.join(timeout=6.0)

                assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
                assert not t1.is_alive()
                assert not t2.is_alive()


# ============================================================================
# Suite 2: Boundary & Malicious Input Fuzzing
# ============================================================================

class TestAdversarialModeFuzzingAndBoundaries:
    """Fuzzes inputs to set_output_mode and manager setters with malicious or extreme values."""

    @pytest.mark.parametrize("invalid_mode", [
        None,
        123,
        3.14159,
        b"ds4",
        b"xbox",
        [],
        {},
        (),
        object(),
        "",
        "   ",
        "\t\r\n",
        "x" * 10000,
        "x_box",
        "dualshock4",
        "xbox360",
        "ds5",
        "switch_pro",
        "nintendo",
        "ds",
        "xb",
        "d s 4",
        "x b o x",
        "ds4\x00extra",
        "\x00xbox",
        "xbox\x00",
        "🎮",
        "xbo\u0445",  # Cyrillic small letter ha (homoglyph for x)
        "ｄｓ４",     # Full-width characters
        "!@#$%^&*()",
    ])
    def test_set_output_mode_rejects_fuzzed_inputs(self, invalid_mode):
        """Ensure invalid types and malformed strings raise ValueError and preserve manager state."""
        manager = ControllerManager()
        initial_mode = manager.output_mode
        initial_gamepad = manager.virtual_gamepad

        with pytest.raises(ValueError):
            manager.set_output_mode(invalid_mode)

        # Confirm manager state is completely uncorrupted
        assert manager.output_mode == initial_mode
        assert manager.virtual_gamepad is initial_gamepad

    @pytest.mark.parametrize("invalid_mode", [
        None,
        456,
        "dualshock",
        "",
        "   ",
    ])
    def test_output_mode_property_setter_rejects_invalid(self, invalid_mode):
        """Verify the property setter manager.output_mode = ... validates inputs."""
        manager = ControllerManager()
        with pytest.raises(ValueError):
            manager.output_mode = invalid_mode
        assert manager.output_mode == "xbox"

    @pytest.mark.parametrize("valid_variant, expected", [
        ("XBOX", "xbox"),
        ("Ds4", "ds4"),
        ("  xbox  ", "xbox"),
        ("  ds4  ", "ds4"),
        ("\txbox\n", "xbox"),
        ("\r\nds4\t", "ds4"),
        ("XbOx", "xbox"),
        ("DS4", "ds4"),
    ])
    def test_set_output_mode_accepts_clean_variants(self, valid_variant, expected):
        """Verify case-insensitivity and whitespace trimming for valid modes."""
        manager = ControllerManager()
        manager.set_output_mode(valid_variant)
        assert manager.output_mode == expected

    @pytest.mark.parametrize("enabled_val, expected_bool", [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("yes", True),
        ("", False),
        (None, False),
        ([], False),
        ([1, 2], True),
    ])
    def test_set_dsu_enabled_coerces_truthy_falsy(self, enabled_val, expected_bool):
        """Verify set_dsu_enabled safely coerces values via bool() without raising."""
        manager = ControllerManager()
        real_dsu = DsuServer(host="127.0.0.1", port=0)
        manager.dsu_server = real_dsu

        try:
            manager.set_dsu_enabled(enabled_val)
            assert manager.dsu_enabled is expected_bool
        finally:
            manager.set_dsu_enabled(False)


# ============================================================================
# Suite 3: DSU Bind Collision & Lifecycle Stress
# ============================================================================

class TestAdversarialDsuPortCollisionAndLifecycle:
    """Stress tests Cemuhook DSU server port collision, release, and recovery."""

    def test_dsu_port_blocked_then_released_recovery(self):
        """Verify manager handles port collision gracefully, reports error, and successfully starts when freed."""
        # Find an available ephemeral port
        temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        temp_sock.bind(("127.0.0.1", 0))
        test_port = temp_sock.getsockname()[1]
        temp_sock.close()

        # 1. Block test_port with an external raw socket
        blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        blocker.bind(("127.0.0.1", test_port))

        statuses = []
        manager = ControllerManager(dsu_host="127.0.0.1", dsu_port=test_port)
        manager.on_status_message = lambda msg: statuses.append(msg)

        try:
            # 2. Try enabling DSU while port is blocked
            manager.set_dsu_enabled(True)

            # Verification: should fail gracefully without exception
            assert manager.dsu_enabled is False
            assert manager.dsu_server.is_running is False
            assert any(f"port {test_port} in use" in s for s in statuses)

            # 3. Free the blocking port
            blocker.close()
            blocker = None

            # 4. Retry enabling DSU now that port is released
            manager.set_dsu_enabled(True)

            assert manager.dsu_enabled is True
            assert manager.dsu_server.is_running is True
            assert manager.dsu_server.port == test_port
            assert any(f"DSU Motion Server active (port {test_port})" in s for s in statuses)

            # 5. Verify real client communication on recovered server
            client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            client.settimeout(1.0)
            try:
                # Send Version Request (0x100000)
                req = bytearray(20)
                struct.pack_into("<4sHHII", req, 0, b"DSUC", 1001, 4, 0, 0x11223344)
                struct.pack_into("<I", req, 16, 0x100000)
                crc = calculate_dsu_crc32(req)
                struct.pack_into("<I", req, 8, crc)
                client.sendto(req, ("127.0.0.1", test_port))

                resp, _ = client.recvfrom(1024)
                assert len(resp) == 24
                magic, ver, _, _, _ = struct.unpack_from("<4sHHII", resp, 0)
                assert magic == b"DSUS"
                assert ver == 1001
            finally:
                client.close()

            # 6. Disable DSU cleanly
            manager.set_dsu_enabled(False)
            assert manager.dsu_enabled is False
            assert manager.dsu_server.is_running is False

        finally:
            if blocker:
                blocker.close()
            manager.shutdown()

    def test_rapid_dsu_toggle_stress(self):
        """Rapidly toggle set_dsu_enabled 40 times to test thread and socket cleanup."""
        manager = ControllerManager()
        # Use ephemeral port 0
        manager.dsu_server = DsuServer(host="127.0.0.1", port=0)

        for i in range(40):
            target = (i % 2 == 0)
            manager.set_dsu_enabled(target)
            assert manager.dsu_enabled == target
            assert manager.dsu_server.is_running == target

        # Ensure cleanly disabled at end
        manager.set_dsu_enabled(False)
        assert manager.dsu_enabled is False
        assert manager.dsu_server.is_running is False

    def test_dsu_server_recreation_when_none(self):
        """Verify set_dsu_enabled recreates DsuServer if attribute was set to None."""
        manager = ControllerManager()
        manager.dsu_server = None

        with mock.patch("app_core.DsuServer") as mock_dsu_cls:
            mock_inst = mock.MagicMock()
            mock_inst.is_running = False

            def fake_start():
                mock_inst.is_running = True
                return True

            mock_inst.start.side_effect = fake_start
            mock_inst.port = 26760
            mock_dsu_cls.return_value = mock_inst

            manager.set_dsu_enabled(True)
            assert manager.dsu_enabled is True
            assert manager.dsu_server is mock_inst
            mock_inst.start.assert_called_once()

            manager.set_dsu_enabled(False)
            assert manager.dsu_enabled is False
            mock_inst.stop.assert_called_once()


# ============================================================================
# Suite 4: State Dispatch Fault Tolerance & Boundary Handling
# ============================================================================

class TestAdversarialStateDispatchFaultTolerance:
    """Stress tests ControllerManager._on_controller_state under abnormal conditions."""

    def test_state_dispatch_survives_crashing_ui_callback(self):
        """Manager must never raise or drop input when GUI callback throws."""
        manager = ControllerManager()
        mock_gp = MockVX360Gamepad()
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True

        def exploding_callback(state):
            raise RuntimeError("Simulated GUI render failure or dead widget")

        manager.on_input_update = exploding_callback

        st = make_state(a=True)
        # Should NOT raise
        manager._on_controller_state(st)

        assert mock_gp.update_count == 1

    def test_state_dispatch_survives_gamepad_update_exception(self):
        """Manager must catch exceptions from virtual controller without crashing."""
        manager = ControllerManager()
        manager.virtual_gamepad.update_from_state = mock.MagicMock(
            side_effect=Exception("Simulated driver communication fault")
        )

        st = make_state(b=True)
        # Should NOT raise
        manager._on_controller_state(st)

    def test_state_dispatch_survives_dsu_update_exception(self):
        """Manager must catch exceptions from DSU server without crashing."""
        manager = ControllerManager()
        mock_dsu = mock.MagicMock()
        mock_dsu.is_running = True
        mock_dsu.update_state.side_effect = Exception("Simulated network send error")
        manager.dsu_server = mock_dsu
        manager.dsu_enabled = True

        st = make_state(accel_z=1.0)
        # Should NOT raise
        manager._on_controller_state(st)

    @pytest.mark.parametrize("corrupt_state", [
        None,
        make_state(),  # Default neutral state
        make_state(accel_x=float("nan"), accel_y=float("inf"), accel_z=-float("inf")),
        make_state(gyro_x=float("nan"), gyro_y=1e30, gyro_z=-1e30),
        make_state(left_stick_x=999.0, left_stick_y=-999.0),
    ])
    def test_state_dispatch_with_corrupt_or_extreme_states(self, corrupt_state):
        """Extreme floating point states (NaN, Inf, overflow) must be handled safely."""
        manager = ControllerManager()
        mock_gp = MockVDS4Gamepad()
        manager.set_output_mode("ds4")
        manager.virtual_gamepad._gamepad = mock_gp
        manager.virtual_gamepad.is_connected = True

        # Should NOT raise any uncaught exceptions
        manager._on_controller_state(corrupt_state)


# ============================================================================
# Suite 5: Heavy Multi-Threaded Concurrency Swarm
# ============================================================================

class TestAdversarialHeavyConcurrencySwarm:
    """Extreme multi-threaded swarm running 5 concurrent workloads simultaneously."""

    def test_five_thread_concurrency_swarm(self):
        manager = ControllerManager()
        mock_xbox = MockVX360Gamepad()
        mock_ds4 = MockVDS4Gamepad()
        mock_dsu = mock.MagicMock()
        mock_dsu.is_running = False
        mock_dsu.start.return_value = True
        mock_dsu.port = 26760
        manager.dsu_server = mock_dsu

        with mock.patch("vgamepad.VX360Gamepad", return_value=mock_xbox), \
             mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):

            manager.is_connected = True
            manager.virtual_gamepad.connect()

            exceptions = []
            stop_event = threading.Event()

            # Thread 1: 250 Hz Input State Producer
            def state_producer():
                try:
                    for i in range(600):
                        st = make_state(
                            a=(i % 2 == 0),
                            b=(i % 3 == 0),
                            accel_x=((i % 20) - 10) / 10.0,
                            accel_z=1.0,
                            gyro_x=30.0,
                            motion_timestamp=1000 + i * 4000,
                        )
                        manager._on_controller_state(st)
                        time.sleep(0.001)
                except Exception as e:
                    exceptions.append(("state_producer", e))

            # Thread 2: Output Mode Switcher
            def mode_switcher():
                try:
                    for i in range(80):
                        target = "ds4" if (i % 2 == 0) else "xbox"
                        manager.set_output_mode(target)
                        time.sleep(0.006)
                except Exception as e:
                    exceptions.append(("mode_switcher", e))

            # Thread 3: DSU Toggler
            def dsu_toggler():
                try:
                    for i in range(60):
                        target = (i % 2 == 0)
                        mock_dsu.is_running = target
                        manager.set_dsu_enabled(target)
                        time.sleep(0.008)
                except Exception as e:
                    exceptions.append(("dsu_toggler", e))

            # Thread 4: Layout & Paddle Remapper
            def layout_remapper():
                layouts = ["nintendo", "xbox"]
                paddles = ["L3", "LB", "Disabled", "RT", "R3", "RB"]
                try:
                    for i in range(50):
                        manager.set_button_layout(layouts[i % 2])
                        manager.set_gl_mapping(paddles[i % len(paddles)])
                        manager.set_gr_mapping(paddles[(i + 1) % len(paddles)])
                        time.sleep(0.010)
                except Exception as e:
                    exceptions.append(("layout_remapper", e))

            # Thread 5: Telemetry Inspector
            def telemetry_reader():
                try:
                    while not stop_event.is_set():
                        _ = manager.output_mode
                        _ = manager.dsu_enabled
                        _ = manager.button_layout
                        _ = manager.virtual_gamepad.gl_mapping
                        _ = manager.virtual_gamepad.gr_mapping
                        time.sleep(0.002)
                except Exception as e:
                    exceptions.append(("telemetry_reader", e))

            threads = [
                threading.Thread(target=state_producer, name="SwarmProducer"),
                threading.Thread(target=mode_switcher, name="SwarmModeSwitcher"),
                threading.Thread(target=dsu_toggler, name="SwarmDsuToggler"),
                threading.Thread(target=layout_remapper, name="SwarmLayoutRemapper"),
                threading.Thread(target=telemetry_reader, name="SwarmTelemetryReader"),
            ]

            for t in threads:
                t.start()

            # Wait for bounded workers
            threads[0].join(timeout=8.0)
            threads[1].join(timeout=8.0)
            threads[2].join(timeout=8.0)
            threads[3].join(timeout=8.0)

            # Signal reader to stop
            stop_event.set()
            threads[4].join(timeout=3.0)

            # Assert all threads completed cleanly
            for t in threads:
                assert not t.is_alive(), f"Thread {t.name} hung"

            assert len(exceptions) == 0, f"Exceptions in swarm: {exceptions}"


# ============================================================================
# Suite 6: Shutdown & Disconnect Lifecycle Idempotency
# ============================================================================

class TestAdversarialShutdownLifecycle:
    """Verifies manager shutdown, teardown, and clean resource release."""

    def test_shutdown_cleans_up_all_subsystems(self):
        manager = ControllerManager()
        mock_dsu = mock.MagicMock()
        mock_dsu.is_running = True
        manager.dsu_server = mock_dsu
        manager._dsu_enabled = True

        mock_ds4 = MockVDS4Gamepad()
        with mock.patch("vgamepad.VDS4Gamepad", return_value=mock_ds4):
            manager.set_output_mode("ds4")
            manager.is_connected = True
            manager.virtual_gamepad.connect()

            assert manager.is_connected is True
            assert manager.virtual_gamepad.is_connected is True
            assert manager.dsu_enabled is True

            manager.shutdown()

            assert manager.is_connected is False
            assert manager.virtual_gamepad.is_connected is False
            assert manager.dsu_enabled is False
            mock_dsu.stop.assert_called_once()
            assert mock_ds4.reset_count >= 1

    def test_shutdown_idempotency(self):
        """Calling shutdown() multiple times consecutively must not raise exceptions."""
        manager = ControllerManager()
        manager.shutdown()
        manager.shutdown()
        manager.shutdown()
        assert manager.is_connected is False
        assert manager.dsu_enabled is False
