# Engineering & Agent Execution Standards

You are operating in a high-velocity, high-trust codebase. Your goal is to maximize shipping speed while strictly maintaining or improving code quality. Do not produce "slop" (quick workarounds, unverified patches, or unidiomatic code).

## 1. Core Hierarchy of Constraints
Rely on deterministic checks first and prose second:
1. **Architecture & Types (Highest Priority):** Follow established module boundaries, strict types, and framework conventions so invalid states are unrepresentable.
2. **Linters & Static Analysis:** Never bypass, disable, or ignore lint rules, compiler errors, or dependency graph checks. Treat linter error messages as architectural guidance to fix the root cause.
3. **Runtime & Formal Verification:** Prove your code works before finishing.
4. **Prose & Comments (Lowest Priority):** Only use text where judgment is required.

## 2. Codebase Hygiene ("Gardening")
* **Do Not Mimic Anti-Patterns:** Because you learn from existing code, actively avoid copying legacy workarounds or deprecated patterns. When touching an area with tech debt, leave it cleaner than you found it.
* **Fix Root Causes, Never Patch:** Do not use quick fixes (e.g., arbitrary `setTimeout` calls, `requestAnimationFrame` hacks, or complex `useEffect` chains) to paper over architectural mismatches. Fix the underlying state or data flow.
* **Respect Directory & Boundary Conventions:** Strictly adhere to folder-level separation of concerns (e.g., never import main-thread/backend modules into renderer/client code).
* **No Comment Excuses:** Do not write inline comments to justify a suboptimal workaround or explain why you broke a pattern. Code comments go stale and are unenforceable; express intent through clean structure, naming, and types instead.
* **Encode Lessons Deterministically:** Whenever you encounter a recurring mistake or receive a correction from the user, propose or implement a deterministic guardrail (a lint rule, type constraint, or automated test) rather than just adding another line of prose to documentation.

## 3. Verification-First Workflow
Most of your effort and token budget should go toward proving correctness, not just writing initial code:
* **Run & Inspect Like a User:** Do not assume code works because it compiles. Use available runtime tools, CLIs, browser/DevTools protocols, or test runners to execute the application and verify behavior end-to-end.
* **Use Shared Vocabulary:** Refer to UI elements and user flows using the repository's established feature maps, state machines, or component names.
* **Check Invariants & Edge Cases:** Validate system properties using property-based tests, state machine checks, or formal verification tools where configured.
* **Produce Reviewable Artifacts:** When completing a task or PR, provide clear verification proof-such as test outputs, screenshots, recordings, or state-chart diagrams-so human reviewers can verify intent without reading every line of boilerplate.

---

## 4. Project Context: Switch2ProBridge (`v2.0.0`)

### Overview & Runtime Environment
* **Purpose:** Windows 10/11 (x64) bridge translating Nintendo Switch 2 Pro Controller (`VID 0x057E`, `PID 0x2069`, plus `0x2009` / `0x2073`) inputs, 6-axis IMU motion, and HD Rumble 2 force-feedback over USB-C and Bluetooth LE into virtual Xbox 360 (`VX360Gamepad`) or motion-enabled DualShock 4 (`VDS4Gamepad`) controllers via ViGEmBus, plus a Cemuhook DSU UDP motion server.
* **Toolchain:** Python `>=3.11,<3.13` managed strictly with **`uv`** (do not use bare system `python` or `pip`):
  * Launch GUI: `uv run python main.py`
  * Launch headless CLI: `uv run python main.py --cli`
  * Run test suite: `uv run pytest -rs`
  * Build portable `.exe` & zip: `uv run python build_standalone.py` (driven by `Switch2ProBridge.spec`)

### Module Boundaries
* `constants.py`: Hardware VIDs/PIDs, BLE UUIDs, pairing payloads, USB startup sequence, report IDs (`0x05`, `0x09`, `0x0A`, output `0x02`/`0x03`), button bitmasks (`BTN_GL`, `BTN_GR`), and HD Rumble 2 constants (`RUMBLE_MAX_AMPLITUDE`, `RUMBLE_HIGH_FREQ`, `RUMBLE_LOW_FREQ`, `RUMBLE_INTERVAL_SEC`).
* `report_parser.py`: Pure stateless decoding and packet serialization (`ControllerState`, `parse_input_report`, `encode_hd_rumble`, `build_switch2_rumble_packet`). Unpacks 12-bit sticks with radial deadzone clamping and the 18-byte ICM-42670-P 6-axis IMU block (`accel_*` in `g`, `gyro_*` in `deg/s`, `imu_temperature`, `motion_timestamp`, `has_motion`). Must have zero UI, USB/BLE I/O, or socket side effects.
* `usb_connector.py`: `UsbProConnector` and WinUSB `MI_01` initialization (`send_winusb_startup_commands`). Handles high-frequency HID reading, `has_report_id=True` parsing, 15 ms rumble output refreshes (`_send_rumble_if_due`), and 5.0 s motor watchdog safety.
* `ble_connector.py`: `BleProConnector` running an isolated `bleak` WinRT asyncio event loop. Handles pairing handshake (`COMMAND_PAIR`), Player 1 LED setup, `_enable_imu_features()`, `has_report_id=False` notifications, and 15 ms GATT rumble writes (`VIBRATION_WRITE_PRO_UUID`).
* `virtual_gamepad.py`, `virtual_xinput.py`, `virtual_ds4.py`: `BaseVirtualController` polymorphic interface and ViGEmBus backends (`VirtualXboxController`, `VirtualDS4Controller`). `VirtualDS4Controller` packs `DS4_REPORT_EX` motion (`8192` LSB/g accel, `16` LSB/dps gyro), Sony stick Y-inversion (`128` center), 8-way D-Pad hats, and GL/GR back paddles.
* `dsu_server.py`: Zero-dependency Cemuhook DSU (`v1001`) UDP server on `127.0.0.1:26760` (`0x100000`, `0x100001`, `0x100002` with IEEE 802.3 CRC32), Windows `SIO_UDP_CONNRESET` / `WSAECONNRESET` / `WSAEMSGSIZE` protection, and 5.0 s subscriber pruning.
* `app_core.py`: `ControllerManager` coordinating USB/BLE transports, lock-safe runtime `set_output_mode("xbox"|"ds4")`, `set_dsu_enabled(bool)`, `trigger_rumble()` / `test_rumble_pulse()`, and 250 Hz `_on_controller_state()` dual dispatch.
* `gui.py`: CustomTkinter UI (`Switch2ProApp`). Keeps the 250 Hz transport thread decoupled via atomic `_latest_state` storage and a 60 FPS `_visualizer_tick` loop with deadband dirty-checking. Handles atomic JSON persistence (`settings.json`) and virtualenv Tcl/Tk discovery (`_ensure_tcl_tk_env`).
* `main.py`: Application CLI/GUI launcher.
* `tests/`: `pytest` unit, stress, GUI, E2E motion, and adversarial suites (plus offline `drivers/ViGEmBusSetup_x64.msi`).

### Project-Specific Invariants & Guardrails
* **WinUSB Bulk IN Drain:** When modifying `send_winusb_startup_commands()` in `usb_connector.py`, always configure `PIPE_TRANSFER_TIMEOUT` and drain command replies from bulk IN endpoint `0x82` (`WinUsb_ReadPipe`) after writes; otherwise the controller's command FIFO stalls and IMU telemetry remains zeroed.
* **ViGEmBus Callback Lifetime:** Register ViGEm force-feedback notifications only via `BaseVirtualController._register_rumble_notification()` when `on_rumble` is configured, preserve exact unannotated parameter names `(client, target, large_motor, small_motor, led_number, user_data)` required by `vgamepad`, and call `_unregister_rumble_notification()` before dropping `_gamepad` to prevent ViGEmClient C DLL access violations during garbage collection.
* **Tkinter Threading & Settings Schema:** Never mutate CustomTkinter widgets directly from 250 Hz USB/BLE callbacks (marshal via `self.after(0, ...)` or atomic `_latest_state`). Keep `DEFAULT_SETTINGS` keys aligned with `tests/test_gui.py` (`test_default_settings_schema`) or update schema tests alongside any intentional additions.
* **Mocking & Ephemeral Ports in Tests:** Mock `vgamepad.VX360Gamepad` and `vgamepad.VDS4Gamepad` in concurrency/thrashing unit tests so real ViGEm targets are not torn down across threads without cleanup; use `port=0` for `DsuServer` network tests to avoid port `26760` conflicts.
* **Terminology & Repo Conventions:** Keep user-facing docs and commit messages concise, humane, and free of emojis or em-dashes. Describe switch protocol handshakes using standard USB/HID/BLE interoperability terms (avoid loaded terms like proprietary keys) and track `Switch2ProBridge.spec` / `drivers/ViGEmBusSetup_x64.msi` while ignoring `dist/`, `build/`, `.venv/`, and local `settings.json`.
