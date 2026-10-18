# Switch 2 Pro Connect

<div align="center">

[![Platform](https://img.shields.io/badge/Platform-Windows%2010%20%7C%2011%20(x64)-0078D6?style=flat-square&logo=windows&logoColor=white)](https://github.com/maxbruk)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Package Manager: uv](https://img.shields.io/badge/Package%20Manager-uv-DE5FE9?style=flat-square&logo=astral&logoColor=white)](https://github.com/astral-sh/uv)
[![Driver: ViGEmBus](https://img.shields.io/badge/Driver-ViGEmBus%20(XInput)-107C41?style=flat-square&logo=xbox&logoColor=white)](https://github.com/nefarius/ViGEmBus)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Author](https://img.shields.io/badge/Developed%20by-Max%20Bruk-0284C7?style=flat-square&logo=github&logoColor=white)](https://github.com/maxbruk)

<p>
  Connect your Nintendo Switch 2 Pro Controller (and Switch 1 Pro Controller) to Windows PC as a virtual Xbox 360 controller over USB-C and Bluetooth.
</p>

[Quick Start](#quick-start) • [Features](#features) • [How to Connect](#how-to-connect) • [Developer Setup](#developer-setup) • [Technical Details](#how-it-works-technical-details) • [Disclaimer](#disclaimer)

</div>

---

## Quick Start

No Python or programming knowledge required.

1. **Download**: Grab the latest `Switch2ProBridge-v1.0.0-Windows-x64.zip` from the [Releases](https://github.com/maxbruk) page.
2. **Extract**: Unzip the folder anywhere on your PC and run `Switch2ProBridge.exe`.
3. **Driver Setup**: If prompted, click "Install Driver" to install the ViGEmBus virtual controller driver (or run `drivers/ViGEmBusSetup_x64.msi` once).
4. **Connect & Play**: Plug in via USB-C or hold the Sync button for Bluetooth. Launch Steam, Xbox Game Pass, Epic Games, or any emulator - your controller will work right away.

---

## Features

- **Wired USB-C (250 Hz)**: Direct low-latency connection over WinUSB with Windows 1 ms multimedia scheduling (`timeBeginPeriod(1)`).
- **Wireless Bluetooth LE**: Automated pairing using standard Bluetooth Low Energy (BLE) handshake and bonding without entering PINs or configuring Windows Bluetooth menus.
- **Solid Player 1 LED**: Automatically assigns Player 1 upon connection, turning LED 1 solid green and stopping the sweeping search flash.
- **12-Bit Analog Sticks**: High-resolution stick coordinate decoding with smooth radial deadzones and zero-flicker packet parsing.
- **Rear Paddle Mapping (GL & GR)**: Map the Switch 2 Pro Controller's back grip paddles to L3 (Sprint), R3 (Crouch), bumpers, triggers, face buttons, or disable them in settings.
- **Layout Toggle**: One-click switch between physical Nintendo button positions (A on right, B on bottom) and standard Xbox letter layout (A=A, B=B).
- **Real-Time Visualizer**: 60 FPS dashboard showing stick crosshairs, button presses, battery percentage, and connection latency.
- **Standalone Portable App**: Single `.exe` with all dependencies bundled - zero external installations needed.

---

## How to Connect

### USB-C (Wired)
1. Plug your controller into any USB port using a standard USB-C cable.
2. Open `Switch2ProBridge.exe`.
3. The controller initializes automatically and begins streaming at 250 Hz.

### Bluetooth (Wireless)
1. Press and hold the small circular **Sync button** on the top of the controller until the 4 LEDs cycle back and forth.
2. In `Switch2ProBridge.exe`, click **"Connect Controller"**.
3. The app pairs automatically and the Player 1 LED turns solid green.

---

## Developer Setup

The project is built with Python 3.12 and managed with [uv](https://github.com/astral-sh/uv).

### Running from Source
```powershell
# Clone the repository
git clone https://github.com/maxbruk/switch2pro-bridge.git
cd switch2pro-bridge

# Run the GUI
uv run python main.py

# Or run in headless CLI mode
uv run python main.py --cli
```

### Running Tests
```powershell
uv run pytest
```

### Building the Standalone Executable
```powershell
uv run python build_standalone.py
```
Output files are placed in `dist/`:
- `dist/Switch2ProBridge.exe`
- `dist/Switch2ProBridge-v1.0.0-Windows-x64.zip`

---

## How It Works (Technical Details)

For those interested in the technical implementation details:

- **USB Handshake**: The Switch 2 Pro Controller (VID `0x057E`, PID `0x2069`) starts in an idle composite state (Report `0x09`). Sending initialization control packets `0x80 0x02` and `0x80 0x04` over WinUSB interface 1 enables full-speed 250 Hz input streaming (Report `0x05`).
- **Bluetooth LE Bonding**: The controller communicates via custom GATT services (`ab7de9be-89fe-49ad-828f-118f09df7fd0`). The connector completes the two-stage host pairing exchange (host MAC address registration and confirmation handshake) directly in user space.
- **XInput Emulation**: Decoded button states and 12-bit analog coordinates are translated into a virtual Xbox 360 controller via the ViGEmBus kernel driver (`vgamepad`), making the controller natively recognized by all Windows games.

---

## Project Structure

```
switch2pro-bridge/
├── constants.py          # Hardware VIDs, PIDs, BLE UUIDs, and packet definitions
├── report_parser.py      # Input report decoding (0x05 / 0x09), stick math, and deadzones
├── usb_connector.py      # WinUSB communication loop & 250 Hz packet pipeline
├── ble_connector.py      # Bleak WinRT GATT connection, pairing handshake & notification stream
├── virtual_xinput.py     # Virtual Xbox 360 emulation via ViGEmBus (vgamepad)
├── app_core.py           # Coordinator handling auto-reconnect and device state
├── gui.py                # CustomTkinter Dark Mode GUI with real-time visualizer
├── build_standalone.py   # Automated PyInstaller distribution packager
├── Switch2ProBridge.spec  # PyInstaller packaging configuration
├── drivers/              # Official ViGEmBus 64-bit driver installer
└── tests/                # Automated pytest regression suite
```

---

## Disclaimer

This project is an independent open-source utility developed solely for hardware interoperability on Windows PCs. It is not affiliated with, endorsed by, or sponsored by Nintendo Co., Ltd., Microsoft Corporation, or any of their subsidiaries.

All product names, trademarks, and registered trademarks (including "Nintendo Switch", "Switch Pro Controller", "Joy-Con", and "Xbox") are property of their respective owners and are used here exclusively for nominative identification and compatibility purposes.

---

## License

Distributed under the [MIT License](LICENSE).

Developed by [Max Bruk](https://github.com/maxbruk).
