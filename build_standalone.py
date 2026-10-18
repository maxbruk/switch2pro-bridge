"""Automated build and packaging script for Switch 2 Pro Controller Bridge.

Builds a self-contained, single-file Windows executable (.exe) with PyInstaller.
Requires no Python or dependencies on the end-user's machine.
"""

import os
import sys
import shutil
import zipfile
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(PROJECT_DIR, "dist")
BUILD_DIR = os.path.join(PROJECT_DIR, "build")
SPEC_FILE = os.path.join(PROJECT_DIR, "Switch2ProBridge.spec")

QUICKSTART_TEXT = """======================================================================
  Nintendo Switch 2 Pro Controller Connector for Windows (XInput Bridge)
======================================================================

Thank you for downloading Switch 2 Pro Controller Connector!
This program allows you to use your Switch 2 Pro Controller (and Switch 1 
Pro Controller) on any Windows game as a virtual Xbox 360 controller.

----------------------------------------------------------------------
1. FIRST-TIME SETUP (VIGEMBUS DRIVER)
----------------------------------------------------------------------
If you have never installed the ViGEmBus virtual controller driver:
  - Run 'drivers/ViGEmBusSetup_x64.msi' once and follow the on-screen prompt.
  - Or simply launch 'Switch2ProBridge.exe' and click the orange 
    "Install Driver" button in the app if prompted.

----------------------------------------------------------------------
2. HOW TO CONNECT
----------------------------------------------------------------------
* VIA USB-C CABLE:
  1. Plug your controller into any USB port on your PC.
  2. Launch 'Switch2ProBridge.exe'.
  3. The app will immediately recognize the controller at 250 Hz!

* VIA BLUETOOTH:
  1. Hold the small circular 'Sync' button on the top edge of your 
     controller until the 4 LEDs start flashing in a back-and-forth wave.
  2. Launch 'Switch2ProBridge.exe' and click "Connect (Auto)".
  3. When connected, Player 1 LED will stay solid green.

----------------------------------------------------------------------
3. PLAYING GAMES
----------------------------------------------------------------------
All Windows games (Steam, Xbox Game Pass, Epic, EA, Ubisoft, emulators) 
will automatically recognize your controller as an Xbox 360 controller.

Rear Paddle Buttons (GL & GR):
You can map the rear grip buttons to any controller button (or disable them) 
under the Settings menu in the app.

----------------------------------------------------------------------
DISCLAIMER
----------------------------------------------------------------------
This project is an independent open-source utility developed solely 
for hardware interoperability on Windows PCs. It is not affiliated 
with, endorsed by, or sponsored by Nintendo Co., Ltd. or Microsoft 
Corporation. All trademarks belong to their respective owners.
"""


def main():
    print("=" * 70)
    print(" Switch 2 Pro Controller Connector - Standalone Executable Builder")
    print("=" * 70)

    # 1. Clean previous build artifacts
    for d in [DIST_DIR, BUILD_DIR]:
        if os.path.exists(d):
            print(f"Cleaning previous build directory: {d}...")
            shutil.rmtree(d, ignore_errors=True)

    # 2. Run PyInstaller
    print(f"\n[1/3] Compiling standalone executable using PyInstaller...")
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--clean",
        SPEC_FILE,
    ]
    res = subprocess.run(cmd, cwd=PROJECT_DIR)
    if res.returncode != 0:
        print("\n[ERROR] PyInstaller build failed!")
        sys.exit(res.returncode)

    exe_path = os.path.join(DIST_DIR, "Switch2ProBridge.exe")
    if not os.path.exists(exe_path):
        print(f"\n[ERROR] Expected output not found: {exe_path}")
        sys.exit(1)

    exe_size_mb = os.path.getsize(exe_path) / (1024 * 1024)
    print(f"\n>>> Build Successful! Output: {exe_path} ({exe_size_mb:.1f} MB)")

    # 3. Create Distribution Package
    print("\n[2/3] Assembling distribution release package...")
    release_name = "Switch2ProBridge-v1.0.0-Windows-x64"
    release_dir = os.path.join(DIST_DIR, release_name)
    os.makedirs(release_dir, exist_ok=True)

    # Copy binary
    shutil.copy2(exe_path, os.path.join(release_dir, "Switch2ProBridge.exe"))

    # Copy driver installer
    drivers_dir = os.path.join(release_dir, "drivers")
    os.makedirs(drivers_dir, exist_ok=True)
    shutil.copy2(
        os.path.join(PROJECT_DIR, "drivers", "ViGEmBusSetup_x64.msi"),
        os.path.join(drivers_dir, "ViGEmBusSetup_x64.msi"),
    )

    # Write QuickStart instructions
    quickstart_path = os.path.join(release_dir, "README_QUICKSTART.txt")
    with open(quickstart_path, "w", encoding="utf-8") as f:
        f.write(QUICKSTART_TEXT)

    # Copy LICENSE
    if os.path.exists(os.path.join(PROJECT_DIR, "LICENSE")):
        shutil.copy2(os.path.join(PROJECT_DIR, "LICENSE"), os.path.join(release_dir, "LICENSE.txt"))

    # 4. Create ZIP archive for easy sharing / GitHub Releases
    print("\n[3/3] Creating portable ZIP archive for release distribution...")
    zip_path = os.path.join(DIST_DIR, f"{release_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(release_dir):
            for file in files:
                abs_file = os.path.join(root, file)
                rel_path = os.path.relpath(abs_file, DIST_DIR)
                zf.write(abs_file, arcname=rel_path)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(" ALL PACKAGING COMPLETE!")
    print("=" * 70)
    print(f" Standalone Binary:  {exe_path} ({exe_size_mb:.1f} MB)")
    print(f" Release Directory:  {release_dir}")
    print(f" Release ZIP:        {zip_path} ({zip_size_mb:.1f} MB)")
    print("=" * 70)
    print("Non-technical users can now simply download and run Switch2ProBridge.exe")
    print("without having Python, uv, or any libraries installed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
