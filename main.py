"""Entry point for Switch 2 Pro Controller Connector (GUI & CLI modes)."""

import sys
import time
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Switch2ProConnect")


def run_cli_mode():
    """Run in headless / command-line mode without GUI."""
    from app_core import ControllerManager

    print("=" * 60)
    print(" Switch 2 Pro Controller Connector (CLI Mode)")
    print("=" * 60)
    print("Auto-connecting to Nintendo Switch 2 Pro Controller...")
    print("Connect via USB-C or press the Sync button for Bluetooth.")
    print("Press Ctrl+C to quit.\n")

    manager = ControllerManager()

    def on_conn_changed(connected: bool, transport: str, name: str, battery: int):
        if connected:
            print(f"\n>>> [CONNECTED] {name} via {transport} (Battery: {battery}%)")
            print(">>> Virtual Xbox 360 controller active and ready for all Windows games!")
        else:
            print("\n>>> [DISCONNECTED] Controller disconnected. Waiting for connection...")

    def on_status(msg: str):
        print(f"[*] {msg}")

    manager.on_connection_changed = on_conn_changed
    manager.on_status_message = on_status

    manager.start_auto_detector()
    manager.connect_single_click()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down connector...")
        manager.stop_auto_detector()
        manager.disconnect()
        print("Goodbye!")


def main():
    parser = argparse.ArgumentParser(description="One-Click Nintendo Switch 2 Pro Controller Connector for Windows")
    parser.add_argument("--cli", "--headless", action="store_true", help="Run in headless CLI mode without GUI")
    args = parser.parse_args()

    if args.cli:
        run_cli_mode()
    else:
        try:
            from gui import run_gui
            run_gui()
        except Exception as e:
            logger.error("GUI failed to start: %s. Falling back to CLI mode.", e)
            run_cli_mode()


if __name__ == "__main__":
    main()
