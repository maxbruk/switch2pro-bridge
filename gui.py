"""Modern CustomTkinter GUI for Switch 2 Pro Controller Connector."""

import sys
import os
import json
import ctypes
import tkinter as tk
import customtkinter as ctk
from typing import Optional

# Enable 1ms high-resolution timer for Windows scheduler
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

# Set Windows AppUserModelID so taskbar resolves custom icon properly
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MaxBruk.Switch2ProConnect.Bridge")
except Exception:
    pass

from app_core import ControllerManager
from report_parser import ControllerState
from virtual_xinput import check_vigem_driver_available, install_vigem_driver

# Set application theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class Switch2ProApp(ctk.CTk):
    """Main Application Window for Switch 2 Pro Controller Connector."""

    def __init__(self):
        super().__init__()

        self.title("Switch 2 Pro Connect")
        self.geometry("640x800")
        self.minsize(580, 720)

        # High-DPI Windows Taskbar and Window Icon Setup
        self._setup_window_icons()

        # Core Manager
        self.manager = ControllerManager()
        self.manager.on_connection_changed = self._on_connection_changed
        self.manager.on_input_update = self._on_input_update
        self.manager.on_status_message = self._on_status_message

        # GUI State
        self.is_connected = False
        self.vigem_ready = check_vigem_driver_available()

        # Ultra-low latency visualizer state (decoupled from 250Hz input thread)
        self._latest_state: Optional[ControllerState] = None
        self._prev_buttons: dict[str, bool] = {}
        self._prev_lx: float = 0.0
        self._prev_ly: float = 0.0
        self._prev_rx: float = 0.0
        self._prev_ry: float = 0.0
        self._prev_battery: int = -1

        self._build_ui()
        self._load_settings()

        # Start smooth 60 FPS visualizer tick loop
        self._schedule_visualizer_tick()

        # Start auto detector by default
        self.manager.start_auto_detector()

        # Initial driver check
        self._refresh_driver_status()

    def _setup_window_icons(self):
        """Set up razor-sharp native icons for both Taskbar and Window Titlebar."""
        for icon_path in [
            os.path.join(getattr(sys, "_MEIPASS", ""), "app_icon.ico"),
            os.path.join(os.path.dirname(sys.executable), "app_icon.ico"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "app_icon.ico"),
        ]:
            if os.path.exists(icon_path):
                # Baseline Tkinter iconbitmap
                try:
                    self.iconbitmap(icon_path)
                except Exception:
                    pass

                # Explicit Win32 WM_SETICON for native taskbar and titlebar resolution
                try:
                    abs_path = os.path.abspath(icon_path)
                    hwnd = self.winfo_id()
                    user32 = ctypes.windll.user32
                    parent = user32.GetParent(hwnd)
                    toplevel = parent if parent else hwnd

                    IMAGE_ICON = 1
                    LR_LOADFROMFILE = 0x00000010
                    LR_DEFAULTSIZE = 0x00000040
                    WM_SETICON = 0x0080

                    # 1. Big Icon (Taskbar: loads system native SM_CXICON, avoiding 16x16 stretch blur)
                    h_big = user32.LoadImageW(None, abs_path, IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE)
                    if h_big:
                        user32.SendMessageW(toplevel, WM_SETICON, 1, h_big)

                    # 2. Small Icon (Titlebar: system native SM_CXSMICON)
                    sm_cx = user32.GetSystemMetrics(49)
                    sm_cy = user32.GetSystemMetrics(50)
                    h_sm = user32.LoadImageW(None, abs_path, IMAGE_ICON, sm_cx, sm_cy, LR_LOADFROMFILE)
                    if h_sm:
                        user32.SendMessageW(toplevel, WM_SETICON, 0, h_sm)
                except Exception:
                    pass
                break

    def _build_ui(self):
        """Construct the UI widgets."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # 1. Header Frame
        header_frame = ctk.CTkFrame(self, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=24, pady=(20, 10), sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)

        title_label = ctk.CTkLabel(
            header_frame,
            text="Switch 2 Pro Connect",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#E6E6E6",
        )
        title_label.grid(row=0, column=0, sticky="w")

        subtitle = ctk.CTkLabel(
            header_frame,
            text="One-Click Nintendo Switch 2 Controller Bridge for Windows",
            font=ctk.CTkFont(size=13),
            text_color="#8E8E93",
        )
        subtitle.grid(row=1, column=0, sticky="w")

        self.status_pill = ctk.CTkLabel(
            header_frame,
            text="● DISCONNECTED",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#8E8E93",
            fg_color="#2C2C2E",
            corner_radius=12,
            padx=14,
            pady=4,
        )
        self.status_pill.grid(row=0, column=1, rowspan=2, sticky="e")

        # 2. Main Content Frame (Scrollable)
        content_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        content_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        content_frame.grid_columnconfigure(0, weight=1)

        # --- Driver Alert Banner (if ViGEmBus missing) ---
        self.driver_card = ctk.CTkFrame(content_frame, fg_color="#3A2E10", border_color="#D97706", border_width=1)
        self.driver_card.grid_columnconfigure(0, weight=1)

        driver_label = ctk.CTkLabel(
            self.driver_card,
            text="⚠️ Xbox Virtual Controller Driver Required\nInstall ViGEmBus driver so Windows games recognize the controller as an Xbox 360 controller.",
            font=ctk.CTkFont(size=12),
            text_color="#FDE68A",
            justify="left",
        )
        driver_label.grid(row=0, column=0, padx=16, pady=(12, 6), sticky="w")

        self.btn_install_driver = ctk.CTkButton(
            self.driver_card,
            text="🛠 Install Driver (1-Click)",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#D97706",
            hover_color="#B45309",
            command=self._install_driver_clicked,
        )
        self.btn_install_driver.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

        # --- HERO ACTION SECTION ---
        hero_frame = ctk.CTkFrame(content_frame, fg_color="#1C1C1E", corner_radius=16)
        hero_frame.grid(row=1, column=0, pady=(10, 16), sticky="ew")
        hero_frame.grid_columnconfigure(0, weight=1)

        self.btn_main = ctk.CTkButton(
            hero_frame,
            text="⚡ CONNECT CONTROLLER",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=60,
            corner_radius=14,
            fg_color="#0284C7",
            hover_color="#0369A1",
            command=self._main_action_clicked,
        )
        self.btn_main.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.lbl_status = ctk.CTkLabel(
            hero_frame,
            text="Ready. Plug in via USB-C or hold Sync on Bluetooth.",
            font=ctk.CTkFont(size=13),
            text_color="#9CA3AF",
        )
        self.lbl_status.grid(row=1, column=0, padx=20, pady=(0, 16))

        # --- Device Details Card ---
        self.info_card = ctk.CTkFrame(content_frame, fg_color="#1C1C1E", corner_radius=14)
        self.info_card.grid(row=2, column=0, pady=(0, 16), sticky="ew")
        self.info_card.grid_columnconfigure((0, 1, 2), weight=1)

        # Connection Mode
        ctk.CTkLabel(self.info_card, text="CONNECTION", font=ctk.CTkFont(size=11, weight="bold"), text_color="#6B7280").grid(row=0, column=0, padx=12, pady=(12, 2))
        self.lbl_mode = ctk.CTkLabel(self.info_card, text="None", font=ctk.CTkFont(size=14, weight="bold"), text_color="#E5E7EB")
        self.lbl_mode.grid(row=1, column=0, padx=12, pady=(0, 12))

        # Device Model
        ctk.CTkLabel(self.info_card, text="CONTROLLER", font=ctk.CTkFont(size=11, weight="bold"), text_color="#6B7280").grid(row=0, column=1, padx=12, pady=(12, 2))
        self.lbl_device = ctk.CTkLabel(self.info_card, text="Not Connected", font=ctk.CTkFont(size=14, weight="bold"), text_color="#E5E7EB")
        self.lbl_device.grid(row=1, column=1, padx=12, pady=(0, 12))

        # Battery
        ctk.CTkLabel(self.info_card, text="BATTERY", font=ctk.CTkFont(size=11, weight="bold"), text_color="#6B7280").grid(row=0, column=2, padx=12, pady=(12, 2))
        self.lbl_battery = ctk.CTkLabel(self.info_card, text="-- %", font=ctk.CTkFont(size=14, weight="bold"), text_color="#10B981")
        self.lbl_battery.grid(row=1, column=2, padx=12, pady=(0, 12))

        # --- Interactive Controller Visualizer ---
        viz_frame = ctk.CTkFrame(content_frame, fg_color="#1C1C1E", corner_radius=14)
        viz_frame.grid(row=3, column=0, pady=(0, 16), sticky="ew")
        viz_frame.grid_columnconfigure((0, 1), weight=1)

        viz_title = ctk.CTkLabel(
            viz_frame,
            text="INPUT TESTER & VISUALIZER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#6B7280",
        )
        viz_title.grid(row=0, column=0, columnspan=2, padx=16, pady=(12, 8), sticky="w")

        # Canvas for Left and Right Analog Sticks
        sticks_container = ctk.CTkFrame(viz_frame, fg_color="transparent")
        sticks_container.grid(row=1, column=0, padx=16, pady=8, sticky="nsew")

        ctk.CTkLabel(sticks_container, text="Left Stick", font=ctk.CTkFont(size=12), text_color="#9CA3AF").pack()
        self.canvas_ls = tk.Canvas(sticks_container, width=100, height=100, bg="#111827", highlightthickness=0)
        self.canvas_ls.pack(pady=4)
        self._init_stick_canvas(self.canvas_ls)

        sticks_r_container = ctk.CTkFrame(viz_frame, fg_color="transparent")
        sticks_r_container.grid(row=1, column=1, padx=16, pady=8, sticky="nsew")

        ctk.CTkLabel(sticks_r_container, text="Right Stick", font=ctk.CTkFont(size=12), text_color="#9CA3AF").pack()
        self.canvas_rs = tk.Canvas(sticks_r_container, width=100, height=100, bg="#111827", highlightthickness=0)
        self.canvas_rs.pack(pady=4)
        self._init_stick_canvas(self.canvas_rs)

        # Button badges grid (Face, Shoulders, D-Pad, System, Clicks, and Grip Paddles)
        btn_grid = ctk.CTkFrame(viz_frame, fg_color="transparent")
        btn_grid.grid(row=2, column=0, columnspan=2, padx=16, pady=(8, 16), sticky="ew")
        for col_i in range(8):
            btn_grid.grid_columnconfigure(col_i, weight=1)

        self.btn_badges = {}
        button_layout = [
            ["A", "B", "X", "Y", "L", "R", "ZL", "ZR"],
            ["D-Up", "D-Dn", "D-Lt", "D-Rt", "-", "+", "Home", "Cap"],
            ["L3", "R3", "GL", "GR"],
        ]

        for row_idx, row_btns in enumerate(button_layout):
            start_col = (8 - len(row_btns)) // 2 if len(row_btns) < 8 else 0
            for i, name in enumerate(row_btns):
                badge = ctk.CTkLabel(
                    btn_grid,
                    text=name,
                    font=ctk.CTkFont(size=11, weight="bold"),
                    text_color="#6B7280",
                    fg_color="#262626",
                    corner_radius=6,
                    padx=8,
                    pady=4,
                )
                badge.grid(row=row_idx, column=start_col + i, padx=3, pady=3, sticky="ew")
                self.btn_badges[name] = badge

        # --- Preferences Card ---
        pref_frame = ctk.CTkFrame(content_frame, fg_color="#1C1C1E", corner_radius=14)
        pref_frame.grid(row=4, column=0, pady=(0, 20), sticky="ew")
        pref_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            pref_frame,
            text="SETTINGS",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#6B7280",
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(12, 8), sticky="w")

        # Button Layout
        ctk.CTkLabel(pref_frame, text="Button Layout:", font=ctk.CTkFont(size=13)).grid(row=1, column=0, padx=16, pady=8, sticky="w")
        self.layout_selector = ctk.CTkSegmentedButton(
            pref_frame,
            values=["Nintendo (Physical)", "Xbox (Letters)"],
            command=self._on_layout_changed,
        )
        self.layout_selector.set("Nintendo (Physical)")
        self.layout_selector.grid(row=1, column=1, padx=16, pady=8, sticky="ew")

        # Auto Connect Switch
        ctk.CTkLabel(pref_frame, text="Auto-Connect on USB/Sync:", font=ctk.CTkFont(size=13)).grid(row=2, column=0, padx=16, pady=8, sticky="w")
        self.switch_auto = ctk.CTkSwitch(
            pref_frame,
            text="",
            command=self._on_auto_switch_changed,
        )
        self.switch_auto.select()
        self.switch_auto.grid(row=2, column=1, padx=16, pady=8, sticky="e")

        # Deadzone Slider
        ctk.CTkLabel(pref_frame, text="Stick Deadzone:", font=ctk.CTkFont(size=13)).grid(row=3, column=0, padx=16, pady=8, sticky="w")
        deadzone_box = ctk.CTkFrame(pref_frame, fg_color="transparent")
        deadzone_box.grid(row=3, column=1, padx=16, pady=8, sticky="ew")
        deadzone_box.grid_columnconfigure(0, weight=1)

        self.slider_deadzone = ctk.CTkSlider(
            deadzone_box,
            from_=0.0,
            to=0.20,
            number_of_steps=20,
            command=self._on_deadzone_changed,
        )
        self.slider_deadzone.set(0.05)
        self.slider_deadzone.grid(row=0, column=0, sticky="ew")
        self.lbl_deadzone_val = ctk.CTkLabel(deadzone_box, text="5%", font=ctk.CTkFont(size=12), width=35)
        self.lbl_deadzone_val.grid(row=0, column=1, padx=(6, 0))

        # Paddle choices dictionary for flexible mapping
        self._paddle_choices = {
            "L3 (Left Stick Click)": "L3",
            "R3 (Right Stick Click)": "R3",
            "LB (Left Bumper)": "LB",
            "RB (Right Bumper)": "RB",
            "LT (Left Trigger)": "LT",
            "RT (Right Trigger)": "RT",
            "A (Bottom Face)": "A",
            "B (Right Face)": "B",
            "X (Left Face)": "X",
            "Y (Top Face)": "Y",
            "Disabled": "Disabled",
        }
        paddle_options_list = list(self._paddle_choices.keys())

        # Grip Left (GL) Back Paddle Mapping
        ctk.CTkLabel(pref_frame, text="GL Paddle (Back-L):", font=ctk.CTkFont(size=13)).grid(
            row=4, column=0, padx=16, pady=8, sticky="w"
        )
        self.gl_selector = ctk.CTkOptionMenu(
            pref_frame,
            values=paddle_options_list,
            command=self._on_gl_mapping_changed,
            fg_color="#262626",
            button_color="#0284C7",
            button_hover_color="#0369A1",
        )
        self.gl_selector.set("L3 (Left Stick Click)")
        self.gl_selector.grid(row=4, column=1, padx=16, pady=8, sticky="ew")

        # Grip Right (GR) Back Paddle Mapping
        ctk.CTkLabel(pref_frame, text="GR Paddle (Back-R):", font=ctk.CTkFont(size=13)).grid(
            row=5, column=0, padx=16, pady=(8, 16), sticky="w"
        )
        self.gr_selector = ctk.CTkOptionMenu(
            pref_frame,
            values=paddle_options_list,
            command=self._on_gr_mapping_changed,
            fg_color="#262626",
            button_color="#0284C7",
            button_hover_color="#0369A1",
        )
        self.gr_selector.set("R3 (Right Stick Click)")
        self.gr_selector.grid(row=5, column=1, padx=16, pady=(8, 16), sticky="ew")

        # 3. Footer Frame (Author Attribution & Link)
        footer_frame = ctk.CTkFrame(self, fg_color="transparent")
        footer_frame.grid(row=2, column=0, padx=24, pady=(4, 12), sticky="ew")

        def open_github(event=None):
            import webbrowser
            webbrowser.open_new_tab("https://github.com/maxbruk")

        lbl_footer = ctk.CTkLabel(
            footer_frame,
            text="Developed by Max Bruk",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#9CA3AF",
            cursor="hand2",
        )
        lbl_footer.pack(anchor="center")
        lbl_footer.bind("<Button-1>", open_github)
        lbl_footer.bind("<Enter>", lambda e: lbl_footer.configure(text_color="#38BDF8"))
        lbl_footer.bind("<Leave>", lambda e: lbl_footer.configure(text_color="#9CA3AF"))

    def _init_stick_canvas(self, canvas: tk.Canvas):
        """Draw circular guide and dot on stick canvas."""
        canvas.delete("all")
        # Center circle guide
        canvas.create_oval(10, 10, 90, 90, outline="#374151", width=2)
        canvas.create_line(50, 10, 50, 90, fill="#1F2937", dash=(2, 2))
        canvas.create_line(10, 50, 90, 50, fill="#1F2937", dash=(2, 2))
        # Analog thumb position
        canvas.create_oval(44, 44, 56, 56, fill="#0284C7", outline="#38BDF8", tags="dot")

    def _update_stick_canvas(self, canvas: tk.Canvas, x_norm: float, y_norm: float):
        """Update position of the stick indicator."""
        # x_norm in [-1.0, 1.0], y_norm in [-1.0, 1.0] (invert y for canvas coords)
        cx = 50 + x_norm * 36
        cy = 50 - y_norm * 36
        canvas.coords("dot", cx - 6, cy - 6, cx + 6, cy + 6)

    def _refresh_driver_status(self):
        """Show or hide the ViGEmBus installation banner."""
        self.vigem_ready = check_vigem_driver_available()
        if self.vigem_ready:
            self.driver_card.grid_remove()
        else:
            self.driver_card.grid(row=0, column=0, pady=(0, 12), sticky="ew")

    def _install_driver_clicked(self):
        """Trigger elevated ViGEmBus driver installer."""
        self.btn_install_driver.configure(state="disabled", text="Installing ViGEmBus...")
        success = install_vigem_driver()
        if success:
            self._on_status_message("Installer launched. Please confirm the Windows UAC prompt.")
            # Check for driver readiness periodically
            self.after(3000, self._check_driver_installed_poll)
        else:
            self.btn_install_driver.configure(state="normal", text="Retry Install Driver")
            self._on_status_message("Failed to launch installer. Please check drivers/ViGEmBusSetup_x64.msi.")

    def _check_driver_installed_poll(self):
        """Poll until ViGEmBus driver is ready."""
        if check_vigem_driver_available():
            self._refresh_driver_status()
            self._on_status_message("ViGEmBus installed and ready! Virtual Xbox controller active.")
        else:
            self.after(2000, self._check_driver_installed_poll)

    def _main_action_clicked(self):
        """Handle single-click connect/disconnect button."""
        if self.is_connected:
            self.manager.disconnect()
        else:
            self.btn_main.configure(state="disabled", text="CONNECTING...")
            self.manager.connect_single_click()

    def _get_settings_path(self) -> str:
        app_data = os.getenv("APPDATA")
        if app_data:
            folder = os.path.join(app_data, "Switch2ProConnect")
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, "settings.json")
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

    def _save_settings(self):
        try:
            settings = {
                "layout": self.manager.button_layout,
                "deadzone": self.manager.deadzone,
                "auto_connect": self.manager.auto_connect_enabled,
                "gl_mapping": self.manager.virtual_gamepad.gl_mapping,
                "gr_mapping": self.manager.virtual_gamepad.gr_mapping,
            }
            with open(self._get_settings_path(), "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        try:
            path = self._get_settings_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                settings = json.load(f)

            layout = settings.get("layout", "nintendo")
            self.manager.set_button_layout(layout)
            if hasattr(self, "layout_selector"):
                self.layout_selector.set("Nintendo (Physical)" if layout == "nintendo" else "Xbox (Letters)")

            deadzone = float(settings.get("deadzone", 0.05))
            self.manager.set_deadzone(deadzone)
            if hasattr(self, "slider_deadzone"):
                self.slider_deadzone.set(deadzone)
                self.lbl_deadzone_val.configure(text=f"{int(deadzone * 100)}%")

            auto_conn = bool(settings.get("auto_connect", True))
            self.manager.auto_connect_enabled = auto_conn
            if hasattr(self, "switch_auto"):
                if auto_conn:
                    self.switch_auto.select()
                else:
                    self.switch_auto.deselect()

            gl = settings.get("gl_mapping", "L3")
            self.manager.set_gl_mapping(gl)
            if hasattr(self, "gl_selector"):
                for k, v in self._paddle_choices.items():
                    if v == gl:
                        self.gl_selector.set(k)
                        break

            gr = settings.get("gr_mapping", "R3")
            self.manager.set_gr_mapping(gr)
            if hasattr(self, "gr_selector"):
                for k, v in self._paddle_choices.items():
                    if v == gr:
                        self.gr_selector.set(k)
                        break
        except Exception:
            pass

    def _on_layout_changed(self, value):
        layout = "nintendo" if "Nintendo" in value else "xbox"
        self.manager.set_button_layout(layout)
        self._save_settings()

    def _on_gl_mapping_changed(self, value: str):
        action = self._paddle_choices.get(value, "L3")
        self.manager.set_gl_mapping(action)
        self._on_status_message(f"GL Back Paddle mapped to {action}")
        self._save_settings()

    def _on_gr_mapping_changed(self, value: str):
        action = self._paddle_choices.get(value, "R3")
        self.manager.set_gr_mapping(action)
        self._on_status_message(f"GR Back Paddle mapped to {action}")
        self._save_settings()

    def _on_auto_switch_changed(self):
        self.manager.auto_connect_enabled = bool(self.switch_auto.get())
        self._save_settings()

    def _on_deadzone_changed(self, val):
        self.lbl_deadzone_val.configure(text=f"{int(val * 100)}%")
        self.manager.set_deadzone(float(val))
        self._save_settings()

    def _on_connection_changed(self, connected: bool, transport: str, device_name: str, battery: int):
        """Called thread-safely by ControllerManager."""
        try:
            self.after(0, self._update_connection_ui, connected, transport, device_name, battery)
        except Exception:
            pass

    def _update_connection_ui(self, connected: bool, transport: str, device_name: str, battery: int):
        self.is_connected = connected
        self.btn_main.configure(state="normal")

        if connected:
            self.btn_main.configure(
                text="⏹ DISCONNECT",
                fg_color="#DC2626",
                hover_color="#B91C1C",
            )
            color = "#10B981" if "USB" in transport else "#06B6D4"
            self.status_pill.configure(
                text=f"● CONNECTED ({transport})",
                text_color=color,
                fg_color="#064E3B" if "USB" in transport else "#164E63",
            )
            self.lbl_mode.configure(text=transport, text_color=color)
            self.lbl_device.configure(text=device_name)
            self.lbl_battery.configure(text=f"{battery}%")
        else:
            self.btn_main.configure(
                text="⚡ CONNECT CONTROLLER",
                fg_color="#0284C7",
                hover_color="#0369A1",
            )
            self.status_pill.configure(
                text="● DISCONNECTED",
                text_color="#8E8E93",
                fg_color="#2C2C2E",
            )
            self.lbl_mode.configure(text="None", text_color="#E5E7EB")
            self.lbl_device.configure(text="Not Connected")
            self.lbl_battery.configure(text="-- %")
            self._init_stick_canvas(self.canvas_ls)
            self._init_stick_canvas(self.canvas_rs)
            self._reset_button_badges()

    def _on_status_message(self, msg: str):
        """Update status message label."""
        try:
            self.after(0, lambda: self.lbl_status.configure(text=msg))
        except Exception:
            pass

    def _schedule_visualizer_tick(self):
        """Run smooth 60 FPS periodic visualizer tick decoupled from 250Hz input thread."""
        self._visualizer_tick()

    def _visualizer_tick(self):
        try:
            state = self._latest_state
            if state is not None and self.is_connected:
                self._render_live_inputs_optimized(state)
            self.after(16, self._visualizer_tick)
        except Exception:
            pass

    def _on_input_update(self, state: ControllerState):
        """Receive live input data from controller (called on high-speed USB/BLE thread).
        
        Zero GUI queue flooding: atomic store of the latest state reference.
        Processing overhead: < 50 nanoseconds, zero GIL starvation.
        """
        self._latest_state = state

    def _render_live_inputs_optimized(self, state: ControllerState):
        """Update visualizer widgets only when their underlying values change."""
        # 1. Analog sticks (update canvas only when stick moves by > 0.015)
        lx, ly = state.left_stick_x, state.left_stick_y
        if abs(lx - self._prev_lx) > 0.015 or abs(ly - self._prev_ly) > 0.015:
            self._update_stick_canvas(self.canvas_ls, lx, ly)
            self._prev_lx, self._prev_ly = lx, ly

        rx, ry = state.right_stick_x, state.right_stick_y
        if abs(rx - self._prev_rx) > 0.015 or abs(ry - self._prev_ry) > 0.015:
            self._update_stick_canvas(self.canvas_rs, rx, ry)
            self._prev_rx, self._prev_ry = rx, ry

        # 2. Battery percentage (update text only when integer changes)
        if state.battery_percentage > 0 and state.battery_percentage != self._prev_battery:
            self.lbl_battery.configure(text=f"{state.battery_percentage}%")
            self._prev_battery = state.battery_percentage

        # 3. Button highlight dirty-checking: only call badge.configure() on toggled states
        highlights = {
            "A": state.a, "B": state.b, "X": state.x, "Y": state.y,
            "L": state.l, "R": state.r, "ZL": state.zl_analog > 0.5, "ZR": state.zr_analog > 0.5,
            "D-Up": state.dpad_up, "D-Dn": state.dpad_down, "D-Lt": state.dpad_left, "D-Rt": state.dpad_right,
            "-": state.minus, "+": state.plus, "Home": state.home, "Cap": state.capture,
            "L3": state.l_thumb, "R3": state.r_thumb,
            "GL": state.gl, "GR": state.gr,
        }

        active_color = "#38BDF8"
        dim_color = "#262626"
        dim_text = "#6B7280"
        active_text = "#0F172A"

        for name, active in highlights.items():
            if self._prev_buttons.get(name) != active:
                self._prev_buttons[name] = active
                badge = self.btn_badges.get(name)
                if badge:
                    badge.configure(
                        fg_color=active_color if active else dim_color,
                        text_color=active_text if active else dim_text,
                    )

    def _reset_button_badges(self):
        """Reset badges and caches to initial dimmed state."""
        self._prev_buttons.clear()
        self._prev_lx = 0.0
        self._prev_ly = 0.0
        self._prev_rx = 0.0
        self._prev_ry = 0.0
        self._prev_battery = -1
        self._latest_state = None
        for badge in self.btn_badges.values():
            badge.configure(fg_color="#262626", text_color="#6B7280")


def run_gui():
    app = Switch2ProApp()
    app.mainloop()
