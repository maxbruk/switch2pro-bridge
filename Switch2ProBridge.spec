# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas = [
    ('drivers/ViGEmBusSetup_x64.msi', 'drivers'),
    ('app_icon.ico', '.'),
]
binaries = []
hiddenimports = [
    'winrt.windows.devices.bluetooth',
    'winrt.windows.devices.bluetooth.genericattributeprofile',
    'winrt.windows.devices.enumeration',
    'winrt.windows.storage.streams',
    'winrt.windows.foundation',
    'ctypes',
    'ctypes.wintypes',
    'vgamepad',
    'customtkinter',
    'PIL',
    'PIL._tkinter_finder',
    'hid',
    'yaml',
    'darkdetect',
]

for pkg in ['customtkinter', 'vgamepad', 'bleak']:
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas.extend(pkg_datas)
    binaries.extend(pkg_binaries)
    hiddenimports.extend(pkg_hidden)

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['unittest', 'pytest', 'tests'],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Switch2ProBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app_icon.ico'],
)
