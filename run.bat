@echo off
title Switch 2 Pro Connect
cd /d "%~dp0"
echo Starting Switch 2 Pro Controller Connector via uv...
uv run python main.py
if errorlevel 1 (
    echo.
    echo Application exited with an error.
    pause
)
