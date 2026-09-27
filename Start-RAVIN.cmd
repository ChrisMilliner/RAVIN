@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-RAVIN.ps1"
if errorlevel 1 (
    echo.
    echo RAVIN failed to start.
    pause
)
