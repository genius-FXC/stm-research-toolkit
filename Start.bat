@echo off
cd /d "%~dp0"
if not exist "%~dp0.venv\Scripts\pythonw.exe" (
  echo Run Install.bat first.
  pause
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\configure.py"
if errorlevel 1 (pause & exit /b 1)
start "STM Research Toolkit" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app\main.py"
