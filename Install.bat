@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -InstallPython %*
set "setup_result=%errorlevel%"
pause
exit /b %setup_result%
