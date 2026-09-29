@echo off
title AI Driver Drowsiness Detection ^& Alert System
echo =====================================================================
echo    AI DRIVER DROWSINESS DETECTION ^& ALERT SYSTEM
echo =====================================================================
echo.
echo Launching application...
echo.

cd /d "%~dp0"
python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =====================================================================
    echo [ERROR] The application stopped unexpectedly with error code %ERRORLEVEL%.
    echo Please make sure dependencies are installed:
    echo    pip install -r requirements.txt
    echo =====================================================================
    echo.
    pause
)
