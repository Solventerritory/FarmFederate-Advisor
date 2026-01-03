@echo off
REM ESP32 Firmware Upload Helper Script
echo ============================================================
echo ESP32 FIRMWARE UPLOAD HELPER
echo ============================================================
echo.

REM Check if Arduino IDE is installed
echo [1/4] Checking for Arduino IDE...
if exist "C:\Program Files\Arduino IDE\Arduino IDE.exe" (
    echo     FOUND: Arduino IDE 2.x
    set ARDUINO_PATH=C:\Program Files\Arduino IDE\Arduino IDE.exe
    goto :found
)
if exist "C:\Program Files (x86)\Arduino\arduino.exe" (
    echo     FOUND: Arduino IDE 1.x
    set ARDUINO_PATH=C:\Program Files (x86)\Arduino\arduino.exe
    goto :found
)

echo     NOT FOUND: Arduino IDE not installed
echo.
echo Please install Arduino IDE from:
echo https://www.arduino.cc/en/software
echo.
pause
exit /b 1

:found
echo.
echo [2/4] Opening firmware file for editing...
echo     File: esp32_complete_sensors.ino
echo.
echo     YOU MUST UPDATE THESE LINES (30-31):
echo     const char* WIFI_SSID = "YOUR_WIFI_NAME";
echo     const char* WIFI_PASSWORD = "YOUR_PASSWORD";
echo.
start "" "%ARDUINO_PATH%" "%~dp0backend\hardware\esp32_sensor_node\esp32_complete_sensors.ino"
echo     Waiting for Arduino IDE to open...
timeout /t 3 /nobreak >nul
echo.

echo [3/4] UPLOAD STEPS IN ARDUINO IDE:
echo ============================================================
echo.
echo   Step 1: UPDATE WIFI CREDENTIALS (lines 30-31)
echo           - Replace YOUR_WIFI_SSID with your WiFi name
echo           - Replace YOUR_WIFI_PASSWORD with your password
echo           - IMPORTANT: WiFi name is case-sensitive!
echo.
echo   Step 2: SELECT BOARD
echo           - Tools ^> Board ^> ESP32 Dev Module
echo.
echo   Step 3: SELECT PORT
echo           - Tools ^> Port ^> COM7
echo.
echo   Step 4: UPLOAD
echo           - Click Upload button (arrow icon)
echo           - Wait for "Done uploading"
echo.
echo   Step 5: VERIFY
echo           - Tools ^> Serial Monitor
echo           - Set baud rate to 115200
echo           - Press Reset button on ESP32
echo           - Look for "WiFi connected" and "MQTT connected"
echo.
echo ============================================================

echo.
echo [4/4] After successful upload, test with:
echo     python backend\check_sensors_usb.py
echo.
echo ============================================================
echo.
echo Arduino IDE should now be open with the firmware file.
echo Follow the steps above to upload to ESP32.
echo.
pause
