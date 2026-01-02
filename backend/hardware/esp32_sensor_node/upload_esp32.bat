@echo off
echo ╔════════════════════════════════════════════════════════════╗
echo ║  ESP32 FIRMWARE UPLOAD SCRIPT                             ║
echo ╚════════════════════════════════════════════════════════════╝
echo.
echo 📂 Project: esp32_sensor_node
echo 🔌 Port: COM7
echo 📦 Files: bootloader.bin, partitions.bin, firmware.bin
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
echo 🔴 IMPORTANT: Put ESP32 in BOOT MODE
echo.
echo    Method 1 (With BOOT button):
echo       1. HOLD the BOOT button
echo       2. While holding BOOT, press and release EN/RST
echo       3. Release BOOT
echo.
echo    Method 2 (No BOOT button):
echo       Unplug USB, hold GPIO0 to GND, plug USB back in
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.
pause
echo.
echo 🚀 Starting upload...
echo.

C:\Users\USER_HP\miniconda3\python.exe "C:\Users\USER_HP\.platformio\packages\tool-esptoolpy\esptool.py" --chip esp32 --port COM7 --baud 460800 --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 40m --flash_size 4MB 0x1000 .pio\build\esp32dev\bootloader.bin 0x8000 .pio\build\esp32dev\partitions.bin 0x10000 .pio\build\esp32dev\firmware.bin

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ═══════════════════════════════════════════════════════════
    echo    ✅ UPLOAD SUCCESSFUL!
    echo ═══════════════════════════════════════════════════════════
    echo.
    echo 🔄 ESP32 will reset automatically
    echo 📡 Data should start flowing to COM7
    echo.
    echo Next steps:
    echo    1. Close this window
    echo    2. Run USB Serial Reader to receive sensor data
    echo.
) else (
    echo.
    echo ═══════════════════════════════════════════════════════════
    echo    ❌ UPLOAD FAILED
    echo ═══════════════════════════════════════════════════════════
    echo.
    echo Troubleshooting:
    echo    1. Check COM7 is not used by another program
    echo    2. Try unplugging and replugging USB
    echo    3. Make sure ESP32 entered boot mode correctly
    echo    4. Try using Arduino IDE instead
    echo.
)

pause
