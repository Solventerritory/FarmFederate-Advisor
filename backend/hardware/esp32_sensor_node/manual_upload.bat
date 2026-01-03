@echo off
echo ========================================
echo ESP32 Manual Upload Script
echo ========================================
echo.
echo INSTRUCTIONS:
echo 1. HOLD DOWN the BOOT button on ESP32
echo 2. Press any key to start upload
echo 3. KEEP HOLDING BOOT until "Writing at..." appears
echo 4. Release BOOT once upload progress shows
echo.
pause

echo.
echo Starting upload...
echo.

"%USERPROFILE%\.platformio\packages\tool-esptoolpy\esptool.py" --chip esp32 --port COM7 --baud 115200 --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 40m --flash_size detect 0x1000 .pio\build\esp32dev\bootloader.bin 0x8000 .pio\build\esp32dev\partitions.bin 0xe000 "%USERPROFILE%\.platformio\packages\framework-arduinoespressif32\tools\partitions\boot_app0.bin" 0x10000 .pio\build\esp32dev\firmware.bin

echo.
if errorlevel 1 (
    echo ========================================
    echo UPLOAD FAILED!
    echo ========================================
    echo.
    echo Common issues:
    echo - BOOT button not held at the right time
    echo - ESP32 disconnected
    echo - Wrong COM port
    echo.
    echo Try again!
) else (
    echo ========================================
    echo UPLOAD SUCCESSFUL!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Press RESET button on ESP32
    echo 2. Open Serial Monitor
    echo 3. Check sensor data
)

pause
