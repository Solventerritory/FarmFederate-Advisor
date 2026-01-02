@echo off
echo ================================================================
echo CP2102 USB Driver Installer for ESP32
echo ================================================================
echo.
echo Opening Silicon Labs driver download page...
echo.
echo Download: CP210x Universal Windows Driver
echo File: CP210x_Universal_Windows_Driver.zip
echo.
start https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers
echo.
echo ================================================================
echo INSTALLATION STEPS:
echo ================================================================
echo 1. Download the CP210x Universal Windows Driver ZIP
echo 2. Extract the ZIP file
echo 3. Right-click on "CP210xVCPInstaller_x64.exe"
echo 4. Select "Run as administrator"
echo 5. Follow the installation wizard
echo 6. Restart your PC (or unplug/replug ESP32)
echo.
echo After installation:
echo - Open Device Manager
echo - Your ESP32 should appear as "Silicon Labs CP210x USB to UART Bridge (COM#)"
echo - Note the COM port number
echo.
pause
