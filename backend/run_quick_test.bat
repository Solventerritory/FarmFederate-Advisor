@echo off
REM Quick start script for Federated Learning Comparison
REM Windows Batch Script

echo ========================================
echo Federated Learning Comprehensive Test
echo LLM + ViT + VLM Comparison
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo [Step 1/3] Checking dependencies...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] PyTorch not found. Installing dependencies...
    echo This may take several minutes...
    pip install -r requirements_federated.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [OK] Dependencies found
)

echo.
echo [Step 2/3] Running quick test...
echo This will train 3 models (Flan-T5-Small, ViT-Base, CLIP-Base)
echo Expected duration: 5-15 minutes depending on hardware
echo.

python run_federated_comprehensive.py --quick_test

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [Step 3/3] Opening results...
echo.

REM Check if results directory exists
if exist "results\comparisons" (
    echo [SUCCESS] Training completed!
    echo.
    echo Results saved to: results\
    echo Plots saved to: results\comparisons\
    echo.
    echo Opening results directory...
    start explorer "results\comparisons"
) else (
    echo [WARNING] Results directory not found
)

echo.
echo ========================================
echo QUICK TEST COMPLETE
echo ========================================
echo.
echo To run full comparison, execute:
echo   python run_federated_comprehensive.py --full
echo.
echo To customize training, see:
echo   README_FEDERATED_COMPARISON.md
echo.

pause
