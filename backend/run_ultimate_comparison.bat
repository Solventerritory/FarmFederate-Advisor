@echo off
REM ============================================================================
REM Ultimate Model Comparison - Execution Script
REM ============================================================================
REM Runs complete model comparison and generates all visualizations
REM 
REM Author: FarmFederate Research Team
REM Date: 2026-01-08
REM ============================================================================

echo.
echo ================================================================================
echo FARMFEDERATE - ULTIMATE MODEL COMPARISON
echo ================================================================================
echo.
echo This script will:
echo   1. Train ALL models (LLM, ViT, VLM) in both centralized and federated modes
echo   2. Compare against 15+ state-of-the-art papers
echo   3. Generate 25+ comprehensive visualization plots
echo.
echo Estimated time: 1-3 hours (depending on hardware)
echo Required space: ~5GB
echo.

pause

echo.
echo [Step 1/3] Checking Python environment...
echo.

python --version
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Running model comparison...
echo.
echo This will train and evaluate all models. Please be patient...
echo.

python ultimate_model_comparison.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model comparison failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Model comparison completed!
echo.
echo [Step 3/3] Generating visualizations...
echo.

python ultimate_plotting_suite.py
if errorlevel 1 (
    echo.
    echo [ERROR] Plotting failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo COMPLETE! All comparisons and visualizations generated.
echo ================================================================================
echo.
echo Results saved to:
echo   - outputs_ultimate_comparison\results\comparison_results.csv
echo   - outputs_ultimate_comparison\results\comparison_results_*.json
echo.
echo Plots saved to:
echo   - outputs_ultimate_comparison\plots\  (25 PNG files)
echo.
echo Next steps:
echo   1. Review the summary dashboard: 25_summary_dashboard.png
echo   2. Check paper comparisons: 14_paper_comparison_bars.png
echo   3. Analyze results CSV for detailed metrics
echo   4. Read ULTIMATE_COMPARISON_README.md for interpretation
echo.

pause
