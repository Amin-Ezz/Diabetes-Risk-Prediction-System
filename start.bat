@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ============================================================
:: Diabetes Risk Prediction - Windows Application Launcher
:: ============================================================

:: 1. Detect project root reliably (handles spaces in path)
cd /d "%~dp0"
title Diabetes Risk Prediction System

echo ============================================================
echo     DIABETES RISK PREDICTION SYSTEM - Deep Learning
echo ============================================================
echo.

:: 2. Check if Python is installed
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in your system PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('python --version 2^>^&1') do set "PY_VER=%%v"
echo [INFO] Detected Python: %PY_VER%

:: 3. Determine Python Environment (use .venv if available, or active Python if packages exist)
set "RUN_PYTHON="
if exist ".venv\Scripts\python.exe" (
    set "RUN_PYTHON=%~dp0.venv\Scripts\python.exe"
    echo [INFO] Using existing local virtual environment: .venv
) else (
    :: Check if active python already has Streamlit and TensorFlow installed
    python -c "import streamlit, tensorflow, sklearn" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] Found required libraries in active Python environment.
        set "RUN_PYTHON=python"
    ) else (
        echo [INFO] Initializing local virtual environment in .venv ...
        python -m venv .venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment.
            pause
            exit /b 1
        )
        set "RUN_PYTHON=%~dp0.venv\Scripts\python.exe"
    )
)

:: 4. Check dependencies in the selected python environment
if "%RUN_PYTHON%"=="python" (
    echo [OK] Using active Python environment with installed dependencies.
) else (
    if not exist ".venv\.deps_installed" (
        echo [INFO] Installing required dependencies from requirements.txt ...
        "%RUN_PYTHON%" -m pip install --upgrade pip
        "%RUN_PYTHON%" -m pip install -r requirements.txt
        if errorlevel 1 (
            echo [ERROR] Dependency installation encountered an error.
            pause
            exit /b 1
        )
        echo installed > ".venv\.deps_installed"
        echo [OK] Dependencies installed successfully.
    ) else (
        echo [OK] Dependencies already verified in .venv.
    )
)

:: 5. Check whether required trained model files exist
if not exist "models\multioutput_nn.keras" (
    echo.
    echo [WARNING] Trained model artifact models\multioutput_nn.keras was not found.
    echo [INFO] Launching automated model training pipeline now ...
    echo.
    "%RUN_PYTHON%" scripts\train.py
    if errorlevel 1 (
        echo [ERROR] Model training failed.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Model training completed successfully.
)

if not exist "models\preprocessor.joblib" (
    echo [ERROR] Preprocessor artifact models\preprocessor.joblib is missing.
    echo Please run: "%RUN_PYTHON%" scripts\train.py
    pause
    exit /b 1
)

:: 6. Launch Streamlit application and open default browser
echo.
echo ============================================================
echo [INFO] Launching Streamlit GUI at: http://localhost:8501
echo [INFO] Press Ctrl+C in this terminal window to stop the server.
echo ============================================================
echo.

start "" "http://localhost:8501"
"%RUN_PYTHON%" -m streamlit run app\streamlit_app.py --server.headless=true --server.port=8501

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
    exit /b 1
)

endlocal
