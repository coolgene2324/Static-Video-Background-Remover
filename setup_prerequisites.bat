@echo off
REM =============================================================================
REM Setup Prerequisites for the Video Background Remover Python Program
REM =============================================================================
echo Starting prerequisite setup...
echo.

REM --- Check for Python Installation ---
echo Checking for Python installation...
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: Python is not installed or not found in PATH!
    echo Please download and install the latest version of Python from:
    echo https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    goto :EOF
) ELSE (
    echo Python detected:
    python --version
)
echo.

REM --- Recommend Virtual Environment (Optional but good practice) ---
echo NOTE: It is recommended to run Python projects in a virtual environment
echo to avoid conflicts between package versions.
echo You can create one by navigating to the script's folder in the command prompt
echo and running: python -m venv venv
echo Then activate it using: venv\Scripts\activate
echo Install packages (next step) *after* activating the environment.
echo.
pause

REM --- Upgrade Pip ---
echo Upgrading pip (Python Package Installer)...
python -m pip install --upgrade pip
IF ERRORLEVEL 1 (
 echo ERROR: Failed to upgrade pip. Check your internet connection or Python setup.
 pause
 goto :EOF
)
echo Pip upgraded successfully.
echo.

REM --- Install Required Python Packages ---
echo Installing required Python packages...
echo   - opencv-python (for video/image processing)
echo   - Pillow (for image handling, GUI preview)
echo   - numpy (numerical library, dependency for OpenCV)
echo   - ttkthemes (for optional GUI themes)
echo.
python -m pip install opencv-python Pillow numpy ttkthemes
IF ERRORLEVEL 1 (
 echo ERROR: Failed to install one or more Python packages.
 echo Check your internet connection and pip output above for details.
 pause
 goto :EOF
)
echo Required Python packages installed successfully.
echo.

REM --- Check for FFmpeg ---
echo Checking for FFmpeg...
ffmpeg -version >nul 2>&1
IF ERRORLEVEL 1 (
    echo WARNING: FFmpeg is not installed or not found in PATH!
    echo The program requires FFmpeg for video encoding.
    echo Please download FFmpeg from: https://ffmpeg.org/download.html
    echo (Recommended: Download a static build, e.g., from gyan.dev for Windows)
    echo After downloading, unzip it and add the 'bin' folder inside it to your system's PATH environment variable.
    echo You may need to restart your command prompt or PC for the PATH change to take effect.
    echo.
) ELSE (
    echo FFmpeg detected in PATH.
)
echo.

REM --- Setup Complete ---
echo ====================================
echo Setup complete.
echo Remember to add FFmpeg to your PATH if you haven't already.
echo You should now be able to run the Python script.
echo ====================================
echo.
pause
