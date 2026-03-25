@echo off
echo ============================================
echo   House Price Prediction - Auto Launcher
echo ============================================
echo.

:: Find conda
SET CONDA_PATH=
FOR %%P IN (
  "%USERPROFILE%\anaconda3\Scripts\conda.exe"
  "%USERPROFILE%\Anaconda3\Scripts\conda.exe"
  "C:\ProgramData\Anaconda3\Scripts\conda.exe"
  "%USERPROFILE%\miniconda3\Scripts\conda.exe"
) DO (
  IF EXIST %%P SET CONDA_PATH=%%P
)

IF "%CONDA_PATH%"=="" (
  echo [ERROR] Anaconda not found. Please open Anaconda Prompt manually.
  pause
  exit /b 1
)

:: Initialize conda for this session
CALL "%USERPROFILE%\anaconda3\Scripts\activate.bat" 2>nul
IF ERRORLEVEL 1 CALL "%USERPROFILE%\Anaconda3\Scripts\activate.bat" 2>nul
IF ERRORLEVEL 1 CALL "C:\ProgramData\Anaconda3\Scripts\activate.bat" 2>nul

:: Check if environment exists
conda env list | findstr /C:"houseprice" >nul 2>&1
IF ERRORLEVEL 1 (
  echo [INFO] Creating conda environment from environment.yml...
  conda env create -f environment.yml
  IF ERRORLEVEL 1 (
    echo [ERROR] Failed to create environment. Trying pip install instead...
    pip install -r requirements.txt
    python house_price_prediction.py
    pause
    exit /b
  )
)

echo [INFO] Activating environment...
CALL conda activate houseprice

echo [INFO] Running script...
echo.
python house_price_prediction.py

echo.
echo ============================================
echo   Done! Check the folder for output files.
echo ============================================
pause
