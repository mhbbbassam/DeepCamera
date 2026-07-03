@echo off
REM Enhanced deploy.bat for icebag-counter (Windows)
REM Usage: run from skills/detection/icebag-counter or let SKILL runner call it
SETLOCAL ENABLEDELAYEDEXPANSION

REM Default venv path - change if your venv is elsewhere
IF NOT DEFINED VENV_PATH (
  SET VENV_PATH=D:\commange\venv
)

ECHO Deploying icebag-counter skill
ECHO VENV_PATH=%VENV_PATH%

IF EXIST "%VENV_PATH%\Scripts\activate.bat" (
  ECHO Activating venv at %VENV_PATH%
  CALL "%VENV_PATH%\Scripts\activate.bat"
) ELSE (
  ECHO No venv found at %VENV_PATH%, creating a new venv at this location
  python -m venv "%VENV_PATH%"
  IF ERRORLEVEL 1 (
    ECHO Failed to create venv. Ensure python is on PATH.
    GOTO :EOF
  )
  CALL "%VENV_PATH%\Scripts\activate.bat"
)

REM Upgrade pip and install requirements
python -m pip install --upgrade pip setuptools wheel
IF EXIST "requirements_cuda.txt" (
  pip install -r "requirements_cuda.txt" || ECHO pip install failed — inspect output
) ELSE (
  ECHO requirements_cuda.txt not found
)

REM Optional: Install TensorRT wheel if env var provided
IF DEFINED TENSORRT_WHEEL (
  IF EXIST "%TENSORRT_WHEEL%" (
    ECHO Installing TensorRT wheel %TENSORRT_WHEEL%
    pip install "%TENSORRT_WHEEL%" || ECHO Failed to pip install TensorRT wheel
  ) ELSE (
    ECHO TENSORRT_WHEEL defined but file not found: %TENSORRT_WHEEL%
  )
)

ECHO Deploy complete. Use scripts\run_skill.bat to run under this venv.
ENDLOCAL
