@echo off
REM Wrapper to activate the specified venv and run detect.py (for SharpAI/Aegis on Windows)
SETLOCAL ENABLEDELAYEDEXPANSION

REM Prefer environment variable SKILL_VENV or VENV_PATH, else default to D:\commange\venv
if defined SKILL_VENV (
  set VENV=%SKILL_VENV%
) else if defined VENV_PATH (
  set VENV=%VENV_PATH%
) else (
  set VENV=D:\commange\venv
)

REM Resolve script directory
set SCRIPT_DIR=%~dp0
set SKILL_DIR=%SCRIPT_DIR%\..
pushd %SKILL_DIR%

REM Activate venv
if exist "%VENV%\Scripts\activate.bat" (
  call "%VENV%\Scripts\activate.bat"
) else (
  echo [icebag-counter] ERROR: Activate script not found at %VENV%\Scripts\activate.bat >&2
  popd
  exit /b 1
)

REM Use python from venv explicitly
set PYEXE=%VENV%\Scripts\python.exe
if not exist "%PYEXE%" (
  echo [icebag-counter] ERROR: python.exe not found in %PYEXE% >&2
  popd
  exit /b 1
)

REM Run detect.py - forward any args and pipe I/O through
"%PYEXE%" "%SKILL_DIR%\scripts\detect.py" %*

popd
ENDLOCAL
exit /b 0
