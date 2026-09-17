@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: STOP_LKA.bat
:: APEX LKA — Clean Shutdown (Windows)
::
:: Stops the Streamlit process started by START_LKA.bat.
:: Only kills the specific process on port 8501.
:: Does NOT kill unrelated Python processes.
::
:: Does NOT modify any ML pipeline, model, or dataset.
:: ============================================================

title APEX LKA — Stopping...
color 0C

:: ── Locate project root ─────────────────────────────────────────────────
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

set "PORT=8501"
set "PID_FILE=%PROJECT_ROOT%\.lka_pid"

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║          APEX LKA  —  Shutdown                       ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ── Check if anything is running on the port ───────────────────────────
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% neq 0 (
    echo  [INFO]  No process found listening on port %PORT%.
    echo  [INFO]  APEX LKA is not running.
    echo.
    :: Clean up stale PID file if present
    if exist "%PID_FILE%" del /f /q "%PID_FILE%"
    echo  Press any key to close.
    pause >nul
    exit /b 0
)

:: ── Find the PID on port 8501 ───────────────────────────────────────────
set "TARGET_PID="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    set "TARGET_PID=%%p"
)

if not defined TARGET_PID (
    echo  [WARN]  Could not determine PID for port %PORT%.
    echo.
    pause >nul
    exit /b 1
)

echo  [STOP]  Terminating APEX LKA process...
echo          Port : %PORT%
echo          PID  : !TARGET_PID!
echo.

:: ── Verify this PID is a Python/Streamlit process (safety check) ────────
set "IS_PYTHON=0"
for /f "tokens=1" %%n in ('tasklist /fi "PID eq !TARGET_PID!" /fo csv /nh 2^>nul') do (
    echo %%n | findstr /i "python\|streamlit" >nul 2>&1
    if !errorlevel! equ 0 set "IS_PYTHON=1"
)

if "!IS_PYTHON!"=="0" (
    echo  [WARN]  PID !TARGET_PID! does not appear to be a Python/Streamlit process.
    echo  [WARN]  Process name:
    tasklist /fi "PID eq !TARGET_PID!" /fo csv /nh 2>nul
    echo.
    echo  Proceed with termination? (Y/N)
    set /p CONFIRM=  > 
    if /i not "!CONFIRM!"=="Y" (
        echo  Aborted.
        pause >nul
        exit /b 0
    )
)

:: ── Terminate the process ───────────────────────────────────────────────
taskkill /PID !TARGET_PID! /F >nul 2>&1
if %errorlevel% equ 0 (
    echo  [OK]    Process !TARGET_PID! terminated.
) else (
    echo  [WARN]  taskkill returned an error for PID !TARGET_PID!.
    echo  [WARN]  The process may have already stopped.
)

:: ── Clean up files ──────────────────────────────────────────────────────
if exist "%PID_FILE%"           del /f /q "%PID_FILE%"
if exist "%PROJECT_ROOT%\.lka_server.log" (
    echo  [LOG]   Server log kept at: .lka_server.log
)

:: ── Verify port is now free ─────────────────────────────────────────────
timeout /t 2 /nobreak >nul
netstat -ano | findstr ":%PORT% " | findstr "LISTENING" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ╔══════════════════════════════════════════════════════╗
    echo  ║  APEX LKA stopped successfully.                      ║
    echo  ║  Port %PORT% is now free.                             ║
    echo  ╚══════════════════════════════════════════════════════╝
) else (
    echo.
    echo  [WARN]  Port %PORT% is still in use — may need a moment to release.
    echo  [WARN]  Try running STOP_LKA.bat again in a few seconds.
)

echo.
echo  Press any key to close.
pause >nul
endlocal
