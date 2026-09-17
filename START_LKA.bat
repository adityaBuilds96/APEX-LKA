@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: START_LKA.bat
:: APEX LKA — One-Click Launcher (Windows)
::
:: Starts the existing Streamlit dashboard (dashboard/app.py)
:: using the system Python where Streamlit is installed.
:: Opens the browser automatically once the server is ready.
::
:: Does NOT modify any ML pipeline, model, or dataset.
:: ============================================================

title APEX LKA — Starting...
color 0B

:: ── Locate project root (same directory as this .bat file) ──────────────
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
cd /d "%PROJECT_ROOT%"

echo.
echo  ==========================================================
echo   APEX LKA  --  Lane Keep Assist
echo   Real-Time Lane Perception and Driver Assistance
echo  ==========================================================
echo.
echo  Project root : %PROJECT_ROOT%
echo.

:: ── Locate Python / Streamlit ────────────────────────────────────────────
set "PYTHON_EXE="
set "STREAMLIT_EXE="

:: 1) Project-local .venv
if exist "%PROJECT_ROOT%\.venv\Scripts\streamlit.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
    set "STREAMLIT_EXE=%PROJECT_ROOT%\.venv\Scripts\streamlit.exe"
    echo  [ENV]  Using project venv: .venv
    goto :env_found
)

:: 2) Project-local venv
if exist "%PROJECT_ROOT%\venv\Scripts\streamlit.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
    set "STREAMLIT_EXE=%PROJECT_ROOT%\venv\Scripts\streamlit.exe"
    echo  [ENV]  Using project venv: venv
    goto :env_found
)

:: 3) System Streamlit
where streamlit >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('where streamlit 2^>nul') do (
        set "STREAMLIT_EXE=%%i"
        goto :env_found
    )
)

echo  [ERROR] Streamlit not found on PATH.
echo  Install it:  pip install streamlit
echo.
pause
exit /b 1

:env_found
echo  [ENV]  Streamlit: %STREAMLIT_EXE%
echo.

:: ── Check dashboard entry point ─────────────────────────────────────────
set "APP_ENTRY=%PROJECT_ROOT%\dashboard\app.py"
if not exist "%APP_ENTRY%" (
    echo  [ERROR] Dashboard not found: dashboard\app.py
    echo.
    pause
    exit /b 1
)

:: ── Port ─────────────────────────────────────────────────────────────────
set "PORT=8501"
set "URL=http://localhost:%PORT%"
set "PID_FILE=%PROJECT_ROOT%\.lka_pid"
set "LOG_FILE=%PROJECT_ROOT%\.lka_server.log"

:: ── Already running? ─────────────────────────────────────────────────────
powershell -NoProfile -Command "(New-Object Net.Sockets.TcpClient).Connect('127.0.0.1',%PORT%)" >nul 2>&1
if %errorlevel% equ 0 (
    echo  [INFO]  APEX LKA already running on port %PORT%.
    echo  [INFO]  Opening browser: %URL%
    start "" "%URL%"
    pause
    exit /b 0
)

:: ── Launch Streamlit in a separate detached window ───────────────────────
echo  [START] Launching dashboard\app.py on port %PORT%...
start "APEX LKA Server" /min cmd /c ^
    "streamlit run "%APP_ENTRY%" --server.port %PORT% --server.headless false --browser.gatherUsageStats false > "%LOG_FILE%" 2>&1"

:: ── Wait for server to become ready (up to 30 s) ─────────────────────────
echo  [WAIT]  Waiting for server on port %PORT%...
set /a TRIES=0

:wait_loop
set /a TRIES+=1
:: 1-second delay via ping
ping 127.0.0.1 -n 2 >nul

:: TCP probe with PowerShell
powershell -NoProfile -Command "try{$c=New-Object Net.Sockets.TcpClient;$c.Connect('127.0.0.1',%PORT%);$c.Close();exit 0}catch{exit 1}" >nul 2>&1
if %errorlevel% equ 0 goto :server_ready

echo  [WAIT]  Attempt %TRIES%/30 -- server not yet ready...
if %TRIES% lss 30 goto :wait_loop

echo.
echo  [ERROR] Server did not start within 30 seconds.
echo  Check log: %LOG_FILE%
echo.
if exist "%LOG_FILE%" (
    echo  --- Last 10 lines of server log ---
    powershell -NoProfile -Command "Get-Content '%LOG_FILE%' -Tail 10"
)
echo.
pause
exit /b 1

:server_ready
:: Record PID so STOP_LKA.bat can target it precisely
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo %%p> "%PID_FILE%"
    set "LKA_PID=%%p"
)

echo.
echo  ==========================================================
echo   APEX LKA is RUNNING
echo  ----------------------------------------------------------
echo   Local URL  :  %URL%
echo   PID        :  !LKA_PID!
echo   Log        :  .lka_server.log
echo  ----------------------------------------------------------
echo   To stop    :  double-click STOP_LKA.bat
echo  ==========================================================
echo.

:: Open browser
start "" "%URL%"
echo  [BROWSER] %URL% opened.
echo.
echo  This window can be closed -- the server keeps running.
echo  Press any key to close this window.
pause >nul
endlocal
