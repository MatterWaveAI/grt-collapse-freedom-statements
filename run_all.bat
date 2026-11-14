@echo off
setlocal
echo =========================================================================
echo MATTERWAVE AI - GRT COLLAPSE-FREEDOM PROOF SUITE RUNNER
echo =========================================================================
echo.

echo [1/4] Running Part 1: Network Rank Preservation (Pure GRT)...
python "%~dp0python\verify_pure_grt.py"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Part 1 verification failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [2/4] Running GRT Model Manifold Inspection...
python "%~dp0python\visualize_manifold_scan.py"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Manifold scan failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [3/4] Running Extension: Loss Anti-Collapse Dynamics (GRT-JEPA)...
python "%~dp0python\verify_grt_jepa.py"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] JEPA dynamics verification failed!
    exit /b %ERRORLEVEL%
)

echo.
echo [4/4] Running GNU Octave Mathematical Verification...
REM 1. Check custom OCTAVE_BIN environment variable
if defined OCTAVE_BIN (
    if exist "%OCTAVE_BIN%" (
        "%OCTAVE_BIN%" -q "%~dp0octave\verify_grt_collapse_freedom.m"
        goto :octave_done
    )
)
REM 2. Check if octave-cli or octave is in system PATH
where octave-cli >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    octave-cli -q "%~dp0octave\verify_grt_collapse_freedom.m"
    goto :octave_done
)
where octave >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    octave -q "%~dp0octave\verify_grt_collapse_freedom.m"
    goto :octave_done
)
REM 3. Check default Program Files installations
if exist "%ProgramFiles%\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe" (
    "%ProgramFiles%\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe" -q "%~dp0octave\verify_grt_collapse_freedom.m"
    goto :octave_done
)
if exist "%ProgramFiles%\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli-11.3.0.exe" (
    "%ProgramFiles%\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli-11.3.0.exe" -q "%~dp0octave\verify_grt_collapse_freedom.m"
    goto :octave_done
)
echo [WARN] GNU Octave CLI not found in PATH or standard Program Files. Set OCTAVE_BIN environment variable to enable Octave verification.

:octave_done
echo.
echo =========================================================================
echo PROOF RUN COMPLETE!
echo Figure 1: "%~dp0figures\figure1_network_rank_preservation.png"
echo Figure 2: "%~dp0figures\figure2_loss_anticollapse_dynamics.png"
echo =========================================================================
pause
