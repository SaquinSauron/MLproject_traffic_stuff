@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto failed
)

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

".venv\Scripts\python.exe" train_april.py
if errorlevel 1 goto failed

echo.
echo Finished. Updated metrics and chart are in april2016_results.
pause
exit /b 0

:failed
echo.
echo Run failed. Check that Python 3 is installed and internet is available for package setup.
pause
exit /b 1
