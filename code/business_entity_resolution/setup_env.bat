@echo off
REM One-time setup on Windows: creates .venv and installs pinned dependencies.
cd /d "%~dp0"
py -3.11 -m venv .venv 2>nul || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Setup done. Next:  run_all.bat
