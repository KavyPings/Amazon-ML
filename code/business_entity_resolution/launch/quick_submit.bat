@echo off
REM Builds a valid submission right now from the country score files written so far.
cd /d "%~dp0.."
"%USERPROFILE%\ber_venv312\Scripts\python.exe" -u "%~dp0quick_submit.py" > "%~dp0quick_log.txt" 2>&1
echo DONE >> "%~dp0quick_log.txt"
