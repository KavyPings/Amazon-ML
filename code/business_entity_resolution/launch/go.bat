@echo off
REM Runs launch\task.py with the project environment; output -> launch\task.log, state -> launch\status.txt
cd /d "%~dp0.."
set VENV=%USERPROFILE%\ber_venv312
echo RUNNING %date% %time% > "%~dp0status.txt"
"%VENV%\Scripts\python.exe" -u "%~dp0task.py" > "%~dp0task.log" 2>&1
echo FINISHED exit=%errorlevel% %date% %time% >> "%~dp0status.txt"
