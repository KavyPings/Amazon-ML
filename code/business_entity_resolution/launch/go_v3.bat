@echo off
REM v3: train on a larger closed subset, predict each country in its own process, merge -> output_v3\
cd /d "%~dp0.."
set VENV=%USERPROFILE%\ber_venv312
echo RUNNING %date% %time% > "%~dp0status_v3.txt"
"%VENV%\Scripts\python.exe" -u "%~dp0task_v3.py" > "%~dp0task_v3.log" 2>&1
echo FINISHED exit=%errorlevel% %date% %time% >> "%~dp0status_v3.txt"
