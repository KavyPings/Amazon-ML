@echo off
REM v4: new blocking (address-heavy re-rank + number bigrams + smarter cap), trained on a closed 30% state subset.
REM Writes artifacts_v4\ and output_v4\ ; progress in launch\task_v4.log, status in launch\status_v4.txt
cd /d "%~dp0.."
set VENV=%USERPROFILE%\ber_venv312
echo RUNNING %date% %time% > "%~dp0status_v4.txt"
"%VENV%\Scripts\python.exe" -u "%~dp0task_run.py" --name v4 --frac 0.30 > "%~dp0task_v4.log" 2>&1
echo FINISHED exit=%errorlevel% %date% %time% >> "%~dp0status_v4.txt"
