@echo off
REM v5: v4 + pruner keeps 99.8% of found true pairs (more recall, ~0.5 more candidates per S1), 30% subset.
REM Writes artifacts_v5\ and output_v5\ ; progress in launch\task_v5.log, status in launch\status_v5.txt
cd /d "%~dp0.."
set VENV=%USERPROFILE%\ber_venv312
echo RUNNING %date% %time% > "%~dp0status_v5.txt"
"%VENV%\Scripts\python.exe" -u "%~dp0task_run.py" --name v5 --frac 0.30 > "%~dp0task_v5.log" 2>&1
echo FINISHED exit=%errorlevel% %date% %time% >> "%~dp0status_v5.txt"
