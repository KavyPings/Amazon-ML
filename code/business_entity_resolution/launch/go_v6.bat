@echo off
REM v6: two-phase training on a closed 50% state subset. The pruner (and so the candidate-set size)
REM is reused from artifacts_v4\, so only ~5 candidates per S1 are ever held in memory; the matcher
REM learns from ~1.7x more data than v4. Writes artifacts_v6\ and output_v6\ ; log in launch\task_v6.log
cd /d "%~dp0.."
set VENV=%USERPROFILE%\ber_venv312
echo RUNNING %date% %time% > "%~dp0status_v6.txt"
"%VENV%\Scripts\python.exe" -u "%~dp0task_run.py" --name v6 --frac 0.50 --set "pruner_dir=%CD%\artifacts_v4" > "%~dp0task_v6.log" 2>&1
echo FINISHED exit=%errorlevel% %date% %time% >> "%~dp0status_v6.txt"
