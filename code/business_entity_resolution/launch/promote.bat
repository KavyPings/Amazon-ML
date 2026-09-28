@echo off
REM Validate a finished run and make it the submission (output\ + artifacts\); the previous ones are
REM kept as output_prev\ and artifacts_prev\. Set RUN to the version to promote (v5 is NOT worth promoting).
set RUN=v6
cd /d "%~dp0.."
"%USERPROFILE%\ber_venv312\Scripts\python.exe" -u "%~dp0promote.py" --out output_%RUN% --art artifacts_%RUN% --name %RUN% > "%~dp0promote_log.txt" 2>&1
echo DONE exit=%errorlevel% >> "%~dp0promote_log.txt"
