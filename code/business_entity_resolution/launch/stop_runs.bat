@echo off
REM Stops a running pipeline task (launched by go.bat) and its worker processes.
"%USERPROFILE%\ber_venv312\Scripts\python.exe" "%~dp0stop_runs.py"
