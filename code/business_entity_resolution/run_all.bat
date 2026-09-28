@echo off
REM Full reproducible run on Windows: train -> predict -> validate.
REM Extra arguments are passed through, e.g.:  run_all.bat --set max_candidates=6
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python run.py all %*
