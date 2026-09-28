@echo off
REM One-time environment setup (Python 3.12 + mature, widely used library versions).
REM Windows Smart App Control blocks brand-new unsigned binaries (e.g. scipy for Python 3.14),
REM so we use long-established releases. The venv lives OUTSIDE OneDrive: %USERPROFILE%\ber_venv312
cd /d "%~dp0.."
set LOG=%~dp0setup_log.txt
set VENV=%USERPROFILE%\ber_venv312
echo SETUP START %date% %time% > "%LOG%"
py -3.12 -c "print('py312 ok')" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo installing Python 3.12 >> "%LOG%"
  py install 3.12 >> "%LOG%" 2>&1
)
py -3.12 -c "print('py312 ok')" >> "%LOG%" 2>&1
if errorlevel 1 (
  winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
)
py -0p >> "%LOG%" 2>&1
if not exist "%VENV%\Scripts\python.exe" py -3.12 -m venv "%VENV%" >> "%LOG%" 2>&1
"%VENV%\Scripts\python.exe" -m pip install --upgrade pip >> "%LOG%" 2>&1
"%VENV%\Scripts\python.exe" -m pip install -r "%~dp0..\requirements.txt" >> "%LOG%" 2>&1
"%VENV%\Scripts\python.exe" "%~dp0check_env.py" >> "%LOG%" 2>&1
"%VENV%\Scripts\python.exe" -m pip freeze > "%~dp0freeze.txt" 2>&1
echo SETUP DONE %date% %time% >> "%LOG%"
