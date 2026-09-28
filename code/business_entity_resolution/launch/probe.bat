@echo off
REM Writes a short system report (Python, RAM, CPU, disk, GPU) to probe.txt next to this file.
cd /d "%~dp0"
set OUT=%~dp0probe.txt
echo === probe %date% %time% > "%OUT%"
echo --- where python / py >> "%OUT%"
where python >> "%OUT%" 2>&1
where py >> "%OUT%" 2>&1
where conda >> "%OUT%" 2>&1
where winget >> "%OUT%" 2>&1
echo --- versions >> "%OUT%"
py -0p >> "%OUT%" 2>&1
python -c "import sys;print(sys.version, sys.executable)" >> "%OUT%" 2>&1
echo --- hardware >> "%OUT%"
powershell -NoProfile -Command "$c=Get-CimInstance Win32_ComputerSystem; 'RAM_GB=' + [math]::Round($c.TotalPhysicalMemory/1GB,1); Get-CimInstance Win32_Processor | ForEach-Object { 'CPU=' + $_.Name + ' cores=' + $_.NumberOfCores + ' threads=' + $_.NumberOfLogicalProcessors }; $d=Get-PSDrive C; 'C_free_GB=' + [math]::Round($d.Free/1GB,1); 'RAM_free_GB=' + [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)" >> "%OUT%" 2>&1
nvidia-smi --query-gpu=name,memory.total --format=csv >> "%OUT%" 2>&1
echo === done >> "%OUT%"
