@echo off
rem SRA CoxBox Overlay: double-click to start (runs offline; see tools\overlay\app.py)
start "" "%~dp0..\..\.venv-tools\Scripts\pythonw.exe" "%~dp0app.py"
