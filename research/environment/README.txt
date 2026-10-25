# Python environments used for research/ results. Regenerate with:
#   .venv/Scripts/python.exe -m pip freeze > research/environment/venv_research.txt
#   .venv-tools/Scripts/python.exe -m pip freeze > research/environment/venv_tools.txt
research venv (.venv): Python 3.13.15
call model venv (research/callmodel/.venv, torch): venv_callmodel.txt
tools venv (.venv-tools, encryption, spreadsheets, whisper): Python 3.13.15
OS: MINGW64_NT-10.0-26200 3.4.10-87d57229.x86_64
captured: 2026-10-24
