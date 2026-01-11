@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
)

call ".venv\Scripts\python.exe" -m pip install --upgrade pip

if exist "requirements.txt" (
  call ".venv\Scripts\python.exe" -m pip install -r requirements.txt
) else (
  call ".venv\Scripts\python.exe" -m pip install fastapi uvicorn pandas sqlalchemy pymysql scikit-learn joblib pydantic
)

call ".venv\Scripts\python.exe" -m pip install pyinstaller

call ".venv\Scripts\python.exe" -m PyInstaller ^
  --onefile ^
  --name lele_ml_api ^
  --collect-submodules uvicorn ^
  --collect-submodules fastapi ^
  --collect-submodules sqlalchemy ^
  --collect-submodules sklearn ^
  --collect-submodules pandas ^
  --collect-submodules pymysql ^
  --collect-submodules joblib ^
  --collect-submodules pydantic ^
  run_api.py

endlocal
