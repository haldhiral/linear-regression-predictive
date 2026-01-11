#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

".venv/bin/python" -m pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  ".venv/bin/python" -m pip install -r requirements.txt
else
  ".venv/bin/python" -m pip install fastapi uvicorn pandas sqlalchemy pymysql scikit-learn joblib pydantic
fi

".venv/bin/python" -m pip install pyinstaller

".venv/bin/python" -m PyInstaller \
  --onefile \
  --name lele_ml_api \
  --collect-submodules uvicorn \
  --collect-submodules fastapi \
  --collect-submodules sqlalchemy \
  --collect-submodules sklearn \
  --collect-submodules pandas \
  --collect-submodules pymysql \
  --collect-submodules joblib \
  --collect-submodules pydantic \
  run_api.py
