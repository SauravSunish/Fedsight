@echo off
setlocal
cd /d "%~dp0"
set "FEDSIGHT_PYTHON=.venv311\Scripts\python.exe"
if not exist "%FEDSIGHT_PYTHON%" set "FEDSIGHT_PYTHON=.venv\Scripts\python.exe"
if not exist "%FEDSIGHT_PYTHON%" (
  echo Python environment missing. See the README setup instructions.
  pause
  exit /b 1
)
rem Open the existing local Streamlit server when one is already listening.
"%FEDSIGHT_PYTHON%" -c "import sys,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=2); sys.exit(0 if r.read()==b'ok' else 1)" >nul 2>&1
if not errorlevel 1 (
  echo A Streamlit app is already running on port 8501. Opening it in your browser.
  start "" "http://127.0.0.1:8501"
  exit /b 0
)
"%FEDSIGHT_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 pause
