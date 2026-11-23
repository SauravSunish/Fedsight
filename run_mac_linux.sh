#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Missing .venv. Create and review the environment once before running this app."
  echo "Example: python3.11 -m venv .venv"
  exit 1
fi
source .venv/bin/activate
python -m pip check
python -m streamlit run app.py --server.address 127.0.0.1
