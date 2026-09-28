#!/usr/bin/env bash
# One-time setup on macOS / Linux / Colab: creates .venv and installs pinned dependencies.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo "Setup done. Next:  ./run_all.sh"
