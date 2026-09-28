#!/usr/bin/env bash
# Full reproducible run: train -> predict -> validate. Extra args pass through.
set -e
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
python run.py all "$@"
