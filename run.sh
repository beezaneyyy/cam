#!/usr/bin/env bash
# AIRFRAME launcher: creates the venv (Python 3.12 via uv, or python3.11/3.12),
# installs dependencies, downloads the hand model if missing, and runs the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "[airframe] creating virtual environment…"
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv -r requirements.txt
  else
    PY=$(command -v python3.12 || command -v python3.11 || true)
    [ -z "$PY" ] && { echo "Python 3.11/3.12 not found. Install: sudo dnf install python3.12"; exit 1; }
    "$PY" -m venv .venv
    .venv/bin/pip install -r requirements.txt
  fi
fi

[ -f models/hand_landmarker.task ] || ./scripts/download_model.sh

export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2
exec .venv/bin/python -m airframe "$@"
