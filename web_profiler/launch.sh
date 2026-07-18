#!/bin/bash
# Launcher for VoxelSim Workbench

set -e
cd "$(dirname "$0")/.."          # project root (web_profiler/..)
PY=venv/bin/python
[ -x "$PY" ] || PY=python3

echo "================================================"
echo "  VoxelSim Workbench"
echo "================================================"
"$PY" -c "import flask" 2>/dev/null || { echo "Installing Flask..."; "$PY" -m pip install -r web_profiler/requirements.txt; }

HOST="${FLASK_HOST:-127.0.0.1}"
PORT="${FLASK_PORT:-5000}"
echo "Serving at: http://$HOST:$PORT   (Ctrl+C to stop)"
echo "================================================"
FLASK_HOST="$HOST" FLASK_PORT="$PORT" exec "$PY" web_profiler/app.py
