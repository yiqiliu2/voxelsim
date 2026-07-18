#!/usr/bin/env python3
"""VoxelSim Workbench — entry point.

Thin launcher around :func:`web_profiler.server.create_app`.  All routes and
logic live in ``web_profiler/server/``; this file only wires up host/port and
starts the dev server.

Usage::

    python app.py                 # 127.0.0.1:5000
    FLASK_HOST=0.0.0.0 FLASK_PORT=8080 python app.py
"""

import os
import sys
from pathlib import Path

# Allow both ``python app.py`` (cwd = web_profiler) and ``python web_profiler/app.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_profiler.server import create_app  # noqa: E402

DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
HOST = os.environ.get("FLASK_HOST", "127.0.0.1")
PORT = int(os.environ.get("FLASK_PORT", "5000"))

app = create_app()

if __name__ == "__main__":
    print(f"VoxelSim Workbench listening on http://{HOST}:{PORT}")
    app.run(debug=DEBUG, host=HOST, port=PORT, threaded=True)
