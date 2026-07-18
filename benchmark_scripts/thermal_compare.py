#!/usr/bin/env python3
"""Compatibility wrapper for the TSIM thermal validation workflow."""

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
