#!/usr/bin/env python3
"""Generate layout visualizations for an exported TSIM thermal package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.visualization import write_layout_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw logic, DRAM, and cross-section views for a thermal package.")
    parser.add_argument("package_dir", type=Path, help="Directory containing logic.flp, dram*.flp, stack.lcf, and metadata.json.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    required = ["logic.flp", "stack.lcf", "metadata.json"]
    missing = [name for name in required if not (package_dir / name).exists()]
    if not (package_dir / "dram0.flp").exists() and not (package_dir / "dram.flp").exists():
        missing.append("dram0.flp or dram.flp")
    if missing:
        print(f"Missing required package files in {package_dir}: {', '.join(missing)}")
        return 2
    paths = write_layout_visualizations(package_dir)
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
