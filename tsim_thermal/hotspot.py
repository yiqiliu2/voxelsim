"""Optional HotSpot 7.0 execution helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class HotSpotResult:
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    ttrace_path: Path | None
    peak_c: float | None = None


def find_hotspot(binary: str = "hotspot") -> str | None:
    if "/" in binary:
        path = Path(binary)
        return str(path.resolve()) if path.exists() else None
    return shutil.which(binary)


def run_hotspot(package_dir: Path, binary: str = "hotspot", timeout_s: float = 300.0) -> HotSpotResult:
    exe = find_hotspot(binary)
    if exe is None:
        return HotSpotResult(False, None, "", f"HotSpot binary not found: {binary}", None)

    ttrace = package_dir / "temperature.ttrace"
    cmd = [
        exe,
        "-c", "hotspot.config",
        "-p", "power.ptrace",
        "-o", ttrace.name,
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=package_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return HotSpotResult(True, 124, stdout, stderr + f"\nTimed out after {timeout_s} seconds.", None)
    ttrace_path = ttrace if ttrace.exists() else None
    return HotSpotResult(True, result.returncode, result.stdout, result.stderr, ttrace_path, _peak_c(ttrace_path))


def _peak_c(ttrace_path: Path | None) -> float | None:
    if ttrace_path is None or not ttrace_path.exists():
        return None
    peak_k: float | None = None
    with open(ttrace_path, encoding="utf-8") as infile:
        next(infile, None)
        for line in infile:
            values = [float(item) for item in line.split()]
            if values:
                row_peak = max(values)
                peak_k = row_peak if peak_k is None else max(peak_k, row_peak)
    return None if peak_k is None else peak_k - 273.15
