"""System API: host status, RAM usage timeline and project log browsing."""

import math
import os
import re
import shutil
from pathlib import Path

from flask import Blueprint, jsonify, request

from .config import (PROJECT_ROOT, RESULTS_DIR, TEST_LOGS_DIR,
                     discover_result_roots)

bp = Blueprint("system", __name__, url_prefix="/api")

_ROOT_LOG_NAMES = ("one_pass.log", "master_runner_output.log",
                   "dram_monitor.log", "dram_usage.log")
_ALLOWED_SUFFIXES = {".log", ".txt", ".csv", ".md", ".json", ".yaml", ".yml"}
_MAX_VIEW_LENGTH = 1024 * 1024  # 1 MB

# ---------------------------------------------------------------------------
# host status
# ---------------------------------------------------------------------------

def _read_meminfo():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    info[parts[0].rstrip(":")] = int(parts[1])  # kB
                except ValueError:
                    pass
    total = info.get("MemTotal", 0) / 1024.0
    avail = info.get("MemAvailable", 0) / 1024.0
    return {"total_mb": round(total, 1), "available_mb": round(avail, 1),
            "used_mb": round(total - avail, 1)}


@bp.route("/system/host")
def system_host():
    try:
        mem = _read_meminfo()
    except OSError:
        mem = {"total_mb": None, "available_mb": None, "used_mb": None}
    try:
        loadavg = list(os.getloadavg())
    except OSError:
        loadavg = None
    du = shutil.disk_usage(PROJECT_ROOT)
    gb = 1024.0 ** 3
    roots = []
    for r in discover_result_roots():
        try:
            rel = Path(r["path"]).relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(r["path"])
        roots.append({"name": r["name"], "path": rel})
    return jsonify({
        "cpu_count": os.cpu_count(),
        "loadavg": loadavg,
        "mem": mem,
        "disk": {"total_gb": round(du.total / gb, 2),
                 "used_gb": round(du.used / gb, 2),
                 "free_gb": round(du.free / gb, 2)},
        "results": {"roots": roots},
    })


# ---------------------------------------------------------------------------
# RAM timeline (dram_usage.log preferred, dram_monitor.log fallback)
# ---------------------------------------------------------------------------

_MONITOR_RE = re.compile(
    r"total=(\d+)\s+used=(\d+)\s+free=(\d+)\s+shared=(\d+)\s+"
    r"buff_cache=(\d+)\s+available=(\d+)")
_MONITOR_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def _parse_dram_usage(path):
    """`ISO used_MB total_MB free_MB buffcache_MB available_MB` per line."""
    t, used, avail, total = [], [], [], []
    with open(path, errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                u, tot, _free, _buff, av = (int(parts[i]) for i in range(1, 6))
            except ValueError:
                continue
            t.append(parts[0])
            used.append(u)
            total.append(tot)
            avail.append(av)
    return t, used, avail, total


def _parse_dram_monitor(path):
    """Two lines per record: timestamp, then `total=G used=G ...` (GB)."""
    t, used, avail, total = [], [], [], []
    pending = None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = _MONITOR_RE.search(line)
            if m:
                if pending is not None:
                    vals = [int(g) * 1024 for g in m.groups()]  # GB -> MB
                    t.append(pending)
                    total.append(vals[0])
                    used.append(vals[1])
                    avail.append(vals[5])
                    pending = None
                continue
            if _MONITOR_TS_RE.match(line):
                pending = line.split()[0]
    return t, used, avail, total


def _decimate(series, max_points):
    """Evenly thin parallel lists to at most ~max_points, keeping the last."""
    n = len(series[0])
    if n <= max_points:
        return series
    stride = math.ceil(n / max_points)
    idx = list(range(0, n, stride))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [[s[i] for i in idx] for s in series]


@bp.route("/system/ram_timeline")
def ram_timeline():
    max_points = min(10000, max(10, request.args.get("max_points", 1000,
                                                     type=int)))
    usage = PROJECT_ROOT / "dram_usage.log"
    monitor = PROJECT_ROOT / "dram_monitor.log"
    if usage.is_file():
        source = "dram_usage.log"
        series = _parse_dram_usage(usage)
    elif monitor.is_file():
        source = "dram_monitor.log"
        series = _parse_dram_monitor(monitor)
    else:
        return jsonify({"t": [], "used_mb": [], "available_mb": [],
                        "total_mb": [], "source": None})
    t, used, avail, total = _decimate(series, max_points)
    return jsonify({"t": t, "used_mb": used, "available_mb": avail,
                    "total_mb": total, "source": source})


# ---------------------------------------------------------------------------
# log listing & viewing
# ---------------------------------------------------------------------------

@bp.route("/logs/list")
def logs_list():
    items = []

    def _add(category, path):
        try:
            st = path.stat()
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except (OSError, ValueError):
            return
        items.append({"category": category, "name": path.name, "path": rel,
                      "size": st.st_size, "mtime": st.st_mtime})

    if TEST_LOGS_DIR.is_dir():
        for p in sorted(TEST_LOGS_DIR.glob("*.log")):
            _add("test_logs", p)
    for name in _ROOT_LOG_NAMES:
        p = PROJECT_ROOT / name
        if p.is_file():
            _add("root", p)
    for p in sorted(PROJECT_ROOT.glob("results_pareto_*/dse_sim.log")):
        _add("dse", p)
    thermal = []
    for d in sorted(RESULTS_DIR.glob("thermal*")):
        if d.is_dir():
            thermal.extend(d.rglob("*.log"))
    for p in sorted(thermal)[:50]:
        _add("thermal", p)
    return jsonify({"logs": items})


@bp.route("/logs/view")
def logs_view():
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "missing path"}), 400
    p = (PROJECT_ROOT / raw).resolve()
    try:
        p.relative_to(PROJECT_ROOT)
    except ValueError:
        return jsonify({"error": "path outside project root"}), 403
    if ".env" in p.name:
        return jsonify({"error": "access refused"}), 403
    if p.suffix not in _ALLOWED_SUFFIXES:
        return jsonify({"error": "file type not allowed"}), 403
    if not p.is_file():
        return jsonify({"error": "file not found"}), 404

    length = request.args.get("length", 65536, type=int)
    length = min(_MAX_VIEW_LENGTH, max(1, length))
    offset = max(0, request.args.get("offset", 0, type=int))
    tail = request.args.get("tail", 0, type=int) == 1

    total = p.stat().st_size
    if tail:
        offset = max(0, total - length)
    with open(p, "rb") as f:
        f.seek(offset)
        data = f.read(length)
    return jsonify({
        "path": p.relative_to(PROJECT_ROOT).as_posix(),
        "total_size": total,
        "offset": offset,
        "length": len(data),
        "eof": offset + len(data) >= total,
        "text": data.decode("utf-8", errors="replace"),
    })
