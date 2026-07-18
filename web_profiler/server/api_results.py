"""Results browsing & per-result detail API.

Result ``id`` is the log file path relative to the project root
(e.g. ``results/logs/llama3-70/bs_32/.../output_cg_8_row_8192.log``).
"""

import csv
import io
import json
import pickle
import threading
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, jsonify, request, Response

from . import parsers, classify
from .config import (PROJECT_ROOT, SUMMARY_METRICS, NPU_FREQ_MHZ_DEFAULT,
                     IMPL_LABELS, TOPO_LABELS)
from .index import get_index

bp = Blueprint("results", __name__, url_prefix="/api")

FILTER_KEYS = ("root", "model", "mode", "batch_size", "num_cores", "sa_size",
               "sram_kb", "dram_bw", "noc_topo", "noc_bw", "core_group",
               "impl", "row", "seq_length")

_pickle_lock = threading.Lock()
_pickle_cache: "dict[str, list]" = {}
_PICKLE_CACHE_MAX = 2  # each pickle can be ~100 MB


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_cfg_or_404(cfg_id: str):
    cfg = get_index().get(cfg_id) if cfg_id else None
    if cfg is None:
        return None, (jsonify({"error": "unknown result id"}), 404)
    return cfg, None


def _with_display(cfg):
    """Config dict stripped of absolute paths, plus display labels."""
    out = {k: v for k, v in cfg.items()
           if k not in ("log_file", "overlap_file", "top_power_file",
                        "pickle_file")}
    out["impl_label"] = IMPL_LABELS.get(cfg["impl"], cfg["impl"])
    out["topo_label"] = TOPO_LABELS.get(cfg["noc_topo"], str(cfg["noc_topo"]))
    return out


def _apply_filters(configs, args):
    out = configs
    for k in FILTER_KEYS:
        raw = args.get(k)
        if raw is None or raw == "":
            continue
        wanted = set()
        for tok in raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                wanted.add(int(tok))
            except ValueError:
                wanted.add(tok)
        out = [c for c in out if c.get(k) in wanted]
    q = args.get("q", "").strip().lower()
    if q:
        out = [c for c in out if q in c["id"].lower()]
    return out


def _enrich_metrics(rows, summaries):
    for row in rows:
        m = summaries.get(row["id"])
        if m:
            row["metrics"] = m
            tt = m.get("total_time")
            if tt:
                row["metrics"]["time_ms"] = tt / (NPU_FREQ_MHZ_DEFAULT * 1e3)
    return rows


# ---------------------------------------------------------------------------
# browsing endpoints
# ---------------------------------------------------------------------------

@bp.route("/overview")
def overview():
    idx = get_index()
    cfgs = idx.all()

    def _count(key):
        out = {}
        for c in cfgs:
            out[c[key]] = out.get(c[key], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: str(kv[0])))

    latest = sorted(cfgs, key=lambda c: c.get("mtime", 0), reverse=True)[:12]
    return jsonify({
        "total": len(cfgs),
        "by_model": _count("model"),
        "by_mode": _count("mode"),
        "by_impl": _count("impl"),
        "by_root": _count("root"),
        "by_cores": _count("num_cores"),
        "roots": idx.roots,
        "built_at": idx.built_at,
        "latest": [_with_display(c) for c in latest],
    })


@bp.route("/filters")
def filters():
    return jsonify(get_index().filter_values())


@bp.route("/metrics_meta")
def metrics_meta():
    return jsonify({k: {"label": v[0], "unit": v[1], "better": v[2]}
                    for k, v in SUMMARY_METRICS.items()})


@bp.route("/index/rebuild", methods=["POST"])
def index_rebuild():
    """Rescan all result roots (picks up web-launched runs) and rebuild."""
    idx = get_index()
    n = idx.rebuild()
    return jsonify({"total": n, "roots": [r["name"] for r in idx.roots],
                    "built_at": idx.built_at})


@bp.route("/results")
def results():
    idx = get_index()
    cfgs = _apply_filters(idx.all(), request.args)
    sort_key = request.args.get("sort", "")
    order = request.args.get("order", "asc")
    with_metrics = request.args.get("with_metrics", "0") == "1"

    summaries = idx.summaries(cfgs) if (with_metrics or
                                        sort_key.startswith("metrics.")) else {}

    def _sort_val(c):
        if sort_key.startswith("metrics."):
            mk = sort_key.split(".", 1)[1]
            m = summaries.get(c["id"]) or {}
            v = m.get(mk)
            if v is None and mk == "time_ms":
                t = m.get("total_time")
                v = t / (NPU_FREQ_MHZ_DEFAULT * 1e3) if t else None
            return (v is None, v)
        return (c.get(sort_key) is None, c.get(sort_key))

    if sort_key:
        cfgs = sorted(cfgs, key=_sort_val, reverse=(order == "desc"))

    total = len(cfgs)
    page = max(1, request.args.get("page", 1, type=int))
    page_size = min(500, max(1, request.args.get("page_size", 50, type=int)))
    rows = cfgs[(page - 1) * page_size: page * page_size]
    rows = [_with_display(c) for c in rows]
    if with_metrics:
        _enrich_metrics(rows, summaries)
    return jsonify({"total": total, "page": page, "page_size": page_size,
                    "rows": rows})


@bp.route("/result/detail")
def result_detail():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    out = _with_display(cfg)
    m = get_index().summary(cfg)
    if m:
        m = dict(m)
        if m.get("total_time"):
            m["time_ms"] = m["total_time"] / (NPU_FREQ_MHZ_DEFAULT * 1e3)
    out["metrics"] = m
    return jsonify(out)


@bp.route("/result/operators")
def result_operators():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    try:
        ops = parsers.parse_operators_file(cfg["log_file"])
    except OSError:
        return jsonify({"error": "log file not found"}), 404
    # attach category when the compile cache lines up
    bd = classify.op_breakdown(cfg, ops)
    if bd:
        for op, cat, typ in zip(ops, bd["op_categories"], bd["op_types"]):
            op["category"] = cat
            op["op_type"] = typ
        return jsonify({"operators": ops, "classified": True})
    return jsonify({"operators": ops, "classified": False})


@bp.route("/result/overlap")
def result_overlap():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    path = cfg["overlap_file"]
    if not Path(path).exists():
        return jsonify({"error": "overlap log not found"}), 404
    max_points = min(20000, request.args.get("max_points", 4000, type=int))
    intervals = parsers.downsample_intervals(
        parsers.parse_overlap_file(path), max_points)
    return jsonify({"intervals": intervals, "count": len(intervals)})


@bp.route("/result/top_power")
def result_top_power():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    path = cfg["top_power_file"]
    if not Path(path).exists():
        return jsonify({"error": "top_power log not found"}), 404
    intervals = parsers.parse_top_power_file(path)
    intervals.sort(key=lambda x: x["power_w"], reverse=True)
    return jsonify({"intervals": intervals})


def _load_pickle(path):
    with _pickle_lock:
        if path in _pickle_cache:
            return _pickle_cache[path]
    with open(path, "rb") as f:
        data = pickle.load(f)
    with _pickle_lock:
        if len(_pickle_cache) >= _PICKLE_CACHE_MAX:
            _pickle_cache.pop(next(iter(_pickle_cache)))
        _pickle_cache[path] = data
    return data


@bp.route("/result/op_energy")
def result_op_energy():
    """Per-operator energy/utilization from the fused-op pickle (if present)."""
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    if not cfg.get("has_pickle"):
        return jsonify({"error": "pickle not available for this result"}), 404
    try:
        ops = _load_pickle(cfg["pickle_file"])
    except (OSError, pickle.UnpicklingError, EOFError) as e:
        return jsonify({"error": f"failed to load pickle: {e}"}), 500
    out = []
    for i, op in enumerate(ops):
        pj_to_mj = 1e-9
        out.append({
            "op_id": getattr(op, "op_id", i),
            "t_start": op.t_dram_ld_start,
            "t_finish": op.t_finish,
            "exec_dur": op.exec_dur,
            "power_w": op.power_W,
            "mm_util": op.mm_util,
            "vu_util": op.vu_util,
            "dram_r_bytes": op.dram_r_bytes,
            "dram_w_bytes": op.dram_w_bytes,
            "energy_total_mj": op.energy_total * pj_to_mj,
            "energy_sa_mj": op.energy_sa * pj_to_mj,
            "energy_vu_mj": op.energy_vu * pj_to_mj,
            "energy_sram_mj": op.energy_sram * pj_to_mj,
            "energy_dram_mj": op.energy_dram * pj_to_mj,
            "energy_noc_mj": op.energy_noc * pj_to_mj,
            "energy_tsv_mj": op.energy_tsv * pj_to_mj,
            "energy_compute_mj": op.energy_compute * pj_to_mj,
        })
    return jsonify({"operators": out})


@bp.route("/result/op_breakdown")
def result_op_breakdown():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    try:
        ops = parsers.parse_operators_file(cfg["log_file"])
    except OSError:
        return jsonify({"error": "log file not found"}), 404
    bd = classify.op_breakdown(cfg, ops)
    if bd is None:
        return jsonify({"error": "operator classification unavailable "
                        "(compile cache not found or op count mismatch)"}), 404
    total = sum(c["cycles"] for c in bd["categories"]) or 1.0
    for c in bd["categories"]:
        c["fraction"] = c["cycles"] / total
        c["time_ms"] = c["cycles"] / (NPU_FREQ_MHZ_DEFAULT * 1e3)
    bd.pop("op_times", None)
    return jsonify(bd)


@bp.route("/result/reproduce")
def result_reproduce():
    """Generate the icbm_launch.py command that would reproduce this result."""
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    from .commands import build_sim_command
    seq = cfg.get("seq_length") or 2048
    cmd = build_sim_command({
        "model": cfg["model"], "num_cores": cfg["num_cores"],
        "batch_size": cfg["batch_size"] if cfg["mode"] == "decode" else 1,
        "sequence_length": seq, "mode": cfg["mode"],
        "sa_size": cfg["sa_size"], "sram_kb": cfg["sram_kb"],
        "dram_bw": cfg["dram_bw"], "noc_topo": cfg["noc_topo"],
        "noc_bw": cfg["noc_bw"], "core_group": cfg["core_group"],
        "impl": cfg["impl"], "root": cfg["root"],
        "trcd": cfg.get("trcd"), "trp": cfg.get("trp"),
    })
    return jsonify({"command": cmd["display"], "argv": cmd["argv"]})


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

@bp.route("/result/export.csv")
def result_export_csv():
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    kind = request.args.get("kind", "operators")
    buf = io.StringIO()
    if kind == "operators":
        ops = parsers.parse_operators_file(cfg["log_file"])
        if not ops:
            return jsonify({"error": "no operators"}), 404
        w = csv.DictWriter(buf, fieldnames=list(ops[0].keys()))
        w.writeheader()
        w.writerows(ops)
    elif kind == "summary":
        m = get_index().summary(cfg) or {}
        w = csv.writer(buf)
        w.writerow(["metric", "value"])
        for k, v in m.items():
            w.writerow([k, v])
        for k in ("model", "mode", "batch_size", "num_cores", "sa_size",
                  "sram_kb", "dram_bw", "noc_topo", "noc_bw", "core_group",
                  "impl", "root"):
            w.writerow([k, cfg.get(k)])
    else:
        return jsonify({"error": "unknown export kind"}), 400
    fname = cfg["id"].replace("/", "_").replace(".log", f"_{kind}.csv")
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition":
                             f"attachment; filename={fname}"})


@bp.route("/result/trace.json")
def result_trace_json():
    """Chrome-Trace JSON of the 5-stage operator pipeline."""
    cfg, err = _get_cfg_or_404(request.args.get("id", ""))
    if err:
        return err
    ops = parsers.parse_operators_file(cfg["log_file"])
    stages = [("DRAM Load", "dur_ld", "start_ld", "DRAM"),
              ("NoC Broadcast", "dur_bcast", "start_bcast", "NoC"),
              ("Compute", "dur_comp", "start_comp", "Compute"),
              ("NoC Reduce", "dur_reduce", "start_reduce", "NoC"),
              ("DRAM Store", "dur_store", "start_store", "DRAM")]
    events = []
    pid = 1
    for op in ops:
        for tid, (label, dur_k, start_k, cat) in enumerate(stages, start=1):
            if op[dur_k] > 0:
                events.append({
                    "name": f"Op{op['op_id']}: {label}", "cat": cat, "ph": "X",
                    "ts": op[start_k], "dur": op[dur_k], "pid": pid, "tid": tid,
                    "args": {"op_id": op["op_id"], "mm_util": op["mm_util"],
                             "vu_util": op["vu_util"]}})
    for tid, (label, *_rest) in enumerate(stages, start=1):
        events.append({"name": "thread_name", "ph": "M", "pid": pid,
                       "tid": tid, "args": {"name": label}})
    events.append({"name": "process_name", "ph": "M", "pid": pid,
                   "args": {"name": request.args.get("name", cfg["id"])}})
    return Response(json.dumps({"traceEvents": events}),
                    mimetype="application/json",
                    headers={"Content-Disposition":
                             "attachment; filename=trace.json"})
