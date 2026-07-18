"""Aggregation, comparison and paper-figure data API.

Provides the sweep/compare/pareto endpoints used by the analysis views,
plus one-call datasets for the paper figures (fig10..fig20).  All data is
derived from the results index (see ``index.py``); per-result metrics come
from the cached summary parser and operator classification from
``classify.py``.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

from . import classify, parsers
from .api_results import FILTER_KEYS, _apply_filters, _with_display
from .config import CYCLE_TO_MS, MODES, PROJECT_ROOT
from .index import get_index

bp = Blueprint("analysis", __name__, url_prefix="/api")

# Dimensions allowed as sweep x / group keys (config fields + trp).
X_KEYS = set(FILTER_KEYS) | {"trp"}

# Default hardware operating point shared by the paper figures.
DEFAULT_FIX = {
    "root": "logs", "num_cores": 256, "sa_size": 32, "sram_kb": 2048,
    "dram_bw": 12288, "noc_topo": 1, "noc_bw": 16, "core_group": 8,
}

# Batch size representative of each mode in the paper experiments.
MODE_BATCH = {"decode": 32, "prefill": 1}

# The four LLMs used throughout the paper evaluation.
LLM_MODELS = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

# Equal-FLOPS (num_cores, sa_size, sram_kb) triples for the fig17 smile
# curve; per-core SRAM keeps the aggregate SRAM budget constant.
SMILE_PAIRS = [(32, 90, 16384), (64, 64, 8192), (128, 45, 4096),
               (256, 32, 2048), (512, 23, 1024), (1024, 16, 512)]

# Stacked energy components for fig19 (summary metric key -> display name).
ENERGY_COMPONENTS = [
    ("static_sa_mj", "SA Static"), ("static_vu_mj", "VU Static"),
    ("static_sram_mj", "SRAM Static"), ("static_dram_mj", "DRAM Static"),
    ("static_tsv_mj", "TSV Static"), ("static_noc_mj", "NoC Static"),
    ("dynamic_sa_mj", "SA Dynamic"), ("dynamic_vu_mj", "VU Dynamic"),
    ("dynamic_sram_mj", "SRAM Dynamic"),
    ("dynamic_dram_mj", "DRAM Dynamic"), ("dynamic_tsv_mj", "TSV Dynamic"),
    ("dynamic_noc_mj", "NoC Dynamic"),
]

# Sequence-length roots and candidate batch sizes for fig20.
FIG20_ROOTS = ["logs_seq1024", "logs_seq2048", "logs_seq4096"]
FIG20_BATCH = {"decode": [16, 32, 64], "prefill": [1, 2, 4]}


# ---------------------------------------------------------------------------
# aggregation helpers
# ---------------------------------------------------------------------------

def _sort_val(v) -> Tuple:
    """Sort key that keeps numeric x values ordered numerically."""
    return (0, float(v)) if isinstance(v, (int, float)) else (1, str(v))


def _metric_value(metrics: Dict, metric: str) -> Optional[float]:
    """Metric value from a summary dict; ``time_ms`` derives from cycles."""
    if metric == "time_ms":
        t = metrics.get("total_time")
        return t * CYCLE_TO_MS if t else None
    return metrics.get(metric)


def _entries(cfgs: List[Dict], summaries: Dict, metric: str) -> List[Tuple]:
    """(value, result id) pairs for configs with a usable metric value."""
    out = []
    for c in cfgs:
        m = summaries.get(c["id"])
        if not m:
            continue
        v = _metric_value(m, metric)
        if v is None:
            continue
        out.append((v, c["id"]))
    return out


def _bucket_point(entries: List[Tuple]) -> Dict:
    """Mean/min/max/n/ids statistics for one bucket of (value, id) pairs."""
    vals = [v for v, _ in entries]
    return {"y": sum(vals) / len(vals), "ymin": min(vals), "ymax": max(vals),
            "n": len(vals), "ids": [i for _, i in entries]}


def aggregate(cfgs: List[Dict], x_key: str, metric: str = "total_time",
              group_key: Optional[str] = None) -> Dict:
    """Bucket configs by (group, x) and average a summary metric per bucket.

    Returns ``{"x": [...], "series": [{"name", "points": [{"x", "y", "ymin",
    "ymax", "n", "ids"}]}], "x_log": bool}``.  Only buckets with data are
    emitted.  ``x_log`` is True when every x value is a positive number and
    max/min >= 8 (i.e. a log axis makes sense).
    """
    summaries = get_index().summaries(cfgs)
    buckets: Dict[Tuple, List[Tuple]] = {}
    for cfg in cfgs:
        x = cfg.get(x_key)
        if x is None:
            continue
        ent = _entries([cfg], summaries, metric)
        if not ent:
            continue
        g = cfg.get(group_key) if group_key else "all"
        buckets.setdefault((g, x), []).extend(ent)
    xs = sorted({x for _, x in buckets}, key=_sort_val)
    series = []
    for g in sorted({g for g, _ in buckets}, key=str):
        points = []
        for x in xs:
            ent = buckets.get((g, x))
            if not ent:
                continue
            points.append({"x": x, **_bucket_point(ent)})
        series.append({"name": str(g), "points": points})
    num_x = [x for x in xs if isinstance(x, (int, float))]
    x_log = (bool(num_x) and len(num_x) == len(xs) and min(num_x) > 0
             and max(num_x) / min(num_x) >= 8)
    return {"x": xs, "series": series, "x_log": x_log}


def _filter(fix: Dict, include_trp: bool = False) -> List[Dict]:
    """Index configs matching a fix dict (values may be comma-separated).

    tRCD/tRP variant logs (``output_cg_8_row_8192_trcd_14_trp_*.log``) are
    excluded unless ``include_trp`` — they duplicate the base config and
    would skew every bucket except an explicit trp sweep.
    """
    args = {k: str(v) for k, v in fix.items() if v is not None}
    cfgs = _apply_filters(get_index().all(), args)
    if not include_trp:
        cfgs = [c for c in cfgs if c.get("trp") is None]
    return cfgs


def _mode_fix(mode: str) -> Dict:
    return {"mode": mode, "batch_size": MODE_BATCH[mode]}


def _req_mode(default: str = "decode") -> str:
    m = request.args.get("mode", default)
    return m if m in MODES else default


# ---------------------------------------------------------------------------
# sweep / compare / pareto endpoints
# ---------------------------------------------------------------------------

@bp.route("/sweep")
def sweep():
    """Aggregated metric over one swept dimension, optionally grouped."""
    x_key = request.args.get("x", "")
    if x_key not in X_KEYS:
        return jsonify({"error": f"x must be one of {sorted(X_KEYS)}"}), 400
    metric = request.args.get("metric", "total_time")
    group_key = request.args.get("group") or None
    if group_key and group_key not in X_KEYS:
        return jsonify({"error":
                        f"group must be one of {sorted(X_KEYS)}"}), 400
    # Fix filters: everything except the swept/grouped dimensions themselves.
    args = {k: v for k, v in request.args.items()
            if k not in ("x", "group", "metric", x_key, group_key)}
    if "sram_bw" in args:  # alias used by the hardware config files
        args["dram_bw"] = args.pop("sram_bw")
    cfgs = _apply_filters(get_index().all(), args)
    if x_key != "trp":  # keep tRP variants out of non-tRP sweeps
        cfgs = [c for c in cfgs if c.get("trp") is None]
    out = aggregate(cfgs, x_key, metric, group_key)
    out.update({"x_key": x_key, "metric": metric, "group_key": group_key,
                "count": len(cfgs)})
    return jsonify(out)


@bp.route("/compare")
def compare():
    """Side-by-side config + metrics for 2..12 result ids."""
    ids = [s.strip() for s in request.args.get("ids", "").split(",")
           if s.strip()]
    if not 2 <= len(ids) <= 12:
        return jsonify({"error": "ids must contain 2..12 result ids"}), 400
    idx = get_index()
    items = []
    for rid in ids:
        cfg = idx.get(rid)
        if cfg is None:
            continue
        m = dict(idx.summary(cfg) or {})
        if m.get("total_time"):
            m["time_ms"] = m["total_time"] * CYCLE_TO_MS
        items.append({"id": rid, "config": _with_display(cfg), "metrics": m})
    if not items:
        return jsonify({"error": "no known result ids"}), 404
    return jsonify({"items": items})


@bp.route("/pareto")
def pareto():
    """DSE pareto results for a mode, read from results_pareto_<mode>/."""
    mode = request.args.get("mode", "decode")
    if mode not in MODES:
        return jsonify({"error": f"mode must be one of {MODES}"}), 400
    path = PROJECT_ROOT / f"results_pareto_{mode}" / "pareto_results.json"
    if not path.is_file():
        return jsonify({"error":
                        f"pareto results not found for mode={mode}"}), 404
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return jsonify({"error": "failed to parse pareto results"}), 500
    data.setdefault("fixed_params", {})
    data.setdefault("num_points", len(data.get("all_points", [])))
    return jsonify(data)


# ---------------------------------------------------------------------------
# paper figure builders
# ---------------------------------------------------------------------------

def build_fig10() -> Dict:
    """Compute-paradigm comparison: x=model, series=impl, decode+prefill.

    Dataflow/SPMD runs sometimes use a different core count or batch size;
    missing (model, impl) buckets are retried without the num_cores and
    batch_size fix and flagged with ``relaxed``.
    """
    impls = ["best", "spmd_compiler", "dataflow"]
    panels = []
    idx = get_index()
    for mode in MODES:
        fix = {**DEFAULT_FIX, **_mode_fix(mode)}
        strict = _filter(fix)
        relaxed = _filter({k: v for k, v in fix.items()
                           if k not in ("num_cores", "batch_size")})
        models = sorted({c["model"] for c in relaxed})
        summaries = idx.summaries(relaxed)
        series = []
        panel_relaxed = False
        for impl in impls:
            points = []
            for model in models:
                cfgs = [c for c in strict
                        if c["impl"] == impl and c["model"] == model]
                rel = False
                if not cfgs:
                    cfgs = [c for c in relaxed
                            if c["impl"] == impl and c["model"] == model]
                    rel = bool(cfgs)
                ent = _entries(cfgs, summaries, "total_time")
                if not ent:
                    continue
                points.append({"x": model, "relaxed": rel,
                               **_bucket_point(ent)})
                panel_relaxed = panel_relaxed or rel
            if points:
                series.append({"name": impl, "points": points})
        if series:
            panels.append({"mode": mode,
                           "title": f"{mode} (batch={MODE_BATCH[mode]})",
                           "x": models, "series": series,
                           "relaxed": panel_relaxed})
    return {"fig": "fig10", "title": "Compute Paradigm Comparison",
            "metric": "total_time", "panels": panels}


def build_fig11(mode: str = "decode") -> Dict:
    """NoC topology x mapping: x=noc_topo, series=impl, one panel per model."""
    panels = []
    for model in LLM_MODELS:
        fix = {**DEFAULT_FIX, **_mode_fix(mode), "model": model,
               "impl": "best,seq_noc"}
        fix.pop("noc_topo", None)  # noc_topo is the swept dimension
        agg = aggregate(_filter(fix), "noc_topo", "total_time", "impl")
        if agg["series"]:
            panels.append({"model": model, "title": f"{model} ({mode})",
                           **agg})
    return {"fig": "fig11", "title": "NoC Topology x Mapping",
            "metric": "total_time", "mode": mode, "panels": panels}


def build_fig12() -> Dict:
    """DRAM bandwidth and tRP sensitivity (llama2-13 decode).

    core_group is intentionally not fixed: it varies across the
    uniform_dram bandwidth sweep.
    """
    base = {**DEFAULT_FIX, "mode": "decode", "batch_size": 32,
            "model": "llama2-13", "impl": "best,uniform_dram"}
    base.pop("core_group")
    panels = []
    bw_fix = dict(base)
    bw_fix.pop("dram_bw", None)  # dram_bw is the swept dimension
    bw = aggregate(_filter(bw_fix), "dram_bw", "total_time", "impl")
    if bw["series"]:
        panels.append({"title": "DRAM bandwidth sweep", "x_key": "dram_bw",
                       **bw})
    trp = aggregate(_filter({**base, "dram_bw": 12288}, include_trp=True),
                    "trp", "total_time", "impl")
    if trp["series"]:
        panels.append({"title": "tRP sweep (dram_bw=12288)", "x_key": "trp",
                       **trp})
    return {"fig": "fig12", "title": "DRAM Bandwidth & tRP Sensitivity",
            "metric": "total_time", "panels": panels}


def build_fig13() -> Dict:
    """Tensor-to-bank placement: x=model, series=impl, decode+prefill."""
    panels = []
    for mode in MODES:
        fix = {**DEFAULT_FIX, **_mode_fix(mode), "impl": "best,uniform_dram"}
        agg = aggregate(_filter(fix), "model", "total_time", "impl")
        if agg["series"]:
            panels.append({"mode": mode,
                           "title": f"{mode} (batch={MODE_BATCH[mode]})",
                           **agg})
    return {"fig": "fig13", "title": "Tensor-to-Bank Placement",
            "metric": "total_time", "panels": panels}


FIG15_DIMS = ["noc_bw", "dram_bw", "sa_size", "num_cores", "sram_kb"]


def build_fig15(mode: str = "decode") -> Dict:
    """Hardware sweep curves: one panel per swept dimension, series=model."""
    panels = []
    for dim in FIG15_DIMS:
        fix = {**DEFAULT_FIX, **_mode_fix(mode), "impl": "best"}
        fix.pop(dim, None)  # never fix the swept dimension itself
        agg = aggregate(_filter(fix), dim, "total_time", "model")
        if agg["series"]:
            panels.append({"sweep": dim, "title": f"{dim} sweep ({mode})",
                           **agg})
    return {"fig": "fig15", "title": "Hardware Sweep",
            "metric": "total_time", "mode": mode, "panels": panels}


def build_fig17() -> Dict:
    """Core-group smile curve over equal-FLOPS (num_cores, sa_size) pairs."""
    labels = [f"{c}/{s}" for c, s, _ in SMILE_PAIRS]
    panels = []
    idx = get_index()
    for model in LLM_MODELS:
        series = []
        for cg in (1, 2, 4, 8):
            points = []
            for cores, sa, sram in SMILE_PAIRS:
                fix = {**DEFAULT_FIX, "mode": "decode", "batch_size": 32,
                       "model": model, "impl": "best", "num_cores": cores,
                       "sa_size": sa, "sram_kb": sram, "core_group": cg}
                cfgs = _filter(fix)
                ent = _entries(cfgs, idx.summaries(cfgs), "total_time")
                if not ent:
                    continue
                points.append({"x": f"{cores}/{sa}", **_bucket_point(ent)})
            if points:
                series.append({"name": f"cg={cg}", "points": points})
        if series:
            panels.append({"model": model, "title": f"{model} (decode)",
                           "x_labels": labels, "series": series})
    return {"fig": "fig17", "title": "Core Group Smile Curve",
            "metric": "total_time", "panels": panels}


def build_fig18(mode: str = "decode") -> Dict:
    """Total energy vs DRAM bandwidth / core count, series=model."""
    panels = []
    for dim in ("dram_bw", "num_cores"):
        fix = {**DEFAULT_FIX, **_mode_fix(mode), "impl": "best"}
        fix.pop(dim, None)
        agg = aggregate(_filter(fix), dim, "total_energy_mj", "model")
        if agg["series"]:
            panels.append({"sweep": dim, "title": f"energy vs {dim} ({mode})",
                           **agg})
    return {"fig": "fig18", "title": "Energy vs Bandwidth / Cores",
            "metric": "total_energy_mj", "mode": mode, "panels": panels}


def build_fig19(model: str = "llama3-70", mode: str = "decode") -> Dict:
    """Stacked component energy breakdown over dram_bw / num_cores.

    Unlike the other figures this averages each energy component directly
    over the summaries in a bucket (missing fields count as 0).
    """
    panels = []
    idx = get_index()
    for dim in ("dram_bw", "num_cores"):
        fix = {**DEFAULT_FIX, **_mode_fix(mode), "impl": "best",
               "model": model}
        fix.pop(dim, None)
        cfgs = _filter(fix)
        summaries = idx.summaries(cfgs)
        xs = sorted({c[dim] for c in cfgs})
        used_x: List = []
        values: Dict[str, List[float]] = {n: [] for _, n in ENERGY_COMPONENTS}
        for x in xs:
            bucket = [summaries[c["id"]] for c in cfgs
                      if c[dim] == x and c["id"] in summaries]
            if not bucket:
                continue
            used_x.append(x)
            for key, name in ENERGY_COMPONENTS:
                values[name].append(
                    sum(b.get(key, 0.0) for b in bucket) / len(bucket))
        if used_x:
            panels.append({
                "sweep": dim, "title": f"{model} energy vs {dim} ({mode})",
                "x": used_x,
                "components": [{"name": n, "values": v}
                               for n, v in values.items()]})
    return {"fig": "fig19", "title": "Component Energy Breakdown",
            "model": model, "mode": mode, "panels": panels}


def build_fig20(model: str = "llama3-70") -> Dict:
    """Operator-category time breakdown across sequence length and batch.

    Panels are (mode, seq_length) combinations; bars are batch sizes.  Time
    per category comes from classify.op_breakdown (busy-interval attribution)
    converted to milliseconds.
    """
    panels = []
    for mode in MODES:
        for root in FIG20_ROOTS:
            m = re.search(r"seq(\d+)", root)
            seq = int(m.group(1)) if m else None
            base = {**DEFAULT_FIX, "root": root, "model": model,
                    "mode": mode, "impl": "best"}
            avail = sorted({c["batch_size"] for c in _filter(base)})
            bs_list = [b for b in FIG20_BATCH[mode] if b in avail] or avail
            rows = []  # (x_label, {cat_key: ms}, {cat_key: label})
            for bs in bs_list:
                cfgs = sorted(_filter({**base, "batch_size": bs}),
                              key=lambda c: c["id"])
                bd = None
                for cfg in cfgs:
                    try:
                        ops = parsers.parse_operators_file(cfg["log_file"])
                    except OSError:
                        continue
                    bd = classify.op_breakdown(cfg, ops)
                    if bd:
                        break
                if bd is None:
                    continue
                rows.append((f"bs{bs}",
                             {c["key"]: c["cycles"] * CYCLE_TO_MS
                              for c in bd["categories"]},
                             {c["key"]: c["label"]
                              for c in bd["categories"]}))
            if not rows:
                continue
            keys: List[str] = []
            for _, ms, _ in rows:
                for k in ms:
                    if k not in keys:
                        keys.append(k)
            categories = [{
                "key": k,
                "label": next((lb[k] for _, _, lb in rows if k in lb), k),
                "values_ms": [ms.get(k, 0.0) for _, ms, _ in rows],
            } for k in keys]
            panels.append({"title": f"{mode} seq={seq}", "mode": mode,
                           "seq_length": seq,
                           "x_labels": [r[0] for r in rows],
                           "categories": categories})
    return {"fig": "fig20", "title": "Operator Type Breakdown",
            "model": model, "panels": panels}


@bp.route("/paper/<fig_id>")
def paper_fig(fig_id: str):
    """One-call plottable dataset for a paper figure."""
    builders = {
        "fig10": lambda: build_fig10(),
        "fig11": lambda: build_fig11(_req_mode()),
        "fig12": lambda: build_fig12(),
        "fig13": lambda: build_fig13(),
        "fig15": lambda: build_fig15(_req_mode()),
        "fig17": lambda: build_fig17(),
        "fig18": lambda: build_fig18(_req_mode()),
        "fig19": lambda: build_fig19(request.args.get("model", "llama3-70"),
                                     _req_mode()),
        "fig20": lambda: build_fig20(request.args.get("model", "llama3-70")),
    }
    fn = builders.get(fig_id)
    if fn is None:
        return jsonify({"error": f"unknown figure id {fig_id!r}",
                        "known": sorted(builders)}), 404
    data = fn()
    if not data["panels"]:
        return jsonify({"error": f"no data available for {fig_id}"}), 404
    return jsonify(data)
