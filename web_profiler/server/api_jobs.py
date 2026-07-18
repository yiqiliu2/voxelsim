"""Job submission & control API (sim / batch / dse / thermal runs).

All parameter validation lives here; :mod:`.jobs` assumes sane input.
Success returns ``201`` with the job JSON, invalid input returns
``400 {"error": ...}``.
"""

import re
from typing import Dict

from flask import Blueprint, jsonify, request

from . import commands
from .commands import (MODEL_INFO, SWEEP_TIERS, DEFAULT_SIM_PARAMS,
                       IMPL_FLAGS, SPLIT_FACTOR_DECODE, SPLIT_FACTOR_PREFILL)
from .config import IMPL_LABELS
from .jobs import get_manager, max_workers, JobManager

bp = Blueprint("jobs", __name__, url_prefix="/api")

MAX_BATCH_UNITS = 64
MAX_LOG_LENGTH = 1024 * 1024
THERMAL_BACKENDS = ("simple", "hotspot", "threedice", "adaptive_grid")

_ROOT_RE = re.compile(r"^[A-Za-z0-9_]+$")
_DIR_RE = re.compile(r"^[A-Za-z0-9_/\-.]+$")

# int sim fields -> (min, max); batch/sequence get a wider range
_INT_RANGES = {
    "num_cores": (1, 65536), "sa_size": (1, 65536), "sram_kb": (1, 65536),
    "dram_bw": (1, 65536), "noc_bw": (1, 65536), "core_group": (1, 65536),
    "row": (1, 65536), "trcd": (1, 65536),
    "batch_size": (1, 1000000), "sequence_length": (1, 1000000),
}


# ---------------------------------------------------------------------------
# validation helpers (raise ValueError -> 400)
# ---------------------------------------------------------------------------

def _coerce_int(value, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field}: expected integer")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field}: expected integer, got {value!r}")
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: expected integer, got {value!r}")
    if not (lo <= iv <= hi):
        raise ValueError(f"{field}: {iv} out of range [{lo}, {hi}]")
    return iv


def _coerce_number(value, field: str):
    """float/int or None; integral floats collapse to int."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field}: expected number")
    try:
        fv = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: expected number, got {value!r}")
    return int(fv) if fv.is_integer() else fv


def _validate_one(key: str, value):
    if key == "model":
        if value not in MODEL_INFO:
            raise ValueError(f"model: unknown {value!r}")
        return value
    if key == "mode":
        if value not in ("decode", "prefill"):
            raise ValueError("mode: must be 'decode' or 'prefill'")
        return value
    if key == "noc_topo":
        return _coerce_int(value, key, 1, 3)
    if key in _INT_RANGES:
        lo, hi = _INT_RANGES[key]
        return _coerce_int(value, key, lo, hi)
    if key == "impl":
        if value not in IMPL_FLAGS:
            raise ValueError(f"impl: unknown {value!r}")
        return value
    if key in ("split_factor", "trp"):
        return _coerce_number(value, key)
    if key == "sim_layers":
        if value is None:
            return None
        return _coerce_int(value, key, 0, 1000000)
    if key == "root":
        if not isinstance(value, str) or not _ROOT_RE.match(value):
            raise ValueError("root: must match ^[A-Za-z0-9_]+$")
        return value
    raise ValueError(f"unknown parameter: {key}")


def _validate_sim_params(raw, allow_lists: bool) -> Dict:
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    known = set(DEFAULT_SIM_PARAMS)
    params = {}
    for key, value in raw.items():
        if key not in known:
            raise ValueError(f"unknown parameter: {key}")
        if isinstance(value, list):
            if not allow_lists:
                raise ValueError(f"{key}: list only allowed for batch")
            if key not in commands._SWEEPABLE:
                raise ValueError(f"{key}: not sweepable")
            if not value:
                raise ValueError(f"{key}: empty list")
            params[key] = [_validate_one(key, v) for v in value]
        else:
            params[key] = _validate_one(key, value)
    # prefill runs are always single-request
    if params.get("mode") == "prefill":
        params["batch_size"] = 1
    return params


def _validate_dse_params(raw) -> Dict:
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    known = {"mode", "models", "num_sweeps", "max_cycles", "core_group",
             "exhaustive"}
    for key in raw:
        if key not in known:
            raise ValueError(f"unknown parameter: {key}")
    out = {}
    mode = raw.get("mode", "decode")
    if mode not in ("decode", "prefill"):
        raise ValueError("mode: must be 'decode' or 'prefill'")
    out["mode"] = mode
    models = raw.get("models")
    if models is not None:
        if not isinstance(models, list) or not models:
            raise ValueError("models: expected non-empty list")
        bad = [m for m in models if m not in MODEL_INFO]
        if bad:
            raise ValueError(f"models: unknown {bad}")
        out["models"] = models
    for k in ("num_sweeps", "max_cycles"):
        if raw.get(k) is not None:
            out[k] = _coerce_int(raw[k], k, 1, 100)
    if raw.get("core_group") is not None:
        out["core_group"] = _coerce_int(raw["core_group"], "core_group",
                                        1, 65536)
    if raw.get("exhaustive"):
        out["exhaustive"] = True
    return out


def _validate_csv(value, field: str, allowed) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected comma-separated string")
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"{field}: empty")
    bad = [t for t in tokens if t not in allowed]
    if bad:
        raise ValueError(f"{field}: unknown {bad}")
    return ",".join(tokens)


def _validate_thermal_params(raw) -> Dict:
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    known = {"models", "modes", "impls", "backends", "results_dir", "out_dir",
             "dram_bws", "rows", "core_groups", "dram_layers"}
    for key in raw:
        if key not in known:
            raise ValueError(f"unknown parameter: {key}")
    out = {}
    out["models"] = _validate_csv(raw.get("models", "llama2-13"), "models",
                                  set(MODEL_INFO) | {"dit-xl"})
    out["modes"] = _validate_csv(raw.get("modes", "decode"), "modes",
                                 {"decode", "prefill"})
    if raw.get("backends") is not None:
        out["backends"] = _validate_csv(raw["backends"], "backends",
                                        set(THERMAL_BACKENDS))
    if raw.get("impls") is not None:
        out["impls"] = _validate_csv(raw["impls"], "impls", set(IMPL_FLAGS))
    for k in ("results_dir", "out_dir"):
        v = raw.get(k)
        if v is not None:
            if not isinstance(v, str) or not _DIR_RE.match(v) or ".." in v:
                raise ValueError(f"{k}: must match ^[A-Za-z0-9_/\\-.]+$ "
                                 "and contain no '..'")
            out[k] = v
    for k in ("dram_bws", "rows", "core_groups"):
        v = raw.get(k)
        if v is not None:
            tokens = [t.strip() for t in str(v).split(",") if t.strip()]
            if not tokens:
                raise ValueError(f"{k}: empty")
            for t in tokens:
                _coerce_int(t, k, 1, 1000000)
            out[k] = ",".join(tokens)
    if raw.get("dram_layers") is not None:
        out["dram_layers"] = _coerce_int(raw["dram_layers"], "dram_layers",
                                         1, 64)
    return out


# ---------------------------------------------------------------------------
# response shaping
# ---------------------------------------------------------------------------

def _err(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


def _strip_units(job: Dict) -> Dict:
    """Job JSON without unit argv (list/submit responses)."""
    out = dict(job)
    keep = ("idx", "label", "display", "expected_output", "status",
            "returncode")
    out["units"] = [{k: u[k] for k in keep if k in u}
                    for u in job["units"]]
    out["progress"] = JobManager.progress(job)
    return out


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@bp.route("/jobs/defs")
def job_defs():
    """Form metadata for the Run page."""
    return jsonify({
        "models": MODEL_INFO,
        "tiers": SWEEP_TIERS,
        "defaults": DEFAULT_SIM_PARAMS,
        "impls": [{"value": k, "label": IMPL_LABELS.get(k, k)}
                  for k in IMPL_FLAGS],
        "modes": ["decode", "prefill"],
        "max_workers": max_workers(),
        "notes": {"decode_split_factor": SPLIT_FACTOR_DECODE,
                  "prefill_split_factor": SPLIT_FACTOR_PREFILL},
    })


@bp.route("/jobs", methods=["POST"])
def job_submit():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return _err("expected JSON body {type, params}")
    jtype = body.get("type")
    raw_params = body.get("params") or {}
    mgr = get_manager()
    try:
        if jtype == "sim":
            params = _validate_sim_params(raw_params, allow_lists=False)
            return jsonify(mgr.submit_sim(params)), 201
        if jtype == "batch":
            params = _validate_sim_params(raw_params, allow_lists=True)
            n = 1
            for v in params.values():
                if isinstance(v, list):
                    n *= len(v)
            if n > MAX_BATCH_UNITS:
                return _err(f"batch expands to {n} runs "
                            f"(max {MAX_BATCH_UNITS})")
            return jsonify(_strip_units(mgr.submit_batch(params))), 201
        if jtype == "dse":
            return jsonify(mgr.submit_dse(
                _validate_dse_params(raw_params))), 201
        if jtype == "thermal":
            return jsonify(mgr.submit_thermal(
                _validate_thermal_params(raw_params))), 201
        return _err("type: must be one of sim|batch|dse|thermal")
    except ValueError as e:
        return _err(str(e))


@bp.route("/jobs")
def job_list():
    limit = min(1000, max(1, request.args.get("limit", 100, type=int)))
    jobs = get_manager().list_jobs(limit=limit)
    return jsonify({"jobs": [_strip_units(j) for j in jobs]})


@bp.route("/jobs/<job_id>")
def job_detail(job_id):
    job = get_manager().get(job_id)
    if job is None:
        return _err("unknown job id", 404)
    out = dict(job)
    out["progress"] = JobManager.progress(job)
    return jsonify(out)


@bp.route("/jobs/<job_id>/log")
def job_log(job_id):
    offset = max(0, request.args.get("offset", 0, type=int))
    length = request.args.get("length", 65536, type=int)
    length = min(MAX_LOG_LENGTH, max(1, length))
    tail = request.args.get("tail", 0, type=int)
    win = get_manager().log_window(job_id, offset=offset, length=length,
                                   tail=tail)
    if win is None:
        return _err("unknown job id", 404)
    return jsonify(win)


@bp.route("/jobs/<job_id>/cancel", methods=["POST"])
def job_cancel(job_id):
    try:
        job = get_manager().cancel(job_id)
    except ValueError as e:
        return _err(str(e), 409)
    if job is None:
        return _err("unknown job id", 404)
    return jsonify(_strip_units(job))


@bp.route("/jobs/<job_id>/delete", methods=["POST"])
def job_delete(job_id):
    try:
        job = get_manager().delete(job_id)
    except ValueError as e:
        return _err(str(e), 409)
    if job is None:
        return _err("unknown job id", 404)
    return jsonify({"deleted": job_id})
