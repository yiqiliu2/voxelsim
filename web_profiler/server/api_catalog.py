"""Catalog API: hardware configs, models, NoC distance tables, serving profiles.

Read-only over project data directories, except ``POST /api/hwconfigs``
which may only *add* new hw_config files — existing files are never
overwritten (mirrors commands.ensure_hw_config).
"""

import csv
import json
import math
import pickle
import re
from collections import Counter

import numpy as np
from flask import Blueprint, jsonify, request

try:
    import yaml
except ImportError:  # PyYAML optional: serving meta.yaml is best-effort
    yaml = None

from .config import (PROJECT_ROOT, HW_CONFIG_DIR, MODELS_DIR,
                     PARSED_MODELS_DIR, ORIGINAL_MODELS_DIR, NOC_TABLES_DIR,
                     LLMSERVING_DIR)

bp = Blueprint("catalog", __name__, url_prefix="/api")

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_\-.]+$")
_TOPO_NAMES = ("MESH", "TORUS", "ALL")
_DIST_NAME_RE = re.compile(r"^Topo\.([A-Za-z]+)_(\d+)\.dist$")
_TEXPR_BATCH_RE = re.compile(r"-b(\d+)\.json$")
_SERVING_CSV_FILES = ("attention.csv", "dense.csv", "per_sequence.csv")

# ---------------------------------------------------------------------------
# hardware configs
# ---------------------------------------------------------------------------

def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _summarize_hw_config(path):
    st = path.stat()
    ent = {"name": path.stem, "size": st.st_size, "mtime": st.st_mtime}
    try:
        cfg = json.loads(path.read_text())
        comp, noc, dram = cfg["compute"], cfg["noc"], cfg["dram"]
        ent.update({
            "sa": comp["mm_pad_shape"][0],
            "dram_bw": dram["bandwidth_GBps"],
            "noc_topo": noc["topology"],
            "noc_bw": noc["bandwidth_bytepc"],
            "row": dram["num_access_per_row"],
            "freq_mhz": dram["npu_freq_MHz"],
            "trp": dram["tRP"],
        })
    except Exception:
        ent["parse_error"] = True
    return ent


@bp.route("/hwconfigs")
def hwconfigs():
    out = [_summarize_hw_config(p) for p in sorted(HW_CONFIG_DIR.glob("*.json"))]
    out.sort(key=lambda e: e["name"])
    return jsonify({"configs": out})


@bp.route("/hwconfigs/<name>")
def hwconfig_detail(name):
    if not _NAME_RE.match(name):
        return jsonify({"error": "invalid config name"}), 400
    path = HW_CONFIG_DIR / f"{name}.json"
    if not path.is_file():
        return jsonify({"error": "config not found"}), 404
    try:
        return jsonify(json.loads(path.read_text()))
    except ValueError as e:
        return jsonify({"error": f"invalid JSON: {e}"}), 500


def _validate_hw_config(cfg):
    """Return an error message, or None if the config schema is acceptable."""
    if not isinstance(cfg, dict):
        return "config must be a JSON object"
    for section in ("compute", "noc", "dram"):
        if not isinstance(cfg.get(section), dict):
            return f"missing or invalid section: {section}"
    comp = cfg["compute"]
    for k in ("ew_pad_len", "ew_reuse_num", "ew_flopc", "mm_flopc",
              "load_store_bw_bytepc", "byte_per_elem", "mm_init_cycle"):
        if not _is_num(comp.get(k)):
            return f"compute.{k} must be a number"
    shape = comp.get("mm_pad_shape")
    if (not isinstance(shape, list) or len(shape) != 3 or
            any(not isinstance(x, int) or isinstance(x, bool) or x <= 0
                for x in shape)):
        return "compute.mm_pad_shape must be a list of 3 positive integers"
    reuse = comp.get("mm_reuse_list")
    if (not isinstance(reuse, list) or not reuse or
            any(not _is_num(x) for x in reuse)):
        return "compute.mm_reuse_list must be a non-empty list of numbers"
    if not isinstance(comp.get("ew_mm_overlap"), bool):
        return "compute.ew_mm_overlap must be a boolean"
    noc = cfg["noc"]
    if not _is_num(noc.get("bandwidth_bytepc")):
        return "noc.bandwidth_bytepc must be a number"
    if noc.get("topology") not in (1, 2, 3):
        return "noc.topology must be 1, 2 or 3"
    if not isinstance(noc.get("default_noc"), bool):
        return "noc.default_noc must be a boolean"
    dram = cfg["dram"]
    for k in ("CL", "tRCD", "tRP", "bandwidth_GBps", "num_access_per_row",
              "npu_freq_MHz"):
        if not _is_num(dram.get(k)):
            return f"dram.{k} must be a number"
    return None


@bp.route("/hwconfigs", methods=["POST"])
def hwconfig_create():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "JSON body required"}), 400
    name = body.get("name")
    if not isinstance(name, str) or not _NAME_RE.match(name):
        return jsonify({"error": "invalid name"}), 400
    cfg = body.get("config")
    err = _validate_hw_config(cfg)
    if err:
        return jsonify({"error": err}), 400
    path = HW_CONFIG_DIR / f"{name}.json"
    if path.exists():
        return jsonify({"error": "config already exists"}), 409
    try:
        with open(path, "x") as f:  # exclusive create: never overwrite
            json.dump(cfg, f, indent=4)
            f.write("\n")
    except FileExistsError:
        return jsonify({"error": "config already exists"}), 409
    return jsonify({"created": name}), 201


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

def _texpr_batches(name):
    batches = []
    for p in MODELS_DIR.glob(f"TExpr/TExpr_{name}-b*.json"):
        m = _TEXPR_BATCH_RE.search(p.name)
        if m:
            batches.append(int(m.group(1)))
    return sorted(batches)


def _parsed_path(name):
    return PARSED_MODELS_DIR / f"parsed_{name}.json"


@bp.route("/models")
def models():
    parsed_names = {p.stem[len("parsed_"):]
                    for p in PARSED_MODELS_DIR.glob("parsed_*.json")}
    original_names = {p.stem for p in ORIGINAL_MODELS_DIR.glob("*.json")}
    out = []
    for name in sorted(parsed_names | original_names):
        ent = {"name": name,
               "has_original": name in original_names,
               "has_parsed": name in parsed_names,
               "op_count": 0, "op_types": {}}
        if name in parsed_names:
            try:
                ops = json.loads(_parsed_path(name).read_text())
                ent["op_count"] = len(ops)
                counts = Counter(op[1] for op in ops
                                 if isinstance(op, list) and len(op) > 1)
                ent["op_types"] = dict(counts.most_common(12))
            except (OSError, ValueError):
                ent["parse_error"] = True
        ent["texpr_batches"] = _texpr_batches(name)
        out.append(ent)
    return jsonify({"models": out})


@bp.route("/models/<name>/ops")
def model_ops(name):
    if not _MODEL_NAME_RE.match(name):
        return jsonify({"error": "invalid model name"}), 400
    path = _parsed_path(name).resolve()
    try:
        path.relative_to(PARSED_MODELS_DIR)
    except ValueError:
        return jsonify({"error": "invalid model name"}), 400
    if not path.is_file():
        return jsonify({"error": "parsed model not found"}), 404
    try:
        ops = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        return jsonify({"error": f"failed to parse model: {e}"}), 500

    q = request.args.get("q", "").strip().lower()
    op_type = request.args.get("type", "").strip()
    rows = []
    for op in ops:
        if not isinstance(op, list) or len(op) < 5:
            continue
        if q and q not in str(op[2]).lower():
            continue
        if op_type and op[1] != op_type:
            continue
        rows.append({"id": op[0], "type": op[1], "expr": op[2],
                     "shapes": op[3], "inputs": op[4]})

    total = len(rows)
    page = max(1, request.args.get("page", 1, type=int))
    page_size = min(500, max(1, request.args.get("page_size", 100, type=int)))
    rows = rows[(page - 1) * page_size: page * page_size]
    return jsonify({"total": total, "page": page, "page_size": page_size,
                    "rows": rows})


# ---------------------------------------------------------------------------
# NoC distance tables
# ---------------------------------------------------------------------------

@bp.route("/noc/tables")
def noc_tables():
    out = []
    for p in sorted(NOC_TABLES_DIR.glob("Topo.*_*.dist")):
        m = _DIST_NAME_RE.match(p.name)
        if m:
            out.append({"topo": m.group(1).upper(), "n": int(m.group(2))})
    return jsonify({"tables": out})


@bp.route("/noc/tables/<topo>/<int:n>")
def noc_table(topo, n):
    topo = topo.upper()
    if topo not in _TOPO_NAMES:
        return jsonify({"error": "topology must be MESH, TORUS or ALL"}), 400
    path = NOC_TABLES_DIR / f"Topo.{topo}_{n}.dist"
    if not path.is_file():
        return jsonify({"error": "table not found"}), 404
    try:
        with open(path, "rb") as f:
            mat = np.asarray(pickle.load(f))
    except Exception as e:
        return jsonify({"error": f"failed to load table: {e}"}), 500
    size = mat.shape[0]
    off_diag = mat[~np.eye(size, dtype=bool)]
    max_hops = int(off_diag.max())
    out = {
        "topo": topo,
        "n": size,
        "stats": {
            "avg_hops": float(off_diag.mean()),
            "max_hops": max_hops,
            "diameter": max_hops,
        },
    }
    if size <= 256:
        out["matrix"] = mat.tolist()
    else:
        block = math.ceil(size / 256)
        crop = (size // block) * block
        small = (mat[:crop, :crop]
                 .reshape(crop // block, block, crop // block, block)
                 .mean(axis=(1, 3)))
        out["matrix"] = np.round(small, 2).tolist()
        out["downsampled"] = True
        out["block"] = block
    return jsonify(out)


# ---------------------------------------------------------------------------
# llmservingsim profiles
# ---------------------------------------------------------------------------

@bp.route("/serving/profiles")
def serving_profiles():
    base = LLMSERVING_DIR / "3D-chip"
    out = []
    if base.is_dir():
        for attn in sorted(base.rglob("attention.csv")):
            tp_dir = attn.parent
            parts = tp_dir.relative_to(base).parts
            # .../<vendor>/<model>/<precision>/<tpDir> -> model is 3rd from end
            model = parts[-3] if len(parts) >= 3 else (parts[0] if parts else "")
            meta = None
            if yaml is not None:
                for cand in (tp_dir / "meta.yaml", tp_dir.parent / "meta.yaml"):
                    if cand.is_file():
                        try:
                            meta = yaml.safe_load(cand.read_text())
                        except Exception:
                            meta = None
                        break
            out.append({
                "path": tp_dir.relative_to(PROJECT_ROOT).as_posix(),
                "model": model,
                "tp": tp_dir.name,
                "files": [f for f in _SERVING_CSV_FILES
                          if (tp_dir / f).is_file()],
                "meta": meta,
            })
    return jsonify({"profiles": out})


def _csv_to_columns(path, max_rows=5000):
    with open(path, newline="", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {"columns": [], "rows": []}
    data = []
    for row in rows[1:max_rows + 1]:
        conv = []
        for cell in row:
            try:
                conv.append(float(cell))
            except ValueError:
                conv.append(cell)
        data.append(conv)
    return {"columns": rows[0], "rows": data}


@bp.route("/serving/profile")
def serving_profile():
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "missing path"}), 400
    p = (PROJECT_ROOT / raw).resolve()
    try:
        p.relative_to(LLMSERVING_DIR)
    except ValueError:
        return jsonify({"error": "path outside llmservingsim"}), 403
    if not p.is_dir():
        return jsonify({"error": "profile directory not found"}), 404
    files = {}
    for name in _SERVING_CSV_FILES:
        f = p / name
        if f.is_file():
            files[name] = _csv_to_columns(f)
    return jsonify({"path": p.relative_to(PROJECT_ROOT).as_posix(),
                    "files": files})
