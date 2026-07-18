"""Operator classification into paper categories (FFN / Attn / Other).

Ported from ``benchmark_scripts/draw_op_breakdown.py`` (same state machine and
busy-interval time attribution) so the web UI can reproduce the Figure-20-style
operator breakdown for any result whose compile cache
(``all_configs_dict.json``) is available.
"""

import json
import os
from typing import Dict, List, Optional

from .config import PICKLES_DIR

FFN_ACT = {"Sigmoid", "Relu", "Erf", "Gelu", "Tanh"}
ATTN_DEF = {"SoftmaxBasic", "BatchMatMul"}

CAT_LABEL = {
    "ffn_up": "FFN (Up/Gate)",
    "ffn_down": "FFN (Down)",
    "attn_flash": "Attn (Flash)",
    "attn_qkv": "Attn (QKV Proj)",
    "attn_score": "Attn (QKT+Softmax)",
    "attn_av": "Attn (AV+Out Proj)",
    "other": "Other",
}
DECODE_CATS = ["ffn_up", "ffn_down", "attn_qkv", "attn_score", "attn_av", "other"]
PREFILL_CATS = ["ffn_up", "ffn_down", "attn_flash", "other"]


def _has_attn_ahead(types, i):
    for j in range(i + 1, len(types)):
        if types[j] == "Sum":
            return False
        if types[j] in ATTN_DEF:
            return True
    return False


def classify_ops(types: List[str], flash: bool) -> List[str]:
    """Label each operator with a sub-category (see draw_op_breakdown.py)."""
    cats = []
    phase = "other"
    seen_attn = False
    softmax_seen = False
    ffn_act_seen = False
    for i, t in enumerate(types):
        if t == "SoftmaxBasic":
            phase = "attn"; seen_attn = True; softmax_seen = True
            c = "attn_flash" if flash else "attn_score"
        elif t == "BatchMatMul":
            phase = "attn"; seen_attn = True
            c = "attn_flash" if flash else \
                ("attn_av" if softmax_seen else "attn_score")
        elif t in FFN_ACT:
            phase = "ffn"; ffn_act_seen = True
            c = "ffn_up"
        elif t == "Sum":
            if phase == "ffn":
                seen_attn = False
            phase = "other"; softmax_seen = False; ffn_act_seen = False
            c = "other"
        elif t == "Dot":
            if phase == "attn":
                c = "attn_flash" if flash else "attn_av"
            elif phase == "ffn":
                c = "ffn_down" if ffn_act_seen else "ffn_up"
            elif not seen_attn:
                if _has_attn_ahead(types, i):
                    phase = "attn"; seen_attn = True; softmax_seen = False
                    c = "attn_flash" if flash else "attn_qkv"
                else:
                    c = "other"
            else:
                phase = "ffn"; ffn_act_seen = False
                c = "ffn_up"
        else:
            if phase == "attn":
                c = "attn_flash" if flash else \
                    ("attn_av" if softmax_seen else "attn_score")
            elif phase == "ffn":
                c = "ffn_up"
            else:
                c = "other"
        cats.append(c)
    return cats


def attribute_op_times(spans: List) -> List[float]:
    """Per-op time attribution by busy interval, split among concurrent ops.

    ``spans`` is a list of (start, finish) cycle pairs.  Every time-slice is
    divided equally among the ops active during it.
    """
    n = len(spans)
    events = []
    for i, (s, f) in enumerate(spans):
        if f <= s:
            f = s + 1
        events.append((s, 1, i))
        events.append((f, 0, i))
    events.sort(key=lambda e: (e[0], e[1]))
    times = [0.0] * n
    active = set()
    prev_t = None
    for t, kind, i in events:
        if prev_t is not None and active and t > prev_t:
            share = (t - prev_t) / len(active)
            for j in active:
                times[j] += share
        if kind == 1:
            active.add(i)
        else:
            active.discard(i)
        prev_t = t
    return times


def find_configs_dict(cfg: Dict) -> Optional[str]:
    """Locate the all_configs_dict.json matching a result config.

    Returns a path whose op count is validated by the caller, or None.
    """
    prefill = cfg["mode"] == "prefill"
    seq = cfg.get("seq_length") or 2048
    sub = f"outputs_icbm_{seq}{'_prefill' if prefill else ''}"
    base = PICKLES_DIR / sub / f"{cfg['num_cores']}cores"
    candidates = []
    if prefill:
        candidates.append(base / f"{cfg['model']}-b1")
        candidates.append(base / f"{cfg['model']}-b{cfg['batch_size']}")
    else:
        candidates.append(base / f"{cfg['model']}-b{cfg['batch_size']}")
        candidates.append(base / f"{cfg['model']}-b32")
    for d in candidates:
        p = d / "all_configs_dict.json"
        if p.is_file():
            return str(p)
    return None


def load_op_types(configs_dict_path: str) -> List[str]:
    """Keys are ``Op_{idx}_{Type}`` in execution order."""
    with open(configs_dict_path) as f:
        names = json.load(f)
    # Sort defensively by numeric index in case dict order is not preserved.
    def _idx(k):
        try:
            return int(k.split("_", 2)[1])
        except (IndexError, ValueError):
            return 0
    keys = sorted(names.keys(), key=_idx)
    return [k.split("_", 2)[2] if len(k.split("_", 2)) > 2 else "Unknown"
            for k in keys]


def op_breakdown(cfg: Dict, operators: List[Dict]) -> Optional[Dict]:
    """Category time breakdown for a result.

    ``operators`` comes from parsers.parse_operators_*.  Returns None when the
    compile cache is unavailable or the op counts do not line up.
    """
    path = find_configs_dict(cfg)
    if not path:
        return None
    try:
        types = load_op_types(path)
    except (OSError, ValueError):
        return None
    if len(types) != len(operators):
        return None
    flash = cfg["mode"] == "prefill"
    cats = classify_ops(types, flash)
    spans = [(op["start_ld"], op["start_finish"]) for op in operators]
    times = attribute_op_times(spans)
    per_cat: Dict[str, float] = {}
    per_cat_count: Dict[str, int] = {}
    for c, t in zip(cats, times):
        per_cat[c] = per_cat.get(c, 0.0) + t
        per_cat_count[c] = per_cat_count.get(c, 0) + 1
    order = PREFILL_CATS if flash else DECODE_CATS
    return {
        "categories": [
            {"key": c, "label": CAT_LABEL[c],
             "cycles": per_cat.get(c, 0.0),
             "op_count": per_cat_count.get(c, 0)}
            for c in order if per_cat.get(c, 0.0) > 0 or c in per_cat
        ],
        "op_categories": cats,
        "op_types": types,
        "op_times": times,
        "source": os.path.relpath(path),
    }
