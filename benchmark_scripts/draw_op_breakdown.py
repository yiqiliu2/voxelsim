"""Per-model fine-grained time-breakdown figure.

Formatted like ``figures/eval_sw_compiler.pdf`` but with a single stacked
column per model per mode (prefill / decode).  Each column shows the
absolute execution time split into fine-grained sub-stages, grouped into
three colour families (one hue per super-category, shaded light -> dark):

  * FFN  (one hue)  -- up/gate projection (activation folded in), down projection
  * ATTN (one hue)  -- attention.  Granularity differs by mode: *prefill* runs
                       as flash attention (QK^T / softmax / attn*V fused on-chip,
                       no HBM round-trip) so it is a single segment; *decode* is
                       NOT flash -- those intermediates round-trip through HBM --
                       so it is split into QKV proj (GEMM only) / QK^T+softmax
                       (incl. RoPE & scaling) / attn*V + output projection
  * Other (one hue) -- layernorms; residuals / embeddings / LM head

Operator categories are recovered by replaying the operator-type sequence
(``all_configs_dict.json``) through a small per-layer state machine, and the
per-operator time is taken from the simulation log via a sort-by-finish
telescoping attribution (which sums exactly to the reported EXE time).
"""

import os
import sys

import numpy as np
import ujson as json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fig_common import *  # colors, modelnames, mpl rcParams
from run_all_tests import default_params, NPU_FREQ_MHZ
from parse_exposed import parse_log_file

TRACE_OUT = "results/logs"

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=20)
plt.rcParams['hatch.linewidth'] = 0.6  # thin FFN hatch lines

# LATENCY_FACTOR is a no-op (1.0).  It exists as a hook for empirical
# calibration against real hardware measurements, but the simulator has
# not been validated against silicon — any value other than 1.0 would be
# arbitrary data manipulation.  Keep this at 1.0 unless you have published
# hardware measurements to calibrate against.
LATENCY_FACTOR = 1.0
assert LATENCY_FACTOR == 1.0, (
    "LATENCY_FACTOR must remain 1.0 unless empirically calibrated "
    "against real hardware measurements.  See comment above."
)

# Cycles -> milliseconds, so the panels read as TTFT/TBT in ms (like Figure 8).
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)

# Models, in the same order as eval_sw_compiler.pdf (the four LLMs).
models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

# One hue per super-category (FFN green, ATTN orange, Other blue); sub-stages
# are shades within the family.
FFN_HUE, ATTN_HUE, OTHER_HUE = colors[2], colors[0], colors[1]

# Attention granularity differs by mode (see module docstring): prefill is
# flash -> one fused segment; decode is not flash -> split into sub-stages.
DECODE_ATTN  = ["attn_qkv", "attn_score", "attn_av"]
PREFILL_ATTN = ["attn_flash"]

# Stacking order (bottom -> top) per mode.  (Activation is tiny -> folded into
# up/gate, so FFN is just up/gate and down.)
_FFN = ["ffn_up", "ffn_down"]
DECODE_CATS  = _FFN + DECODE_ATTN  + ["other"]
PREFILL_CATS = _FFN + PREFILL_ATTN + ["other"]

# Legend = union of both modes (attention grouped: fused block, then stages).
LEGEND_CATS = _FFN + ["attn_flash"] + DECODE_ATTN + ["other"]
ALL_CATS = LEGEND_CATS

CAT_LABEL = {
    "ffn_up":     "FFN (Up/Gate)",
    "ffn_down":   "FFN (Down)",
    "attn_flash": "Attn (Flash)",
    "attn_qkv":   "Attn (QKV Proj)",
    "attn_score": "Attn (QK$^\\mathsf{T}$+Softmax)",
    "attn_av":    "Attn (AV+Out Proj)",
    "other":      "Other",
}


def _mix(base, f):
    """Lighten (``f`` > 0, toward white) or darken (``f`` < 0, toward black)."""
    rgb = np.array(mcolors.to_rgb(base))
    c = rgb + (1.0 - rgb) * f if f >= 0 else rgb * (1.0 + f)
    return tuple(np.clip(c, 0.0, 1.0))


# Per-sub-category colour.  Decode attention uses three orange shades; the
# prefill fused block uses the orange base, kept distinct from those shades.
CAT_COLOR = {
    "ffn_up":     _mix(FFN_HUE, 0.60),    # FFN lighter overall
    "ffn_down":   _mix(FFN_HUE, -0.20),
    "attn_flash": _mix(ATTN_HUE, 0.05),   # Flash slightly lighter
    "attn_qkv":   _mix(ATTN_HUE, 0.42),
    "attn_score": _mix(ATTN_HUE, -0.12),  # QK^T+Softmax lighter
    "attn_av":    _mix(ATTN_HUE, -0.28),  # AV+Out Proj lighter (esp. this one)
    "other":      _mix(OTHER_HUE, 0.0),
}

# Diagonal hatch on FFN so it stays distinguishable from attention in B&W /
# grayscale (texture, not just colour).
CAT_HATCH = {c: "///" for c in _FFN}

# Operator-type markers used by the classifier.
FFN_ACT = {"Sigmoid", "Relu", "Erf", "Gelu", "Tanh"}
ATTN_DEF = {"SoftmaxBasic", "BatchMatMul"}


def _has_attn_ahead(types, i):
    """True if an attention marker occurs before the next norm boundary."""
    for j in range(i + 1, len(types)):
        if types[j] == "Sum":
            return False
        if types[j] in ATTN_DEF:
            return True
    return False


def classify_ops(types, flash):
    """Label each operator with a sub-category.

    A per-layer state machine walks the operator-type sequence.  Norms start
    with a reduction ('Sum'); attention is anchored by BatchMatMul / Softmax;
    FFN by its activation.  Within FFN, the activation (tiny, folded into
    up/gate) marks the boundary that splits the projection 'Dot's into up/gate
    (before) and down (after); a projection 'Dot' with no attention ahead is
    embedding / LM head ('other').

    Attention depends on ``flash``.  When ``flash`` (prefill) the whole block is
    a single 'attn_flash' segment -- QK^T / softmax / attn*V are fused on-chip.
    Otherwise (decode) it is split: 'attn_qkv' is the QKV projection GEMM ONLY;
    everything from there up to softmax (RoPE, QK^T BatchMatMuls, scaling/mask,
    softmax) is score-prep -> 'attn_score'; attn*V BatchMatMuls, the output
    'Dot' and any post-softmax misc -> 'attn_av'.
    """
    cats = []
    phase = "other"
    seen_attn = False     # whether this layer's attention block has begun
    softmax_seen = False  # within the current attention block (decode only)
    ffn_act_seen = False  # within the current FFN block
    for i, t in enumerate(types):
        if t == "SoftmaxBasic":
            phase = "attn"; seen_attn = True; softmax_seen = True
            c = "attn_flash" if flash else "attn_score"
        elif t == "BatchMatMul":
            phase = "attn"; seen_attn = True
            c = "attn_flash" if flash else \
                ("attn_av" if softmax_seen else "attn_score")
        elif t in FFN_ACT:
            # Activation is tiny -> folded into up/gate; still marks up->down.
            phase = "ffn"; ffn_act_seen = True
            c = "ffn_up"
        elif t == "Sum":  # norm boundary
            if phase == "ffn":
                seen_attn = False  # FFN norm -> next layer
            phase = "other"; softmax_seen = False; ffn_act_seen = False
            c = "other"
        elif t == "Dot":
            if phase == "attn":
                c = "attn_flash" if flash else "attn_av"   # output projection
            elif phase == "ffn":
                c = "ffn_down" if ffn_act_seen else "ffn_up"
            elif not seen_attn:
                if _has_attn_ahead(types, i):
                    phase = "attn"; seen_attn = True; softmax_seen = False
                    c = "attn_flash" if flash else "attn_qkv"   # QKV projection
                else:
                    c = "other"                  # embedding / LM head
            else:
                phase = "ffn"; ffn_act_seen = False
                c = "ffn_up"                      # first FFN projection
        else:  # elementwise / misc -> current phase's current sub-stage
            if phase == "attn":
                # Pre-softmax elementwise (RoPE on Q/K, QK^T scaling, masking)
                # is score-prep -> fold into attn_score; only the QKV projection
                # GEMM stays in attn_qkv.  Post-softmax misc -> attn_av.
                c = "attn_flash" if flash else \
                    ("attn_av" if softmax_seen else "attn_score")
            elif phase == "ffn":
                c = "ffn_up"                      # up/gate proj + activation/gating
            else:
                c = "other"                      # norm body (Div/Sqrt/Sub/...)
        cats.append(c)
    return cats


def _config_dir(model, prefill):
    """Directory of the all_configs_dict.json holding operator names.

    Decode uses the unrolled full-model graph (batch 32, KV 2048); prefill is the
    real single-request graph (batch 1, full 2048-token prompt).
    """
    sub = "outputs_icbm_2048_prefill" if prefill else "outputs_icbm_2048"
    bs = "b1" if prefill else "b32"
    return os.path.join("results", "pickles", sub,
                        f"{default_params['num_cores']}cores", f"{model}-{bs}")


def get_op_types(model, prefill):
    cfg = os.path.join(_config_dir(model, prefill), "all_configs_dict.json")
    with open(cfg) as f:
        names = json.load(f)
    # Keys are "Op_{idx}_{Type}" in execution order.
    return [k.split("_", 2)[2] for k in names]


def get_logfile(model, prefill):
    p = default_params
    op = "prefill" if prefill else "decode"
    bs = "bs_1" if prefill else "bs_32"   # prefill = real batch 1
    return os.path.join(
        TRACE_OUT, model, bs, f"core_{p['num_cores']}", op,
        f"sa_{p['sa']}-vu_{p['sa']}",
        f"sram_{p['sram_kb']}-drambw_{p['dram_bw']}_PLACEHOLDER",
        f"topo_{p['noc_topo']}-nocbw{p['noc_bw']}", "best",
        f"output_cg_{p['core_group']}_row_{p['row']}.log")


def read_exe_time(log_content):
    """The reported total EXE time from the first log line."""
    first = log_content.split("\n", 1)[0]
    return float(first.split("EXE time (total, fused):")[1].split(",")[0])


def attribute_op_times(op_logs):
    """Per-op time attribution by busy interval, split among concurrent ops.

    Each op occupies ``[t_dram_ld_start, t_finish]``; every time-slice is
    divided equally among the ops active during it. This is stable under
    finish-order reshuffles, unlike the old telescoping finish-gap method
    (which let a small op's time leak into an adjacent op when their finish
    times coincided -- e.g. QKV-proj collapsing to ~0, or FFN-down time
    landing on the next norm). Returns per-op times summing to the
    union-of-intervals busy span; callers renormalize to the reported EXE.
    """
    n = len(op_logs)
    events = []
    for i in range(n):
        s = op_logs[i].t_dram_ld_start
        f = op_logs[i].t_finish
        if f <= s:
            f = s + 1   # guard zero/negative spans
        events.append((s, 1, i))   # start
        events.append((f, 0, i))   # end (sorts before start at same t)
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


def category_times(model, prefill):
    """Return absolute ms per category (keys in ``ALL_CATS``) for a config.

    Per-operator time is attributed by sorting operators by finish time and
    taking the gap to the previous finish (telescopes exactly to the last
    finish).  The result is rescaled so the categories sum to the log's
    reported EXE time (for prefill this lifts a single-layer trace to the full
    model scale, exactly as the reference figure does), then scaled by
    ``LATENCY_FACTOR`` is a no-op (1.0) unless empirically calibrated
    against real hardware (the factor is folded into every category).
    """
    logfile = get_logfile(model, prefill)
    assert os.path.exists(logfile), f"Missing log file {logfile}"
    with open(logfile) as f:
        content = f.read()
    exe_time = read_exe_time(content)
    op_logs = parse_log_file(content, dram_bw_GBps=default_params['dram_bw'],
                             npu_freq_MHz=NPU_FREQ_MHZ)
    cats = classify_ops(get_op_types(model, prefill), flash=prefill)

    n = len(op_logs)
    times = attribute_op_times(op_logs)

    agg = {c: 0.0 for c in ALL_CATS}
    for i in range(n):
        c = cats[i] if i < len(cats) else "other"
        agg[c] += times[i]
    total = sum(agg.values())
    if total > 0:
        for c in ALL_CATS:
            agg[c] *= exe_time * LATENCY_FACTOR * CYCLE_TO_MS / total
    return agg


def draw_one(ax, prefill):
    cats = PREFILL_CATS if prefill else DECODE_CATS
    breakdowns = [category_times(m, prefill) for m in models]
    x = list(range(len(models)))
    bottoms = [0.0] * len(models)
    for cat in cats:                             # stack bottom -> top
        heights = [bd[cat] for bd in breakdowns]
        ax.bar(x, heights, bottom=bottoms, width=0.45, color=CAT_COLOR[cat],
               edgecolor="black", linewidth=0.5, hatch=CAT_HATCH.get(cat),
               label=CAT_LABEL[cat])
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels([modelnames[m] for m in models], rotation=25,
                       fontsize=22, ha="right", rotation_mode="anchor")
    if prefill:
        ax.set_ylabel("Prefill\nTTFT (ms)", fontsize=23)
    else:
        ax.set_ylabel("Decode\nTBT (ms)", fontsize=23)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5,
            color="lightgrey", zorder=1)
    ax.margins(x=0.05)


def draw_figure():
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.2))
    plt.subplots_adjust(top=0.9, bottom=0.35, left=0.09, right=0.97,
                        wspace=0.385)

    draw_one(axes[0], prefill=False)   # TBT (decode)
    draw_one(axes[1], prefill=True)    # TTFT (prefill)

    # Union legend (decode splits attention, prefill fuses it), built by hand so
    # both modes' attention entries appear regardless of which axis drew them.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CAT_COLOR[c], edgecolor="black", linewidth=0.5,
                     hatch=CAT_HATCH.get(c), label=CAT_LABEL[c])
               for c in LEGEND_CATS]
    fig.legend(handles=handles, ncol=4, fontsize=18, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.195),
               columnspacing=1.0, handlelength=1.3, handletextpad=0.45,
               labelspacing=0.25)

    plt.savefig("figures/eval_op_breakdown.pdf", bbox_inches="tight",
                pad_inches=0.02)
    print("Wrote figures/eval_op_breakdown.pdf")


if __name__ == "__main__":
    draw_figure()
