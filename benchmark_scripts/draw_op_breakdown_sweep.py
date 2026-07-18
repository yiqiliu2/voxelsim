"""Op-breakdown sweep over (sequence_length, batch_size).

Grid:
  rows = seq in {1024, 2048, 4096}
  cols = decode b16 / decode b32 / decode b64 / prefill b1
  each cell = one stacked op-breakdown bar per model (4 LLMs)

Reuses the operator classifier + telescoping attribution from
``draw_op_breakdown.py``; only the data-loading paths are parametrized by
(seq, batch) and the per-seq trace dir (results/logs_seq{S}) produced by
``run_seq_batch_sweep.py``.
"""

import os

import numpy as np
import ujson as json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fig_common import *  # colors, modelnames, mpl rcParams
from run_all_tests import default_params, NPU_FREQ_MHZ
from parse_exposed import parse_log_file

# Reuse the classifier and colour/label machinery verbatim.
from draw_op_breakdown import (
    classify_ops, read_exe_time, attribute_op_times,
    DECODE_CATS, PREFILL_CATS, LEGEND_CATS, ALL_CATS,
    CAT_LABEL, CAT_COLOR, CAT_HATCH, CYCLE_TO_MS, LATENCY_FACTOR,
)

plt.rc('xtick', labelsize=13)
plt.rc('ytick', labelsize=14)
plt.rcParams['hatch.linewidth'] = 0.6

models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

# Single-model figure: one row of four panels for THE_MODEL.
THE_MODEL = "llama3-70"

# Four panels. Each entry: (x-axis label, top held-constant title,
# is_prefill, [(seq,batch),...], bar tick labels).
PANELS = [
    ("Sequence Length",  "Batch Size = 32",      False, [(1024, 32), (2048, 32), (4096, 32)], ["1k", "2k", "4k"]),
    ("Batch Size",       "Sequence = 2k", False, [(2048, 16), (2048, 32), (2048, 64)], ["16", "32", "64"]),
    ("Sequence Length",  "Batch Size = 1",       True,  [(1024, 1),  (2048, 1),  (4096, 1)],  ["1k", "2k", "4k"]),
    ("Batch Size",       "Sequence = 2k", True,  [(2048, 1),  (2048, 2),  (2048, 4)],  ["1", "2", "4"]),
]

CORES = default_params['num_cores']
SA = default_params['sa']
SRAM_KB = default_params['sram_kb']
DRAM_BW = default_params['dram_bw']
NOC_TOPO = default_params['noc_topo']
NOC_BW = default_params['noc_bw']
CG = default_params['core_group']
ROW = default_params['row']


def trace_base(seq):
    return f"results/logs_seq{seq}"


def _config_dir(model, prefill, seq, batch):
    sub = f"outputs_icbm_{seq}_prefill" if prefill else f"outputs_icbm_{seq}"
    return os.path.join("results", "pickles", sub, f"{CORES}cores",
                        f"{model}-b{batch}")


def get_op_types(model, prefill, seq, batch):
    cfg = os.path.join(_config_dir(model, prefill, seq, batch),
                       "all_configs_dict.json")
    with open(cfg) as f:
        names = json.load(f)
    return [k.split("_", 2)[2] for k in names]


def get_logfile(model, prefill, seq, batch):
    op = "prefill" if prefill else "decode"
    return os.path.join(
        trace_base(seq), model, f"bs_{batch}", f"core_{CORES}", op,
        f"sa_{SA}-vu_{SA}",
        f"sram_{SRAM_KB}-drambw_{DRAM_BW}_PLACEHOLDER",
        f"topo_{NOC_TOPO}-nocbw{NOC_BW}", "best",
        f"output_cg_{CG}_row_{ROW}.log")


def category_times(model, prefill, seq, batch):
    """Absolute ms per category for one (model, mode, seq, batch). None if missing."""
    logfile = get_logfile(model, prefill, seq, batch)
    if not os.path.exists(logfile):
        return None
    with open(logfile) as f:
        content = f.read()
    if "Overall Util:" not in content:
        return None
    exe_time = read_exe_time(content)
    op_logs = parse_log_file(content, dram_bw_GBps=DRAM_BW,
                             npu_freq_MHz=NPU_FREQ_MHZ)
    cats = classify_ops(get_op_types(model, prefill, seq, batch), flash=prefill)

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


def draw_panel(ax, model, prefill, sweep, xlabels, scale=1.0):
    """One panel = 3 op-breakdown stacked bars for `model` over `sweep`.

    `scale` multiplies every category height (e.g. 1e-3 to show ms in seconds).
    """
    cats = PREFILL_CATS if prefill else DECODE_CATS
    breakdowns = [category_times(model, prefill, seq, b) for (seq, b) in sweep]
    x = list(range(len(sweep)))
    bottoms = [0.0] * len(sweep)
    for cat in cats:
        heights = [((bd[cat] if bd else 0.0) * scale) for bd in breakdowns]
        ax.bar(x, heights, bottom=bottoms, width=0.65, color=CAT_COLOR[cat],
               edgecolor="black", linewidth=0.5, hatch=CAT_HATCH.get(cat),
               label=CAT_LABEL[cat])
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=12)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5,
            color="lightgrey", zorder=1)
    ax.margins(x=0.12)
    ax.tick_params(axis='y', labelsize=12)


def draw_figure():
    os.makedirs("figures", exist_ok=True)
    # Single model: one row of four panels.
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
    plt.subplots_adjust(top=0.74, bottom=0.30, left=0.06, right=0.995, wspace=0.46)

    for pi, (xlabel, toplabel, prefill, sweep, xticks) in enumerate(PANELS):
        ax = axes[pi]
        # prefill TTFT shown in seconds, decode TBT in milliseconds.
        draw_panel(ax, THE_MODEL, prefill, sweep, xticks,
                   scale=1e-3 if prefill else 1.0)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_title(toplabel, fontsize=13)
        ax.set_ylabel("Prefill TTFT (s)" if prefill else "Decode TBT (ms)",
                      fontsize=14, labelpad=1)

    from matplotlib.patches import Patch
    handles = [Patch(facecolor=CAT_COLOR[c], edgecolor="black", linewidth=0.5,
                     hatch=CAT_HATCH.get(c), label=CAT_LABEL[c])
               for c in LEGEND_CATS]
    fig.legend(handles=handles, ncol=4, fontsize=12.5, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.18),
               columnspacing=1.0, handlelength=1.3, handletextpad=0.45,
               labelspacing=0.25)

    out = "figures/eval_op_breakdown_sweep.pdf"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"Wrote {out}")


if __name__ == "__main__":
    draw_figure()
