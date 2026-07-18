#!/usr/bin/env python3

"""Plot power and energy metrics across architecture sweeps.

This script mirrors the style of the other figure utilities in this folder.
Edit the macros in the configuration block below to choose which metric to
plot on the y-axis, the configuration parameter to sweep on the x-axis, and
which implementation subdirectories to compare.  The script parses summary
power/energy statistics directly from the simulator logs using regular
expressions so it remains robust to large log files.
"""

from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator, NullFormatter

from fig_common import colors1, markers1, modelnames, lines1  # pylint: disable=import-error
from run_all_tests import (  # pylint: disable=import-error
    DF_DRAM_DIV,
    DF_DRAM_MUL,
    DF_FACTOR,
    TRACE_OUT,
    default_params,
    sweep_lists,
)


# ---------------------------------------------------------------------------
# Configuration macros (customize these for your study)
# ---------------------------------------------------------------------------

# Models to include in the figure (order controls subplot placement).
MODELS: List[str] = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

# Optional overrides for models rendered per stage. Leave as None to reuse MODELS.
MODELS_DECODE: List[str] | None = None
MODELS_PREFILL: List[str] | None = None
# MODELS_PREFILL: List[str] | None = ["llama2-13", "gemma2", "opt-30", "llama3-70", "dit-xl"]

# Order used to assign colors/markers. Extend if you add models not listed in MODELS.
MODEL_STYLE_ORDER: List[str] = MODELS.copy()

# Operation types to render.  Always render decode then prefill for paired subplots.
OP_TYPES: Tuple[str, ...] = ("decode", "prefill")

# Implementation subdirectory to read results from (e.g., "best").
IMPLEMENTATION_KEY: str = "best"

# Configuration parameters to sweep on the x-axis (must exist in default_params).
# Provide one entry per metric group. Each entry produces two subplots (decode/prefill).
X_PARAMS: List[str] = ["dram_bw", "num_cores"]

# Optional explicit list of sweep values per parameter.
# Example: {"num_cores": [64, 128], "dram_bw": [4000, 8000, 16000]}.
CUSTOM_X_VALUES: Dict[str, List[int]] | None = None

# Metrics to plot on the y-axis. One entry per configuration parameter above.
Y_METRICS: List[str] = ["total_energy_mj", "total_energy_mj"]

# Optional human-readable group labels. If omitted, the script derives labels from PARAM_TITLES.
GROUP_LABELS: List[str] | None = None

# Default axis scaling. Individual sweep parameters can override these via X_PARAM_SCALE_OVERRIDES.
USE_LOG_X_DEFAULT: bool = True
USE_LOG_Y_DEFAULT: bool = True

# Optional per-parameter overrides for axis scaling:
# {"dram_bw": (True, False)} means log x, linear y for that sweep.
X_PARAM_SCALE_OVERRIDES: Dict[str, Tuple[bool, bool]] = {"dram_bw": (False, True),
                                                         "num_cores": (True, True)}

# Batch size and core group are fixed for the experiments in this plot.
BATCH_SIZE: int = 32
CORE_GROUP: int = default_params["core_group"]

# Figure options.
FIGSIZE = (16, 3.3)
OUTPUT_DIR = "figures"
SAVE_PDF: bool = True
SUBPLOT_WSPACE: float | None = 0.3
YLABEL_PAD: float = 15.0

GROUP_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

# When True, decode subplots share a common y-range and prefill subplots share another.
# When False, each subplot uses the range implied by its own data.
SHARE_STAGE_Y_LIMITS: bool = True

# Hardcoded y-axis bounds per stage. Prefill energy varies less across sweeps
# than decode; a fixed upper bound prevents decade_bounds from pushing it to 100+.
STAGE_Y_LIMIT_OVERRIDES: Dict[str, Tuple[float, float]] = {
    "prefill": (10.0, 250.0),
}




# ---------------------------------------------------------------------------
# Regular expressions and metadata
# ---------------------------------------------------------------------------

EXE_RE = re.compile(r"EXE time .*?:\s*([-+eE0-9\.]+)", re.IGNORECASE)
ENERGY_RE = re.compile(
    r"Energy\s*\(mJ\)\s*:\s*([-+eE0-9\.]+)\s*,\s*Static\s*:\s*([-+eE0-9\.]+)\s*mJ\s*,\s*"
    r"Dyn\.\s*:\s*([-+eE0-9\.]+)\s*mJ(?:\s*\(Comp\s*=\s*([-+eE0-9\.]+)%\))?",
    re.IGNORECASE,
)
POWER_RE = re.compile(
    r"Power\s*\(w\)\s*:\s*([-+eE0-9\.]+)\s*,\s*Static\s*:\s*([-+eE0-9\.]+)\s*W"
    r"(?:\s*\(dram:\s*([-+eE0-9\.]+)\s*logic:\s*([-+eE0-9\.]+)\))?\s*,\s*"
    r"Dyn\.\s*:\s*([-+eE0-9\.]+)\s*W",
    re.IGNORECASE,
)
AVERAGE_POWER_RE = re.compile(r"Average Power \(W\):\s*([-+eE0-9\.]+)", re.IGNORECASE)

METRIC_INFO: Dict[str, Dict[str, object]] = {
    "total_energy_mj": {
        "field": "total_energy_mj",
        "label": "Total Energy (J)",
        "scale": 1e-3,
    },
    "static_energy_mj": {
        "field": "static_energy_mj",
        "label": "Static Energy (J)",
        "scale": 1e-3,
    },
    "dynamic_energy_mj": {
        "field": "dynamic_energy_mj",
        "label": "Dynamic Energy (J)",
        "scale": 1e-3,
    },
    "dynamic_energy_comp_pct": {
        "field": "dynamic_energy_comp_pct",
        "label": "Compute Share of Dynamic Energy (%)",
        "scale": 1.0,
    },
    "total_power_w": {
        "field": "total_power_w",
        "label": "Total Power (W)",
        "scale": 1.0,
    },
    "static_power_w": {
        "field": "static_power_w",
        "label": "Static Power (W)",
        "scale": 1.0,
    },
    "static_power_dram_w": {
        "field": "static_power_dram_w",
        "label": "Static Power – DRAM (W)",
        "scale": 1.0,
    },
    "static_power_logic_w": {
        "field": "static_power_logic_w",
        "label": "Static Power – Logic (W)",
        "scale": 1.0,
    },
    "dynamic_power_w": {
        "field": "dynamic_power_w",
        "label": "Dynamic Power (W)",
        "scale": 1.0,
    },
}

def param_scales(x_param: str) -> Tuple[bool, bool]:
    return X_PARAM_SCALE_OVERRIDES.get(x_param, (USE_LOG_X_DEFAULT, USE_LOG_Y_DEFAULT))


def decade_bounds(values: List[float]) -> Tuple[float, float]:
    positives = [val for val in values if val > 0 and math.isfinite(val)]
    if not positives:
        raise ValueError("No positive finite values for log-scale axis")

    min_val = min(positives)
    max_val = max(positives)

    log_min = math.floor(math.log10(min_val))
    log_max = math.ceil(math.log10(max_val))

    if log_min == log_max:
        log_min -= 1
        log_max += 1

    return 10 ** log_min, 10 ** log_max


def configure_log_ticks(ax: plt.Axes) -> None:
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(0.1, 1.0, 0.1)))
    ax.yaxis.set_minor_formatter(NullFormatter())


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trace_root() -> str:
    return os.path.join(repo_root(), TRACE_OUT)


def apply_impl_overrides(cfg: Dict[str, int], impl_key: str) -> Tuple[Dict[str, int], str]:
    """Return a copy of cfg with implementation-specific adjustments."""

    cfg_copy = dict(cfg)
    target_dir = impl_key

    # NOTE: "dataflow" override is for architectural comparison only.
    if impl_key == "dataflow":
        cfg_copy["num_cores"] = max(1, cfg_copy["num_cores"] // DF_FACTOR)
        cfg_copy["dram_bw"] = max(1, cfg_copy["dram_bw"] // DF_DRAM_DIV * DF_DRAM_MUL)
        cfg_copy["bs"] = max(1, cfg_copy["bs"] // DF_FACTOR)
        target_dir = "best"

    return cfg_copy, target_dir


def build_log_path(model: str, cfg: Dict[str, int], impl_dir: str, op_type: str) -> str:
    comp_dir = f"sa_{int(cfg['sa'])}-vu_{int(cfg['sa'])}"
    mem_dir = f"sram_{int(cfg['sram_kb'])}-drambw_{int(cfg['dram_bw'])}_PLACEHOLDER"
    noc_dir = f"topo_{int(cfg['noc_topo'])}-nocbw{int(cfg['noc_bw'])}"

    bs = 1 if op_type == "prefill" else int(cfg['bs'])  # prefill = real batch 1
    parts = [
        trace_root(),
        model,
        f"bs_{bs}",
        f"core_{int(cfg['num_cores'])}",
        op_type,
        comp_dir,
        mem_dir,
        noc_dir,
        impl_dir,
    ]
    log_dir = os.path.join(*parts)
    log_file = f"output_cg_{int(cfg['core_group'])}_row_{int(cfg['row'])}.log"
    return os.path.join(log_dir, log_file)


def parse_summary_metrics(log_path: str) -> Dict[str, float | List[float] | None]:
    with open(log_path, "r", encoding="utf-8") as infile:
        content = infile.read()

    metrics: Dict[str, float | List[float] | None] = {
        "operator_average_power_w": [],
    }

    exe_match = EXE_RE.search(content)
    if exe_match:
        metrics["exe_time_cycles"] = float(exe_match.group(1))

    energy_match = ENERGY_RE.search(content)
    if energy_match:
        metrics["total_energy_mj"] = float(energy_match.group(1))
        metrics["static_energy_mj"] = float(energy_match.group(2))
        metrics["dynamic_energy_mj"] = float(energy_match.group(3))
        comp_pct = energy_match.group(4)
        metrics["dynamic_energy_comp_pct"] = float(comp_pct) if comp_pct is not None else None
    else:
        raise ValueError(f"Could not locate energy summary in {log_path}")

    power_match = POWER_RE.search(content)
    if power_match:
        metrics["total_power_w"] = float(power_match.group(1))
        metrics["static_power_w"] = float(power_match.group(2))
        dram_static = power_match.group(3)
        logic_static = power_match.group(4)
        metrics["static_power_dram_w"] = float(dram_static) if dram_static else None
        metrics["static_power_logic_w"] = float(logic_static) if logic_static else None
        metrics["dynamic_power_w"] = float(power_match.group(5))
    else:
        metrics["total_power_w"] = np.nan
        metrics["static_power_w"] = np.nan
        metrics["static_power_dram_w"] = None
        metrics["static_power_logic_w"] = None
        metrics["dynamic_power_w"] = np.nan

    avg_power_values = [float(val) for val in AVERAGE_POWER_RE.findall(content)]
    metrics["operator_average_power_w"] = avg_power_values

    return metrics


def get_sweep_values(x_param: str) -> List[int]:
    if CUSTOM_X_VALUES is not None and x_param in CUSTOM_X_VALUES:
        return CUSTOM_X_VALUES[x_param]
    sweep = sweep_lists.get(x_param)
    if sweep is not None:
        return list(sweep)
    default_value = default_params.get(x_param)
    if default_value is None:
        raise KeyError(f"Parameter '{x_param}' not found in sweep lists or defaults.")
    return [default_value]


def get_metric_series(
    model: str, op_type: str, x_param: str, metric: str, log_y: bool
) -> Tuple[List[float], List[float]]:
    if metric not in METRIC_INFO:
        raise KeyError(f"Unknown metric '{metric}'. Update METRIC_INFO.")

    base_cfg = dict(default_params)
    if x_param not in base_cfg:
        raise KeyError(f"Parameter '{x_param}' missing from default configuration.")
    base_cfg.update({"bs": BATCH_SIZE, "core_group": CORE_GROUP, "row": default_params["row"]})

    sweep_vals = get_sweep_values(x_param)
    x_numeric: List[float] = []
    y_values: List[float] = []

    for value in sweep_vals:
        cfg_for_value = dict(base_cfg)
        cfg_for_value[x_param] = value
        cfg_impl, impl_dir = apply_impl_overrides(cfg_for_value, IMPLEMENTATION_KEY)

        try:
            log_path = build_log_path(model, cfg_impl, impl_dir, op_type)
        except KeyError as exc:
            print(f"[WARN] Missing key {exc} for configuration {cfg_impl}")
            continue

        if not os.path.exists(log_path):
            print(f"[WARN] Missing log: {log_path}")
            continue

        try:
            metrics = parse_summary_metrics(log_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            continue

        field_name = METRIC_INFO[metric]["field"]  # type: ignore[index]
        scale = METRIC_INFO[metric]["scale"]  # type: ignore[index]
        raw_value = metrics.get(field_name)  # type: ignore[arg-type]

        if raw_value is None:
            continue
        metric_value = float(raw_value) * float(scale)
        if log_y and metric_value <= 0:
            print(
                f"[WARN] Non-positive value {metric_value} for metric {metric} in {log_path}; skipping"
            )
            continue

        x_numeric.append(float(value))
        y_values.append(metric_value)

    return x_numeric, y_values


def draw_figures() -> None:
    if not X_PARAMS:
        raise ValueError("X_PARAMS must contain at least one configuration parameter.")
    if not Y_METRICS:
        raise ValueError("Y_METRICS must contain at least one metric identifier.")
    if len(X_PARAMS) != len(Y_METRICS):
        raise ValueError("X_PARAMS and Y_METRICS must be the same length (one x-param per metric).")

    n_groups = len(Y_METRICS)
    fig, axes = plt.subplots(1, 2 * n_groups, figsize=FIGSIZE, squeeze=False)
    adjust_kwargs = {"top": 0.775, "bottom": 0.3, "left": 0.05, "right": 0.99}
    if SUBPLOT_WSPACE is not None:
        adjust_kwargs["wspace"] = SUBPLOT_WSPACE
    plt.subplots_adjust(**adjust_kwargs)
    axes_row = axes[0]

    param_titles = {
        "dram_bw": "DRAM Bandwidth (TBps)",
        "num_cores": "Core Count",
        "sa": "Systolic Array Size",
        "vu": "Vector Unit Width",
        "sram_kb": "Per-core SRAM Size (MB)",
        "core_group": "Core Group Size",
        "noc_topo": "NoC Topology",
        "noc_bw": "NoC Link Bandwidth (B/cycle)",
    }

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []
    seen_models: Set[str] = set()
    stage_axes: Dict[Tuple[str, bool], List[plt.Axes]] = {}

    for group_idx, (metric, x_param) in enumerate(zip(Y_METRICS, X_PARAMS)):
        log_x, log_y = param_scales(x_param)

        for op_offset, op_type in enumerate(OP_TYPES):
            ax_idx = 2 * group_idx + op_offset
            ax = axes_row[ax_idx]

            if SHARE_STAGE_Y_LIMITS:
                if len(axes_row) > ax_idx + 2:
                    ax.sharey(axes_row[ax_idx + 2])

            stage_axes.setdefault((op_type, log_y), []).append(ax)

            if op_type == "decode" and MODELS_DECODE is not None:
                stage_models = MODELS_DECODE
            elif op_type == "prefill" and MODELS_PREFILL is not None:
                stage_models = MODELS_PREFILL
            else:
                stage_models = MODELS

            for model in stage_models:
                if model not in MODEL_STYLE_ORDER:
                    MODEL_STYLE_ORDER.append(model)
                style_idx = MODEL_STYLE_ORDER.index(model)
                x_vals, y_vals = get_metric_series(model, op_type, x_param, metric, log_y)
                if not x_vals:
                    continue
                linestyle = lines1[style_idx % len(lines1)]
                color = colors1[style_idx % len(colors1)]
                marker = markers1[style_idx % len(markers1)]
                ax.plot(
                    x_vals,
                    y_vals,
                    linestyle,
                    marker=marker,
                    color=color,
                    linewidth=2,
                    markersize=10,
                    label=modelnames.get(model, model),
                )

                if model not in seen_models:
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            linestyle=linestyle,
                            marker=marker,
                            color=color,
                            linewidth=2,
                            markersize=10,
                        )
                    )
                    legend_labels.append(modelnames.get(model, model))
                    seen_models.add(model)

            if log_x:
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(NullLocator())
            if log_y:
                ax.set_yscale("log")
                configure_log_ticks(ax)

            sweep_vals = get_sweep_values(x_param)
            if sweep_vals:
                if x_param == "dram_bw":
                    tick_labels = [str(v // 1000) for v in sweep_vals]
                elif x_param == "sram_kb":
                    tick_labels = [f"{v/1024:.1f}" if idx else "0.5" for idx, v in enumerate(sweep_vals)]
                else:
                    tick_labels = [str(v) for v in sweep_vals]
                ax.set_xticks(sweep_vals, tick_labels, rotation=20)

            metric_label = METRIC_INFO[metric]["label"]  # type: ignore[index]
            if group_idx == 0 and op_offset == 0:
                ax.set_ylabel(metric_label, fontsize=26, labelpad=YLABEL_PAD)
            else:
                ax.set_ylabel("")

            ax.set_title(f"{op_type.capitalize()}", fontsize=24)
            ax.grid(
                which="major",
                axis="both",
                linestyle="-",
                linewidth=0.5,
                color="grey",
                zorder=1,
            )
            # Keep label font sizes modest but make the tick marks themselves
            # long and thick enough to remain visible in single-column layouts.
            ax.tick_params(
                axis="x",
                labelsize=22,
                length=8,
                width=1.5,
                which="both",
            )
            ax.tick_params(
                axis="y",
                labelsize=22,
                length=8,
                width=1.5,
                which="both",
            )

            if log_y and not SHARE_STAGE_Y_LIMITS:
                axis_values: List[float] = []
                for line in ax.get_lines():
                    data = np.asarray(line.get_ydata())
                    if data.size == 0:
                        continue
                    axis_values.extend(val for val in data if val > 0 and np.isfinite(val))

                if axis_values:
                    try:
                        y_min, y_max = decade_bounds(axis_values)
                        ax.set_ylim(y_min, y_max)
                    except ValueError:
                        pass

        group_label = GROUP_LABELS[group_idx] if GROUP_LABELS and group_idx < len(GROUP_LABELS) else param_titles.get(x_param, x_param.replace("_", " ").title())
        letter = GROUP_LETTERS[group_idx] if group_idx < len(GROUP_LETTERS) else ""
        mid_ax = axes_row[2 * group_idx]
        mid_ax_pos = mid_ax.get_position()
        right_ax_pos = axes_row[2 * group_idx + 1].get_position()
        x_center = (mid_ax_pos.x0 + right_ax_pos.x1) / 2
        fig.text(
            x_center,
            0.05,
            f"{letter} {group_label}",
            ha="center",
            va="center",
            fontsize=26,
        )

    if SHARE_STAGE_Y_LIMITS:
        for (stage, log_y), axes_list in stage_axes.items():
            y_values: List[float] = []
            for ax in axes_list:
                for line in ax.get_lines():
                    data = np.asarray(line.get_ydata())
                    if data.size == 0:
                        continue
                    finite = data[np.isfinite(data)]
                    if finite.size == 0:
                        continue
                    if log_y:
                        finite = finite[finite > 0]
                        if finite.size == 0:
                            continue
                    y_values.extend(finite.tolist())

            if not y_values:
                continue

            if log_y:
                try:
                    y_min, y_max = decade_bounds(y_values)
                except ValueError:
                    continue
            else:
                y_min = min(y_values)
                y_max = max(y_values)

                if y_min == y_max:
                    expansion = 0.05 * y_max if y_max != 0 else 1.0
                    y_min -= expansion
                    y_max += expansion
                else:
                    span = y_max - y_min
                    padding = 0.05 * span
                    y_min -= padding
                    y_max += padding
            if stage in STAGE_Y_LIMIT_OVERRIDES:
                y_min, y_max = STAGE_Y_LIMIT_OVERRIDES[stage]
            for ax in axes_list:
                ax.set_ylim(y_min, y_max)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.1),
        frameon=False,
        ncol=min(len(legend_labels), 5),
        fontsize=26,
        columnspacing=1.0,
        handlelength=1.8,
    )

    if SAVE_PDF:
        os.makedirs(os.path.join(repo_root(), OUTPUT_DIR), exist_ok=True)
        slug = f"eval_{'_'.join(Y_METRICS)}_vs_{'_'.join(X_PARAMS)}"
        outfile = os.path.join(repo_root(), OUTPUT_DIR, f"{slug}.pdf")
        plt.savefig(outfile, bbox_inches="tight")
        print(f"Saved figure to {outfile}")
    else:
        plt.show()


def main() -> None:
    draw_figures()


if __name__ == "__main__":
    main()
