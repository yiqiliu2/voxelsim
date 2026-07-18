#!/usr/bin/env python3

"""Plot component-level energy breakdowns across architecture sweeps.

This script generates stacked bar chart energy breakdowns for each (hardware + model workload)
configuration. It can produce either normalized (0-1 scale) or absolute (J) energy breakdowns,
allowing comparison of energy consumption when varying specific hardware parameters one at a time
(specified by X_PARAMS in the configuration section).

The script traverses logs in the TRACE_OUT directory structure (similar to draw_energy_power.py),
parsing per-component energy numbers (DRAM, SRAM, NoC, SA, VU, TSV) from each log while tracking
static and dynamic energy separately. It generates one figure per model, with subplots for each
stage/X-param combination.

Style: Uses hatches and darker colors for dynamic power, solid and lighter colors for static power.
All static power values are plotted first (bottom layer), then dynamic on top (top layer).

Usage:
    # Generate absolute energy breakdowns (default):
    python3 draw_component_energy_breakdown.py

    # Generate normalized energy breakdowns:
    python3 draw_component_energy_breakdown.py --normalize

Configuration:
    Edit the configuration section below to customize:
    - MODELS: List of models to plot
    - X_PARAMS: Hardware parameters to sweep on the x-axis
    - NORMALIZE: Default normalization mode
    - COMPONENT_ORDER: Order of component stacking
    - COMPONENT_COLORS: Color scheme for each component
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Any, Dict, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullLocator

from fig_common import modelnames  # pylint: disable=import-error
from run_all_tests import (  # pylint: disable=import-error
    DF_DRAM_DIV,
    DF_DRAM_MUL,
    DF_FACTOR,
    TRACE_OUT,
    default_params,
    sweep_lists,
)


# ---------------------------------------------------------------------------
# Configuration macros
# ---------------------------------------------------------------------------

# Models to include in the figure (order controls subplot placement).
MODELS: List[str] = ["llama3-70"]

# Operation types to render. Always render decode then prefill for paired subplots.
OP_TYPES: Tuple[str, ...] = ("decode", "prefill")

# Implementation subdirectory to read results from (e.g., "best").
IMPLEMENTATION_KEY: str = "best"

# Configuration parameters to sweep on the x-axis (must exist in default_params).
# Provide one entry per parameter sweep.
X_PARAMS: List[str] = ["dram_bw", "num_cores"]

# Optional explicit list of sweep values per parameter.
CUSTOM_X_VALUES: Dict[str, List[int]] | None = None

# Whether to normalize energy values (0-1 scale) or use absolute values (J).
NORMALIZE: bool = False

# Batch size and core group are fixed for the experiments in this plot.
BATCH_SIZE: int = 32
CORE_GROUP: int = default_params["core_group"]

# Figure options.
FIGSIZE_PER_SUBPLOT = (3.7, 4.4)  # Width, height per subplot
OUTPUT_DIR = "figures"
SAVE_PDF: bool = True
SUBPLOT_WSPACE: float = 0.25
SUBPLOT_HSPACE: float = 0.3
YLABEL_PAD: float = 15.0

# Component order for stacking (bottom to top within each energy type).
COMPONENT_ORDER: Tuple[str, ...] = ("dram", "tsv", "noc", "sram", "vu", "sa")

# Component colors: lighter for static, darker + hatch for dynamic.
COMPONENT_COLORS: Dict[str, Tuple[str, str]] = {
    "dram": ("#ffaaaa", "#cc0000"),  # Light red to dark red
    "tsv": ("#d4b896", "#8b6914"),   # Light brown to dark brown
    "noc": ("#d4a5d4", "#8b008b"),   # Light purple to dark purple
    "sram": ("#a5d4a5", "#006400"),  # Light green to dark green
    "vu": ("#ffd4a5", "#ff8c00"),    # Light orange to dark orange
    "sa": ("#a5c8ff", "#0000cd"),    # Light blue to dark blue
}
plt.rcParams.update({'hatch.linewidth': 2})
DYNAMIC_HATCH = "\\\\"
EDGE_COLOR = "black"
BAR_LINEWIDTH = 0.8
# BAR_LINEWIDTH = 1.2

# Font sizes for visibility in paper columns.
TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 24
TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 22

GROUP_LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]


# ---------------------------------------------------------------------------
# Regular expressions for parsing component energy
# ---------------------------------------------------------------------------

COMPONENT_RE = re.compile(
    r"(sa|vu|sram|dram|tsv|noc)\s*[=:]\s*([-+eE0-9.]+)",
    re.IGNORECASE
)
EXE_TIME_RE = re.compile(r"EXE time.*?:\s*([-+eE0-9\.]+)", re.IGNORECASE)


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


def parse_component_energies(log_path: str) -> Dict[str, Dict[str, float]]:
    """Parse static and dynamic energy for each component from a log file.

    Returns:
        Dict with keys "static" and "dynamic", each containing component->energy_J mappings.
    """
    static_energy: Dict[str, float] = {comp: 0.0 for comp in COMPONENT_ORDER}
    dynamic_energy: Dict[str, float] = {comp: 0.0 for comp in COMPONENT_ORDER}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as infile:
        for line in infile:
            lowered = line.lower().strip()
            if lowered.startswith("static energy:"):
                for match in COMPONENT_RE.finditer(line):
                    comp = match.group(1).lower()
                    if comp in static_energy:
                        static_energy[comp] += float(match.group(2))
            elif lowered.startswith("dynamic energy:"):
                for match in COMPONENT_RE.finditer(line):
                    comp = match.group(1).lower()
                    if comp in dynamic_energy:
                        dynamic_energy[comp] += float(match.group(2))

    # The logs record energies in millijoules (mJ). Convert to Joules (J) for plotting
    # and downstream consistency by dividing by 1000.0.
    for comp in static_energy:
        static_energy[comp] = static_energy[comp] / 1000.0
        dynamic_energy[comp] = dynamic_energy[comp] / 1000.0

    return {"static": static_energy, "dynamic": dynamic_energy}


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


def get_energy_breakdown_series(
    model: str, op_type: str, x_param: str
) -> Tuple[List[float], Dict[str, List[float]], Dict[str, List[float]]]:
    """Get energy breakdown data for a given model, operation type, and sweep parameter.

    Returns:
        Tuple of (x_values, static_energies_by_component, dynamic_energies_by_component)
    """
    base_cfg = dict(default_params)
    if x_param not in base_cfg:
        raise KeyError(f"Parameter '{x_param}' missing from default configuration.")
    base_cfg.update({"bs": BATCH_SIZE, "core_group": CORE_GROUP, "row": default_params["row"]})

    sweep_vals = get_sweep_values(x_param)
    x_numeric: List[float] = []
    static_series: Dict[str, List[float]] = {comp: [] for comp in COMPONENT_ORDER}
    dynamic_series: Dict[str, List[float]] = {comp: [] for comp in COMPONENT_ORDER}

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
            energies = parse_component_energies(log_path)
        except Exception as exc:
            print(f"[WARN] Error parsing {log_path}: {exc}")
            continue

        x_numeric.append(float(value))
        for comp in COMPONENT_ORDER:
            static_series[comp].append(energies["static"][comp])
            dynamic_series[comp].append(energies["dynamic"][comp])

    return x_numeric, static_series, dynamic_series


def plot_energy_breakdown(
    ax: plt.Axes,
    x_values: List[float],
    static_series: Dict[str, List[float]],
    dynamic_series: Dict[str, List[float]],
    x_param: str,
    normalize: bool,
    first_col: bool,
) -> List[plt.Rectangle]:
    """Plot stacked bar chart of energy breakdown on given axes.

    Returns:
        List of legend patches.
    """
    if not x_values:
        return []

    x_pos = np.arange(len(x_values))
    width = 0.6

    # Compute totals for normalization if needed.
    totals = np.zeros(len(x_values))
    for comp in COMPONENT_ORDER:
        totals += np.array(static_series[comp]) + np.array(dynamic_series[comp])

    # Plot static energy first (bottom layer).
    static_bottoms = np.zeros(len(x_values))
    static_patches = []
    for comp in COMPONENT_ORDER:
        values = np.array(static_series[comp])
        if normalize and np.any(totals > 0):
            # Convert to percentage contribution of total energy.
            values = np.divide(values, totals, where=totals > 0, out=np.zeros_like(values)) * 100.0

        if np.any(values > 0):
            light_color, _ = COMPONENT_COLORS[comp]
            patch = ax.bar(
                x_pos,
                values,
                width,
                bottom=static_bottoms,
                color=light_color,
                edgecolor=EDGE_COLOR,
                linewidth=BAR_LINEWIDTH,
                label=f"{comp.upper()} Static",
            )
            static_patches.append(patch[0])
            static_bottoms += values

    # Plot dynamic energy on top.
    dynamic_bottoms = static_bottoms.copy()
    dynamic_patches = []
    for comp in COMPONENT_ORDER:
        values = np.array(dynamic_series[comp])
        if normalize and np.any(totals > 0):
            # Convert to percentage contribution of total energy.
            values = np.divide(values, totals, where=totals > 0, out=np.zeros_like(values)) * 100.0

        if np.any(values > 0):
            _, dark_color = COMPONENT_COLORS[comp]
            patch = ax.bar(
                x_pos,
                values,
                width,
                bottom=dynamic_bottoms,
                color=dark_color,
                edgecolor=EDGE_COLOR,
                linewidth=BAR_LINEWIDTH,
                hatch=DYNAMIC_HATCH,
                label=f"{comp.upper()} Dyn.",
            )
            # Make the hatch strokes white while keeping the bar edge/border black.
            for p in patch:
                p.set_hatch(DYNAMIC_HATCH)
                p.set_edgecolor(EDGE_COLOR)
                p.set_facecolor(dark_color)
                p.set_linewidth(BAR_LINEWIDTH)
                p._hatch_color = mpl.colors.to_rgba('white')
            dynamic_patches.append(patch[0])
            dynamic_bottoms += values

    # Configure x-axis.
    if x_param == "dram_bw":
        tick_labels = [str(int(v) // 1000) for v in x_values]
    elif x_param == "sram_kb":
        tick_labels = [f"{v/1024:.1f}" if idx else "0.5" for idx, v in enumerate(x_values)]
    else:
        tick_labels = [str(int(v)) for v in x_values]

    ax.set_xticks(x_pos, tick_labels, rotation=20, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis='y', labelsize=TICK_FONTSIZE)

    # Configure y-axis.
    if normalize:
        # Percent scale from 0 to 100.
        ax.set_ylim(0.0, 105.0)
        if first_col:
            ax.set_ylabel("Norm. Energy (%)", fontsize=LABEL_FONTSIZE, labelpad=YLABEL_PAD)
        # Use simple 0–25–50–75–100 ticks.
        norm_ticks = np.linspace(0.0, 100.0, 5)
        ax.set_yticks(norm_ticks)
    else:
        y_max = np.max(dynamic_bottoms) if len(dynamic_bottoms) > 0 else 1.0
        y_top = float(y_max) * 1.1
        ax.set_ylim(0.0, y_top)
        if first_col:
            ax.set_ylabel("Energy (J)", fontsize=LABEL_FONTSIZE, labelpad=YLABEL_PAD)

        # Choose a "nice" step to produce slightly more fine-grained, human-friendly ticks.
        # Strategy: target ~5 intervals, then round the step to 1,2,5 * 10^n.
        target_intervals = 5
        raw_step = y_top / max(1, target_intervals)
        if raw_step <= 0:
            step = y_top
        else:
            exp = math.floor(math.log10(raw_step))
            base = 10 ** exp
            for m in (1, 2, 5, 10):
                if base * m >= raw_step:
                    step = base * m
                    break

        # Generate ticks from 0 up to y_top, inclusive, using the chosen nice step.
        ticks = np.arange(0.0, y_top + 1e-12, step)
        # Ensure at least two ticks (0 and y_top) exist.
        if len(ticks) < 2:
            ticks = np.array([0.0, y_top])
        ax.set_yticks(ticks)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5, zorder=0)

    return static_patches + dynamic_patches


def get_row_major_order(items: List[Any], ncol: int) -> List[Any]:
    """
    Reorders a list of items to produce a row-major layout in
    a Matplotlib legend that fills column-by-column.

    Args:
        items: The list of legend handles or labels.
        ncol: The number of columns in the legend.

    Returns:
        A new list with items reordered.
    """
    if ncol <= 0:
        return items

    n_items = len(items)
    # Calculate the number of rows required
    nrow = math.ceil(n_items / ncol)

    # Create the reordered list
    ordered_items = []

    # Loop over columns (outer loop) and rows (inner loop)
    # This reads the items as if they were already in a
    # row-major grid, but builds the new list in
    # column-major order.
    for c_idx in range(ncol):
        for r_idx in range(nrow):
            # Calculate the index from the original (row-major) list
            item_idx = (r_idx * ncol) + c_idx

            # Add the item if it exists (for incomplete final rows)
            if item_idx < n_items:
                ordered_items.append(items[item_idx])

    return ordered_items



def draw_figures_per_model(normalize: bool = NORMALIZE) -> None:
    """Generate one figure per model, with subplots for each stage/x-param combination."""

    param_titles = {
        "dram_bw": "DRAM Bandwidth (TBps)",
        "num_cores": "Core Count",
        "sa": "Systolic Array Size",
        "vu": "Vector Unit Width",
        "sram_kb": "Per-core SRAM (MB)",
        "core_group": "Core Group Size",
        "noc_topo": "NoC Topology",
        "noc_bw": "NoC Link Bandwidth",
    }

    for model in MODELS:
        n_params = len(X_PARAMS)
        n_stages = len(OP_TYPES)

        # Create subplot grid: single row with decode/prefill pairs for each x_param.
        # Layout: [decode_param1, prefill_param1, decode_param2, prefill_param2, ...]
        n_subplots = n_params * n_stages
        fig_width = FIGSIZE_PER_SUBPLOT[0] * n_subplots
        fig_height = FIGSIZE_PER_SUBPLOT[1]
        fig, axes = plt.subplots(
            1, n_subplots,
            figsize=(fig_width, fig_height),
            squeeze=False
        )
        axes_row = axes[0]

        plt.subplots_adjust(
            left=0.03, right=0.99, top=0.66, bottom=0.22,
            wspace=SUBPLOT_WSPACE
        )

        for param_idx, x_param in enumerate(X_PARAMS):
            for stage_idx, op_type in enumerate(OP_TYPES):
                # Calculate subplot index: 2*param_idx + stage_idx
                ax_idx = 2 * param_idx + stage_idx
                ax = axes_row[ax_idx]

                x_vals, static_series, dynamic_series = get_energy_breakdown_series(
                    model, op_type, x_param
                )

                plot_energy_breakdown(
                    ax,
                    x_vals,
                    static_series,
                    dynamic_series,
                    x_param,
                    normalize,
                    first_col=(ax_idx == 0)
                )

                # Set subplot title.
                ax.set_title(op_type.capitalize(), fontsize=TITLE_FONTSIZE, pad=10)

                # Set x-axis label centered below each decode/prefill pair.
                # Only set on prefill (second) subplot of each pair.
                if stage_idx == 1:
                    xlabel = param_titles.get(x_param, x_param.replace("_", " ").title())
                    # Add letter prefix (a), (b), etc.
                    letter = GROUP_LETTERS[param_idx] if param_idx < len(GROUP_LETTERS) else f"({param_idx})"
                    xlabel_with_letter = f"{letter} {xlabel}"
                    # Position label between the two subplots of this parameter group.
                    mid_ax = axes_row[2 * param_idx]
                    right_ax = axes_row[2 * param_idx + 1]
                    mid_ax_pos = mid_ax.get_position()
                    right_ax_pos = right_ax.get_position()
                    x_center = (mid_ax_pos.x0 + right_ax_pos.x1) / 2
                    fig.text(
                        x_center,
                        0.05,
                        xlabel_with_letter,
                        ha="center",
                        va="center",
                        fontsize=LABEL_FONTSIZE,
                    )

        # Create manual legend with all components in order.
        from matplotlib.patches import Rectangle
        legend_handles = []
        legend_labels = []

        # Add dynamic components.
        for comp in COMPONENT_ORDER:
            _, dark_color = COMPONENT_COLORS[comp]
            lgd_patch =  Rectangle((0, 0), 1, 1, facecolor=dark_color, edgecolor=EDGE_COLOR,
                         linewidth=BAR_LINEWIDTH, hatch=DYNAMIC_HATCH)
            lgd_patch._hatch_color = mpl.colors.to_rgba('white')
            legend_handles.append(
                lgd_patch
            )
            legend_labels.append(f"{comp.upper()} Dyn.")

        # Add static components first.
        for comp in COMPONENT_ORDER:
            light_color, _ = COMPONENT_COLORS[comp]
            legend_handles.append(
                Rectangle((0, 0), 1, 1, facecolor=light_color, edgecolor=EDGE_COLOR, linewidth=BAR_LINEWIDTH)
            )
            legend_labels.append(f"{comp.upper()} Static")
        # print(legend_labels)
        # legend_labels = ["a", "b", "c", "d", "e", "f", "a1", "b1", "c1", "d1", "e1", "f1"]
        new_lgd_handles = get_row_major_order(legend_handles, ncol=6)
        new_lgd_labels = get_row_major_order(legend_labels, ncol=6)
        # Add legend at the top.
        fig.legend(
            new_lgd_handles,
            new_lgd_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.00),
            ncol=6,
            handletextpad=0.5,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            handlelength=1.5,
            handleheight=1.0,
            columnspacing=0.5,
        )

        # Save figure.
        if SAVE_PDF:
            outdir = os.path.join(repo_root(), OUTPUT_DIR)
            os.makedirs(outdir, exist_ok=True)
            mode_str = "normalized" if normalize else "absolute"
            outfile = os.path.join(outdir, f"{model}_energy_breakdown_{mode_str}.pdf")
            plt.savefig(outfile, bbox_inches="tight", dpi=300)
            print(f"Saved figure to {outfile}")
        else:
            plt.show()

        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot component-level energy breakdowns per model."
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize energy values to 0-1 scale (default: absolute J values).",
    )
    args = parser.parse_args()

    draw_figures_per_model(normalize=args.normalize)


if __name__ == "__main__":
    main()
