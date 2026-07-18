#!/usr/bin/env python3
"""Plot peak temperature and component power-density bars."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

try:
    from fig_common import colors1, dark_colors1, markers1, modelnames  # type: ignore
except ImportError:  # pragma: no cover
    colors1 = ["#126b91", "#ec6632", "#93bc38", "#252422", "#c45ab3"] * 2
    dark_colors1 = ["#072836", "#95340e", "#495e1c", "#6a6762", "#873179"] * 2
    markers1 = ["v", "o", "^", "*", ""]
    modelnames = {}

mpl.rcParams.update({"font.family": "serif"})
mpl.rcParams["pdf.fonttype"] = 42


class LegendHatchedPatch(Rectangle):
    """Legend patch whose hatch and border are drawn as separate layers."""


class HandlerLegendHatchedPatch(HandlerBase):
    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        xy = (-xdescent, -ydescent)
        base = Rectangle(
            xy,
            width,
            height,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            transform=trans,
        )
        hatch = Rectangle(
            xy,
            width,
            height,
            facecolor="none",
            edgecolor=BAR_HATCH_COLOR,
            hatch=BAR_HATCH,
            linewidth=BAR_HATCH_EDGE_LINEWIDTH,
            transform=trans,
        )
        return [base, hatch]


MODEL_ORDER = ("llama2-13", "llama3-70", "opt-30", "gemma2", "dit-xl")
STAGE_ORDER = ("prefill", "decode")
STAGE_TITLES = {"prefill": "Prefill", "decode": "Decode"}
PLOT_STAGE_OVERRIDES = {"dit-xl": "prefill"}
COMPONENTS = {"sa", "sram", "tsv", "router", "vu"}

# =============================================================================
# Style macros: adjust plot appearance here.
# =============================================================================
LOGIC_TEMP_COLOR_INDEX = 1
DRAM_TEMP_COLOR_INDEX = 2
POWER_DENSITY_COLOR_INDEX = 0
BAR_EDGE_COLOR = "black"
BAR_EDGE_LINEWIDTH = 0.6
BAR_HATCH = "\\\\"
BAR_HATCH_COLOR = "lightgray"
BAR_HATCH_EDGE_LINEWIDTH = 0.0
GRID_COLOR = "lightgrey"
GRID_LINEWIDTH = 0.5
LIMIT_LINEWIDTH = 1.2
LIMIT_LINE_ALPHA = 0.95
TEMP_LIMIT_C = 85.0
TEMP_LIMIT_LABEL = "85 °C Threshold"
TEMP_LIMIT_LINE_COLOR = "#333333"
LOGIC_LEGEND_LABEL = "Logic"
DRAM_LEGEND_LABEL = "DRAM"
POWER_DENSITY_LIMIT_W_PER_MM2 = 0.7
POWER_DENSITY_LIMIT_LABEL = "0.7 W/mm2"
NORMALIZED_LIMIT_PCT = 100.0
NORMALIZED_LIMIT_COLOR = "#555555"
NORMALIZED_LIMIT_LINEWIDTH = 1.1
NORMALIZED_LIMIT_ALPHA = 0.9

OUTPUT_DPI = 260
BBOX_INCHES = "tight"
PLOT_SCRIPT_COPY_NAME = "plot_thermal_temp_density.py"

SUBPLOT_FIG_HEIGHT_IN = 2.38
SUBPLOT_MIN_WIDTH_IN = 6.0
SUBPLOT_WIDTH_PER_GROUP_IN = 0.64
SUBPLOT_WIDTH_PAD_IN = 0.55
SUBPLOT_TEXT_FONTSIZE = 9.5
SUBPLOT_BAR_WIDTH = 0.34
SUBPLOT_DENSITY_BAR_WIDTH = 0.50
SUBPLOT_XTICK_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_XTICK_ROTATION_DEG = 18
SUBPLOT_XTICK_PAD = 2
SUBPLOT_X_MARGIN_LEFT = 0.55
SUBPLOT_X_MARGIN_RIGHT = 0.45
SUBPLOT_TITLE_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_TITLE_PAD = 2
SUBPLOT_YTICK_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_YLABEL_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_TEMP_YLABEL = "Peak\nTemp.\n(°C)"
SUBPLOT_TEMP_YLABEL_ROTATION = 0
SUBPLOT_TEMP_YLABEL_PAD = 22
SUBPLOT_SUPTITLE_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_SUPTITLE_Y = 0.99
SUBPLOT_LEGEND_FONTSIZE = SUBPLOT_TEXT_FONTSIZE
SUBPLOT_LEGEND_HANDLE_LENGTH = 1.2
SUBPLOT_LEGEND_COLUMN_SPACING = 0.9
SUBPLOT_LEGEND_Y = 1.01
SUBPLOT_TEMP_ADJUST = dict(left=0.115, right=0.995, bottom=0.36, top=0.68, wspace=0.08)
SUBPLOT_DENSITY_ADJUST = dict(left=0.135, right=0.995, bottom=0.29, top=0.78, wspace=0.08)

HORIZONTAL_FIG_WIDTH_IN = 4.15
HORIZONTAL_MIN_HEIGHT_IN = 2.35
HORIZONTAL_HEIGHT_PER_GROUP_IN = 0.24
HORIZONTAL_HEIGHT_PAD_IN = 0.75
HORIZONTAL_BAR_HEIGHT = 0.34
HORIZONTAL_DENSITY_BAR_HEIGHT = 0.52
HORIZONTAL_LABEL_FONTSIZE = 10.5
HORIZONTAL_TICK_FONTSIZE = 9
HORIZONTAL_YTICK_FONTSIZE = 8
HORIZONTAL_LEGEND_FONTSIZE = 7.9
HORIZONTAL_LEGEND_Y = 1.03
HORIZONTAL_ADJUST = dict(left=0.22, right=0.985, bottom=0.12, top=0.84, hspace=0.20)

COMBINED_FIGSIZE = (7.9, 3.2)
COMBINED_XTICK_FONTSIZE = 11
COMBINED_XTICK_ROTATION_DEG = 12
COMBINED_XTICK_PAD = 30
COMBINED_X_MARGIN = 0.2
COMBINED_YLABEL_FONTSIZE = 12
COMBINED_YTICK_FONTSIZE = 10
COMBINED_TITLE_FONTSIZE = 13
COMBINED_TITLE_PAD = 4
COMBINED_LEGEND_Y = 1.03
COMBINED_ABSOLUTE_LEGEND_FONTSIZE = 8.4
COMBINED_RISE_LEGEND_FONTSIZE = 8.0
COMBINED_NORMALIZED_LEGEND_FONTSIZE = 9.0
COMBINED_ABSOLUTE_ADJUST = dict(left=0.105, right=0.82, bottom=0.39, top=0.64)
COMBINED_NORMALIZED_ADJUST = dict(left=0.105, right=0.98, bottom=0.39, top=0.68)

TEMP_AXIS_PAD_FACTOR = 1.12
TEMP_COMBINED_AXIS_PAD_FACTOR = 1.14
TEMP_MIN_YMAX_C = 90.0
TEMP_RISE_AXIS_PAD_FACTOR = 1.24
TEMP_RISE_LIMIT_PAD_FACTOR = 1.12
DENSITY_AXIS_PAD_FACTOR = 1.20
DENSITY_COMBINED_AXIS_PAD_FACTOR = 1.25
DENSITY_LIMIT_PAD_FACTOR = 1.18
NORMALIZED_AXIS_PAD_FACTOR = 1.22
NORMALIZED_MIN_YMAX_PCT = 112.0

TEMP_TICK_STEP_C = 20.0
TEMP_TICK_STEP_LARGE_C = 25.0
TEMP_TICK_LARGE_THRESHOLD_C = 120.0
DENSITY_TICK_STEP_SMALL = 0.2
DENSITY_TICK_STEP_MEDIUM = 0.25
DENSITY_TICK_STEP_LARGE = 0.5
DENSITY_TICK_SMALL_THRESHOLD = 1.0
DENSITY_TICK_MEDIUM_THRESHOLD = 2.0
PERCENT_TICK_STEP = 20.0
PERCENT_TICK_STEP_LARGE = 25.0
PERCENT_TICK_LARGE_THRESHOLD = 140.0

MODEL_X_SPACING = 0.35
STAGE_X_OFFSET = 0.09
BAR_WIDTH = 0.05


def parse_args() -> argparse.Namespace:
    repo_src = Path(__file__).resolve().parents[1]
    default_root = repo_src / "results" / "thermal_matrix_prefill_bs1_seq2048_decode_bs32_iter100_tsim_area"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal-root", type=Path, default=default_root)
    parser.add_argument("--density-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ambient-c", type=float, default=35.0)
    parser.add_argument("--density-window-us", type=int, default=1000)
    parser.add_argument("--density-threshold", type=float, default=POWER_DENSITY_LIMIT_W_PER_MM2)
    parser.add_argument(
        "--normalize-density-peak-to-threshold",
        action="store_true",
        help="Scale all peak power densities so the largest plotted value equals --density-threshold.",
    )
    parser.add_argument("--title", default="")
    return parser.parse_args()


def window_tag(window_us: int) -> str:
    if window_us == 0:
        return "raw"
    if window_us % 1000 == 0:
        return f"{window_us // 1000}ms"
    return f"{window_us}us"


def model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def plot_mode_for(model: str, mode: str) -> str:
    return PLOT_STAGE_OVERRIDES.get(model, mode)


def record_plot_mode(record: Dict[str, object]) -> str:
    return str(record.get("plot_mode") or plot_mode_for(str(record["model"]), str(record["mode"])))


def summary_paths(root: Path) -> Iterable[Path]:
    for mode in STAGE_ORDER:
        path = root / mode / "thermal_cooler_matrix_summary.csv"
        if path.exists():
            yield path


def peak_dram_temp_c(timeseries_csv: Path) -> float:
    with timeseries_csv.open(encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        dram_cols = [
            name
            for name in (reader.fieldnames or [])
            if name.startswith("dram") and name.endswith("_peak_c")
        ]
        if not dram_cols:
            raise ValueError(f"{timeseries_csv} has no dram*_peak_c columns")
        peak = -math.inf
        for row in reader:
            for col in dram_cols:
                value = row.get(col, "")
                if value:
                    peak = max(peak, float(value))
    if not math.isfinite(peak):
        raise ValueError(f"{timeseries_csv} has no DRAM peak samples")
    return peak


def read_temperature_rows(root: Path) -> Dict[Tuple[str, str], Dict[str, object]]:
    rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    for summary_csv in summary_paths(root):
        matrix_dir = summary_csv.parent
        with summary_csv.open(encoding="utf-8", newline="") as infile:
            for row in csv.DictReader(infile):
                model = row["model"]
                mode = row["mode"]
                if model == "dit-xl" and mode == "prefill":
                    continue
                tag = row["tag"]
                timeseries_csv = matrix_dir / f"{tag}_layer_temperature_timeseries.csv"
                rows[(model, mode)] = {
                    "model": model,
                    "mode": mode,
                    "tag": tag,
                    "logic_peak_c": float(row["logic_peak_c"]),
                    "dram_peak_c": peak_dram_temp_c(timeseries_csv),
                    "source_summary": str(summary_csv),
                    "source_timeseries": str(timeseries_csv),
                }
    return rows


def read_density_rows(root: Path, window_us: int) -> Dict[Tuple[str, str], Dict[str, object]]:
    density_csv = root / "aggregate_power_density_smoothed.csv"
    rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    with density_csv.open(encoding="utf-8", newline="") as infile:
        for row in csv.DictReader(infile):
            if int(row["window_us"]) != window_us or row["component"] not in COMPONENTS:
                continue
            key = (row["model"], row["mode"])
            value = float(row["stacked_group_w_per_mm2"])
            if key not in rows or value > float(rows[key]["peak_component_density_w_per_mm2"]):
                rows[key] = {
                    "peak_component_density_w_per_mm2": value,
                    "peak_density_component": row["component"],
                    "source_density_csv": str(density_csv),
                }
    return rows


def ordered_records(
    thermal_root: Path,
    density_root: Path,
    window_us: int,
    ambient_c: float,
) -> List[Dict[str, object]]:
    temp_rows = read_temperature_rows(thermal_root)
    density_rows = read_density_rows(density_root, window_us)
    records: List[Dict[str, object]] = []
    for model in MODEL_ORDER:
        for mode in STAGE_ORDER:
            key = (model, mode)
            if key not in temp_rows:
                continue
            if key not in density_rows:
                raise ValueError(f"No density row for {model} {mode} at {window_us} us")
            record = {**temp_rows[key], **density_rows[key]}
            record["plot_mode"] = plot_mode_for(model, mode)
            record["logic_rise_c"] = max(0.0, float(record["logic_peak_c"]) - ambient_c)
            record["dram_rise_c"] = max(0.0, float(record["dram_peak_c"]) - ambient_c)
            records.append(record)
    return records


def write_source_csv(records: List[Dict[str, object]], out_dir: Path, tag: str) -> Path:
    out_path = out_dir / f"peak_temp_density_source_{tag}.csv"
    fields = [
        "model",
        "mode",
        "plot_mode",
        "tag",
        "logic_peak_c",
        "dram_peak_c",
        "logic_rise_c",
        "dram_rise_c",
        "logic_temp_pct_of_50c_rise",
        "dram_temp_pct_of_50c_rise",
        "raw_peak_component_density_w_per_mm2",
        "density_normalization_scale",
        "peak_component_density_w_per_mm2",
        "density_pct_of_0p7_w_per_mm2",
        "peak_density_component",
        "source_summary",
        "source_timeseries",
        "source_density_csv",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    return out_path


def expose_plot_script(out_dir: Path) -> Path:
    out_path = out_dir / PLOT_SCRIPT_COPY_NAME
    shutil.copyfile(Path(__file__).resolve(), out_path)
    return out_path


def x_positions(records: List[Dict[str, object]]) -> Tuple[Dict[Tuple[str, str], float], List[float], List[str]]:
    models = sorted({str(record["model"]) for record in records}, key=model_sort_key)
    by_model = {model: {record_plot_mode(record) for record in records if record["model"] == model} for model in models}
    positions: Dict[Tuple[str, str], float] = {}
    centers: List[float] = []
    labels: List[str] = []
    for idx, model in enumerate(models):
        center = idx * MODEL_X_SPACING
        centers.append(center)
        labels.append(modelnames.get(model, model))
        if by_model[model] == {"prefill", "decode"}:
            positions[(model, "prefill")] = center - STAGE_X_OFFSET
            positions[(model, "decode")] = center + STAGE_X_OFFSET
        elif "decode" in by_model[model]:
            positions[(model, "decode")] = center
        elif "prefill" in by_model[model]:
            positions[(model, "prefill")] = center
        for record in records:
            if str(record["model"]) == model:
                positions[(model, str(record["mode"]))] = positions[(model, record_plot_mode(record))]
    return positions, centers, labels


def enrich_normalized(records: List[Dict[str, object]], threshold: float) -> None:
    for record in records:
        record.setdefault(
            "raw_peak_component_density_w_per_mm2",
            float(record["peak_component_density_w_per_mm2"]),
        )
        record.setdefault("density_normalization_scale", 1.0)
        record["logic_temp_pct_of_50c_rise"] = 100.0 * float(record["logic_rise_c"]) / 50.0
        record["dram_temp_pct_of_50c_rise"] = 100.0 * float(record["dram_rise_c"]) / 50.0
        record["density_pct_of_0p7_w_per_mm2"] = (
            100.0 * float(record["peak_component_density_w_per_mm2"]) / threshold
        )


def normalize_density_peak_to_threshold(records: List[Dict[str, object]], threshold: float) -> float:
    peak = max(float(record["peak_component_density_w_per_mm2"]) for record in records)
    if peak <= 0.0:
        return 1.0
    scale = threshold / peak
    for record in records:
        raw_density = float(record["peak_component_density_w_per_mm2"])
        record["raw_peak_component_density_w_per_mm2"] = raw_density
        record["density_normalization_scale"] = scale
        record["peak_component_density_w_per_mm2"] = raw_density * scale
    return scale


def apply_common_x_axis(ax, records: List[Dict[str, object]], positions: Dict[Tuple[str, str], float],
                        centers: List[float], labels: List[str]) -> None:
    ax.set_xticks(centers)
    ax.set_xticklabels(
        labels,
        fontsize=COMBINED_XTICK_FONTSIZE,
        rotation=COMBINED_XTICK_ROTATION_DEG,
        ha="right",
    )
    ax.tick_params(axis="x", pad=COMBINED_XTICK_PAD, length=0)
    ax.set_xlim(min(centers) - COMBINED_X_MARGIN, max(centers) + COMBINED_X_MARGIN)


def draw_bar(
    ax,
    x,
    height,
    *,
    width: float,
    color: str,
    zorder: int,
    hatch: bool = False,
):
    ax.bar(
        x,
        height,
        width=width,
        color=color,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        zorder=zorder,
    )
    if hatch:
        ax.bar(
            x,
            height,
            width=width,
            facecolor="none",
            edgecolor=BAR_HATCH_COLOR,
            hatch=BAR_HATCH,
            linewidth=BAR_HATCH_EDGE_LINEWIDTH,
            zorder=zorder + 1,
        )


def draw_barh(
    ax,
    y,
    width_value,
    *,
    height: float,
    color: str,
    zorder: int,
    hatch: bool = False,
):
    ax.barh(
        y,
        width_value,
        height=height,
        color=color,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        zorder=zorder,
    )
    if hatch:
        ax.barh(
            y,
            width_value,
            height=height,
            facecolor="none",
            edgecolor=BAR_HATCH_COLOR,
            hatch=BAR_HATCH,
            linewidth=BAR_HATCH_EDGE_LINEWIDTH,
            zorder=zorder + 1,
        )


def bar_handles(
    logic_color: str,
    dram_color: str,
    density_color: str,
    density_label: str,
    logic_temp_label: str = LOGIC_LEGEND_LABEL,
    dram_temp_label: str = DRAM_LEGEND_LABEL,
):
    from matplotlib.patches import Patch

    return [
        Patch(facecolor=logic_color, edgecolor=BAR_EDGE_COLOR, label=logic_temp_label),
        dram_legend_patch(dram_color, dram_temp_label),
        Patch(facecolor=density_color, edgecolor=BAR_EDGE_COLOR, label=density_label),
    ]


def dram_legend_patch(dram_color: str, label: str = DRAM_LEGEND_LABEL) -> LegendHatchedPatch:
    return LegendHatchedPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        facecolor=dram_color,
        edgecolor=BAR_EDGE_COLOR,
        label=label,
    )


def legend_handler_map():
    return {LegendHatchedPatch: HandlerLegendHatchedPatch()}


def absolute_bar_handles(
    logic_color: str,
    dram_color: str,
    density_color: str,
    density_label: str,
    temp_limit_label: str = TEMP_LIMIT_LABEL,
    logic_temp_label: str = LOGIC_LEGEND_LABEL,
    dram_temp_label: str = DRAM_LEGEND_LABEL,
):
    from matplotlib.lines import Line2D

    return [
        *bar_handles(logic_color, dram_color, density_color, density_label, logic_temp_label, dram_temp_label),
        Line2D([0], [0], color=TEMP_LIMIT_LINE_COLOR, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=temp_limit_label),
        Line2D([0], [0], color=density_color, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=POWER_DENSITY_LIMIT_LABEL),
    ]


def records_for_stage(records: List[Dict[str, object]], mode: str) -> List[Dict[str, object]]:
    return [
        record
        for record in records
        if record_plot_mode(record) == mode
    ]


def model_tick_labels(stage_records: List[Dict[str, object]]) -> List[str]:
    return [modelnames.get(str(record["model"]), str(record["model"])) for record in stage_records]


def apply_subplot_x_axis(ax, count: int, labels: List[str]) -> None:
    ax.set_xticks(np.arange(count))
    ax.set_xticklabels(
        labels,
        fontsize=SUBPLOT_XTICK_FONTSIZE,
        rotation=SUBPLOT_XTICK_ROTATION_DEG,
        ha="right",
    )
    ax.tick_params(axis="x", pad=SUBPLOT_XTICK_PAD, length=0)
    if count:
        ax.set_xlim(-SUBPLOT_X_MARGIN_LEFT, count - SUBPLOT_X_MARGIN_RIGHT)


def stage_counts(records: List[Dict[str, object]]) -> List[int]:
    return [len(records_for_stage(records, mode)) for mode in STAGE_ORDER]


def compact_subplot_figure(records: List[Dict[str, object]], height: float = SUBPLOT_FIG_HEIGHT_IN):
    import matplotlib.pyplot as plt

    counts = stage_counts(records)
    total_groups = max(1, sum(counts))
    # Scale figure width with the total number of model groups. Width ratios
    # then keep one x-unit the same physical size in prefill and decode panels.
    width = max(SUBPLOT_MIN_WIDTH_IN, SUBPLOT_WIDTH_PER_GROUP_IN * total_groups + SUBPLOT_WIDTH_PAD_IN)
    return plt.subplots(
        1,
        2,
        figsize=(width, height),
        sharey=True,
        gridspec_kw={"width_ratios": [max(1, count) for count in counts]},
    )


def compact_horizontal_figure(records: List[Dict[str, object]], width: float = HORIZONTAL_FIG_WIDTH_IN):
    import matplotlib.pyplot as plt

    counts = stage_counts(records)
    total_groups = max(1, sum(counts))
    height = max(HORIZONTAL_MIN_HEIGHT_IN, HORIZONTAL_HEIGHT_PER_GROUP_IN * total_groups + HORIZONTAL_HEIGHT_PAD_IN)
    return plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={"height_ratios": [max(1, count) for count in counts]},
    )


def apply_dense_value_ticks(ax, axis: str, max_value: float, kind: str) -> None:
    from matplotlib.ticker import MultipleLocator

    if kind == "temp":
        step = TEMP_TICK_STEP_C if max_value <= TEMP_TICK_LARGE_THRESHOLD_C else TEMP_TICK_STEP_LARGE_C
    elif kind == "percent":
        step = PERCENT_TICK_STEP if max_value <= PERCENT_TICK_LARGE_THRESHOLD else PERCENT_TICK_STEP_LARGE
    elif max_value <= DENSITY_TICK_SMALL_THRESHOLD:
        step = DENSITY_TICK_STEP_SMALL
    elif max_value <= DENSITY_TICK_MEDIUM_THRESHOLD:
        step = DENSITY_TICK_STEP_MEDIUM
    else:
        step = DENSITY_TICK_STEP_LARGE
    locator = MultipleLocator(step)
    if axis == "y":
        ax.yaxis.set_major_locator(locator)
    else:
        ax.xaxis.set_major_locator(locator)


def plot_peak_temperature_subplots(
    records: List[Dict[str, object]],
    out_dir: Path,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not records:
        raise ValueError("No records to plot")

    logic_color = colors1[LOGIC_TEMP_COLOR_INDEX]
    dram_color = colors1[DRAM_TEMP_COLOR_INDEX]
    max_temp = max(
        max(float(record["logic_peak_c"]), float(record["dram_peak_c"]))
        for record in records
    )
    fig, axes = compact_subplot_figure(records)
    bar_width = SUBPLOT_BAR_WIDTH

    for ax, mode in zip(axes, STAGE_ORDER):
        stage_records = records_for_stage(records, mode)
        x = np.arange(len(stage_records))
        logic = [float(record["logic_peak_c"]) for record in stage_records]
        dram = [float(record["dram_peak_c"]) for record in stage_records]
        draw_bar(
            ax,
            x - bar_width / 2,
            logic,
            width=bar_width,
            color=logic_color,
            zorder=3,
        )
        draw_bar(
            ax,
            x + bar_width / 2,
            dram,
            width=bar_width,
            color=dram_color,
            zorder=3,
            hatch=True,
        )
        ax.axhline(TEMP_LIMIT_C, linestyle="--", color=TEMP_LIMIT_LINE_COLOR, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
        apply_subplot_x_axis(ax, len(stage_records), model_tick_labels(stage_records))
        ax.set_title(STAGE_TITLES[mode], fontsize=SUBPLOT_TITLE_FONTSIZE, pad=SUBPLOT_TITLE_PAD)
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)
        ax.tick_params(axis="y", labelsize=SUBPLOT_YTICK_FONTSIZE)

    axes[0].set_ylabel(
        SUBPLOT_TEMP_YLABEL,
        fontsize=SUBPLOT_YLABEL_FONTSIZE,
        rotation=SUBPLOT_TEMP_YLABEL_ROTATION,
        labelpad=SUBPLOT_TEMP_YLABEL_PAD,
        va="center",
        ha="center",
    )
    y_max = max(TEMP_MIN_YMAX_C, max_temp * TEMP_AXIS_PAD_FACTOR)
    axes[0].set_ylim(0.0, y_max)
    apply_dense_value_ticks(axes[0], "y", y_max, "temp")
    if title:
        fig.suptitle(title, fontsize=SUBPLOT_SUPTITLE_FONTSIZE, y=SUBPLOT_SUPTITLE_Y)

    handles = [
        Patch(facecolor=logic_color, edgecolor=BAR_EDGE_COLOR, label=LOGIC_LEGEND_LABEL),
        dram_legend_patch(dram_color),
        Line2D([0], [0], color=TEMP_LIMIT_LINE_COLOR, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=TEMP_LIMIT_LABEL),
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        fontsize=SUBPLOT_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, SUBPLOT_LEGEND_Y),
        frameon=False,
        handlelength=SUBPLOT_LEGEND_HANDLE_LENGTH,
        columnspacing=SUBPLOT_LEGEND_COLUMN_SPACING,
        handler_map=legend_handler_map(),
    )

    fig.subplots_adjust(**SUBPLOT_TEMP_ADJUST)
    pdf_path = out_dir / f"peak_temperature_subplots_{tag}.pdf"
    png_path = out_dir / f"peak_temperature_subplots_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_peak_density_subplots(
    records: List[Dict[str, object]],
    out_dir: Path,
    threshold: float,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not records:
        raise ValueError("No records to plot")

    density_color = colors1[POWER_DENSITY_COLOR_INDEX]
    max_density = max(float(record["peak_component_density_w_per_mm2"]) for record in records)
    fig, axes = compact_subplot_figure(records)

    for ax, mode in zip(axes, STAGE_ORDER):
        stage_records = records_for_stage(records, mode)
        x = np.arange(len(stage_records))
        density = [float(record["peak_component_density_w_per_mm2"]) for record in stage_records]
        ax.bar(
            x,
            density,
            width=SUBPLOT_DENSITY_BAR_WIDTH,
            color=density_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=3,
        )
        ax.axhline(threshold, linestyle="--", color=density_color, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
        apply_subplot_x_axis(ax, len(stage_records), model_tick_labels(stage_records))
        ax.set_title(STAGE_TITLES[mode], fontsize=SUBPLOT_TITLE_FONTSIZE, pad=SUBPLOT_TITLE_PAD)
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="y", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)
        ax.tick_params(axis="y", labelsize=SUBPLOT_YTICK_FONTSIZE)

    axes[0].set_ylabel("Density (W/mm2)", fontsize=SUBPLOT_YLABEL_FONTSIZE)
    y_max = max(threshold * DENSITY_LIMIT_PAD_FACTOR, max_density * DENSITY_AXIS_PAD_FACTOR)
    axes[0].set_ylim(0.0, y_max)
    apply_dense_value_ticks(axes[0], "y", y_max, "density")
    if title:
        fig.suptitle(title, fontsize=SUBPLOT_SUPTITLE_FONTSIZE, y=SUBPLOT_SUPTITLE_Y)

    handles = [
        Patch(facecolor=density_color, edgecolor=BAR_EDGE_COLOR, label="Peak power density"),
        Line2D([0], [0], color=density_color, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=POWER_DENSITY_LIMIT_LABEL),
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        fontsize=SUBPLOT_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, SUBPLOT_LEGEND_Y),
        frameon=False,
        handlelength=SUBPLOT_LEGEND_HANDLE_LENGTH,
        columnspacing=SUBPLOT_LEGEND_COLUMN_SPACING,
    )

    fig.subplots_adjust(**SUBPLOT_DENSITY_ADJUST)
    pdf_path = out_dir / f"peak_power_density_subplots_{tag}.pdf"
    png_path = out_dir / f"peak_power_density_subplots_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_peak_temperature_horizontal(
    records: List[Dict[str, object]],
    out_dir: Path,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not records:
        raise ValueError("No records to plot")

    logic_color = colors1[LOGIC_TEMP_COLOR_INDEX]
    dram_color = colors1[DRAM_TEMP_COLOR_INDEX]
    max_temp = max(
        max(float(record["logic_peak_c"]), float(record["dram_peak_c"]))
        for record in records
    )
    fig, axes = compact_horizontal_figure(records)
    bar_height = HORIZONTAL_BAR_HEIGHT

    for ax, mode in zip(axes, STAGE_ORDER):
        stage_records = records_for_stage(records, mode)
        y = np.arange(len(stage_records))
        logic = [float(record["logic_peak_c"]) for record in stage_records]
        dram = [float(record["dram_peak_c"]) for record in stage_records]
        draw_barh(
            ax,
            y - bar_height / 2,
            logic,
            height=bar_height,
            color=logic_color,
            zorder=3,
        )
        draw_barh(
            ax,
            y + bar_height / 2,
            dram,
            height=bar_height,
            color=dram_color,
            zorder=3,
            hatch=True,
        )
        ax.axvline(TEMP_LIMIT_C, linestyle="--", color=TEMP_LIMIT_LINE_COLOR, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(model_tick_labels(stage_records), fontsize=HORIZONTAL_YTICK_FONTSIZE)
        ax.invert_yaxis()
        ax.set_title(STAGE_TITLES[mode], fontsize=SUBPLOT_TITLE_FONTSIZE, pad=SUBPLOT_TITLE_PAD)
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="x", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)
        ax.tick_params(axis="x", labelsize=HORIZONTAL_TICK_FONTSIZE)

    axes[-1].set_xlabel("Peak Temp. (°C)", fontsize=HORIZONTAL_LABEL_FONTSIZE)
    x_max = max(TEMP_MIN_YMAX_C, max_temp * TEMP_AXIS_PAD_FACTOR)
    axes[0].set_xlim(0.0, x_max)
    apply_dense_value_ticks(axes[0], "x", x_max, "temp")
    if title:
        fig.suptitle(title, fontsize=SUBPLOT_SUPTITLE_FONTSIZE, y=SUBPLOT_SUPTITLE_Y)

    handles = [
        Patch(facecolor=logic_color, edgecolor=BAR_EDGE_COLOR, label=LOGIC_LEGEND_LABEL),
        dram_legend_patch(dram_color),
        Line2D([0], [0], color=TEMP_LIMIT_LINE_COLOR, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=TEMP_LIMIT_LABEL),
    ]
    fig.legend(
        handles=handles,
        ncol=3,
        fontsize=HORIZONTAL_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, HORIZONTAL_LEGEND_Y),
        frameon=False,
        handler_map=legend_handler_map(),
    )

    fig.subplots_adjust(**HORIZONTAL_ADJUST)
    pdf_path = out_dir / f"peak_temperature_horizontal_{tag}.pdf"
    png_path = out_dir / f"peak_temperature_horizontal_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_peak_density_horizontal(
    records: List[Dict[str, object]],
    out_dir: Path,
    threshold: float,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not records:
        raise ValueError("No records to plot")

    density_color = colors1[POWER_DENSITY_COLOR_INDEX]
    max_density = max(float(record["peak_component_density_w_per_mm2"]) for record in records)
    fig, axes = compact_horizontal_figure(records)

    for ax, mode in zip(axes, STAGE_ORDER):
        stage_records = records_for_stage(records, mode)
        y = np.arange(len(stage_records))
        density = [float(record["peak_component_density_w_per_mm2"]) for record in stage_records]
        ax.barh(
            y,
            density,
            height=HORIZONTAL_DENSITY_BAR_HEIGHT,
            color=density_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=3,
        )
        ax.axvline(threshold, linestyle="--", color=density_color, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(model_tick_labels(stage_records), fontsize=HORIZONTAL_YTICK_FONTSIZE)
        ax.invert_yaxis()
        ax.set_title(STAGE_TITLES[mode], fontsize=SUBPLOT_TITLE_FONTSIZE, pad=SUBPLOT_TITLE_PAD)
        ax.set_axisbelow(True)
        ax.grid(which="major", axis="x", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)
        ax.tick_params(axis="x", labelsize=HORIZONTAL_TICK_FONTSIZE)

    axes[-1].set_xlabel("Density (W/mm2)", fontsize=HORIZONTAL_LABEL_FONTSIZE)
    x_max = max(threshold * DENSITY_LIMIT_PAD_FACTOR, max_density * DENSITY_AXIS_PAD_FACTOR)
    axes[0].set_xlim(0.0, x_max)
    apply_dense_value_ticks(axes[0], "x", x_max, "density")
    if title:
        fig.suptitle(title, fontsize=SUBPLOT_SUPTITLE_FONTSIZE, y=SUBPLOT_SUPTITLE_Y)

    handles = [
        Patch(facecolor=density_color, edgecolor=BAR_EDGE_COLOR, label="Peak power density"),
        Line2D([0], [0], color=density_color, linestyle="--", linewidth=LIMIT_LINEWIDTH, label=POWER_DENSITY_LIMIT_LABEL),
    ]
    fig.legend(
        handles=handles,
        ncol=2,
        fontsize=HORIZONTAL_LEGEND_FONTSIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, HORIZONTAL_LEGEND_Y),
        frameon=False,
    )

    fig.subplots_adjust(**HORIZONTAL_ADJUST)
    pdf_path = out_dir / f"peak_power_density_horizontal_{tag}.pdf"
    png_path = out_dir / f"peak_power_density_horizontal_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_absolute_bars(
    records: List[Dict[str, object]],
    out_dir: Path,
    ambient_c: float,
    threshold: float,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No records to plot")

    positions, centers, labels = x_positions(records)
    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    ax2 = ax.twinx()

    logic_color = colors1[LOGIC_TEMP_COLOR_INDEX]
    dram_color = colors1[DRAM_TEMP_COLOR_INDEX]
    density_color = colors1[POWER_DENSITY_COLOR_INDEX]
    bar_width = BAR_WIDTH

    max_temp = ambient_c
    max_density = 0.0
    for record in records:
        key = (str(record["model"]), str(record["mode"]))
        x = positions[key]
        logic_peak = float(record["logic_peak_c"])
        dram_peak = float(record["dram_peak_c"])
        density = float(record["peak_component_density_w_per_mm2"])
        max_temp = max(max_temp, logic_peak, dram_peak)
        max_density = max(max_density, density)
        draw_bar(
            ax,
            x - bar_width,
            logic_peak,
            width=bar_width,
            color=logic_color,
            zorder=3,
        )
        draw_bar(
            ax,
            x,
            dram_peak,
            width=bar_width,
            color=dram_color,
            zorder=3,
            hatch=True,
        )
        ax2.bar(
            x + bar_width,
            density,
            width=bar_width,
            color=density_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=4,
        )

    ax.axhline(TEMP_LIMIT_C, linestyle="--", color=TEMP_LIMIT_LINE_COLOR, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
    ax2.axhline(threshold, linestyle="--", color=density_color, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)

    apply_common_x_axis(ax, records, positions, centers, labels)
    temp_y_max = max(TEMP_MIN_YMAX_C, max_temp * TEMP_COMBINED_AXIS_PAD_FACTOR)
    density_y_max = max(threshold * DENSITY_LIMIT_PAD_FACTOR, max_density * DENSITY_COMBINED_AXIS_PAD_FACTOR)
    ax.set_ylim(0.0, temp_y_max)
    ax2.set_ylim(0.0, density_y_max)
    apply_dense_value_ticks(ax, "y", temp_y_max, "temp")
    apply_dense_value_ticks(ax2, "y", density_y_max, "density")

    ax.set_ylabel("Temp. (°C)", fontsize=COMBINED_YLABEL_FONTSIZE)
    ax2.set_ylabel("Power density (W/mm2)", fontsize=COMBINED_YLABEL_FONTSIZE, labelpad=8)
    if title:
        ax.set_title(title, fontsize=COMBINED_TITLE_FONTSIZE, pad=COMBINED_TITLE_PAD)

    ax.tick_params(axis="y", labelsize=COMBINED_YTICK_FONTSIZE)
    ax2.tick_params(axis="y", labelsize=COMBINED_YTICK_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)

    ax.legend(
        handles=absolute_bar_handles(logic_color, dram_color, density_color, "Peak power density"),
        ncol=3,
        fontsize=COMBINED_ABSOLUTE_LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, COMBINED_LEGEND_Y),
        frameon=False,
        handler_map=legend_handler_map(),
    )

    fig.subplots_adjust(**COMBINED_ABSOLUTE_ADJUST)
    pdf_path = out_dir / f"peak_temp_density_bars_{tag}.pdf"
    png_path = out_dir / f"peak_temp_density_bars_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_temperature_rise_bars(
    records: List[Dict[str, object]],
    out_dir: Path,
    ambient_c: float,
    threshold: float,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No records to plot")

    positions, centers, labels = x_positions(records)
    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    ax2 = ax.twinx()

    logic_color = colors1[LOGIC_TEMP_COLOR_INDEX]
    dram_color = colors1[DRAM_TEMP_COLOR_INDEX]
    density_color = colors1[POWER_DENSITY_COLOR_INDEX]
    bar_width = BAR_WIDTH
    temp_rise_limit_c = TEMP_LIMIT_C - ambient_c

    max_rise = 0.0
    max_density = 0.0
    for record in records:
        key = (str(record["model"]), str(record["mode"]))
        x = positions[key]
        logic_rise = float(record["logic_rise_c"])
        dram_rise = float(record["dram_rise_c"])
        density = float(record["peak_component_density_w_per_mm2"])
        max_rise = max(max_rise, logic_rise, dram_rise)
        max_density = max(max_density, density)
        draw_bar(
            ax,
            x - bar_width,
            logic_rise,
            width=bar_width,
            color=logic_color,
            zorder=3,
        )
        draw_bar(
            ax,
            x,
            dram_rise,
            width=bar_width,
            color=dram_color,
            zorder=3,
            hatch=True,
        )
        ax2.bar(
            x + bar_width,
            density,
            width=bar_width,
            color=density_color,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_LINEWIDTH,
            zorder=4,
        )

    ax.axhline(temp_rise_limit_c, linestyle="--", color=logic_color, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)
    ax2.axhline(threshold, linestyle="--", color=density_color, linewidth=LIMIT_LINEWIDTH, alpha=LIMIT_LINE_ALPHA, zorder=2)

    apply_common_x_axis(ax, records, positions, centers, labels)
    temp_y_max = max(temp_rise_limit_c * TEMP_RISE_LIMIT_PAD_FACTOR, max_rise * TEMP_RISE_AXIS_PAD_FACTOR)
    density_y_max = max(threshold * DENSITY_LIMIT_PAD_FACTOR, max_density * DENSITY_COMBINED_AXIS_PAD_FACTOR)
    ax.set_ylim(0.0, temp_y_max)
    ax2.set_ylim(0.0, density_y_max)
    apply_dense_value_ticks(ax, "y", temp_y_max, "temp")
    apply_dense_value_ticks(ax2, "y", density_y_max, "density")

    ax.set_ylabel("Temp. Rise (°C)", fontsize=COMBINED_YLABEL_FONTSIZE)
    ax2.set_ylabel("Power density (W/mm2)", fontsize=COMBINED_YLABEL_FONTSIZE, labelpad=8)
    if title:
        ax.set_title(title, fontsize=COMBINED_TITLE_FONTSIZE, pad=COMBINED_TITLE_PAD)

    ax.tick_params(axis="y", labelsize=COMBINED_YTICK_FONTSIZE)
    ax2.tick_params(axis="y", labelsize=COMBINED_YTICK_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)

    temp_limit_label = f"{temp_rise_limit_c:g}°C rise"
    ax.legend(
        handles=absolute_bar_handles(
            logic_color,
            dram_color,
            density_color,
            "Peak power density",
            temp_limit_label,
            "Logic peak temp. rise",
            "DRAM peak temp. rise",
        ),
        ncol=3,
        fontsize=COMBINED_RISE_LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, COMBINED_LEGEND_Y),
        frameon=False,
        handler_map=legend_handler_map(),
    )

    fig.subplots_adjust(**COMBINED_ABSOLUTE_ADJUST)
    pdf_path = out_dir / f"peak_temp_rise_density_bars_{tag}.pdf"
    png_path = out_dir / f"peak_temp_rise_density_bars_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def plot_normalized_bars(
    records: List[Dict[str, object]],
    out_dir: Path,
    threshold: float,
    tag: str,
    title: str,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not records:
        raise ValueError("No records to plot")

    positions, centers, labels = x_positions(records)
    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    logic_color = colors1[LOGIC_TEMP_COLOR_INDEX]
    dram_color = colors1[DRAM_TEMP_COLOR_INDEX]
    density_color = colors1[POWER_DENSITY_COLOR_INDEX]
    bar_width = BAR_WIDTH
    max_pct = 0.0

    for record in records:
        key = (str(record["model"]), str(record["mode"]))
        x = positions[key]
        logic_pct = float(record["logic_temp_pct_of_50c_rise"])
        dram_pct = float(record["dram_temp_pct_of_50c_rise"])
        density_pct = float(record["density_pct_of_0p7_w_per_mm2"])
        max_pct = max(max_pct, logic_pct, dram_pct, density_pct)
        draw_bar(ax, x - bar_width, logic_pct, width=bar_width, color=logic_color, zorder=3)
        draw_bar(ax, x, dram_pct, width=bar_width, color=dram_color, zorder=3, hatch=True)
        draw_bar(ax, x + bar_width, density_pct, width=bar_width, color=density_color, zorder=3)

    ax.axhline(
        NORMALIZED_LIMIT_PCT,
        linestyle="--",
        color=NORMALIZED_LIMIT_COLOR,
        linewidth=NORMALIZED_LIMIT_LINEWIDTH,
        alpha=NORMALIZED_LIMIT_ALPHA,
        zorder=2,
    )
    apply_common_x_axis(ax, records, positions, centers, labels)
    y_max = max(NORMALIZED_MIN_YMAX_PCT, max_pct * NORMALIZED_AXIS_PAD_FACTOR)
    ax.set_ylim(0.0, y_max)
    apply_dense_value_ticks(ax, "y", y_max, "percent")
    ax.set_ylabel("Normalized (%)", fontsize=COMBINED_YLABEL_FONTSIZE)
    if title:
        ax.set_title(title, fontsize=COMBINED_TITLE_FONTSIZE, pad=COMBINED_TITLE_PAD)
    ax.tick_params(axis="y", labelsize=COMBINED_YTICK_FONTSIZE)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=GRID_LINEWIDTH, color=GRID_COLOR, zorder=1)
    ax.legend(
        handles=bar_handles(logic_color, dram_color, density_color, "Peak power density"),
        ncol=3,
        fontsize=COMBINED_NORMALIZED_LEGEND_FONTSIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, COMBINED_LEGEND_Y),
        frameon=False,
        handler_map=legend_handler_map(),
    )
    fig.subplots_adjust(**COMBINED_NORMALIZED_ADJUST)

    pdf_path = out_dir / f"peak_temp_density_normalized_bars_{tag}.pdf"
    png_path = out_dir / f"peak_temp_density_normalized_bars_{tag}.png"
    fig.savefig(pdf_path, bbox_inches=BBOX_INCHES)
    fig.savefig(png_path, dpi=OUTPUT_DPI, bbox_inches=BBOX_INCHES)
    plt.close(fig)
    return pdf_path, png_path


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.thermal_root / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = window_tag(args.density_window_us)
    density_root = args.density_root or args.thermal_root
    records = ordered_records(args.thermal_root, density_root, args.density_window_us, args.ambient_c)
    if args.normalize_density_peak_to_threshold:
        normalize_density_peak_to_threshold(records, args.density_threshold)
    enrich_normalized(records, args.density_threshold)
    plot_script_path = expose_plot_script(out_dir)
    source_csv = write_source_csv(records, out_dir, tag)
    temp_pdf_path, temp_png_path = plot_peak_temperature_subplots(
        records,
        out_dir,
        tag,
        args.title,
    )
    density_pdf_path, density_png_path = plot_peak_density_subplots(
        records,
        out_dir,
        args.density_threshold,
        tag,
        args.title,
    )
    temp_h_pdf_path, temp_h_png_path = plot_peak_temperature_horizontal(
        records,
        out_dir,
        tag,
        args.title,
    )
    density_h_pdf_path, density_h_png_path = plot_peak_density_horizontal(
        records,
        out_dir,
        args.density_threshold,
        tag,
        args.title,
    )
    print(f"Wrote {plot_script_path}")
    print(f"Wrote {source_csv}")
    print(f"Wrote {temp_pdf_path}")
    print(f"Wrote {temp_png_path}")
    print(f"Wrote {density_pdf_path}")
    print(f"Wrote {density_png_path}")
    print(f"Wrote {temp_h_pdf_path}")
    print(f"Wrote {temp_h_png_path}")
    print(f"Wrote {density_h_pdf_path}")
    print(f"Wrote {density_h_png_path}")


if __name__ == "__main__":
    main()
