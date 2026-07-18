#!/usr/bin/env python3
"""Plot SXM-cooling peak logic and DRAM temperatures from cooler-matrix runs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

import matplotlib as mpl

try:
    from fig_common import colors1, modelnames  # type: ignore
except ImportError:  # pragma: no cover - keeps the script usable outside benchmark_scripts.
    colors1 = ["#126b91", "#ec6632", "#93bc38", "#252422", "#c45ab3"] * 2
    modelnames = {}

mpl.rcParams.update({"font.family": "serif"})
mpl.rcParams["pdf.fonttype"] = 42


DEFAULT_MODEL_ORDER = ("llama2-13", "llama3-70", "opt-30", "gemma2", "dit-xl")
STAGE_ORDER = ("prefill", "decode")
STAGE_LABELS = {"prefill": "P", "decode": "D"}


def parse_args() -> argparse.Namespace:
    repo_src = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Create a grouped bar chart of SXM-cooling peak logic/DRAM temperatures."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_src / "results",
        help="Root containing thermal_cooler_matrix* result directories.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_src / "results" / "thermal_cooler_matrix_plots",
        help="Directory for the generated plot and source CSV.",
    )
    parser.add_argument(
        "--summary-csv",
        action="append",
        type=Path,
        default=[],
        help="Specific thermal_cooler_matrix_summary.csv file to include. May be repeated.",
    )
    parser.add_argument("--cooling-profile", default="sxm_air")
    parser.add_argument("--ambient-c", type=float, default=35.0)
    parser.add_argument("--title", default="SXM cooling peak logic/DRAM temperature")
    return parser.parse_args()


def read_summary_rows(results_root: Path, summary_csvs: List[Path]) -> Iterable[Tuple[Path, Dict[str, str]]]:
    paths = summary_csvs or sorted(results_root.glob("thermal_cooler_matrix*/thermal_cooler_matrix_summary.csv"))
    for summary_csv in paths:
        with summary_csv.open(encoding="utf-8", newline="") as infile:
            for row in csv.DictReader(infile):
                yield summary_csv.parent, row


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
        raise ValueError(f"{timeseries_csv} has no DRAM temperature samples")
    return peak


def collect_peaks(args: argparse.Namespace) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    candidates: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
    for matrix_dir, row in read_summary_rows(args.results_root, args.summary_csv):
        if row.get("cooling_profile") != args.cooling_profile:
            continue
        tag = row["tag"]
        timeseries_csv = matrix_dir / f"{tag}_layer_temperature_timeseries.csv"
        if not timeseries_csv.exists():
            raise FileNotFoundError(f"Missing layer temperature time series for {tag}: {timeseries_csv}")
        record = {
            "model": row["model"],
            "mode": row["mode"],
            "logic_peak_c": float(row["logic_peak_c"]),
            "dram_peak_c": peak_dram_temp_c(timeseries_csv),
            "duration_s": float(row.get("duration_s") or 0.0),
            "source_summary": str(matrix_dir / "thermal_cooler_matrix_summary.csv"),
            "source_timeseries": str(timeseries_csv),
        }
        candidates[(str(record["model"]), str(record["mode"]))].append(record)

    for key, records in candidates.items():
        # If duplicate runs exist for the same model/stage, keep the run with the
        # highest package-observed peak so the chart reflects "maximum achieved".
        merged[key] = max(records, key=lambda item: max(float(item["logic_peak_c"]), float(item["dram_peak_c"])))
    rows = list(merged.values())
    rows.sort(key=lambda item: (model_sort_key(str(item["model"])), STAGE_ORDER.index(str(item["mode"]))))
    return rows


def model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return (DEFAULT_MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(DEFAULT_MODEL_ORDER), model)


def write_source_csv(rows: List[Dict[str, object]], out_dir: Path) -> Path:
    out_path = out_dir / "sxm_peak_logic_dram_temperatures.csv"
    fieldnames = [
        "model",
        "mode",
        "logic_peak_c",
        "dram_peak_c",
        "duration_s",
        "source_summary",
        "source_timeseries",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return out_path


def plot(rows: List[Dict[str, object]], out_dir: Path, ambient_c: float, title: str) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("No SXM cooling rows were found")

    by_model_stage = {(str(row["model"]), str(row["mode"])): row for row in rows}
    models = sorted({str(row["model"]) for row in rows}, key=model_sort_key)

    fig_width = max(6.5, 1.12 * len(models) + 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 2.75))
    logic_color = colors1[0]
    dram_color = colors1[1]

    bar_width = 0.135
    offsets = {
        ("prefill", "logic"): -0.27,
        ("prefill", "dram"): -0.12,
        ("decode", "logic"): 0.12,
        ("decode", "dram"): 0.27,
    }
    stage_centers = {"prefill": -0.195, "decode": 0.195}
    x_centers = np.arange(len(models), dtype=float)
    y_floor = ambient_c
    y_top = y_floor

    first_logic = True
    first_dram = True
    for model_idx, model in enumerate(models):
        for stage in STAGE_ORDER:
            row = by_model_stage.get((model, stage))
            if row is None:
                continue
            logic = float(row["logic_peak_c"])
            dram = float(row["dram_peak_c"])
            y_top = max(y_top, logic, dram)
            for component, value, color in (
                ("logic", logic, logic_color),
                ("dram", dram, dram_color),
            ):
                x = x_centers[model_idx] + offsets[(stage, component)]
                height = max(0.0, value - y_floor)
                is_logic = component == "logic"
                ax.bar(
                    x,
                    height,
                    width=bar_width,
                    bottom=y_floor,
                    color=color,
                    edgecolor="black",
                    linewidth=0.65,
                    hatch="" if is_logic else "\\\\",
                    label=(
                        "Logic" if is_logic and first_logic
                        else "DRAM" if (not is_logic and first_dram)
                        else None
                    ),
                    zorder=3,
                )
                if is_logic:
                    first_logic = False
                else:
                    first_dram = False
            ax.text(
                x_centers[model_idx] + stage_centers[stage],
                y_floor - 0.95,
                STAGE_LABELS[stage],
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                clip_on=False,
            )

    ax.axhline(85.0, linestyle="--", color="#d62728", linewidth=1.25, alpha=0.8, label="85 C")
    ax.set_xticks(x_centers)
    ax.set_xticklabels([modelnames.get(model, model) for model in models], fontsize=12, rotation=12, ha="right")
    ax.set_xlim(-0.65, len(models) - 0.35)
    ax.set_ylim(y_floor - 2.6, max(92.0, y_top + 7.0))
    ax.set_ylabel("Peak Temp. (C)", fontsize=15)
    ax.set_xlabel("P = prefill, D = decode", fontsize=13, labelpad=8)
    if title:
        ax.set_title(title, fontsize=16, pad=6)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax.legend(ncol=3, fontsize=12, loc="upper left", frameon=False, handlelength=1.25, columnspacing=0.9)
    fig.tight_layout(pad=0.25)

    png_path = out_dir / "sxm_peak_logic_dram_temperatures.png"
    pdf_path = out_dir / "sxm_peak_logic_dram_temperatures.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_peaks(args)
    source_csv = write_source_csv(rows, args.out_dir)
    png_path, pdf_path = plot(rows, args.out_dir, args.ambient_c, args.title)
    print(f"Wrote {source_csv}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
