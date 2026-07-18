#!/usr/bin/env python3
"""Plot recent SXM thermal peaks and normalized component power density."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import numpy as np

try:
    from fig_common import colors1, modelnames  # type: ignore
except ImportError:  # pragma: no cover
    colors1 = ["#126b91", "#ec6632", "#93bc38", "#252422", "#c45ab3"] * 2
    modelnames = {}

mpl.rcParams.update({"font.family": "serif"})
mpl.rcParams["pdf.fonttype"] = 42


MODEL_ORDER = ("llama2-13", "llama3-70", "opt-30", "gemma2", "dit-xl")
STAGE_ORDER = ("prefill", "decode")
STAGE_LABELS = {"prefill": "P", "decode": "D"}
COMPONENT_ORDER = ("SA", "VU", "SRAM", "Router", "DRAM")


@dataclass(frozen=True)
class PackageRecord:
    model: str
    mode: str
    tag: str
    package_dir: Path
    metadata_path: Path
    mtime: float
    timeseries_csv: Path


def parse_args() -> argparse.Namespace:
    repo_src = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=repo_src / "results")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_src / "results" / "thermal_recent_sxm_plots",
    )
    parser.add_argument("--cooling-profile", default="sxm_air")
    parser.add_argument("--ambient-c", type=float, default=35.0)
    parser.add_argument(
        "--normalized-power-density-max",
        type=float,
        default=0.7,
        help="Scale all plotted power densities so the global maximum equals this value.",
    )
    return parser.parse_args()


def model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(MODEL_ORDER), model)


def include_record(model: str, mode: str) -> bool:
    if mode not in STAGE_ORDER:
        return False
    if model == "dit-xl" and mode == "prefill":
        return False
    return True


def read_metadata(path: Path) -> Dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def iter_candidate_packages(results_root: Path, cooling_profile: str) -> Iterable[PackageRecord]:
    for metadata_path in results_root.glob("thermal_cooler_matrix*/packages/*/metadata.json"):
        metadata = read_metadata(metadata_path)
        if not metadata:
            continue
        cooling = metadata.get("cooling_model", {})
        if not isinstance(cooling, dict) or cooling.get("profile") != cooling_profile:
            continue
        run = metadata.get("run", {})
        if not isinstance(run, dict):
            continue
        model = str(run.get("model", ""))
        mode = str(run.get("mode", ""))
        if not include_record(model, mode):
            continue
        package_dir = metadata_path.parent
        tag = package_dir.name
        timeseries_csv = package_dir.parent.parent / f"{tag}_layer_temperature_timeseries.csv"
        if not timeseries_csv.exists():
            continue
        yield PackageRecord(
            model=model,
            mode=mode,
            tag=tag,
            package_dir=package_dir,
            metadata_path=metadata_path,
            mtime=metadata_path.stat().st_mtime,
            timeseries_csv=timeseries_csv,
        )


def select_latest_packages(results_root: Path, cooling_profile: str) -> List[PackageRecord]:
    latest: Dict[Tuple[str, str], PackageRecord] = {}
    for record in iter_candidate_packages(results_root, cooling_profile):
        key = (record.model, record.mode)
        if key not in latest or record.mtime > latest[key].mtime:
            latest[key] = record
    records = list(latest.values())
    records.sort(key=lambda item: (model_sort_key(item.model), STAGE_ORDER.index(item.mode)))
    return records


def read_temperature_peaks(record: PackageRecord) -> Dict[str, object]:
    logic_peak = -math.inf
    dram_peak = -math.inf
    duration_s = 0.0
    with record.timeseries_csv.open(encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        dram_cols = [
            name
            for name in (reader.fieldnames or [])
            if name.startswith("dram") and name.endswith("_peak_c")
        ]
        for row in reader:
            duration_s = max(duration_s, float(row.get("time_s") or 0.0))
            if row.get("logic_peak_c"):
                logic_peak = max(logic_peak, float(row["logic_peak_c"]))
            for col in dram_cols:
                if row.get(col):
                    dram_peak = max(dram_peak, float(row[col]))
    if not math.isfinite(logic_peak):
        raise ValueError(f"No logic_peak_c values in {record.timeseries_csv}")
    if not math.isfinite(dram_peak):
        raise ValueError(f"No dram*_peak_c values in {record.timeseries_csv}")
    return {
        "model": record.model,
        "mode": record.mode,
        "tag": record.tag,
        "logic_peak_c": logic_peak,
        "dram_peak_c": dram_peak,
        "duration_s": duration_s,
        "source_timeseries": str(record.timeseries_csv),
        "source_package": str(record.package_dir),
    }


def read_floorplan_areas_mm2(package_dir: Path) -> Dict[str, float]:
    areas: Dict[str, float] = {}
    for flp_path in [package_dir / "logic.flp", *sorted(package_dir.glob("dram*.flp"))]:
        if not flp_path.exists():
            continue
        with flp_path.open(encoding="utf-8") as infile:
            for line in infile:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()
                if len(parts) < 5:
                    continue
                name = parts[0]
                width_m = float(parts[1])
                height_m = float(parts[2])
                areas[name] = width_m * height_m * 1.0e6
    return areas


def classify_component(name: str) -> str | None:
    if name.endswith("_sa"):
        return "SA"
    if name.endswith("_vu"):
        return "VU"
    if name.endswith("_sram"):
        return "SRAM"
    if name.endswith("_router"):
        return "Router"
    if "_pkg" in name and "_bank" in name and name.startswith("dram"):
        return "DRAM"
    return None


def peak_component_power_density(record: PackageRecord) -> Dict[str, object]:
    spatial_power = record.package_dir / "spatial_power.csv"
    if not spatial_power.exists():
        raise FileNotFoundError(spatial_power)
    areas = read_floorplan_areas_mm2(record.package_dir)
    raw_peak = {component: 0.0 for component in COMPONENT_ORDER}
    with spatial_power.open(encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)
        columns: List[Tuple[int, str, float]] = []
        for idx, name in enumerate(header):
            if idx == 0 or name == "total":
                continue
            component = classify_component(name)
            if component is None:
                continue
            area = areas.get(name)
            if area is None or area <= 0.0:
                continue
            columns.append((idx, component, area))
        for row in reader:
            for idx, component, area in columns:
                if idx >= len(row) or not row[idx]:
                    continue
                density = float(row[idx]) / area
                if density > raw_peak[component]:
                    raw_peak[component] = density
    result: Dict[str, object] = {
        "model": record.model,
        "mode": record.mode,
        "tag": record.tag,
        "source_package": str(record.package_dir),
    }
    for component in COMPONENT_ORDER:
        result[f"{component.lower()}_peak_w_per_mm2_raw"] = raw_peak[component]
    return result


def write_rows_csv(rows: List[Dict[str, object]], path: Path, fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def group_positions(records: List[PackageRecord]) -> Tuple[Dict[Tuple[str, str], float], List[float], List[str]]:
    models = sorted({record.model for record in records}, key=model_sort_key)
    positions: Dict[Tuple[str, str], float] = {}
    centers: List[float] = []
    labels: List[str] = []
    x = 0.0
    for model in models:
        present = [record.mode for record in records if record.model == model]
        if model == "dit-xl":
            if "decode" in present:
                positions[(model, "decode")] = x
                centers.append(x)
                labels.append(modelnames.get(model, model))
                x += 1.0
            continue
        offsets = {"prefill": -0.18, "decode": 0.18}
        for mode in STAGE_ORDER:
            if mode in present:
                positions[(model, mode)] = x + offsets[mode]
        centers.append(x)
        labels.append(modelnames.get(model, model))
        x += 1.0
    return positions, centers, labels


def plot_temperature(records: List[PackageRecord], rows: List[Dict[str, object]], out_dir: Path, ambient_c: float) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_by_key = {(str(row["model"]), str(row["mode"])): row for row in rows}
    positions, centers, labels = group_positions(records)
    fig, ax = plt.subplots(figsize=(6.7, 2.75))
    y_top = ambient_c
    bar_width = 0.13
    first_logic = True
    first_dram = True
    for record in records:
        row = row_by_key[(record.model, record.mode)]
        center = positions[(record.model, record.mode)]
        for component, xoff, color, hatch in (
            ("logic", -0.075, colors1[0], ""),
            ("dram", 0.075, colors1[1], "\\\\"),
        ):
            value = float(row[f"{component}_peak_c"])
            y_top = max(y_top, value)
            ax.bar(
                center + xoff,
                max(0.0, value - ambient_c),
                width=bar_width,
                bottom=ambient_c,
                color=color,
                edgecolor="black",
                linewidth=0.65,
                hatch=hatch,
                label=(
                    "Logic" if component == "logic" and first_logic
                    else "DRAM" if component == "dram" and first_dram
                    else None
                ),
                zorder=3,
            )
            first_logic = first_logic and component != "logic"
            first_dram = first_dram and component != "dram"
        if record.model != "dit-xl":
            ax.text(
                center,
                ambient_c - 0.95,
                STAGE_LABELS[record.mode],
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                clip_on=False,
            )
    ax.axhline(85.0, linestyle="--", color="#d62728", linewidth=1.25, alpha=0.8, label="85 C")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=12, rotation=12, ha="right")
    ax.set_xlim(min(centers) - 0.65, max(centers) + 0.65)
    ax.set_ylim(ambient_c - 2.6, max(92.0, y_top + 7.0))
    ax.set_ylabel("Peak Temp. (C)", fontsize=15)
    ax.set_xlabel("P = prefill, D = decode", fontsize=13, labelpad=8)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax.legend(ncol=3, fontsize=12, loc="upper left", frameon=False, handlelength=1.25, columnspacing=0.9)
    fig.tight_layout(pad=0.25)
    png_path = out_dir / "recent_sxm_peak_logic_dram_temperatures.png"
    pdf_path = out_dir / "recent_sxm_peak_logic_dram_temperatures.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_power_density(
    records: List[PackageRecord],
    rows: List[Dict[str, object]],
    out_dir: Path,
    *,
    normalized: bool,
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_by_key = {(str(row["model"]), str(row["mode"])): row for row in rows}
    positions, centers, labels = group_positions(records)
    fig, ax = plt.subplots(figsize=(6.9, 2.75))
    bar_width = 0.055
    offsets = np.linspace(-0.13, 0.13, len(COMPONENT_ORDER))
    component_colors = {
        "SA": colors1[0],
        "VU": colors1[1],
        "SRAM": colors1[2],
        "Router": colors1[3],
        "DRAM": colors1[4],
    }
    for comp_idx, component in enumerate(COMPONENT_ORDER):
        first = True
        for record in records:
            row = row_by_key[(record.model, record.mode)]
            suffix = "norm" if normalized else "raw"
            value = float(row[f"{component.lower()}_peak_w_per_mm2_{suffix}"])
            ax.bar(
                positions[(record.model, record.mode)] + float(offsets[comp_idx]),
                value,
                width=bar_width,
                color=component_colors[component],
                edgecolor="black",
                linewidth=0.55,
                label=component if first else None,
                zorder=3,
            )
            first = False
    for record in records:
        if record.model != "dit-xl":
            ax.text(
                positions[(record.model, record.mode)],
                -0.038,
                STAGE_LABELS[record.mode],
                ha="center",
                va="top",
                fontsize=13,
                fontweight="bold",
                clip_on=False,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, fontsize=12, rotation=12, ha="right")
    ax.set_xlim(min(centers) - 0.65, max(centers) + 0.65)
    max_value = max(
        float(row[f"{component.lower()}_peak_w_per_mm2_{'norm' if normalized else 'raw'}"])
        for row in rows
        for component in COMPONENT_ORDER
    )
    ax.set_ylim(-0.09 * max(1.0, max_value), max(0.78, max_value * 1.12))
    ax.set_ylabel("Peak Power Density\n(W/mm$^2$)", fontsize=14)
    if normalized:
        ax.set_xlabel("P = prefill, D = decode; normalized to max = 0.7 W/mm$^2$", fontsize=12, labelpad=8)
        stem = "recent_sxm_normalized_component_power_density"
    else:
        ax.set_xlabel("P = prefill, D = decode", fontsize=12, labelpad=8)
        stem = "recent_sxm_component_power_density_raw"
    ax.tick_params(axis="y", labelsize=12)
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax.legend(ncol=5, fontsize=10.5, loc="upper left", frameon=False, handlelength=1.1, columnspacing=0.75)
    fig.tight_layout(pad=0.25)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = select_latest_packages(args.results_root, args.cooling_profile)
    if not records:
        raise SystemExit("No recent SXM package records found")

    temp_rows = [read_temperature_peaks(record) for record in records]
    temp_fields = [
        "model",
        "mode",
        "tag",
        "logic_peak_c",
        "dram_peak_c",
        "duration_s",
        "source_timeseries",
        "source_package",
    ]
    temp_csv = args.out_dir / "recent_sxm_peak_logic_dram_temperatures.csv"
    write_rows_csv(temp_rows, temp_csv, temp_fields)

    power_rows = [peak_component_power_density(record) for record in records]
    raw_fields = [f"{component.lower()}_peak_w_per_mm2_raw" for component in COMPONENT_ORDER]
    max_raw = max(float(row[field]) for row in power_rows for field in raw_fields)
    scale = float(args.normalized_power_density_max) / max_raw if max_raw > 0.0 else 0.0
    for row in power_rows:
        row["normalization_scale"] = scale
        for component in COMPONENT_ORDER:
            raw_field = f"{component.lower()}_peak_w_per_mm2_raw"
            norm_field = f"{component.lower()}_peak_w_per_mm2_norm"
            row[norm_field] = float(row[raw_field]) * scale
    power_fields = [
        "model",
        "mode",
        "tag",
        *raw_fields,
        *[f"{component.lower()}_peak_w_per_mm2_norm" for component in COMPONENT_ORDER],
        "normalization_scale",
        "source_package",
    ]
    power_csv = args.out_dir / "recent_sxm_component_power_density.csv"
    write_rows_csv(power_rows, power_csv, power_fields)

    temp_png, temp_pdf = plot_temperature(records, temp_rows, args.out_dir, float(args.ambient_c))
    power_png, power_pdf = plot_power_density(records, power_rows, args.out_dir, normalized=True)
    raw_power_png, raw_power_pdf = plot_power_density(records, power_rows, args.out_dir, normalized=False)
    print(f"Wrote {temp_csv}")
    print(f"Wrote {power_csv}")
    print(f"Wrote {temp_png}")
    print(f"Wrote {temp_pdf}")
    print(f"Wrote {power_png}")
    print(f"Wrote {power_pdf}")
    print(f"Wrote {raw_power_png}")
    print(f"Wrote {raw_power_pdf}")


if __name__ == "__main__":
    main()
