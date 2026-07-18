#!/usr/bin/env python3
"""Sweep 3D-ICE cooling resistance for stacked and logic-direct packages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.artifacts import REPO_SRC, load_artifacts
from tsim_thermal.defaults import DEFAULT_AMBIENT_C
from tsim_thermal.threedice import run_threedice
from tsim_thermal.trace import TraceConfig, build_component_power_trace, export_trace_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate r_convec for stacked vs logic-direct 3D-ICE thermal packages.")
    parser.add_argument("--results-dir", type=Path, default=REPO_SRC / "results" / "logs_full_model")
    parser.add_argument("--out-dir", type=Path, default=REPO_SRC / "results" / "thermal_cooling_ablation")
    parser.add_argument("--model", default="llama2-13")
    parser.add_argument("--mode", default="prefill")
    parser.add_argument("--impl", default="best")
    parser.add_argument("--dram-bw", type=int, default=12288)
    parser.add_argument("--row", type=int, default=8192)
    parser.add_argument("--core-group", type=int, default=8)
    parser.add_argument("--ambient-c", type=float, default=DEFAULT_AMBIENT_C)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--hotspot-grid", type=int, default=128)
    parser.add_argument("--dram-layers", type=int, default=8)
    parser.add_argument("--r-convecs", default="0.02,0.04,0.06,0.08,0.10,0.13,0.16")
    parser.add_argument("--threedice-bin", default=str(REPO_SRC.parent / "external" / "3d-ice-src" / "bin" / "3D-ICE-Emulator"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    return parser.parse_args()


def split_floats(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_temperature_stats(package_dir: Path) -> Dict[str, float]:
    trace = package_dir / "threedice_temperature.ttrace"
    with trace.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = [[float(item) for item in line.split()] for line in infile if line.strip()]
    temps_c = np.asarray(rows, dtype=float) - 273.15
    logic_idx = [idx for idx, name in enumerate(names) if name.startswith("core_") or name.startswith("logic")]
    dram_idx = [idx for idx, name in enumerate(names) if name.startswith("dram")]
    if not logic_idx or not dram_idx:
        raise ValueError(f"could not classify logic/dram columns in {trace}")
    logic = temps_c[:, logic_idx]
    dram = temps_c[:, dram_idx]
    return {
        "logic_peak_c": float(np.max(logic)),
        "logic_avg_c": float(np.mean(logic)),
        "dram_peak_c": float(np.max(dram)),
        "dram_avg_c": float(np.mean(dram)),
    }


def plot_results(csv_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[Dict[str, str]] = []
    with csv_path.open(encoding="utf-8") as infile:
        rows.extend(csv.DictReader(infile))

    def series(stack: str, metric: str, component: str) -> Tuple[List[float], List[float]]:
        selected = [row for row in rows if row["stack"] == stack]
        selected.sort(key=lambda row: float(row["r_convec_k_per_w"]))
        return (
            [float(row["r_convec_k_per_w"]) for row in selected],
            [float(row[f"{component}_{metric}_c"]) for row in selected],
        )

    panels = [
        ("stacked", "peak", "Stacked Peak"),
        ("stacked", "avg", "Stacked Average"),
        ("logic_direct", "peak", "Logic-Direct Peak"),
        ("logic_direct", "avg", "Logic-Direct Average"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (stack, metric, title) in zip(axes.flat, panels):
        for component, color in (("logic", "#d62728"), ("dram", "#1f77b4")):
            x, y = series(stack, metric, component)
            ax.plot(x, y, marker="o", linewidth=2, color=color, label=component.upper())
        ax.set_title(title)
        ax.set_ylabel("Temperature (C)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("r_convec (K/W)")
    fig.suptitle("Cooling Resistance Ablation: Full Prefill, 35 C Boundary")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TSIM_THREEDICE_CORES", "48")
    os.environ.setdefault("OMP_NUM_THREADS", "48")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log("loading TSIM artifacts")
    artifacts = load_artifacts(args.results_dir, models=args.model, modes=args.mode, impls=args.impl)
    selected = [
        artifact for artifact in artifacts
        if artifact.run_id.dram_bw == args.dram_bw
        and artifact.run_id.row == args.row
        and artifact.run_id.core_group == args.core_group
    ]
    if not selected:
        raise SystemExit("No matching artifact found.")
    artifact = selected[0]
    log(f"selected {artifact.run_id.label()}")

    base_cfg = TraceConfig(
        ambient_c=args.ambient_c,
        max_bins=args.bins,
        major_op_samples=0,
        dram_layers=args.dram_layers,
        dram_floorplan_granularity="bank",
        dram_bank_mapping="address_trace",
        logic_floorplan="intra_core",
        hotspot_grid=args.hotspot_grid,
    )
    log("building shared power trace")
    trace = build_component_power_trace(artifact, base_cfg)

    rows: List[Dict[str, object]] = []
    r_values = split_floats(args.r_convecs)
    for logic_direct in (False, True):
        stack = "logic_direct" if logic_direct else "stacked"
        for r_convec in r_values:
            tag = f"{stack}_r{str(r_convec).replace('.', 'p')}"
            package_dir = args.out_dir / "packages" / tag
            cfg = TraceConfig(
                ambient_c=args.ambient_c,
                max_bins=args.bins,
                major_op_samples=0,
                dram_layers=args.dram_layers,
                dram_floorplan_granularity="bank",
                dram_bank_mapping="address_trace",
                logic_floorplan="intra_core",
                hotspot_grid=args.hotspot_grid,
                r_convec_k_per_w=r_convec,
                cooling_profile=f"ablation_r_convec_{r_convec:g}",
                logic_direct_to_heatsink=logic_direct,
            )
            log(f"exporting {tag}")
            export_trace_package(artifact, trace, cfg, package_dir, write_visualizations=False)
            log(f"running 3D-ICE {tag}")
            result = run_threedice(package_dir, args.threedice_bin, args.timeout_s, heatmap_count=0)
            stats = read_temperature_stats(package_dir)
            metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
            rows.append({
                "stack": stack,
                "logic_direct_to_heatsink": logic_direct,
                "r_convec_k_per_w": r_convec,
                "ambient_c": args.ambient_c,
                "h_w_per_m2_k": metadata["cooling_model"]["heat_transfer_coefficient_w_per_m2_k"],
                "threedice_runtime_s": result.runtime_s,
                "package_dir": str(package_dir),
                **stats,
            })

    csv_path = args.out_dir / "cooling_ablation.csv"
    fieldnames = [
        "stack",
        "logic_direct_to_heatsink",
        "r_convec_k_per_w",
        "ambient_c",
        "h_w_per_m2_k",
        "logic_peak_c",
        "logic_avg_c",
        "dram_peak_c",
        "dram_avg_c",
        "threedice_runtime_s",
        "package_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    png_path = args.out_dir / "cooling_ablation_temperature_vs_rconvec.png"
    plot_results(csv_path, png_path)
    print(json.dumps({"csv": str(csv_path), "plot": str(png_path), "points": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
