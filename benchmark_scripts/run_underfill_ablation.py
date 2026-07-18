#!/usr/bin/env python3
"""Sweep HBM bond/underfill thickness under several cooling assumptions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


COOLING_PROFILES: Tuple[Tuple[str, float], ...] = (
    ("pcie_air", 0.13),
    ("sxm_air", 0.06),
    ("high_perf_liquid", 0.02),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate underfill thickness for stacked 3D-ICE packages.")
    parser.add_argument("--results-dir", type=Path, default=REPO_SRC / "results" / "logs_full_model")
    parser.add_argument("--out-dir", type=Path, default=REPO_SRC / "results" / "thermal_underfill_ablation")
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
    parser.add_argument("--thicknesses-um", default="1-20")
    parser.add_argument("--threedice-bin", default=str(REPO_SRC.parent / "external" / "3d-ice-src" / "bin" / "3D-ICE-Emulator"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cores-per-job", type=int, default=8)
    return parser.parse_args()


def parse_thicknesses(value: str) -> List[float]:
    value = value.strip()
    if "-" in value and "," not in value:
        start, end = [int(item.strip()) for item in value.split("-", 1)]
        step = 1 if end >= start else -1
        return [float(item) for item in range(start, end + step, step)]
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def patch_lcf_thickness(path: Path, thickness_um: float) -> None:
    thickness_m = max(float(thickness_um), 1e-6) * 1e-6
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    in_bond_record = False
    record_field = 0
    patched = 0
    for line in lines:
        if "polymer/solder microbump bonding layer" in line:
            in_bond_record = True
            record_field = 0
            out.append(line)
            continue

        stripped = line.strip()
        if in_bond_record and stripped and not stripped.startswith("#"):
            record_field += 1
        if in_bond_record and record_field == 6:
            out.append(f"{thickness_m:.9g}")
            in_bond_record = False
            patched += 1
            continue
        out.append(line)
    if patched == 0:
        raise ValueError(f"did not patch any bond layer thicknesses in {path}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def patch_hotspot_config(path: Path, r_convec: float) -> None:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("-r_convec "):
            lines.append(f"-r_convec {r_convec:.9g}")
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def patch_metadata(path: Path, profile: str, r_convec: float, thickness_um: float) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    area_mm2 = float(data["cooling_model"]["effective_sink_area_mm2"])
    h_w_per_um2_k = 1.0 / (float(r_convec) * area_mm2 * 1.0e6)
    data["cooling_model"]["profile"] = profile
    data["cooling_model"]["r_convec_k_per_w"] = float(r_convec)
    data["cooling_model"]["heat_transfer_coefficient_w_per_um2_k"] = h_w_per_um2_k
    data["cooling_model"]["heat_transfer_coefficient_w_per_m2_k"] = h_w_per_um2_k * 1.0e12
    bond = data["hotspot_material_assumptions"]["hbm_polymer_solder_bond"]
    bond["thickness_um"] = float(thickness_um)
    bond["thickness_m"] = float(thickness_um) * 1.0e-6
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def read_temperature_stats(package_dir: Path) -> Dict[str, float]:
    trace = package_dir / "threedice_temperature.ttrace"
    with trace.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = [[float(item) for item in line.split()] for line in infile if line.strip()]
    temps_c = np.asarray(rows, dtype=float) - 273.15
    logic_idx = [idx for idx, name in enumerate(names) if name.startswith("core_") or name.startswith("logic")]
    dram_idx = [idx for idx, name in enumerate(names) if name.startswith("dram")]
    logic = temps_c[:, logic_idx]
    dram = temps_c[:, dram_idx]
    return {
        "logic_peak_c": float(np.max(logic)),
        "logic_avg_c": float(np.mean(logic)),
        "dram_peak_c": float(np.max(dram)),
        "dram_avg_c": float(np.mean(dram)),
    }


def first_bond_height_um(package_dir: Path) -> float:
    path = package_dir / "threedice_stack.stk"
    lines = path.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("layer LAYER_TYPE_L1"):
            for next_line in lines[idx + 1:idx + 6]:
                stripped = next_line.strip()
                if stripped.startswith("height "):
                    return float(stripped.split()[1])
    return float("nan")


def run_case(task: Dict[str, object]) -> Dict[str, object]:
    package_dir = Path(task["package_dir"])
    base_dir = Path(task["base_dir"])
    profile = str(task["profile"])
    r_convec = float(task["r_convec"])
    thickness_um = float(task["thickness_um"])
    threedice_bin = str(task["threedice_bin"])
    timeout_s = float(task["timeout_s"])
    cores_per_job = int(task["cores_per_job"])

    os.environ["TSIM_THREEDICE_CORES"] = str(cores_per_job)
    os.environ["OMP_NUM_THREADS"] = str(cores_per_job)
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    if package_dir.exists():
        shutil.rmtree(package_dir)
    shutil.copytree(base_dir, package_dir)
    patch_lcf_thickness(package_dir / "stack.lcf", thickness_um)
    patch_hotspot_config(package_dir / "hotspot.config", r_convec)
    patch_metadata(package_dir / "metadata.json", profile, r_convec, thickness_um)
    result = run_threedice(package_dir, threedice_bin, timeout_s, heatmap_count=0)
    stats = read_temperature_stats(package_dir)
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "cooling_profile": profile,
        "r_convec_k_per_w": r_convec,
        "underfill_thickness_um": thickness_um,
        "actual_bond_height_um": first_bond_height_um(package_dir),
        "ambient_c": float(task["ambient_c"]),
        "h_w_per_m2_k": metadata["cooling_model"]["heat_transfer_coefficient_w_per_m2_k"],
        "threedice_runtime_s": result.runtime_s,
        "package_dir": str(package_dir),
        **stats,
    }


def plot_results(csv_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    colors = {"pcie_air": "#d62728", "sxm_air": "#ff7f0e", "high_perf_liquid": "#1f77b4"}
    labels = {"pcie_air": "PCIe air", "sxm_air": "SXM air", "high_perf_liquid": "High-perf liquid"}
    panels = [
        ("logic_peak_c", "Logic Peak"),
        ("logic_avg_c", "Logic Average"),
        ("dram_peak_c", "DRAM Peak"),
        ("dram_avg_c", "DRAM Average"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (metric, title) in zip(axes.flat, panels):
        for profile, _r in COOLING_PROFILES:
            selected = [row for row in rows if row["cooling_profile"] == profile]
            selected.sort(key=lambda row: float(row["underfill_thickness_um"]))
            ax.plot(
                [float(row["underfill_thickness_um"]) for row in selected],
                [float(row[metric]) for row in selected],
                marker="o",
                markersize=3.5,
                linewidth=2,
                color=colors[profile],
                label=labels[profile],
            )
        ax.set_title(title)
        ax.set_ylabel("Temperature (C)")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Underfill thickness per layer (um)")
    fig.suptitle("Stacked Underfill Thickness Ablation: Full Prefill, 35 C Boundary")
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    total_threads = max(1, int(args.jobs)) * max(1, int(args.cores_per_job))
    log(f"parallelism: jobs={args.jobs}, cores_per_job={args.cores_per_job}, total_threads={total_threads}")
    os.environ.setdefault("TSIM_THREEDICE_CORES", str(args.cores_per_job))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.cores_per_job))
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

    base_dir = args.out_dir / "base_package"
    cfg = TraceConfig(
        ambient_c=args.ambient_c,
        max_bins=args.bins,
        major_op_samples=0,
        dram_layers=args.dram_layers,
        dram_floorplan_granularity="bank",
        dram_bank_mapping="address_trace",
        logic_floorplan="intra_core",
        hotspot_grid=args.hotspot_grid,
        logic_direct_to_heatsink=False,
    )
    log("building shared power trace")
    trace = build_component_power_trace(artifact, cfg)
    log("exporting base package")
    export_trace_package(artifact, trace, cfg, base_dir, write_visualizations=False)

    thicknesses = parse_thicknesses(args.thicknesses_um)
    tasks: List[Dict[str, object]] = []
    for profile, r_convec in COOLING_PROFILES:
        for thickness_um in thicknesses:
            tag = f"{profile}_uf{thickness_um:g}um".replace(".", "p")
            package_dir = args.out_dir / "packages" / tag
            tasks.append({
                "profile": profile,
                "r_convec": r_convec,
                "thickness_um": thickness_um,
                "ambient_c": args.ambient_c,
                "base_dir": str(base_dir),
                "package_dir": str(package_dir),
                "threedice_bin": args.threedice_bin,
                "timeout_s": args.timeout_s,
                "cores_per_job": args.cores_per_job,
            })

    rows: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
        future_to_task = {executor.submit(run_case, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            row = future.result()
            rows.append(row)
            log(
                f"done {task['profile']} {task['thickness_um']:g}um: "
                f"logic_peak={row['logic_peak_c']:.2f}C, bond={row['actual_bond_height_um']:.1f}um"
            )

    rows.sort(key=lambda row: (
        str(row["cooling_profile"]),
        float(row["underfill_thickness_um"]),
    ))

    csv_path = args.out_dir / "underfill_ablation.csv"
    fieldnames = [
        "cooling_profile",
        "r_convec_k_per_w",
        "underfill_thickness_um",
        "actual_bond_height_um",
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

    png_path = args.out_dir / "underfill_ablation_temperature_vs_thickness.png"
    plot_results(csv_path, png_path)
    print(json.dumps({"csv": str(csv_path), "plot": str(png_path), "points": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
