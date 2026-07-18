#!/usr/bin/env python3
"""Run a focused TSIM -> 3D-ICE thermal probe for one artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.artifacts import REPO_SRC, load_artifacts
from tsim_thermal.defaults import DEFAULT_AMBIENT_C, DEFAULT_BOND_THICKNESS_UM, DEFAULT_R_CONVEC_K_PER_W
from tsim_thermal.threedice import run_threedice
from tsim_thermal.trace import TraceConfig, build_component_power_trace, export_trace_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused 3D-ICE backend probe.")
    parser.add_argument("--results-dir", type=Path, default=REPO_SRC / "results" / "logs")
    parser.add_argument("--out-dir", type=Path, default=REPO_SRC / "results" / "thermal_3dice_probe")
    parser.add_argument("--model", default="llama2-13")
    parser.add_argument("--mode", default="prefill")
    parser.add_argument("--impl", default="best")
    parser.add_argument("--dram-bw", type=int, default=12288)
    parser.add_argument("--row", type=int, default=8192)
    parser.add_argument("--core-group", type=int, default=8)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--duration-ms", type=float, default=250.0)
    parser.add_argument("--ambient-c", type=float, default=DEFAULT_AMBIENT_C)
    parser.add_argument("--r-convec-k-per-w", type=float, default=DEFAULT_R_CONVEC_K_PER_W)
    parser.add_argument("--cooling-profile", default="")
    parser.add_argument("--dram-layers", type=int, default=8)
    parser.add_argument("--bond-thickness-um", type=float, default=DEFAULT_BOND_THICKNESS_UM)
    parser.add_argument(
        "--logic-direct-to-heatsink",
        action="store_true",
        help="What-if model: place the logic die on the top heat-sink side of the 3D-ICE stack.",
    )
    parser.add_argument("--threedice-bin", default=str(REPO_SRC.parent / "external" / "3d-ice-src" / "bin" / "3D-ICE-Emulator"))
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--heatmap-count", type=int, default=4)
    parser.add_argument("--heatmap-layers", default="logic,dram0,dram3,dram7")
    parser.add_argument("--hotspot-grid", type=int, default=128)
    parser.add_argument("--skip-layout-png", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TSIM_THREEDICE_CORES", "48")
    os.environ.setdefault("OMP_NUM_THREADS", "48")
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")

    started = time.perf_counter()
    log("loading TSIM artifacts")
    artifacts = load_artifacts(
        args.results_dir,
        models=args.model,
        modes=args.mode,
        impls=args.impl,
    )
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

    cooling_profile = args.cooling_profile.strip()
    if not cooling_profile:
        cooling_profile = (
            "average_h100_sxm_air"
            if abs(float(args.r_convec_k_per_w) - 0.06) < 1e-12
            else "conservative_h100_pcie_dual_slot_air"
        )

    trace_cfg = TraceConfig(
        duration_ms=args.duration_ms,
        ambient_c=args.ambient_c,
        r_convec_k_per_w=args.r_convec_k_per_w,
        cooling_profile=cooling_profile,
        max_bins=args.bins,
        major_op_samples=0,
        dram_layers=args.dram_layers,
        dram_floorplan_granularity="bank",
        dram_bank_mapping="address_trace",
        logic_floorplan="intra_core",
        hotspot_grid=args.hotspot_grid,
        logic_direct_to_heatsink=args.logic_direct_to_heatsink,
        bond_thickness_um=args.bond_thickness_um,
    )

    log("building power trace")
    trace = build_component_power_trace(artifact, trace_cfg)
    safe_name = (
        f"{artifact.run_id.model}_{artifact.run_id.mode}_{artifact.run_id.impl}_"
        f"c{artifact.run_id.num_cores}_sa{artifact.run_id.sa}_bw{artifact.run_id.dram_bw}_"
        f"sram{artifact.run_id.sram_kb}_cg{artifact.run_id.core_group}_row{artifact.run_id.row}_3dice_probe"
    )
    package_dir = args.out_dir / "packages" / safe_name

    log(f"exporting package to {package_dir}")
    export_trace_package(
        artifact,
        trace,
        trace_cfg,
        package_dir,
        write_visualizations=not args.skip_layout_png,
    )

    log("running 3D-ICE")
    result = run_threedice(
        package_dir,
        args.threedice_bin,
        args.timeout_s,
        args.heatmap_count,
        args.heatmap_layers,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "status": "ok" if result.returncode == 0 else f"failed:{result.returncode}",
        "returncode": result.returncode,
        "available": result.available,
        "peak_c": result.peak_c,
        "runtime_s": result.runtime_s,
        "elapsed_s": elapsed,
        "num_cores": result.num_cores,
        "model_path": str(result.model_path) if result.model_path else None,
        "temperature_trace": str(result.temperature_trace) if result.temperature_trace else None,
        "heatmap_dir": str(result.heatmap_dir) if result.heatmap_dir else None,
    }
    (args.out_dir / "probe_summary.json").parent.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log(json.dumps(summary, indent=2))
    if result.stdout:
        log("3D-ICE stdout:")
        print(result.stdout, flush=True)
    if result.stderr:
        log("3D-ICE stderr:")
        print(result.stderr, flush=True)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
