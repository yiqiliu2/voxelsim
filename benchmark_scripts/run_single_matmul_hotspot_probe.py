#!/usr/bin/env python3
"""Run a one-matmul HotSpot probe and emit time-sliced heatmaps."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.artifacts import RunArtifacts, RunSummary, load_artifacts
from tsim_thermal.defaults import DEFAULT_AMBIENT_C
from tsim_thermal.hotspot import run_hotspot
from tsim_thermal.models import ThermalConfig
from tsim_thermal.operator_analysis import write_operator_hotspot_analysis
from tsim_thermal.trace import TraceConfig, build_component_power_trace, export_trace_package
from tsim_thermal.visualization import FlpBlock, read_flp, write_layout_visualizations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and run a single-large-matmul HotSpot package.")
    parser.add_argument("--results-dir", type=Path, default=SRC_DIR / "results" / "logs")
    parser.add_argument("--out-dir", type=Path, default=SRC_DIR / "results" / "single_matmul_hotspot_probe")
    parser.add_argument("--models", default="", help="Comma-separated model filter. Empty searches all models.")
    parser.add_argument("--modes", default="prefill,decode")
    parser.add_argument("--impls", default="best")
    parser.add_argument("--op-index", type=int, default=None,
                        help="Use a specific operator index after filtering instead of the largest SA-energy matmul.")
    parser.add_argument("--npu-freq-mhz", type=float, default=1500.0)
    parser.add_argument("--max-bins", type=int, default=160)
    parser.add_argument("--major-op-samples", type=int, default=32)
    parser.add_argument("--target-duration-s", type=float, default=0.0,
                        help="Stretch the selected matmul to this runtime and scale dynamic energy to preserve stage power.")
    parser.add_argument("--hotspot-grid", type=int, default=64)
    parser.add_argument("--hotspot-bin", default=str(SRC_DIR.parent / "external" / "hotspot-7.0" / "hotspot"))
    parser.add_argument("--hotspot-timeout-s", type=float, default=1800.0)
    parser.add_argument("--time-samples", type=int, default=12,
                        help="Number of time positions to visualize within the single operator.")
    parser.add_argument("--layers", default="logic,dram0,dram7",
                        help="Comma-separated layer names to visualize, or 'all'.")
    parser.add_argument("--dram-layers", type=int, default=8)
    parser.add_argument("--dram-capacity-gb", type=float, default=192.0)
    parser.add_argument("--hbm-package-capacity-gb", type=float, default=16.0)
    parser.add_argument("--hbm-package-area-mm2", type=float, default=87.62745402745404)
    parser.add_argument("--hbm-banks-per-package", type=int, default=16)
    parser.add_argument("--hbm-interleave-stripe-bytes", type=int, default=256)
    parser.add_argument("--dram-bank-mapping",
                        choices=("address_trace", "hbm_interleave", "hbm-interleave",
                                 "fine_interleave", "fine-interleave", "bank_interleave", "bank-interleave",
                                 "from_impl", "uniform", "interleave_size", "software_aware"),
                        default="address_trace")
    parser.add_argument("--noc-power-backend", choices=("tsim_simple", "simple", "dsent", "orion"), default="tsim_simple")
    parser.add_argument("--noc-power-flit-bits", type=int, default=64)
    parser.add_argument("--noc-power-injection-rate", type=float, default=0.3)
    parser.add_argument("--noc-power-link-length-mm", type=float, default=1.0)
    parser.add_argument("--noc-power-dsent-tech", default="TG11LVT")
    parser.add_argument(
        "--die-size-mm",
        type=float,
        default=None,
        help="Optional square logic die side; omitted by default so thermal export derives it from simulator logic area.",
    )
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def select_matmul(artifacts: List[RunArtifacts], op_index: int | None) -> Tuple[RunArtifacts, int, object]:
    candidates = []
    for artifact in artifacts:
        for idx, op in enumerate(artifact.op_logs):
            if getattr(op, "mm_dur", 0) > 0 and getattr(op, "energy_sa", 0.0) > 0:
                score = float(getattr(op, "energy_sa", 0.0))
                candidates.append((score, artifact, idx, op))
    if not candidates:
        raise SystemExit("No matmul operators with SA energy were found.")
    if op_index is not None:
        indexed = [(artifact, idx, op) for _score, artifact, idx, op in candidates if idx == op_index]
        if not indexed:
            raise SystemExit(f"No filtered artifact contains matmul op index {op_index}.")
        artifact, idx, op = indexed[0]
        return artifact, idx, op
    _score, artifact, idx, op = max(candidates, key=lambda item: item[0])
    return artifact, idx, op


def normalize_op(op: object) -> object:
    cloned = copy.deepcopy(op)
    origin = int(getattr(cloned, "t_dram_ld_start", 0))
    time_attrs = (
        "t_dram_ld_start",
        "t_bcast_start",
        "t_comp_shift_start",
        "t_reduce_start",
        "t_dram_st_start",
        "t_finish",
        "t_exec_enter",
    )
    for attr in time_attrs:
        if hasattr(cloned, attr):
            setattr(cloned, attr, max(0, int(getattr(cloned, attr)) - origin))
    cloned.exec_dur = int(getattr(cloned, "t_finish", 0)) - int(getattr(cloned, "t_dram_ld_start", 0))
    for mapping_name in ("start_times", "end_times"):
        mapping = getattr(cloned, mapping_name, None)
        if isinstance(mapping, dict):
            for key, value in list(mapping.items()):
                mapping[key] = max(0, int(value) - origin)
    return cloned


def scale_op_runtime_and_energy(op: object, target_duration_s: float, npu_freq_mhz: float) -> Tuple[object, float]:
    if target_duration_s <= 0:
        return op, 1.0
    target_cycles = max(1, int(round(target_duration_s * npu_freq_mhz * 1e6)))
    current_cycles = max(1, int(getattr(op, "t_finish", 1)) - int(getattr(op, "t_dram_ld_start", 0)))
    scale = target_cycles / current_cycles
    cloned = copy.deepcopy(op)

    time_attrs = (
        "t_dram_ld_start",
        "t_bcast_start",
        "t_comp_shift_start",
        "t_reduce_start",
        "t_dram_st_start",
        "t_finish",
        "t_exec_enter",
    )
    duration_attrs = (
        "dram_ld_dur",
        "bcast_dur",
        "mm_dur",
        "ew_dur",
        "sram_r_dur",
        "sram_w_dur",
        "shift_dur",
        "comp_dur",
        "comp_sh_dur",
        "reduce_dur",
        "dram_st_dur",
        "exec_dur",
    )
    energy_attrs = (
        "energy_total",
        "energy_compute",
        "energy_sss",
        "energy_sa",
        "energy_vu",
        "energy_noc",
        "energy_sram",
        "energy_dram",
        "energy_tsv",
    )
    for attr in time_attrs:
        if hasattr(cloned, attr):
            setattr(cloned, attr, int(round(int(getattr(cloned, attr)) * scale)))
    for attr in duration_attrs:
        if hasattr(cloned, attr):
            setattr(cloned, attr, max(0, int(round(int(getattr(cloned, attr)) * scale))))
    for attr in energy_attrs:
        if hasattr(cloned, attr):
            setattr(cloned, attr, float(getattr(cloned, attr)) * scale)

    if hasattr(cloned, "t_dram_ld_start"):
        cloned.t_dram_ld_start = 0
    cloned.t_finish = target_cycles
    cloned.exec_dur = target_cycles
    if hasattr(cloned, "power_W"):
        cloned.power_W = (float(getattr(cloned, "energy_total", 0.0)) / 1e12) / (target_cycles / (npu_freq_mhz * 1e6))

    for mapping_name in ("start_times", "end_times"):
        mapping = getattr(cloned, mapping_name, None)
        if isinstance(mapping, dict):
            for key, value in list(mapping.items()):
                mapping[key] = int(round(int(value) * scale))
            if mapping_name == "end_times":
                for key, value in list(mapping.items()):
                    mapping[key] = min(target_cycles, value)
    return cloned, scale


def single_op_artifact(artifact: RunArtifacts, op_index: int, op: object, npu_freq_mhz: float) -> RunArtifacts:
    runtime_s = max(1e-15, int(getattr(op, "t_finish", 1)) / (npu_freq_mhz * 1e6))
    dynamic_energy_mj = float(getattr(op, "energy_total", 0.0)) / 1e9
    static_power_w = artifact.summary.static_power_w
    static_energy_mj = static_power_w * runtime_s * 1e3
    dynamic_power_w = (dynamic_energy_mj / 1e3) / runtime_s
    total_power_w = static_power_w + dynamic_power_w
    run_id = replace(artifact.run_id, model=f"{artifact.run_id.model}-single-matmul")
    summary = RunSummary(
        exec_cycles=max(1, int(getattr(op, "t_finish", 1))),
        total_energy_mj=static_energy_mj + dynamic_energy_mj,
        static_energy_mj=static_energy_mj,
        dynamic_energy_mj=dynamic_energy_mj,
        total_power_w=total_power_w,
        static_power_w=static_power_w,
        dram_static_power_w=artifact.summary.dram_static_power_w,
        logic_static_power_w=artifact.summary.logic_static_power_w,
        dynamic_power_w=dynamic_power_w,
        static_component_w=dict(artifact.summary.static_component_w),
    )
    return RunArtifacts(
        run_id=run_id,
        log_path=artifact.log_path,
        pickle_path=artifact.pickle_path,
        summary=summary,
        op_logs=[op],
    )


def read_trace_matrix(path: Path, kelvin_to_c: bool) -> Tuple[List[str], np.ndarray]:
    with path.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = []
        for line in infile:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != len(names):
                continue
            rows.append([float(item) for item in parts])
    matrix = np.asarray(rows, dtype=float)
    if kelvin_to_c:
        matrix -= 273.15
    return names, matrix


def active_layers(package_dir: Path) -> Dict[str, List[FlpBlock]]:
    layers = {"logic": read_flp(package_dir / "logic.flp")}
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers[path.stem] = read_flp(path)
    return layers


def choose_bins(n_bins: int, count: int) -> List[int]:
    if n_bins <= 0 or count <= 0:
        return []
    count = min(count, n_bins)
    return sorted({int(round(item)) for item in np.linspace(0, n_bins - 1, count)})


def layer_indices(block_names: List[str], blocks: Iterable[FlpBlock]) -> List[int]:
    index = {name: idx for idx, name in enumerate(block_names)}
    return [index[block.name] for block in blocks if block.name in index]


def fixed_ranges(
    block_names: List[str],
    temps_c: np.ndarray,
    power_density: np.ndarray,
    layers: Dict[str, List[FlpBlock]],
    selected_bins: List[int],
) -> Dict[str, Tuple[float, float, float, float]]:
    ranges = {}
    for name, blocks in layers.items():
        idxs = layer_indices(block_names, blocks)
        if not idxs:
            continue
        temp_values = temps_c[np.ix_(selected_bins, idxs)]
        pd_values = power_density[np.ix_(selected_bins, idxs)]
        ranges[name] = (
            float(np.nanmin(temp_values)),
            float(np.nanmax(temp_values)),
            float(np.nanmin(pd_values)),
            float(np.nanmax(pd_values)),
        )
    return ranges


def draw_axis(ax, blocks: List[FlpBlock], values: Dict[str, float], vmin: float, vmax: float, title: str, cmap_name: str, rectangle_cls, plt) -> None:
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(cmap_name)
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    for block in blocks:
        value = values.get(block.name, vmin)
        ax.add_patch(
            rectangle_cls(
                (block.x_mm, block.y_mm),
                block.width_mm,
                block.height_mm,
                facecolor=cmap(norm(value)),
                edgecolor="#222222",
                linewidth=0.12,
            )
        )
    width = max(block.right_mm for block in blocks)
    height = max(block.top_mm for block in blocks)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nscale {vmin:.2f} to {vmax:.2f}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = ax.figure.colorbar(scalar, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title)


def write_time_heatmaps(
    package_dir: Path,
    out_dir: Path,
    requested_layers: List[str] | str,
    sample_count: int,
    dt_s: float,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    block_names, temps_c = read_trace_matrix(package_dir / "temperature.ttrace", kelvin_to_c=True)
    power_names, power_w = read_trace_matrix(package_dir / "power.ptrace", kelvin_to_c=False)
    if block_names != power_names:
        raise ValueError("temperature.ttrace and power.ptrace headers differ")
    layers = active_layers(package_dir)
    block_lookup = {block.name: block for blocks in layers.values() for block in blocks}
    block_areas = np.asarray([
        max(1e-12, block_lookup[name].width_mm * block_lookup[name].height_mm)
        for name in block_names
    ])
    power_density = power_w / block_areas[None, :]
    selected_bins = choose_bins(temps_c.shape[0], sample_count)
    if requested_layers != "all":
        keep = set(requested_layers)
        layers = {name: blocks for name, blocks in layers.items() if name in keep}
    ranges = fixed_ranges(block_names, temps_c, power_density, layers, selected_bins)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "time_heatmap_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "sample_index", "bin", "time_s", "layer", "png",
            "max_temp_c", "max_temp_block", "max_power_density_w_per_mm2", "max_power_density_block",
        ])
        index = {name: idx for idx, name in enumerate(block_names)}
        for sample_idx, bin_idx in enumerate(selected_bins):
            for layer_name, blocks in layers.items():
                idxs = layer_indices(block_names, blocks)
                if not idxs:
                    continue
                temp_values = {block.name: float(temps_c[bin_idx, index[block.name]]) for block in blocks if block.name in index}
                pd_values = {block.name: float(power_density[bin_idx, index[block.name]]) for block in blocks if block.name in index}
                temp_block = max(temp_values, key=temp_values.get)
                pd_block = max(pd_values, key=pd_values.get)
                temp_min, temp_max, pd_min, pd_max = ranges[layer_name]
                fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
                draw_axis(axes[0], blocks, temp_values, temp_min, temp_max, "Temperature (C)", "inferno", Rectangle, plt)
                draw_axis(axes[1], blocks, pd_values, pd_min, pd_max, "Power density (W/mm^2)", "viridis", Rectangle, plt)
                fig.suptitle(f"{layer_name} bin {bin_idx} t={bin_idx * dt_s * 1e6:.2f} us")
                fig.tight_layout()
                png = out_dir / f"t{sample_idx:02d}_bin{bin_idx:04d}_{layer_name}.png"
                fig.savefig(png, dpi=180, bbox_inches="tight")
                plt.close(fig)
                writer.writerow([
                    sample_idx,
                    bin_idx,
                    bin_idx * dt_s,
                    layer_name,
                    str(png),
                    temp_values[temp_block],
                    temp_block,
                    pd_values[pd_block],
                    pd_block,
                ])
    plt.close("all")
    return manifest


def main() -> int:
    args = parse_args()
    artifacts = load_artifacts(args.results_dir, args.models, args.modes, args.impls, args.npu_freq_mhz)
    if not artifacts:
        print(f"No analyzable TSIM logs found under {args.results_dir}", file=sys.stderr)
        return 1
    source_artifact, source_op_index, source_op = select_matmul(artifacts, args.op_index)
    op = normalize_op(source_op)
    op, duration_scale = scale_op_runtime_and_energy(op, args.target_duration_s, args.npu_freq_mhz)
    artifact = single_op_artifact(source_artifact, source_op_index, op, args.npu_freq_mhz)

    exec_s = max(1e-15, artifact.summary.exec_cycles / (args.npu_freq_mhz * 1e6))
    trace_cfg = TraceConfig(
        npu_freq_mhz=args.npu_freq_mhz,
        duration_ms=exec_s * 1e3,
        max_bins=args.max_bins,
        major_op_samples=args.major_op_samples,
        major_op_percentile=100.0,
        grid=4,
        spatial_policy="uniform",
        die_size_mm=args.die_size_mm,
        ambient_c=DEFAULT_AMBIENT_C,
        dram_layers=args.dram_layers,
        dram_capacity_mb=int(round(args.dram_capacity_gb * 1024)),
        hbm_package_capacity_mb=int(round(args.hbm_package_capacity_gb * 1024)),
        hbm_package_area_mm2=args.hbm_package_area_mm2,
        hbm_banks_per_package=args.hbm_banks_per_package,
        hbm_interleave_stripe_bytes=args.hbm_interleave_stripe_bytes,
        dram_floorplan_granularity="bank",
        dram_bank_mapping=args.dram_bank_mapping,
        logic_floorplan="intra_core",
        hotspot_grid=args.hotspot_grid,
        noc_power_backend=args.noc_power_backend,
        noc_power_flit_bits=args.noc_power_flit_bits,
        noc_power_injection_rate=args.noc_power_injection_rate,
        noc_power_link_length_mm=args.noc_power_link_length_mm,
        noc_power_dsent_tech=args.noc_power_dsent_tech,
    )
    thermal_cfg = ThermalConfig()
    trace = build_component_power_trace(artifact, trace_cfg)

    package_dir = args.out_dir / "package"
    package_paths = export_trace_package(artifact, trace, trace_cfg, package_dir)
    write_layout_visualizations(package_dir)
    hs = run_hotspot(package_dir, args.hotspot_bin, args.hotspot_timeout_s)
    (package_dir / "hotspot_run.json").write_text(
        json.dumps({
            "status": "missing" if not hs.available else ("ok" if hs.returncode == 0 else f"failed:{hs.returncode}"),
            "peak_c": hs.peak_c,
            "stdout": hs.stdout,
            "stderr": hs.stderr,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    if not hs.available or hs.returncode != 0:
        print((package_dir / "hotspot_run.json").read_text(encoding="utf-8"))
        return 2

    write_operator_hotspot_analysis(artifact, trace, trace_cfg, thermal_cfg, package_dir, heatmap_count=1)
    layers: List[str] | str = "all" if args.layers.strip().lower() == "all" else split_csv(args.layers)
    time_manifest = write_time_heatmaps(
        package_dir,
        package_dir / "operator_time_heatmaps",
        layers,
        args.time_samples,
        trace.dt_s,
    )

    summary = {
        "source_run": source_artifact.run_id.__dict__,
        "source_op_index": source_op_index,
        "source_op_id": int(getattr(source_op, "op_id", source_op_index)),
        "single_op_exec_cycles": artifact.summary.exec_cycles,
        "duration_scale": duration_scale,
        "target_duration_s": args.target_duration_s,
        "single_op_exec_us": exec_s * 1e6,
        "single_op_mm_cycles": int(getattr(op, "mm_dur", 0)),
        "single_op_mm_us": int(getattr(op, "mm_dur", 0)) / (args.npu_freq_mhz * 1e6) * 1e6,
        "energy_total_pj": float(getattr(op, "energy_total", 0.0)),
        "energy_sa_pj": float(getattr(op, "energy_sa", 0.0)),
        "trace_bins": int(trace.component_power_w.shape[0]),
        "trace_dt_us": trace.dt_s * 1e6,
        "hotspot_peak_c": hs.peak_c,
        "package_dir": str(package_dir),
        "power_trace": str(package_paths["ptrace"]),
        "time_heatmap_manifest": str(time_manifest),
    }
    summary_path = args.out_dir / "single_matmul_probe_summary.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
