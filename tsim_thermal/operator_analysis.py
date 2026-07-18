"""Per-operator HotSpot post-processing and heatmap generation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .artifacts import RunArtifacts
from .models import ThermalConfig, simulate_lumped, throttle_slowdown
from .trace import PowerTrace, TraceConfig
from .visualization import FlpBlock, read_flp


@dataclass(frozen=True)
class LayerBlocks:
    name: str
    flp_path: Path
    blocks: List[FlpBlock]


def write_operator_hotspot_analysis(
    artifact: RunArtifacts,
    trace: PowerTrace,
    trace_cfg: TraceConfig,
    thermal_cfg: ThermalConfig,
    package_dir: Path,
    heatmap_count: int = 12,
) -> Dict[str, Path]:
    ttrace_path = package_dir / "temperature.ttrace"
    ptrace_path = package_dir / "power.ptrace"
    if not ttrace_path.exists() or not ptrace_path.exists():
        return {}

    temp_names, temps_c = _read_trace_matrix(ttrace_path, kelvin_to_c=True)
    power_names, power_w = _read_trace_matrix(ptrace_path, kelvin_to_c=False)
    if temp_names != power_names:
        raise ValueError("temperature.ttrace and power.ptrace headers differ")

    layers = _read_active_layers(package_dir)
    block_lookup: Dict[str, Tuple[str, FlpBlock]] = {}
    for layer in layers:
        for block in layer.blocks:
            block_lookup[block.name] = (layer.name, block)
    block_areas = np.asarray([
        max(1e-12, block_lookup[name][1].width_mm * block_lookup[name][1].height_mm)
        for name in temp_names
    ])
    power_density = power_w / block_areas[None, :]

    summary_path = package_dir / "operator_hotspot_summary.csv"
    simple_temp_c = simulate_lumped(trace, thermal_cfg)
    rows = _operator_summary_rows(
        artifact,
        trace,
        trace_cfg,
        thermal_cfg,
        simple_temp_c,
        temp_names,
        temps_c,
        power_density,
        block_lookup,
    )
    with open(summary_path, "w", newline="", encoding="utf-8") as outfile:
        fieldnames = list(rows[0].keys()) if rows else [
            "op_index", "op_id", "start_cycle", "end_cycle", "runtime_s",
            "max_temp_c", "max_temp_block", "max_temp_layer",
            "max_power_density_w_per_mm2", "max_power_density_block",
            "max_power_density_layer", "hotspot_slowdown", "simple_slowdown",
            "additional_slowdown_pct",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    unit_power_path = package_dir / "operator_unit_power_density.csv"
    _write_unit_power_density(unit_power_path, artifact, trace, trace_cfg, temp_names, power_density)

    heatmap_dir = package_dir / "operator_heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    selected = _selected_operator_indices(artifact, heatmap_count)
    heatmap_paths = _write_operator_heatmaps(
        heatmap_dir,
        selected,
        artifact,
        trace,
        trace_cfg,
        temp_names,
        temps_c,
        power_density,
        layers,
    )

    return {
        "operator_hotspot_summary": summary_path,
        "operator_unit_power_density": unit_power_path,
        "operator_heatmap_dir": heatmap_dir,
        **{f"operator_heatmap_{idx}": path for idx, path in enumerate(heatmap_paths)},
    }


def _read_trace_matrix(path: Path, kelvin_to_c: bool) -> Tuple[List[str], np.ndarray]:
    with open(path, encoding="utf-8") as infile:
        header = next(infile).split()
        rows = [[float(item) for item in line.split()] for line in infile if line.strip()]
    matrix = np.asarray(rows, dtype=float)
    if kelvin_to_c:
        matrix = matrix - 273.15
    return header, matrix


def _read_active_layers(package_dir: Path) -> List[LayerBlocks]:
    layers = [LayerBlocks("logic", package_dir / "logic.flp", read_flp(package_dir / "logic.flp"))]
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers.append(LayerBlocks(path.stem, path, read_flp(path)))
    return layers


def _operator_window_bins(op: object, trace: PowerTrace, trace_cfg: TraceConfig) -> Tuple[int, int]:
    start_cycle = int(getattr(op, "t_dram_ld_start", 0))
    end_cycle = int(getattr(op, "t_finish", start_cycle))
    cycle_to_s = 1.0 / (float(trace_cfg.npu_freq_mhz) * 1e6)
    start_s = start_cycle * cycle_to_s
    end_s = max(start_s + trace.dt_s, end_cycle * cycle_to_s)
    start_bin = max(0, min(trace.component_power_w.shape[0] - 1, int(np.floor(start_s / trace.dt_s))))
    end_bin = max(start_bin + 1, min(trace.component_power_w.shape[0], int(np.ceil(end_s / trace.dt_s))))
    return start_bin, end_bin


def _operator_summary_rows(
    artifact: RunArtifacts,
    trace: PowerTrace,
    trace_cfg: TraceConfig,
    thermal_cfg: ThermalConfig,
    simple_temp_c: np.ndarray,
    block_names: List[str],
    temps_c: np.ndarray,
    power_density: np.ndarray,
    block_lookup: Dict[str, Tuple[str, FlpBlock]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for op_index, op in enumerate(artifact.op_logs):
        start_bin, end_bin = _operator_window_bins(op, trace, trace_cfg)
        op_temps = temps_c[start_bin:end_bin]
        op_pd = power_density[start_bin:end_bin]
        if op_temps.size == 0:
            continue
        temp_flat = int(np.argmax(op_temps))
        temp_row, temp_col = np.unravel_index(temp_flat, op_temps.shape)
        pd_flat = int(np.argmax(op_pd))
        pd_row, pd_col = np.unravel_index(pd_flat, op_pd.shape)
        temp_block = block_names[temp_col]
        pd_block = block_names[pd_col]
        temp_layer = block_lookup[temp_block][0]
        pd_layer = block_lookup[pd_block][0]
        max_temp_c = float(op_temps[temp_row, temp_col])
        hotspot_slowdown = throttle_slowdown(max_temp_c, thermal_cfg.throttle_c)
        simple_window = simple_temp_c[start_bin:end_bin]
        simple_peak_c = float(simple_window.max()) if simple_window.size else float(simple_temp_c.max())
        simple_slowdown = throttle_slowdown(simple_peak_c, thermal_cfg.throttle_c)
        start_cycle = int(getattr(op, "t_dram_ld_start", 0))
        end_cycle = int(getattr(op, "t_finish", start_cycle))
        runtime_s = max(0.0, (end_cycle - start_cycle) / (float(trace_cfg.npu_freq_mhz) * 1e6))
        rows.append({
            "op_index": op_index,
            "op_id": int(getattr(op, "op_id", op_index)),
            "start_cycle": start_cycle,
            "end_cycle": end_cycle,
            "runtime_s": runtime_s,
            "start_bin": start_bin,
            "end_bin": end_bin,
            "energy_pj": float(getattr(op, "energy_total", 0.0)),
            "energy_sa_pj": float(getattr(op, "energy_sa", 0.0)),
            "energy_dram_pj": float(getattr(op, "energy_dram", 0.0)),
            "dram_r_bytes": int(getattr(op, "dram_r_bytes", 0)),
            "dram_w_bytes": int(getattr(op, "dram_w_bytes", 0)),
            "max_temp_c": max_temp_c,
            "max_temp_block": temp_block,
            "max_temp_layer": temp_layer,
            "max_temp_time_s": (start_bin + temp_row) * trace.dt_s,
            "max_power_density_w_per_mm2": float(op_pd[pd_row, pd_col]),
            "max_power_density_block": pd_block,
            "max_power_density_layer": pd_layer,
            "max_power_density_time_s": (start_bin + pd_row) * trace.dt_s,
            "hotspot_slowdown": hotspot_slowdown,
            "simple_peak_c": simple_peak_c,
            "simple_slowdown": simple_slowdown,
            "additional_slowdown_pct": (hotspot_slowdown / simple_slowdown - 1.0) * 100.0,
        })
    return rows


def _write_unit_power_density(
    path: Path,
    artifact: RunArtifacts,
    trace: PowerTrace,
    trace_cfg: TraceConfig,
    block_names: List[str],
    power_density: np.ndarray,
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["op_index", "op_id", "block", "max_power_density_w_per_mm2", "avg_power_density_w_per_mm2"])
        for op_index, op in enumerate(artifact.op_logs):
            start_bin, end_bin = _operator_window_bins(op, trace, trace_cfg)
            window = power_density[start_bin:end_bin]
            if window.size == 0:
                continue
            max_pd = window.max(axis=0)
            avg_pd = window.mean(axis=0)
            for block, max_value, avg_value in zip(block_names, max_pd, avg_pd):
                if max_value <= 0 and avg_value <= 0:
                    continue
                writer.writerow([op_index, int(getattr(op, "op_id", op_index)), block, float(max_value), float(avg_value)])


def _selected_operator_indices(artifact: RunArtifacts, count: int) -> List[int]:
    count = max(0, int(count))
    if count == 0:
        return []
    scored = [
        (
            float(getattr(op, "energy_total", 0.0))
            + float(getattr(op, "energy_sa", 0.0))
            + float(getattr(op, "energy_dram", 0.0)),
            idx,
        )
        for idx, op in enumerate(artifact.op_logs)
    ]
    return [idx for _score, idx in sorted(scored, reverse=True)[:count]]


def _write_operator_heatmaps(
    heatmap_dir: Path,
    selected_indices: List[int],
    artifact: RunArtifacts,
    trace: PowerTrace,
    trace_cfg: TraceConfig,
    block_names: List[str],
    temps_c: np.ndarray,
    power_density: np.ndarray,
    layers: List[LayerBlocks],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    block_index = {name: idx for idx, name in enumerate(block_names)}
    paths: List[Path] = []
    for op_index in selected_indices:
        op = artifact.op_logs[op_index]
        start_bin, end_bin = _operator_window_bins(op, trace, trace_cfg)
        temp_window = temps_c[start_bin:end_bin]
        pd_window = power_density[start_bin:end_bin]
        if temp_window.size == 0:
            continue
        for layer in layers:
            layer_indices = [block_index[block.name] for block in layer.blocks if block.name in block_index]
            if not layer_indices:
                continue
            layer_temp = {block_names[idx]: float(temp_window[:, idx].max()) for idx in layer_indices}
            layer_pd = {block_names[idx]: float(pd_window[:, idx].max()) for idx in layer_indices}
            out = heatmap_dir / f"op{op_index:04d}_{layer.name}_thermal.png"
            _draw_layer_heatmap(layer, layer_temp, layer_pd, out, Rectangle, plt)
            paths.append(out)
    plt.close("all")
    return paths


def _draw_layer_heatmap(layer: LayerBlocks, temps: Dict[str, float], power_density: Dict[str, float], path: Path, rectangle_cls, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    _draw_metric_axis(axes[0], layer, temps, "Max temperature (C)", "inferno", rectangle_cls, plt)
    _draw_metric_axis(axes[1], layer, power_density, "Max power density (W/mm^2)", "viridis", rectangle_cls, plt)
    fig.suptitle(layer.name)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_metric_axis(ax, layer: LayerBlocks, values: Dict[str, float], title: str, cmap_name: str, rectangle_cls, plt) -> None:
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(cmap_name)
    finite_values = [value for value in values.values() if np.isfinite(value)]
    vmin = min(finite_values) if finite_values else 0.0
    vmax = max(finite_values) if finite_values else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    for block in layer.blocks:
        value = values.get(block.name, vmin)
        ax.add_patch(
            rectangle_cls(
                (block.x_mm, block.y_mm),
                block.width_mm,
                block.height_mm,
                facecolor=cmap(norm(value)),
                edgecolor="#222222",
                linewidth=0.2,
            )
        )
    width = max(block.right_mm for block in layer.blocks)
    height = max(block.top_mm for block in layer.blocks)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nmin {vmin:.2f}, max {vmax:.2f}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = ax.figure.colorbar(scalar, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title)
