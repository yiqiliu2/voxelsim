"""Internal non-uniform block-grid thermal backend.

This is not a replacement for 3D-ICE.  It is a lightweight TSIM-native
backend that uses the exported rectangular floorplan blocks as variable-size
thermal cells so we can exercise the non-uniform-grid path and visualization
workflow without requiring an external solver binary.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .visualization import FlpBlock, read_flp


@dataclass
class AdaptiveGridResult:
    status: str
    peak_c: float | None
    temperature_trace: Path | None
    summary_path: Path | None
    heatmap_dir: Path | None


@dataclass(frozen=True)
class LayerBlocks:
    name: str
    blocks: List[FlpBlock]


def run_adaptive_grid(
    package_dir: Path,
    ambient_c: float,
    dt_s: float,
    heatmap_count: int = 6,
) -> AdaptiveGridResult:
    ptrace = package_dir / "power.ptrace"
    if not ptrace.exists():
        return AdaptiveGridResult("missing_power_trace", None, None, None, None)

    names, power_w = _read_power_trace(ptrace)
    layers = _read_active_layers(package_dir)
    block_lookup: Dict[str, Tuple[str, FlpBlock]] = {}
    for layer in layers:
        for block in layer.blocks:
            block_lookup[block.name] = (layer.name, block)

    missing = [name for name in names if name not in block_lookup]
    if missing:
        summary = package_dir / "adaptive_grid_summary.json"
        summary.write_text(
            json.dumps({"status": "missing_floorplan_blocks", "missing": missing[:20]}, indent=2) + "\n",
            encoding="utf-8",
        )
        return AdaptiveGridResult("missing_floorplan_blocks", None, None, summary, None)

    areas_mm2 = np.asarray([
        max(1e-9, block_lookup[name][1].width_mm * block_lookup[name][1].height_mm)
        for name in names
    ], dtype=float)
    temps_c = _simulate_block_temperatures(power_w, areas_mm2, ambient_c, dt_s)
    peak_c = float(np.max(temps_c)) if temps_c.size else ambient_c

    ttrace = package_dir / "adaptive_temperature.ttrace"
    _write_temperature_trace(ttrace, names, temps_c)

    power_density = power_w / areas_mm2[None, :]
    heatmap_dir = package_dir / "adaptive_grid_heatmaps"
    heatmaps = _write_heatmaps(heatmap_dir, names, temps_c, power_density, layers, dt_s, heatmap_count)

    summary_path = package_dir / "adaptive_grid_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "peak_c": peak_c,
                "ambient_c": ambient_c,
                "dt_s": dt_s,
                "bins": int(power_w.shape[0]),
                "blocks": int(power_w.shape[1]),
                "min_cell_area_mm2": float(np.min(areas_mm2)),
                "max_cell_area_mm2": float(np.max(areas_mm2)),
                "temperature_trace": str(ttrace),
                "heatmaps": [str(path) for path in heatmaps],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return AdaptiveGridResult("ok", peak_c, ttrace, summary_path, heatmap_dir)


def _read_power_trace(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = [[float(item) for item in line.split()] for line in infile if line.strip()]
    return names, np.asarray(rows, dtype=float)


def _read_active_layers(package_dir: Path) -> List[LayerBlocks]:
    layers = [LayerBlocks("logic", read_flp(package_dir / "logic.flp"))]
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers.append(LayerBlocks(path.stem, read_flp(path)))
    return layers


def _simulate_block_temperatures(
    power_w: np.ndarray,
    areas_mm2: np.ndarray,
    ambient_c: float,
    dt_s: float,
) -> np.ndarray:
    if power_w.size == 0:
        return np.empty_like(power_w)
    # Variable-size cells: smaller blocks get higher thermal resistance and
    # lower capacitance, while large pads/packages respond more slowly.
    r_k_per_w = 1.15 / np.sqrt(np.maximum(areas_mm2, 1e-9))
    r_k_per_w = np.clip(r_k_per_w, 0.18, 12.0)
    c_j_per_k = np.clip(0.035 * areas_mm2, 0.002, 2.5)
    tau_s = np.maximum(r_k_per_w * c_j_per_k, max(dt_s, 1e-12) * 4.0)
    alpha = 1.0 - np.exp(-dt_s / tau_s)

    temps = np.empty_like(power_w)
    current = np.full(power_w.shape[1], ambient_c, dtype=float)
    for idx, row in enumerate(power_w):
        target = ambient_c + row * r_k_per_w
        current += alpha * (target - current)
        if current.size > 1:
            # Weak global package coupling avoids completely independent cells
            # while keeping this inexpensive and deterministic.
            mean_delta = float(np.mean(current - ambient_c))
            current += 0.015 * (ambient_c + mean_delta - current)
        temps[idx] = current
    return temps


def _write_temperature_trace(path: Path, names: List[str], temps_c: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        outfile.write(" ".join(names) + "\n")
        for row in temps_c:
            outfile.write(" ".join(f"{value + 273.15:.6f}" for value in row) + "\n")


def _write_heatmaps(
    heatmap_dir: Path,
    names: List[str],
    temps_c: np.ndarray,
    power_density: np.ndarray,
    layers: List[LayerBlocks],
    dt_s: float,
    heatmap_count: int,
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    heatmap_dir.mkdir(exist_ok=True)
    if temps_c.shape[0] == 0:
        return []
    count = max(1, min(int(heatmap_count), temps_c.shape[0]))
    peak_bins = sorted(
        set(int(idx) for idx in np.linspace(0, temps_c.shape[0] - 1, count))
        | {int(np.argmax(np.max(temps_c, axis=1)))}
    )
    block_index = {name: idx for idx, name in enumerate(names)}
    paths: List[Path] = []
    for bin_idx in peak_bins:
        for layer in layers:
            layer_indices = [block_index[block.name] for block in layer.blocks if block.name in block_index]
            if not layer_indices:
                continue
            layer_temps = {names[idx]: float(temps_c[bin_idx, idx]) for idx in layer_indices}
            layer_pd = {names[idx]: float(power_density[bin_idx, idx]) for idx in layer_indices}
            out = heatmap_dir / f"bin{bin_idx:05d}_{layer.name}_thermal.png"
            _draw_layer_heatmap(
                layer,
                layer_temps,
                layer_pd,
                out,
                f"{layer.name} t={bin_idx * dt_s * 1e6:.2f} us",
                Rectangle,
                plt,
            )
            paths.append(out)
    plt.close("all")
    return paths


def _draw_layer_heatmap(layer: LayerBlocks, temps: Dict[str, float], power_density: Dict[str, float],
                        path: Path, title: str, rectangle_cls, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    _draw_metric_axis(axes[0], layer.blocks, temps, "Temperature (C)", "inferno", rectangle_cls, plt)
    _draw_metric_axis(axes[1], layer.blocks, power_density, "Power density (W/mm^2)", "viridis", rectangle_cls, plt)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_metric_axis(ax, blocks: List[FlpBlock], values: Dict[str, float], title: str,
                      cmap_name: str, rectangle_cls, plt) -> None:
    finite_values = [value for value in values.values() if math.isfinite(value)]
    vmin = min(finite_values) if finite_values else 0.0
    vmax = max(finite_values) if finite_values else 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    cmap = plt.get_cmap(cmap_name)
    for block in blocks:
        value = values.get(block.name, vmin)
        norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        ax.add_patch(
            rectangle_cls(
                (block.x_mm, block.y_mm),
                block.width_mm,
                block.height_mm,
                facecolor=cmap(norm),
                edgecolor="#222222",
                linewidth=0.2,
            )
        )
    width = max(block.right_mm for block in blocks)
    height = max(block.top_mm for block in blocks)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}\nmin {vmin:.2f}, max {vmax:.2f}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
