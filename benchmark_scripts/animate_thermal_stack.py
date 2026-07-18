#!/usr/bin/env python3
"""Create an animated GIF from an exported TSIM thermal package."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.visualization import FlpBlock, read_flp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate layer heatmaps from a TSIM thermal package.")
    parser.add_argument("package_dir", type=Path, help="Directory containing FLPs and a temperature or power trace.")
    parser.add_argument("--quantity", choices=("temperature", "power", "power-density"), default="temperature",
                        help="Trace quantity to animate. Default: temperature.")
    parser.add_argument("--trace", type=Path, default=None,
                        help="Trace path. Defaults to temperature traces for temperature and power.ptrace otherwise.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output GIF path. Defaults to <package>/thermal_stack.gif.")
    parser.add_argument("--layers", default="logic,dram0,dram3,dram7",
                        help="Comma-separated layer names to show, or 'all'.")
    parser.add_argument("--samples", type=int, default=16, help="Number of time samples in the GIF.")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--cmap", default="inferno")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--logic-region-stats", action="store_true",
                        help="Annotate each frame with logic average and SA/VU/SRAM/NoC/TSV power density.")
    parser.add_argument("--stats-out", type=Path, default=None,
                        help="Optional CSV path for per-frame logic-region power-density statistics.")
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def choose_trace(package_dir: Path, trace_arg: Path | None, quantity: str) -> Path:
    if trace_arg is not None:
        return trace_arg if trace_arg.is_absolute() else package_dir / trace_arg
    if quantity in {"power", "power-density"}:
        path = package_dir / "power.ptrace"
        if path.exists():
            return path
        raise FileNotFoundError(f"No power.ptrace found in {package_dir}")
    for name in ("adaptive_temperature.ttrace", "temperature.ttrace"):
        path = package_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No adaptive_temperature.ttrace or temperature.ttrace found in {package_dir}")


def read_trace(path: Path, quantity: str) -> Tuple[List[str], np.ndarray]:
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
    if quantity == "temperature" and matrix.size and float(np.nanmedian(matrix)) > 200.0:
        matrix -= 273.15
    return names, matrix


def read_dt_s(package_dir: Path) -> float:
    metadata = package_dir / "metadata.json"
    if not metadata.exists():
        return 0.0
    data = json.loads(metadata.read_text(encoding="utf-8"))
    return float(data.get("dt_s") or 0.0)


def read_component_power(package_dir: Path) -> List[Dict[str, float]]:
    path = package_dir / "component_power.csv"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append({key: float(value) for key, value in row.items() if value != ""})
    return rows


def active_layers(package_dir: Path) -> Dict[str, List[FlpBlock]]:
    layers = {"logic": read_flp(package_dir / "logic.flp")}
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers[path.stem] = read_flp(path)
    return layers


def choose_bins(n_bins: int, count: int) -> List[int]:
    if n_bins <= 0:
        return []
    count = max(1, min(int(count), n_bins))
    return sorted({int(round(item)) for item in np.linspace(0, n_bins - 1, count)})


def layer_indices(block_names: List[str], blocks: Iterable[FlpBlock]) -> List[int]:
    index = {name: idx for idx, name in enumerate(block_names)}
    return [index[block.name] for block in blocks if block.name in index]


def apply_quantity_transform(
    block_names: List[str],
    matrix: np.ndarray,
    layers: Dict[str, List[FlpBlock]],
    quantity: str,
) -> np.ndarray:
    if quantity != "power-density":
        return matrix
    block_lookup = {block.name: block for blocks in layers.values() for block in blocks}
    areas = np.asarray([
        max(1e-12, block_lookup[name].width_mm * block_lookup[name].height_mm)
        for name in block_names
    ])
    return matrix / areas[None, :]


def quantity_labels(quantity: str) -> Tuple[str, str, str]:
    if quantity == "temperature":
        return "Temperature", "C", "TSIM 3D Thermal Stack"
    if quantity == "power":
        return "Power", "W", "TSIM 3D Power Trace"
    return "Power density", "W/mm^2", "TSIM 3D Power-Density Trace"


def logic_region_stats(
    block_names: List[str],
    power_w: np.ndarray,
    layers: Dict[str, List[FlpBlock]],
    bin_idx: int,
    component_power_rows: List[Dict[str, float]],
) -> Dict[str, float]:
    logic_blocks = layers.get("logic", [])
    index = {name: idx for idx, name in enumerate(block_names)}
    groups = {
        "sram": [],
        "sa": [],
        "vu": [],
        "router": [],
        "tsv": [],
    }
    for block in logic_blocks:
        if block.name.startswith("tsv_") and block.name in index:
            groups["tsv"].append(block)
            continue
        if block.name not in index or not block.name.startswith("core_"):
            continue
        unit = block.name.rsplit("_", 1)[-1]
        if unit in groups:
            groups[unit].append(block)

    def density(blocks: List[FlpBlock]) -> float:
        area = sum(max(1e-12, block.width_mm * block.height_mm) for block in blocks)
        if area <= 0:
            return 0.0
        watts = sum(float(power_w[bin_idx, index[block.name]]) for block in blocks if block.name in index)
        return watts / area

    active_blocks = [block for blocks in groups.values() for block in blocks]
    stats = {
        "logic_avg_w_per_mm2": density(active_blocks),
        "sram_w_per_mm2": density(groups["sram"]),
        "sa_w_per_mm2": density(groups["sa"]),
        "vu_w_per_mm2": density(groups["vu"]),
        "router_w_per_mm2": density(groups["router"]),
        "tsv_region_w_per_mm2": density(groups["tsv"]),
    }
    router_area = sum(max(1e-12, block.width_mm * block.height_mm) for block in groups["router"])
    tsv_area = sum(max(1e-12, block.width_mm * block.height_mm) for block in groups["tsv"])
    component_row = component_power_rows[bin_idx] if bin_idx < len(component_power_rows) else {}
    if router_area > 0:
        stats["noc_w_per_mm2"] = float(component_row.get("noc", 0.0)) / router_area
    else:
        stats["noc_w_per_mm2"] = 0.0
    if tsv_area > 0:
        stats["tsv_w_per_mm2"] = float(component_row.get("tsv", 0.0)) / tsv_area
    else:
        stats["tsv_w_per_mm2"] = 0.0
    return stats


def format_logic_stats(stats: Dict[str, float]) -> str:
    return (
        f"Logic avg: {stats['logic_avg_w_per_mm2']:.3f} W/mm^2\n"
        f"SA: {stats['sa_w_per_mm2']:.3f}   "
        f"VU: {stats['vu_w_per_mm2']:.3f}   "
        f"SRAM: {stats['sram_w_per_mm2']:.3f}   "
        f"NoC: {stats['noc_w_per_mm2']:.3f}   "
        f"TSV: {stats['tsv_w_per_mm2']:.3f}"
    )


def draw_layer(ax, blocks: List[FlpBlock], values: Dict[str, float], vmin: float, vmax: float, cmap, norm) -> None:
    from matplotlib.patches import Rectangle

    for block in blocks:
        value = values.get(block.name, vmin)
        ax.add_patch(
            Rectangle(
                (block.x_mm, block.y_mm),
                block.width_mm,
                block.height_mm,
                facecolor=cmap(norm(value)),
                edgecolor="#1f1f1f",
                linewidth=0.08,
            )
        )
    width = max(block.right_mm for block in blocks)
    height = max(block.top_mm for block in blocks)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])


def render_frame(
    block_names: List[str],
    matrix: np.ndarray,
    layers: Dict[str, List[FlpBlock]],
    bin_idx: int,
    dt_s: float,
    vmin: float,
    vmax: float,
    cmap_name: str,
    dpi: int,
    quantity: str,
    logic_stats_text: str | None = None,
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    n_layers = len(layers)
    cols = min(4, n_layers)
    rows = int(np.ceil(n_layers / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False)
    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-9))
    index = {name: idx for idx, name in enumerate(block_names)}
    quantity_name, unit, title = quantity_labels(quantity)

    for ax in axes.flat:
        ax.axis("off")
    for ax, (layer_name, blocks) in zip(axes.flat, layers.items()):
        values = {
            block.name: float(matrix[bin_idx, index[block.name]])
            for block in blocks
            if block.name in index
        }
        draw_layer(ax, blocks, values, vmin, vmax, cmap, norm)
        layer_max = max(values.values()) if values else float("nan")
        ax.set_title(f"{layer_name} max {layer_max:.2f} {unit}", fontsize=10)

    time_text = f"bin {bin_idx}"
    if dt_s > 0:
        time_text += f"   t={bin_idx * dt_s * 1e6:.2f} us"
    fig.suptitle(f"{title}  {time_text}", fontsize=14)
    if logic_stats_text:
        fig.text(
            0.5,
            0.925,
            logic_stats_text,
            ha="center",
            va="top",
            fontsize=12,
            fontfamily="monospace",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#333333",
                "alpha": 0.96,
            },
        )
        fig.subplots_adjust(top=0.82, bottom=0.04)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=list(axes.flat), fraction=0.024, pad=0.02)
    cbar.set_label(f"{quantity_name} ({unit})")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    adaptive_palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE", Image.ADAPTIVE)
    return Image.open(buf).convert("P", palette=adaptive_palette)


def main() -> int:
    args = parse_args()
    package_dir = args.package_dir.resolve()
    trace_path = choose_trace(package_dir, args.trace, args.quantity)
    out_path = (args.out if args.out is not None else package_dir / "thermal_stack.gif")
    if not out_path.is_absolute():
        out_path = package_dir / out_path

    block_names, matrix = read_trace(trace_path, args.quantity)
    if matrix.size == 0:
        raise SystemExit(f"No trace rows in {trace_path}")
    layers = active_layers(package_dir)
    power_w = matrix
    matrix = apply_quantity_transform(block_names, matrix, layers, args.quantity)
    if args.layers.strip().lower() != "all":
        keep = split_csv(args.layers)
        layers = {name: layers[name] for name in keep if name in layers}
    if not layers:
        raise SystemExit("No requested layers matched the package floorplans.")

    selected_bins = choose_bins(matrix.shape[0], args.samples)
    selected_indices = []
    for blocks in layers.values():
        selected_indices.extend(layer_indices(block_names, blocks))
    if not selected_indices:
        raise SystemExit("Temperature trace columns do not match selected floorplan blocks.")
    selected_values = matrix[np.ix_(selected_bins, selected_indices)]
    vmin = float(args.vmin) if args.vmin is not None else float(np.nanmin(selected_values))
    vmax = float(args.vmax) if args.vmax is not None else float(np.nanmax(selected_values))
    dt_s = read_dt_s(package_dir)
    component_power_rows = read_component_power(package_dir) if args.logic_region_stats else []
    stats_by_bin: Dict[int, Dict[str, float]] = {}
    if args.logic_region_stats:
        for bin_idx in selected_bins:
            stats_by_bin[bin_idx] = logic_region_stats(block_names, power_w, active_layers(package_dir), bin_idx, component_power_rows)

    frames = [
        render_frame(
            block_names,
            matrix,
            layers,
            bin_idx,
            dt_s,
            vmin,
            vmax,
            args.cmap,
            args.dpi,
            args.quantity,
            format_logic_stats(stats_by_bin[bin_idx]) if bin_idx in stats_by_bin else None,
        )
        for bin_idx in selected_bins
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(20, int(round(1000.0 / max(args.fps, 0.1))))
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    stats_path = None
    if args.logic_region_stats:
        stats_path = args.stats_out
        if stats_path is None:
            stats_path = out_path.with_suffix(".logic_region_stats.csv")
        if not stats_path.is_absolute():
            stats_path = package_dir / stats_path
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", newline="", encoding="utf-8") as outfile:
            fieldnames = [
                "sample_index",
                "bin",
                "time_s",
                "logic_avg_w_per_mm2",
                "sa_w_per_mm2",
                "vu_w_per_mm2",
                "sram_w_per_mm2",
                "router_w_per_mm2",
                "noc_w_per_mm2",
                "tsv_w_per_mm2",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for sample_index, bin_idx in enumerate(selected_bins):
                row = {
                    "sample_index": sample_index,
                    "bin": bin_idx,
                    "time_s": bin_idx * dt_s,
                    **stats_by_bin[bin_idx],
                }
                writer.writerow(row)
    print(json.dumps({
        "gif": str(out_path),
        "trace": str(trace_path),
        "quantity": args.quantity,
        "frames": len(frames),
        "layers": list(layers),
        "vmin": vmin,
        "vmax": vmax,
        "logic_region_stats": str(stats_path) if stats_path else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
