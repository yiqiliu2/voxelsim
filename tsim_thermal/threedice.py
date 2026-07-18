"""3D-ICE backend helpers for TSIM thermal packages."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .defaults import DEFAULT_AMBIENT_C, DEFAULT_R_CONVEC_K_PER_W
from .visualization import FlpBlock, StackLayer, floorplan_bounds, read_flp, read_lcf


@dataclass
class ThreeDICEResult:
    available: bool
    returncode: int | None
    stdout: str
    stderr: str
    model_path: Path | None
    output_path: Path | None
    peak_c: float | None = None
    temperature_trace: Path | None = None
    heatmap_dir: Path | None = None
    runtime_s: float | None = None
    num_cores: int | None = None


def find_threedice(binary: str = "3D-ICE") -> str | None:
    if "/" in binary:
        path = Path(binary)
        return str(path.resolve()) if path.exists() else None
    return shutil.which(binary)


def run_threedice(
    package_dir: Path,
    binary: str = "3D-ICE",
    timeout_s: float = 300.0,
    heatmap_count: int | None = None,
    heatmap_layers: List[str] | str | None = None,
) -> ThreeDICEResult:
    model_path = export_threedice_stack(package_dir)
    exe = find_threedice(binary)
    if exe is None:
        return ThreeDICEResult(False, None, "", f"3D-ICE binary not found: {binary}", model_path, None)

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    num_cores = int(os.environ.get("TSIM_THREEDICE_CORES", "48"))
    output_path = package_dir / "threedice_stdout.txt"
    cmd = [exe, model_path.name]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(num_cores))
    env.setdefault("OMP_PROC_BIND", "close")
    env.setdefault("OMP_PLACES", "cores")
    try:
        result = subprocess.run(
            cmd,
            cwd=package_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return ThreeDICEResult(
            True,
            124,
            stdout,
            stderr + f"\nTimed out after {timeout_s} seconds.",
            model_path,
            None,
            num_cores=num_cores,
        )

    clean_stdout = _sanitize_stdout(result.stdout)
    output_path.write_text(clean_stdout, encoding="utf-8")
    trace_path = None
    heatmap_dir = None
    peak_c = None
    max_trace_path = None
    if heatmap_count is None:
        heatmap_count = max(0, int(os.environ.get("TSIM_THREEDICE_HEATMAP_COUNT", "4")))
    if isinstance(heatmap_layers, str):
        heatmap_layers = [item.strip() for item in heatmap_layers.split(",") if item.strip()]
    if heatmap_layers is None:
        heatmap_layers = [
            item.strip()
            for item in os.environ.get("TSIM_THREEDICE_HEATMAP_LAYERS", "").split(",")
            if item.strip()
        ]
    if result.returncode == 0:
        names, temps_k = parse_tflp_outputs(package_dir, quantity="average")
        max_names, max_temps_k = parse_tflp_outputs(package_dir, quantity="maximum")
        if temps_k.size:
            trace_path = package_dir / "threedice_temperature.ttrace"
            _write_temperature_trace(trace_path, names, temps_k)
            peak_source_k = max_temps_k if max_temps_k.size else temps_k
            peak_c = float(np.max(peak_source_k) - 273.15)
            if max_temps_k.size:
                max_trace_path = package_dir / "threedice_temperature_max.ttrace"
                _write_temperature_trace(max_trace_path, max_names, max_temps_k)
            heatmap_dir = write_threedice_heatmaps(
                package_dir,
                names,
                temps_k - 273.15,
                _read_power_trace(package_dir / "power.ptrace")[1],
                float(metadata.get("dt_s") or 0.0),
                heatmap_count=heatmap_count,
                layer_filter=heatmap_layers or None,
            )
            if max_temps_k.size and heatmap_count > 0:
                write_threedice_heatmaps(
                    package_dir,
                    max_names,
                    max_temps_k - 273.15,
                    _read_power_trace(package_dir / "power.ptrace")[1],
                    float(metadata.get("dt_s") or 0.0),
                    heatmap_count=heatmap_count,
                    layer_filter=heatmap_layers or None,
                    heatmap_dir_name="threedice_heatmaps_max",
                )

    summary = {
        "status": "ok" if result.returncode == 0 else f"failed:{result.returncode}",
        "returncode": result.returncode,
        "peak_c": peak_c,
        "num_cores": num_cores,
        "model_path": str(model_path),
        "stdout_path": str(output_path),
        "temperature_trace": str(trace_path) if trace_path else None,
        "temperature_max_trace": str(max_trace_path) if max_trace_path else None,
        "heatmap_dir": str(heatmap_dir) if heatmap_dir else None,
        "runtime_s": _parse_runtime_s(clean_stdout),
    }
    (package_dir / "threedice_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return ThreeDICEResult(
        True,
        result.returncode,
        clean_stdout,
        result.stderr,
        model_path,
        output_path,
        peak_c,
        trace_path,
        heatmap_dir,
        summary["runtime_s"],
        num_cores,
    )


def export_threedice_stack(package_dir: Path) -> Path:
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    block_names, power_w = _read_power_trace(package_dir / "power.ptrace")
    power_by_name = {name: power_w[:, idx] for idx, name in enumerate(block_names)}
    layers = read_lcf(package_dir / "stack.lcf")
    logic_blocks = read_flp(package_dir / "logic.flp")
    outline_w_mm, outline_h_mm = floorplan_bounds(logic_blocks)
    for layer in layers:
        outline_w_mm = max(outline_w_mm, floorplan_bounds(read_flp(package_dir / layer.floorplan))[0])
        outline_h_mm = max(outline_h_mm, floorplan_bounds(read_flp(package_dir / layer.floorplan))[1])

    written_flps: Dict[str, str] = {}
    for layer in layers:
        if not layer.power:
            continue
        source = package_dir / layer.floorplan
        out_name = f"threedice_{source.name}"
        write_threedice_floorplan(source, package_dir / out_name, power_by_name)
        written_flps[layer.floorplan] = out_name

    ambient_k = float(metadata.get("ambient_c", DEFAULT_AMBIENT_C)) + 273.15
    dt_s = float(metadata.get("dt_s") or 1e-6)
    num_cores = int(os.environ.get("TSIM_THREEDICE_CORES", "48"))
    die_discr_x = max(1, int(os.environ.get("TSIM_THREEDICE_DIE_DISCR_X", "1")))
    die_discr_y = max(1, int(os.environ.get("TSIM_THREEDICE_DIE_DISCR_Y", "1")))
    bond_discr = _bond_discretization(metadata, outline_w_mm, outline_h_mm)
    cell_w_um = max(1.0, outline_w_mm * 1000.0 / max(16.0, float(metadata.get("hotspot_grid") or 128)))
    cell_h_um = max(1.0, outline_h_mm * 1000.0 / max(16.0, float(metadata.get("hotspot_grid") or 128)))
    htc = _heat_transfer_coefficient_um2(metadata, outline_w_mm, outline_h_mm)

    lines: List[str] = [
        "material SILICON_TSIM :",
        "   thermal conductivity     1.00e-4 ;",
        "   volumetric heat capacity 1.75e-12 ;",
        "",
        "material BOND_TSIM :",
        "   thermal conductivity     2.00e-6 ;",
        "   volumetric heat capacity 4.00e-12 ;",
        "",
        "top heat sink :",
        f"   heat transfer coefficient {htc:.8e} ;",
        f"   temperature               {ambient_k:.6f} ;",
        "",
        "dimensions :",
        f"   chip length {outline_w_mm * 1000.0:.6f}, width {outline_h_mm * 1000.0:.6f} ;",
        f"   cell length  {cell_w_um:.6f}, width  {cell_h_um:.6f} ;",
        "   non-uniform true;",
        "",
    ]

    passive_layer_defs: List[str] = []
    die_defs: List[str] = []
    stack_entries: Dict[int, str] = {}
    output_lines: List[str] = ["output:"]
    for layer in layers:
        material = "SILICON_TSIM" if layer.power else "BOND_TSIM"
        if layer.power:
            source_um = max(0.5, min(2.0, layer.thickness_um * 0.5))
            body_um = max(0.5, layer.thickness_um - source_um)
            die_name = f"DIE_TYPE_L{layer.index}"
            instance = _layer_instance_name(layer)
            die_defs.extend([
                f"die {die_name} :",
                f"   source {source_um:.6f} {material} ;",
                f"   layer  {body_um:.6f} {material} ;",
                "",
            ])
            stack_entries[layer.index] = (
                f'die {instance} {die_name} floorplan "./{written_flps[layer.floorplan]}" '
                f"discretization {die_discr_x} {die_discr_y} ;"
            )
            output_lines.append(
                f'Tflp( {instance}, "threedice_{instance}_tflp.txt", average, slot );'
            )
            output_lines.append(
                f'Tflp( {instance}, "threedice_{instance}_tflp_max.txt", maximum, slot );'
            )
        else:
            layer_name = f"LAYER_TYPE_L{layer.index}"
            instance = _layer_instance_name(layer)
            passive_layer_defs.extend([
                f"layer {layer_name} :",
                f"   height {layer.thickness_um:.6f} ;",
                f"   material {material} ;",
                "",
            ])
            stack_entries[layer.index] = (
                f"layer {instance} {layer_name} discretization {bond_discr[0]} {bond_discr[1]} ;"
            )

    lines.extend(passive_layer_defs)
    lines.extend(die_defs)
    cooling = metadata.get("cooling_model", {})
    logic_direct_to_heatsink = bool(cooling.get("logic_direct_to_heatsink", False))
    stack_order = layers if logic_direct_to_heatsink else list(reversed(layers))
    stack_lines: List[str] = ["stack:"]
    for layer in stack_order:
        stack_lines.append(stack_entries[layer.index])
    lines.extend(stack_lines)
    lines.extend([
        "",
        "solver:",
        f"   transient step {dt_s:.12g}, slot {dt_s:.12g} ;",
        f"   initial temperature {ambient_k:.6f} ;",
        f"   numofcores {num_cores} ;",
        "",
    ])
    lines.extend(output_lines)
    lines.append("")

    stk = package_dir / "threedice_stack.stk"
    stk.write_text("\n".join(lines), encoding="utf-8")
    return stk


def write_threedice_floorplan(source_flp: Path, out_flp: Path, power_by_name: Dict[str, np.ndarray]) -> None:
    blocks = read_flp(source_flp)
    n_steps = max((len(values) for values in power_by_name.values()), default=1)
    zeros = np.zeros(n_steps, dtype=float)
    lines: List[str] = []
    for block in blocks:
        values = power_by_name.get(block.name, zeros)
        lines.extend([
            f"{block.name} :",
            "",
            f"  position       {block.x_mm * 1000.0:.6f}, {block.y_mm * 1000.0:.6f} ;",
            f"  dimension      {block.width_mm * 1000.0:.6f}, {block.height_mm * 1000.0:.6f} ;",
            "",
            "  power values " + ", ".join(f"{float(value):.8f}" for value in values) + " ;",
            "",
        ])
    out_flp.write_text("\n".join(lines), encoding="utf-8")


def parse_tflp_outputs(package_dir: Path, quantity: str = "average") -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    rows_by_layer: List[np.ndarray] = []
    layers = [layer for layer in read_lcf(package_dir / "stack.lcf") if layer.power]
    suffix = "" if quantity == "average" else "_max"
    for layer in layers:
        instance = _layer_instance_name(layer)
        path = package_dir / f"threedice_{instance}_tflp{suffix}.txt"
        if not path.exists():
            continue
        layer_names, values = _read_tflp(path)
        names.extend(layer_names)
        rows_by_layer.append(values)
    if not rows_by_layer:
        return [], np.empty((0, 0), dtype=float)
    min_rows = min(values.shape[0] for values in rows_by_layer)
    joined = np.concatenate([values[:min_rows] for values in rows_by_layer], axis=1)
    return names, joined


def write_threedice_heatmaps(
    package_dir: Path,
    names: List[str],
    temps_c: np.ndarray,
    power_w: np.ndarray,
    dt_s: float,
    heatmap_count: int,
    layer_filter: List[str] | None = None,
    heatmap_dir_name: str = "threedice_heatmaps",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    layers = _read_active_layers(package_dir)
    if layer_filter:
        keep = set(layer_filter)
        layers = [(name, blocks) for name, blocks in layers if name in keep]
    index = {name: idx for idx, name in enumerate(names)}
    areas = np.ones(len(names), dtype=float)
    for layer_name, blocks in layers:
        for block in blocks:
            if block.name in index:
                areas[index[block.name]] = max(1e-9, block.width_mm * block.height_mm)
    power_density = power_w[: temps_c.shape[0], : len(names)] / areas[None, :]

    heatmap_dir = package_dir / heatmap_dir_name
    heatmap_dir.mkdir(exist_ok=True)
    if temps_c.shape[0] == 0 or heatmap_count <= 0 or not layers:
        return heatmap_dir
    count = max(1, min(int(heatmap_count), temps_c.shape[0]))
    bins = sorted(set(int(idx) for idx in np.linspace(0, temps_c.shape[0] - 1, count))
                  | {int(np.argmax(np.max(temps_c, axis=1)))})
    for bin_idx in bins:
        for layer_name, blocks in layers:
            layer_indices = [index[block.name] for block in blocks if block.name in index]
            if not layer_indices:
                continue
            temp_values = {names[idx]: float(temps_c[bin_idx, idx]) for idx in layer_indices}
            pd_values = {names[idx]: float(power_density[bin_idx, idx]) for idx in layer_indices}
            out = heatmap_dir / f"bin{bin_idx:05d}_{layer_name}_3dice.png"
            _draw_heatmap_pair(
                blocks,
                temp_values,
                pd_values,
                out,
                f"{layer_name} t={bin_idx * dt_s * 1e6:.2f} us",
                Rectangle,
                plt,
            )
    plt.close("all")
    return heatmap_dir


def _read_power_trace(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = [[float(item) for item in line.split()] for line in infile if line.strip()]
    return names, np.asarray(rows, dtype=float)


def _read_tflp(path: Path) -> Tuple[List[str], np.ndarray]:
    names: List[str] = []
    rows: List[List[float]] = []
    with path.open(encoding="utf-8") as infile:
        for raw in infile:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("% Time"):
                parts = [part.strip() for part in line[1:].split("\t") if part.strip()]
                names = [part[:-3] if part.endswith("(K)") else part for part in parts[1:]]
                continue
            if line.startswith("%"):
                continue
            values = [float(item) for item in line.split()]
            if len(values) > 1:
                rows.append(values[1:])
    return names, np.asarray(rows, dtype=float)


def _write_temperature_trace(path: Path, names: List[str], temps_k: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as outfile:
        outfile.write(" ".join(names) + "\n")
        for row in temps_k:
            outfile.write(" ".join(f"{float(value):.6f}" for value in row) + "\n")


def _read_active_layers(package_dir: Path) -> List[Tuple[str, List[FlpBlock]]]:
    layers: List[Tuple[str, List[FlpBlock]]] = [("logic", read_flp(package_dir / "logic.flp"))]
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers.append((path.stem, read_flp(path)))
    return layers


def _layer_instance_name(layer: StackLayer) -> str:
    stem = Path(layer.floorplan).stem.upper()
    if layer.power:
        if stem == "LOGIC":
            return "LOGIC_DIE"
        return f"{stem.upper()}_DIE"
    return f"BOND_LAYER_{layer.index}"


def _heat_transfer_coefficient_um2(metadata: dict, outline_w_mm: float, outline_h_mm: float) -> float:
    sink_area_um2 = max(1.0, _effective_sink_area_mm2(metadata, outline_w_mm, outline_h_mm) * 1.0e6)
    cooling = metadata.get("cooling_model", {})
    r_convec_k_per_w = float(cooling.get("r_convec_k_per_w", DEFAULT_R_CONVEC_K_PER_W))
    return 1.0 / (r_convec_k_per_w * sink_area_um2)


def _effective_sink_area_mm2(metadata: dict, outline_w_mm: float, outline_h_mm: float) -> float:
    layout = metadata.get("hbm_package_layout", {})
    candidates: List[float] = []

    explicit_area_mm2 = layout.get("effective_sink_area_mm2")
    if explicit_area_mm2 is not None:
        area = float(explicit_area_mm2)
        if area > 0:
            candidates.append(area)

    logic_side_mm = metadata.get("die_size_mm")
    if logic_side_mm is not None:
        side = float(logic_side_mm)
        if side > 0:
            candidates.append(side * side)

    dram_w_mm = layout.get("footprint_width_mm")
    dram_h_mm = layout.get("footprint_height_mm")
    if dram_w_mm is not None and dram_h_mm is not None:
        width = float(dram_w_mm)
        height = float(dram_h_mm)
        if width > 0 and height > 0:
            candidates.append(width * height)

    if not candidates and outline_w_mm > 0 and outline_h_mm > 0:
        candidates.append(outline_w_mm * outline_h_mm)
    return max(candidates) if candidates else 1.0e-6


def _bond_discretization(metadata: dict, outline_w_mm: float, outline_h_mm: float) -> Tuple[int, int]:
    env_x = os.environ.get("TSIM_THREEDICE_BOND_DISCR_X")
    env_y = os.environ.get("TSIM_THREEDICE_BOND_DISCR_Y")
    if env_x and env_y:
        return max(1, int(env_x)), max(1, int(env_y))
    layout = metadata.get("hbm_package_layout", {})
    cols = max(1, int(layout.get("grid_cols") or 4))
    rows = max(1, int(layout.get("grid_rows") or 3))
    # Four cells per HBM package footprint side preserves package-scale vertical
    # conduction without exploding 3D-ICE's all-pairs non-uniform connection build.
    return max(4, cols * 4), max(4, rows * 4)


def _parse_runtime_s(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if "Emulation took" not in line:
            continue
        parts = line.replace("sec", "").split()
        for idx, part in enumerate(parts):
            if part == "took" and idx + 1 < len(parts):
                try:
                    return float(parts[idx + 1])
                except ValueError:
                    return None
    return None


def _sanitize_stdout(stdout: str) -> str:
    warning = "Connection length/width of cells between layers < EPSILON"
    count = stdout.count(warning)
    if count == 0:
        return stdout
    return stdout.replace(warning, "") + f"\n[TSIM filtered {count} repeated 3D-ICE EPSILON overlap warnings]\n"


def _draw_heatmap_pair(blocks: List[FlpBlock], temps: Dict[str, float], power_density: Dict[str, float],
                       path: Path, title: str, rectangle_cls, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    _draw_metric_axis(axes[0], blocks, temps, "Temperature (C)", "inferno", rectangle_cls, plt)
    _draw_metric_axis(axes[1], blocks, power_density, "Power density (W/mm^2)", "viridis", rectangle_cls, plt)
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
                linewidth=0.12,
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
