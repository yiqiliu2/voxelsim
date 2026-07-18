#!/usr/bin/env python3
"""Compute component and stacked power-density summaries for thermal packages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.visualization import FlpBlock, floorplan_bounds, read_flp


COMPONENT_SUFFIXES = {
    "sram": "_sram",
    "sa": "_sa",
    "vu": "_vu",
    "router": "_router",
    "tsv": "_tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Thermal matrix output root.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--windows-us", default="0,500,1000,2000")
    return parser.parse_args()


def split_windows(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def block_area_mm2(block: FlpBlock) -> float:
    return block.width_mm * block.height_mm


def intersection_mm2(a: FlpBlock, b: FlpBlock) -> float:
    x0 = max(a.x_mm, b.x_mm)
    y0 = max(a.y_mm, b.y_mm)
    x1 = min(a.right_mm, b.right_mm)
    y1 = min(a.top_mm, b.top_mm)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def component_for_block(name: str) -> str | None:
    if name.startswith("logic_pad") or "_pad_" in name:
        return None
    for component, suffix in COMPONENT_SUFFIXES.items():
        if name.endswith(suffix):
            return component
    return None


def read_power(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8") as infile:
        names = next(infile).split()
    matrix = np.loadtxt(path, skiprows=1, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return names, matrix


def smooth_peak(series: np.ndarray, dt_s: float, window_us: int) -> float:
    if series.size == 0:
        return float("nan")
    if window_us <= 0 or dt_s <= 0:
        return float(np.max(series))
    samples = int(math.ceil((window_us * 1e-6) / dt_s))
    samples = max(1, samples)
    if samples >= series.size:
        return float(np.mean(series))
    kernel = np.ones(samples, dtype=float) / samples
    return float(np.max(np.convolve(series, kernel, mode="valid")))


def dram_overlap_weights(
    logic_blocks: list[FlpBlock],
    dram_blocks: list[FlpBlock],
    name_to_col: dict[str, int],
) -> dict[str, list[tuple[int, float]]]:
    weights: dict[str, list[tuple[int, float]]] = {block.name: [] for block in logic_blocks}
    for logic_block in logic_blocks:
        for dram_block in dram_blocks:
            col = name_to_col.get(dram_block.name)
            if col is None:
                continue
            overlap = intersection_mm2(logic_block, dram_block)
            if overlap <= 0:
                continue
            area = block_area_mm2(dram_block)
            if area > 0:
                weights[logic_block.name].append((col, overlap / area))
    return weights


def weighted_dram_series(matrix: np.ndarray, weights: list[tuple[int, float]]) -> np.ndarray:
    if not weights:
        return np.zeros(matrix.shape[0], dtype=float)
    cols = np.fromiter((col for col, _ in weights), dtype=int)
    factors = np.fromiter((weight for _, weight in weights), dtype=float)
    return matrix[:, cols] @ factors


def package_rows(package_dir: Path, windows_us: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    run = metadata["run"]
    dt_s = float(metadata.get("dt_s") or 0.0)

    names, matrix = read_power(package_dir / "power.ptrace")
    name_to_col = {name: idx for idx, name in enumerate(names)}
    logic_blocks_all = read_flp(package_dir / "logic.flp")
    logic_blocks = [block for block in logic_blocks_all if component_for_block(block.name)]
    dram_blocks: list[FlpBlock] = []
    for dram_path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        dram_blocks.extend(block for block in read_flp(dram_path) if "_pad_" not in block.name)

    overlap = dram_overlap_weights(logic_blocks, dram_blocks, name_to_col)
    block_records: list[dict[str, object]] = []
    component_power: dict[str, np.ndarray] = {}
    component_stacked_power: dict[str, np.ndarray] = {}
    component_area: dict[str, float] = {}
    component_block_peak: dict[str, dict[int, float]] = {
        component: {window: 0.0 for window in windows_us}
        for component in COMPONENT_SUFFIXES
    }

    for block in logic_blocks:
        component = component_for_block(block.name)
        if component is None:
            continue
        col = name_to_col.get(block.name)
        if col is None:
            continue
        area = block_area_mm2(block)
        if area <= 0:
            continue
        logic_power = matrix[:, col]
        stacked_power = logic_power + weighted_dram_series(matrix, overlap[block.name])
        logic_density = logic_power / area
        stacked_density = stacked_power / area

        component_power.setdefault(component, np.zeros(matrix.shape[0], dtype=float))
        component_stacked_power.setdefault(component, np.zeros(matrix.shape[0], dtype=float))
        component_power[component] += logic_power
        component_stacked_power[component] += stacked_power
        component_area[component] = component_area.get(component, 0.0) + area

        for window_us in windows_us:
            component_block_peak[component][window_us] = max(
                component_block_peak[component][window_us],
                smooth_peak(stacked_density, dt_s, window_us),
            )
            block_records.append(
                {
                    **run,
                    "package": package_dir.name,
                    "window_us": window_us,
                    "component": component,
                    "block": block.name,
                    "area_mm2": area,
                    "component_only_w_per_mm2": smooth_peak(logic_density, dt_s, window_us),
                    "stacked_w_per_mm2": smooth_peak(stacked_density, dt_s, window_us),
                }
            )

    total_width_mm, total_height_mm = floorplan_bounds(logic_blocks_all)
    die_area_mm2 = total_width_mm * total_height_mm
    active_area_mm2 = sum(block_area_mm2(block) for block in logic_blocks)
    all_power = matrix.sum(axis=1)
    active_logic_power = sum(component_power.values(), np.zeros(matrix.shape[0], dtype=float))
    active_stacked_power = sum(component_stacked_power.values(), np.zeros(matrix.shape[0], dtype=float))

    rows: list[dict[str, object]] = []
    for window_us in windows_us:
        for component in sorted(component_power):
            area = component_area[component]
            rows.append(
                {
                    **run,
                    "package": package_dir.name,
                    "window_us": window_us,
                    "component": component,
                    "area_mm2": area,
                    "component_only_group_w_per_mm2": smooth_peak(component_power[component] / area, dt_s, window_us),
                    "stacked_group_w_per_mm2": smooth_peak(component_stacked_power[component] / area, dt_s, window_us),
                    "stacked_block_peak_w_per_mm2": component_block_peak[component][window_us],
                    "threshold_w_per_mm2": 0.7,
                }
            )
        rows.append(
            {
                **run,
                "package": package_dir.name,
                "window_us": window_us,
                "component": "active_logic",
                "area_mm2": active_area_mm2,
                "component_only_group_w_per_mm2": smooth_peak(active_logic_power / active_area_mm2, dt_s, window_us),
                "stacked_group_w_per_mm2": smooth_peak(active_stacked_power / active_area_mm2, dt_s, window_us),
                "stacked_block_peak_w_per_mm2": "",
                "threshold_w_per_mm2": 0.7,
            }
        )
        rows.append(
            {
                **run,
                "package": package_dir.name,
                "window_us": window_us,
                "component": "full_stack_outline",
                "area_mm2": die_area_mm2,
                "component_only_group_w_per_mm2": "",
                "stacked_group_w_per_mm2": smooth_peak(all_power / die_area_mm2, dt_s, window_us),
                "stacked_block_peak_w_per_mm2": "",
                "threshold_w_per_mm2": 0.7,
            }
        )
    return rows, block_records


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.root
    windows_us = split_windows(args.windows_us)
    package_dirs = sorted(path for path in args.root.rglob("packages/*") if (path / "power.ptrace").exists())
    all_rows: list[dict[str, object]] = []
    all_block_rows: list[dict[str, object]] = []
    for package_dir in package_dirs:
        rows, block_rows = package_rows(package_dir, windows_us)
        all_rows.extend(rows)
        all_block_rows.extend(block_rows)
        print(f"processed {package_dir}", flush=True)
    write_csv(out_dir / "aggregate_power_density_smoothed.csv", all_rows)
    write_csv(out_dir / "aggregate_power_density_blocks.csv", all_block_rows)
    print(out_dir / "aggregate_power_density_smoothed.csv")
    print(out_dir / "aggregate_power_density_blocks.csv")


if __name__ == "__main__":
    main()
