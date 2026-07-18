#!/usr/bin/env python3
"""Visualize tensor-level DRAM access records as HBM bank traffic heatmaps."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.artifacts import RunArtifacts, load_artifacts
from tsim_thermal.trace import (
    TraceConfig,
    dram_mapping_record_events,
    hbm_package_layout,
    meta_from_op,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render per-tensor synthetic address-trace DRAM traffic over HBM bank floorplan."
    )
    parser.add_argument("--results-dir", type=Path, default=SRC_DIR / "results" / "logs")
    parser.add_argument("--out-dir", type=Path, default=SRC_DIR / "results" / "dram_bank_traffic")
    parser.add_argument("--model", default="llama2-13")
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--impl", default="best")
    parser.add_argument("--dram-bw", type=int, default=12288)
    parser.add_argument("--row", type=int, default=8192)
    parser.add_argument("--core-group", type=int, default=8)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--npu-freq-mhz", type=float, default=1500.0)
    parser.add_argument("--dram-capacity-gb", type=float, default=192.0)
    parser.add_argument("--hbm-package-capacity-gb", type=float, default=16.0)
    parser.add_argument("--hbm-package-area-mm2", type=float, default=87.62745402745404)
    parser.add_argument("--hbm-package-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--hbm-banks-per-package", type=int, default=16)
    parser.add_argument("--stripe-bytes", type=int, default=256)
    parser.add_argument("--bank-mapping",
                        choices=("address_trace", "hbm_interleave", "uniform", "interleave_size", "software_aware"),
                        default="address_trace")
    parser.add_argument("--die-size-mm", type=float, default=12.0)
    return parser.parse_args()


def select_artifact(args: argparse.Namespace) -> RunArtifacts:
    artifacts = load_artifacts(
        args.results_dir,
        models=args.model,
        modes=args.mode,
        impls=args.impl,
        npu_freq_mhz=args.npu_freq_mhz,
    )
    matches = [
        artifact
        for artifact in artifacts
        if artifact.run_id.dram_bw == args.dram_bw
        and artifact.run_id.row == args.row
        and artifact.run_id.core_group == args.core_group
    ]
    if not matches:
        raise SystemExit(
            "No artifact matched "
            f"model={args.model} mode={args.mode} impl={args.impl} "
            f"dram_bw={args.dram_bw} row={args.row} core_group={args.core_group}"
        )
    return matches[0]


def build_trace_config(args: argparse.Namespace) -> TraceConfig:
    return TraceConfig(
        npu_freq_mhz=args.npu_freq_mhz,
        die_size_mm=args.die_size_mm,
        dram_capacity_mb=int(round(args.dram_capacity_gb * 1024)),
        hbm_package_capacity_mb=int(round(args.hbm_package_capacity_gb * 1024)),
        hbm_package_area_mm2=args.hbm_package_area_mm2,
        hbm_package_aspect_ratio=args.hbm_package_aspect_ratio,
        hbm_banks_per_package=args.hbm_banks_per_package,
        hbm_interleave_stripe_bytes=args.stripe_bytes,
        dram_floorplan_granularity="bank",
        dram_bank_mapping=args.bank_mapping,
    )


def collect_events(artifact: RunArtifacts, cfg: TraceConfig, package_count: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for op_position, op in enumerate(artifact.op_logs):
        meta = meta_from_op(op)
        for stage in ("read", "write"):
            for event in dram_mapping_record_events(meta, stage, cfg, artifact, op, op_position, package_count):
                event["op_position"] = op_position
                event["op_id"] = int(getattr(op, "op_id", op_position))
                event["stage"] = stage
                events.append(event)
    return events


def bank_grid_dims(package_layout: Dict[str, Any], banks_per_package: int) -> Tuple[int, int, int, int]:
    bank_cols = max(1, int(math.ceil(math.sqrt(max(1, banks_per_package)))))
    bank_rows = max(1, int(math.ceil(max(1, banks_per_package) / bank_cols)))
    grid_rows = int(package_layout["rows"]) * bank_rows
    grid_cols = int(package_layout["cols"]) * bank_cols
    return bank_rows, bank_cols, grid_rows, grid_cols


def bank_to_grid(bank: int, package_layout: Dict[str, Any], banks_per_package: int) -> Tuple[int, int]:
    bank_rows, bank_cols, _grid_rows, _grid_cols = bank_grid_dims(package_layout, banks_per_package)
    package = int(bank) // banks_per_package
    in_package_bank = int(bank) % banks_per_package
    package_row, package_col = divmod(package, int(package_layout["cols"]))
    bank_row, bank_col = divmod(in_package_bank, bank_cols)
    return package_row * bank_rows + bank_row, package_col * bank_cols + bank_col


def event_matrix(
    events: List[Dict[str, Any]],
    package_layout: Dict[str, Any],
    banks_per_package: int,
    frame_count: int,
    end_cycle: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Dict[str, Any]]]:
    _bank_rows, _bank_cols, grid_rows, grid_cols = bank_grid_dims(package_layout, banks_per_package)
    frame_count = max(1, int(frame_count))
    end_cycle = max(1, int(end_cycle))
    edges = np.linspace(0, end_cycle, frame_count + 1)
    traffic = np.zeros((frame_count, grid_rows, grid_cols), dtype=float)
    frame_ops: List[Dict[str, Any]] = []

    for frame_idx in range(frame_count):
        frame_start = int(math.floor(edges[frame_idx]))
        frame_end = int(math.ceil(edges[frame_idx + 1]))
        frame_events = []
        op_bytes: Dict[Tuple[int, str], float] = defaultdict(float)

        for event in events:
            start = int(event.get("start_cycle", 0))
            duration = max(1, int(event.get("duration_cycle", 0)))
            end = start + duration
            overlap = max(0, min(frame_end, end) - max(frame_start, start))
            if overlap <= 0:
                continue
            frac = overlap / duration
            frame_events.append(event)
            op_key = (int(event.get("op_position", 0)), str(event.get("stage", "")))
            op_bytes[op_key] += float(event.get("total_bytes", 0.0)) * frac
            for bank, weight in event.get("bank_weights", {}).items():
                row, col = bank_to_grid(int(bank), package_layout, banks_per_package)
                traffic[frame_idx, row, col] += float(weight) * frac

        top = sorted(op_bytes.items(), key=lambda item: item[1], reverse=True)[:4]
        frame_ops.append({
            "frame": frame_idx,
            "cycle_start": frame_start,
            "cycle_end": frame_end,
            "event_count": len(frame_events),
            "top_ops": "; ".join(
                f"op{op_idx}:{stage}:{bytes_value / (1024 * 1024):.2f}MiB"
                for (op_idx, stage), bytes_value in top
            ),
        })

    frame_ranges = [(int(math.floor(edges[i])), int(math.ceil(edges[i + 1]))) for i in range(frame_count)]
    return traffic, frame_ranges, frame_ops


def draw_heatmap(
    matrix: np.ndarray,
    package_layout: Dict[str, Any],
    banks_per_package: int,
    path: Path,
    title: str,
    subtitle: str,
    vmax: float,
) -> None:
    bank_rows, bank_cols, grid_rows, grid_cols = bank_grid_dims(package_layout, banks_per_package)
    fig_w = max(8.0, grid_cols * 0.34)
    fig_h = max(5.0, grid_rows * 0.34 + 1.1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    image = ax.imshow(
        matrix / (1024 * 1024),
        cmap="inferno",
        vmin=0.0,
        vmax=max(vmax, 1e-12) / (1024 * 1024),
        interpolation="nearest",
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("HBM package/bank columns")
    ax.set_ylabel("HBM package/bank rows")
    ax.set_xticks([])
    ax.set_yticks([])

    for row in range(grid_rows + 1):
        is_package_boundary = row % bank_rows == 0
        ax.axhline(
            row - 0.5,
            color="white" if is_package_boundary else "#202020",
            linewidth=1.15 if is_package_boundary else 0.32,
            alpha=0.9 if is_package_boundary else 0.45,
        )
    for col in range(grid_cols + 1):
        is_package_boundary = col % bank_cols == 0
        ax.axvline(
            col - 0.5,
            color="white" if is_package_boundary else "#202020",
            linewidth=1.15 if is_package_boundary else 0.32,
            alpha=0.9 if is_package_boundary else 0.45,
        )

    for package in range(int(package_layout["package_count"])):
        package_row, package_col = divmod(package, int(package_layout["cols"]))
        ax.text(
            package_col * bank_cols + 0.12,
            package_row * bank_rows + 0.35,
            f"pkg{package:02d}",
            color="white",
            fontsize=7,
            ha="left",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.35, "linewidth": 0, "pad": 1},
        )

    cbar = fig.colorbar(image, ax=ax, shrink=0.84)
    cbar.set_label("DRAM access traffic (MiB)")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_csvs(
    out_dir: Path,
    traffic: np.ndarray,
    frame_ranges: List[Tuple[int, int]],
    frame_ops: List[Dict[str, Any]],
    package_layout: Dict[str, Any],
    banks_per_package: int,
) -> None:
    totals_path = out_dir / "bank_traffic_totals.csv"
    with totals_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["logical_bank", "package", "bank_in_package", "traffic_bytes"])
        total_matrix = traffic.sum(axis=0)
        for bank in range(int(package_layout["package_count"]) * banks_per_package):
            row, col = bank_to_grid(bank, package_layout, banks_per_package)
            writer.writerow([bank, bank // banks_per_package, bank % banks_per_package, float(total_matrix[row, col])])

    frames_path = out_dir / "bank_traffic_frames.csv"
    with frames_path.open("w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["frame", "cycle_start", "cycle_end", "logical_bank", "package", "bank_in_package", "traffic_bytes"])
        for frame_idx, (cycle_start, cycle_end) in enumerate(frame_ranges):
            for bank in range(int(package_layout["package_count"]) * banks_per_package):
                row, col = bank_to_grid(bank, package_layout, banks_per_package)
                value = float(traffic[frame_idx, row, col])
                if value > 0:
                    writer.writerow([frame_idx, cycle_start, cycle_end, bank, bank // banks_per_package, bank % banks_per_package, value])

    ops_path = out_dir / "frame_active_ops.csv"
    with ops_path.open("w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["frame", "cycle_start", "cycle_end", "event_count", "top_ops"])
        writer.writeheader()
        writer.writerows(frame_ops)


def write_gif(frame_paths: List[Path], gif_path: Path, fps: int) -> None:
    images = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    duration_ms = int(round(1000 / max(1, fps)))
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    for image in images:
        image.close()


def main() -> None:
    args = parse_args()
    artifact = select_artifact(args)
    cfg = build_trace_config(args)
    package_layout = hbm_package_layout(cfg, args.die_size_mm)
    package_count = int(package_layout["package_count"])
    events = collect_events(artifact, cfg, package_count)
    if not events:
        raise SystemExit(f"No dram_access_records found in {artifact.pickle_path}")

    end_cycle = max(int(event.get("start_cycle", 0)) + int(event.get("duration_cycle", 0)) for event in events)
    traffic, frame_ranges, frame_ops = event_matrix(
        events,
        package_layout,
        cfg.hbm_banks_per_package,
        args.frames,
        end_cycle,
    )

    label = (
        f"{artifact.run_id.model}_{artifact.run_id.mode}_{artifact.run_id.impl}_"
        f"bw{artifact.run_id.dram_bw}_cg{artifact.run_id.core_group}_row{artifact.run_id.row}_{args.bank_mapping}"
    )
    out_dir = args.out_dir / label
    frames_dir = out_dir / "gif_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    vmax = float(np.percentile(traffic[traffic > 0], 99.0)) if np.any(traffic > 0) else 1.0
    title_prefix = f"{artifact.run_id.model} {artifact.run_id.mode} DRAM access traffic"
    frame_paths: List[Path] = []
    for frame_idx, (cycle_start, cycle_end) in enumerate(frame_ranges):
        top_ops = frame_ops[frame_idx]["top_ops"] or "no active DRAM access"
        path = frames_dir / f"frame_{frame_idx:03d}.png"
        draw_heatmap(
            traffic[frame_idx],
            package_layout,
            cfg.hbm_banks_per_package,
            path,
            title_prefix,
            f"frame {frame_idx:03d}, cycles {cycle_start}-{cycle_end}, {top_ops}",
            vmax,
        )
        frame_paths.append(path)

    draw_heatmap(
        traffic.sum(axis=0),
        package_layout,
        cfg.hbm_banks_per_package,
        out_dir / "dram_access_heatmap_total.png",
        title_prefix,
        f"total over {len(events)} tensor access events",
        float(np.max(traffic.sum(axis=0))),
    )
    write_gif(frame_paths, out_dir / "dram_access_heatmap_timeseries.gif", args.fps)
    write_csvs(out_dir, traffic, frame_ranges, frame_ops, package_layout, cfg.hbm_banks_per_package)

    summary_path = out_dir / "metadata.txt"
    summary_path.write_text(
        "\n".join([
            f"artifact={artifact.pickle_path}",
            f"events={len(events)}",
            f"frames={args.frames}",
            f"package_count={package_count}",
            f"banks_per_package={cfg.hbm_banks_per_package}",
            f"stripe_bytes={cfg.hbm_interleave_stripe_bytes}",
            f"mapping={args.bank_mapping}",
            f"gif={out_dir / 'dram_access_heatmap_timeseries.gif'}",
        ]) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {out_dir / 'dram_access_heatmap_timeseries.gif'}")
    print(f"Wrote {out_dir / 'dram_access_heatmap_total.png'}")
    print(f"Wrote {out_dir / 'frame_active_ops.csv'}")


if __name__ == "__main__":
    main()
