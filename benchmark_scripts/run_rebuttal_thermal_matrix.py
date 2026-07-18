#!/usr/bin/env python3
"""Run default-chip thermal rebuttal experiments across workloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = SRC_DIR.parent
DEFAULT_MODELS = "llama2-13,llama3-70,opt-30,gemma2,dit-xl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch simple and HotSpot thermal validation jobs for rebuttal data.")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--modes", default="decode,prefill")
    parser.add_argument("--out-dir", type=Path, default=SRC_DIR / "results" / "thermal_rebuttal")
    parser.add_argument("--results-dir", type=Path, default=SRC_DIR / "results" / "logs")
    parser.add_argument("--hotspot-bin", default=str(REPO_DIR / "external" / "hotspot-7.0" / "hotspot"))
    parser.add_argument("--hotspot-grid", type=int, default=64,
                        help="HotSpot thermal mesh resolution. 64 gives about 4.0 x 3.0 cells per core and 5.3 x 4.0 cells per HBM bank block in the default layout; use 128 for longer spot checks.")
    parser.add_argument("--hotspot-timeout-s", type=float, default=21600.0)
    parser.add_argument("--duration-ms", type=float, default=2.0)
    parser.add_argument("--max-bins", type=int, default=10000)
    parser.add_argument("--major-op-samples", type=int, default=4)
    parser.add_argument("--major-op-percentile", type=float, default=25.0)
    parser.add_argument("--max-trace-power-error-pct", type=float, default=100.0,
                        help="Forwarded validation threshold; high by default because prefill traces may represent sampled layers.")
    parser.add_argument("--logic-floorplan", choices=("intra_core", "coarse_grid"), default="intra_core",
                        help="Logic-layer HotSpot floorplan granularity. Defaults to intra-core blocks.")
    parser.add_argument("--noc-power-backend", choices=("tsim_simple", "simple", "dsent", "orion"), default="tsim_simple")
    parser.add_argument("--noc-power-flit-bits", type=int, default=64)
    parser.add_argument("--noc-power-injection-rate", type=float, default=0.3)
    parser.add_argument("--noc-power-link-length-mm", type=float, default=1.0)
    parser.add_argument("--noc-power-dsent-tech", default="TG11LVT")
    parser.add_argument("--parallel-jobs", type=int, default=2)
    parser.add_argument("--operator-heatmap-count", type=int, default=2)
    parser.add_argument("--skip-simple", action="store_true")
    parser.add_argument("--skip-hotspot", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only regenerate aggregate reports from existing per-run HotSpot CSVs.")
    parser.add_argument("--no-aggregate", action="store_true",
                        help="Do not write aggregate reports after the matrix finishes.")
    parser.add_argument("--run-tsim", action="store_true",
                        help="Generate missing mode-9 TSIM artifacts before thermal analysis.")
    parser.add_argument("--tsim-decode-parallel-limit", type=int, default=3)
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_DIR / path


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def launch_matrix(cmds: list[list[str]], cwd: Path, env: dict[str, str], parallel_jobs: int) -> None:
    active: list[tuple[list[str], subprocess.Popen]] = []
    pending = list(cmds)
    failed: list[tuple[list[str], int]] = []
    while pending or active:
        while pending and len(active) < max(1, parallel_jobs):
            cmd = pending.pop(0)
            print(" ".join(cmd), flush=True)
            active.append((cmd, subprocess.Popen(cmd, cwd=cwd, env=env)))
        next_active = []
        for cmd, proc in active:
            ret = proc.poll()
            if ret is None:
                next_active.append((cmd, proc))
            elif ret != 0:
                failed.append((cmd, ret))
        active = next_active
        if pending or active:
            time.sleep(5)
    if failed:
        for cmd, ret in failed:
            print(f"FAILED({ret}): {' '.join(cmd)}", file=sys.stderr)
        raise SystemExit(3)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "n/a", "None"):
        return math.nan
    return float(value)


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx
        while end + 1 < len(order) and values[order[end + 1]] == values[order[idx]]:
            end += 1
        avg_rank = (idx + end + 2) / 2.0
        for pos in range(idx, end + 1):
            ranks[order[pos]] = avg_rank
        idx = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_delta = [item - left_mean for item in left]
    right_delta = [item - right_mean for item in right]
    denom = math.sqrt(sum(item * item for item in left_delta) * sum(item * item for item in right_delta))
    if denom == 0:
        return math.nan
    return sum(a * b for a, b in zip(left_delta, right_delta)) / denom


def _spearman(rows: list[dict[str, str]], left_key: str, right_key: str) -> float:
    pairs = [
        (_float(row, left_key), _float(row, right_key))
        for row in rows
    ]
    pairs = [(left, right) for left, right in pairs if not math.isnan(left) and not math.isnan(right)]
    if len(pairs) < 2:
        return math.nan
    left, right = zip(*pairs)
    return _pearson(_rank(list(left)), _rank(list(right)))


def _fmt(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def write_aggregate_reports(out_dir: Path) -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in sorted((out_dir / "hotspot").glob("*/*/thermal_compare.csv")):
        with path.open(newline="", encoding="utf-8") as infile:
            for row in csv.DictReader(infile):
                row["source_csv"] = str(path)
                rows.append(row)
    if not rows:
        print(f"No HotSpot thermal_compare.csv files found under {out_dir / 'hotspot'}", flush=True)
        return {"runs": 0}

    fields = [
        "model", "mode", "logic_floorplan", "hotspot_grid", "dram_floorplan_granularity",
        "dram_bank_mapping", "noc_power_backend", "thermal_bins", "thermal_dt_us", "major_ops_meeting_target_fraction",
        "simple_power_density_w_per_mm2", "simple_peak_c", "stack_peak_c", "hotspot_peak_c",
        "hotspot_delta_vs_stack_c", "hotspot_margin_c", "simple_slowdown", "stack_slowdown",
        "slowdown_delta_pct", "trace_avg_power_error_pct", "spatial_ops_with_metadata",
        "inferred_spatial_events", "fallback_events", "package_dir", "source_csv",
    ]
    aggregate_csv = out_dir / "aggregate_thermal_compare.csv"
    with aggregate_csv.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["model"], item["mode"])):
            writer.writerow({field: row.get(field, "") for field in fields})

    hottest = max(rows, key=lambda row: _float(row, "hotspot_peak_c"))
    summary = {
        "runs": len(rows),
        "models": sorted({row["model"] for row in rows}),
        "modes": sorted({row["mode"] for row in rows}),
        "max_hotspot_peak_c": _float(hottest, "hotspot_peak_c"),
        "max_hotspot_peak_run": f"{hottest['model']}/{hottest['mode']}",
        "min_hotspot_margin_c": min(_float(row, "hotspot_margin_c") for row in rows),
        "max_slowdown_delta_pct": max(abs(_float(row, "slowdown_delta_pct")) for row in rows),
        "max_trace_avg_power_error_pct": max(abs(_float(row, "trace_avg_power_error_pct")) for row in rows),
        "max_simple_power_density_w_per_mm2": max(_float(row, "simple_power_density_w_per_mm2") for row in rows),
        "hotspot_vs_simple_peak_spearman": _spearman(rows, "simple_peak_c", "hotspot_peak_c"),
        "hotspot_vs_stack_peak_spearman": _spearman(rows, "stack_peak_c", "hotspot_peak_c"),
        "fallback_events_total": sum(int(float(row.get("fallback_events") or 0)) for row in rows),
        "spatial_ops_with_metadata_total": sum(int(float(row.get("spatial_ops_with_metadata") or 0)) for row in rows),
        "inferred_spatial_events_total": sum(int(float(row.get("inferred_spatial_events") or 0)) for row in rows),
    }
    (out_dir / "aggregate_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Rebuttal Thermal Matrix Aggregate\n",
        f"Runs: {summary['runs']}\n",
        f"Max HotSpot peak: {summary['max_hotspot_peak_c']:.2f} C ({summary['max_hotspot_peak_run']})\n",
        f"Minimum HotSpot thermal margin to 85 C: {summary['min_hotspot_margin_c']:.2f} C\n",
        f"Max slowdown delta: {summary['max_slowdown_delta_pct']:.3f}%\n",
        f"Spearman(simple peak, HotSpot peak): {_fmt(float(summary['hotspot_vs_simple_peak_spearman']), 4)}\n",
        f"Spearman(stack peak, HotSpot peak): {_fmt(float(summary['hotspot_vs_stack_peak_spearman']), 4)}\n",
        f"Max trace average-power error: {summary['max_trace_avg_power_error_pct']:.3f}%\n",
        f"Total TSIM spatial op records: {summary['spatial_ops_with_metadata_total']}\n",
        f"Total inferred spatial op records: {summary['inferred_spatial_events_total']}\n",
        f"Total fallback dynamic events: {summary['fallback_events_total']}\n",
        "\n| Model | Mode | Logic floorplan | HotSpot grid | DRAM mapping | Bins | dt us | Major-op hit | Simple peak C | Stack peak C | HotSpot peak C | HotSpot margin C | Slowdown delta % | Spatial ops | Inferred ops | Trace avg err % |\n",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in sorted(rows, key=lambda item: (item["model"], item["mode"])):
        lines.append(
            f"| {row['model']} | {row['mode']} | {row.get('logic_floorplan', '')} | "
            f"{int(float(row.get('hotspot_grid') or 0))} | {row.get('dram_bank_mapping', '')} | "
            f"{int(float(row['thermal_bins']))} | "
            f"{_float(row, 'thermal_dt_us'):.3f} | {_float(row, 'major_ops_meeting_target_fraction'):.3f} | "
            f"{_float(row, 'simple_peak_c'):.2f} | {_float(row, 'stack_peak_c'):.2f} | "
            f"{_float(row, 'hotspot_peak_c'):.2f} | {_float(row, 'hotspot_margin_c'):.2f} | "
            f"{_float(row, 'slowdown_delta_pct'):.3f} | "
            f"{int(float(row.get('spatial_ops_with_metadata') or 0))} | "
            f"{int(float(row.get('inferred_spatial_events') or 0))} | "
            f"{_float(row, 'trace_avg_power_error_pct'):.3f} |\n"
        )
    aggregate_md = out_dir / "aggregate_thermal_compare.md"
    aggregate_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {aggregate_csv}", flush=True)
    print(f"Wrote {aggregate_md}", flush=True)
    print(f"Wrote {out_dir / 'aggregate_summary.json'}", flush=True)
    return summary


def thermal_cmd(args: argparse.Namespace, model: str, mode: str, backend: str, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(SRC_DIR / "benchmark_scripts" / "thermal_compare.py"),
        "--results-dir", str(args.results_dir),
        "--out-dir", str(out_dir),
        "--models", model,
        "--modes", mode,
        "--impls", "best",
        "--duration-ms", str(args.duration_ms),
        "--max-bins", str(args.max_bins),
        "--major-op-samples", str(args.major_op_samples),
        "--major-op-percentile", str(args.major_op_percentile),
        "--max-trace-power-error-pct", str(args.max_trace_power_error_pct),
        "--dram-layers", "8",
        "--dram-capacity-gb", "192",
        "--hbm-package-capacity-gb", "16",
        "--hbm-package-area-mm2", "87.62745402745404",
        "--dram-floorplan-granularity", "bank",
        "--hbm-banks-per-package", "16",
        "--dram-bank-mappings", "hbm_interleave",
        "--dram-bws", "12288",
        "--rows", "8192",
        "--core-groups", "8",
        "--spatial-policies", "uniform",
        "--die-size-mm", "28.325779313106533",
        "--logic-floorplan", args.logic_floorplan,
        "--hotspot-grid", str(args.hotspot_grid),
        "--thermal-backends", backend,
        "--noc-power-backend", args.noc_power_backend,
        "--noc-power-flit-bits", str(args.noc_power_flit_bits),
        "--noc-power-injection-rate", str(args.noc_power_injection_rate),
        "--noc-power-link-length-mm", str(args.noc_power_link_length_mm),
        "--noc-power-dsent-tech", args.noc_power_dsent_tech,
        "--operator-heatmap-count", str(args.operator_heatmap_count),
    ]
    if "hotspot" in backend:
        cmd.extend([
            "--hotspot-bin", args.hotspot_bin,
            "--hotspot-timeout-s", str(args.hotspot_timeout_s),
        ])
    return cmd


def main() -> int:
    args = parse_args()
    args.out_dir = repo_path(args.out_dir)
    args.results_dir = repo_path(args.results_dir)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/3dstack-mpl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        write_aggregate_reports(args.out_dir)
        return 0

    if args.run_tsim:
        run_checked([
            sys.executable,
            "run_all_modes.py",
            "--modes", "9",
            "--run-both",
            "--decode-parallel-limit", str(args.tsim_decode_parallel_limit),
        ], SRC_DIR, env)

    models = split_csv(args.models)
    modes = split_csv(args.modes)

    if not args.skip_simple:
        run_checked([
            sys.executable,
            str(SRC_DIR / "benchmark_scripts" / "thermal_compare.py"),
            "--results-dir", str(args.results_dir),
            "--out-dir", str(args.out_dir / "simple"),
            "--models", ",".join(models),
            "--modes", ",".join(modes),
            "--impls", "best",
            "--duration-ms", str(args.duration_ms),
            "--max-bins", str(args.max_bins),
            "--major-op-samples", str(args.major_op_samples),
            "--major-op-percentile", str(args.major_op_percentile),
            "--max-trace-power-error-pct", str(args.max_trace_power_error_pct),
            "--thermal-backends", "simple",
            "--noc-power-backend", args.noc_power_backend,
            "--noc-power-flit-bits", str(args.noc_power_flit_bits),
            "--noc-power-injection-rate", str(args.noc_power_injection_rate),
            "--noc-power-link-length-mm", str(args.noc_power_link_length_mm),
            "--noc-power-dsent-tech", args.noc_power_dsent_tech,
            "--dram-bws", "12288",
            "--rows", "8192",
            "--core-groups", "8",
            "--spatial-policies", "uniform",
        ], SRC_DIR, env)

    if not args.skip_hotspot:
        cmds = []
        for model in models:
            for mode in modes:
                cmds.append(thermal_cmd(args, model, mode, "simple,hotspot", args.out_dir / "hotspot" / model / mode))
        launch_matrix(cmds, SRC_DIR, env, args.parallel_jobs)
    if not args.no_aggregate:
        write_aggregate_reports(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
