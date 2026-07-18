#!/usr/bin/env python3
"""Run 3D-ICE thermal packages across cooler profiles and models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tsim_thermal.artifacts import REPO_SRC, RunArtifacts, load_artifacts
from tsim_thermal.defaults import DEFAULT_AMBIENT_C
from tsim_thermal.threedice import run_threedice
from tsim_thermal.trace import TraceConfig, build_component_power_trace, export_trace_package
from tsim_thermal.visualization import FlpBlock, read_flp


COOLING_PROFILES: Tuple[Tuple[str, float], ...] = (
    ("pcie_air", 0.13),
    ("sxm_air", 0.06),
    ("high_perf_liquid", 0.02),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cooler matrix thermal experiments.")
    parser.add_argument("--results-dir", type=Path, default=REPO_SRC / "results" / "logs")
    parser.add_argument("--out-dir", type=Path, default=REPO_SRC / "results" / "thermal_cooler_matrix")
    parser.add_argument("--models", default="llama3-70")
    parser.add_argument("--modes", default="decode,prefill")
    parser.add_argument("--cooling-profiles", default=",".join(name for name, _ in COOLING_PROFILES))
    parser.add_argument("--impl", default="best")
    parser.add_argument("--batch-size", type=int, default=None, help="Only use artifacts from this bs_* directory.")
    parser.add_argument("--dram-bw", type=int, default=12288)
    parser.add_argument("--row", type=int, default=8192)
    parser.add_argument("--core-group", type=int, default=8)
    parser.add_argument("--ambient-c", type=float, default=DEFAULT_AMBIENT_C)
    parser.add_argument("--bond-thickness-um", type=float, default=10.0)
    parser.add_argument("--decode-iterations", type=int, default=100)
    parser.add_argument("--decode-samples-per-iteration", type=int, default=64)
    parser.add_argument("--prefill-bins", type=int, default=1024)
    parser.add_argument("--hotspot-grid", type=int, default=128)
    parser.add_argument("--dram-layers", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cores-per-job", type=int, default=16)
    parser.add_argument("--gif-samples", type=int, default=32)
    parser.add_argument("--gif-layers", default="all")
    parser.add_argument("--gif-fps", type=float, default=8.0)
    parser.add_argument("--gif-dpi", type=int, default=110)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--noc-power-backend", choices=("tsim_simple", "simple", "dsent", "orion"), default="tsim_simple")
    parser.add_argument("--noc-power-flit-bits", type=int, default=64)
    parser.add_argument("--noc-power-injection-rate", type=float, default=0.3)
    parser.add_argument("--noc-power-link-length-mm", type=float, default=1.0)
    parser.add_argument("--noc-power-dsent-tech", default="TG11LVT")
    parser.add_argument("--postprocess-only", action="store_true", help="Reuse existing completed 3D-ICE packages.")
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--threedice-bin", default=str(REPO_SRC.parent / "external" / "3d-ice-src" / "bin" / "3D-ICE-Emulator"))
    return parser.parse_args()


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def safe_tag(value: str) -> str:
    return value.replace("/", "_").replace(".", "p").replace("-", "_")


def select_artifacts(args: argparse.Namespace) -> List[RunArtifacts]:
    artifacts = load_artifacts(args.results_dir, models=args.models, modes=args.modes, impls=args.impl)
    selected = [
        artifact for artifact in artifacts
        if artifact.run_id.dram_bw == args.dram_bw
        and artifact.run_id.row == args.row
        and artifact.run_id.core_group == args.core_group
        and (args.batch_size is None or artifact.run_id.batch_size == args.batch_size)
    ]
    selected.sort(key=lambda item: (item.run_id.model, item.run_id.batch_size, item.run_id.mode))
    return selected


def trace_config_for_task(task: Dict[str, Any], artifact: RunArtifacts) -> TraceConfig:
    base_runtime_s = artifact.summary.exec_cycles / (1500.0 * 1e6)
    mode = artifact.run_id.mode
    if mode == "decode":
        duration_ms = base_runtime_s * int(task["decode_iterations"]) * 1e3
        bins = int(task["decode_iterations"]) * int(task["decode_samples_per_iteration"])
    else:
        duration_ms = base_runtime_s * 1e3
        bins = int(task["prefill_bins"])
    return TraceConfig(
        duration_ms=duration_ms,
        ambient_c=float(task["ambient_c"]),
        r_convec_k_per_w=float(task["r_convec"]),
        cooling_profile=str(task["cooling_profile"]),
        max_bins=max(1, bins),
        major_op_samples=0,
        dram_layers=int(task["dram_layers"]),
        dram_floorplan_granularity="bank",
        dram_bank_mapping="address_trace",
        logic_floorplan="intra_core",
        hotspot_grid=int(task["hotspot_grid"]),
        bond_thickness_um=float(task["bond_thickness_um"]),
        noc_power_backend=str(task["noc_power_backend"]),
        noc_power_flit_bits=int(task["noc_power_flit_bits"]),
        noc_power_injection_rate=float(task["noc_power_injection_rate"]),
        noc_power_link_length_mm=float(task["noc_power_link_length_mm"]),
        noc_power_dsent_tech=str(task["noc_power_dsent_tech"]),
    )


def read_trace_matrix(path: Path) -> Tuple[List[str], np.ndarray]:
    with path.open(encoding="utf-8") as infile:
        names = next(infile).split()
        rows = []
        for line in infile:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) == len(names):
                rows.append([float(item) for item in parts])
    matrix = np.asarray(rows, dtype=float)
    if matrix.size and float(np.nanmedian(matrix)) > 200.0:
        matrix -= 273.15
    return names, matrix


def real_blocks(path: Path, layer: str) -> List[FlpBlock]:
    blocks = read_flp(path)
    if layer == "logic":
        return [block for block in blocks if not block.name.startswith("logic_pad")]
    return [block for block in blocks if "_pad_" not in block.name]


def package_layers(package_dir: Path) -> Dict[str, List[FlpBlock]]:
    layers = {"logic": real_blocks(package_dir / "logic.flp", "logic")}
    for path in sorted(package_dir.glob("dram*.flp"), key=lambda item: item.name):
        layers[path.stem] = real_blocks(path, path.stem)
    return layers


def layer_temperature_series(package_dir: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    names, temps_c = read_trace_matrix(package_dir / "threedice_temperature.ttrace")
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    dt_s = float(metadata.get("dt_s") or 0.0)
    times = np.arange(temps_c.shape[0], dtype=float) * dt_s
    index = {name: idx for idx, name in enumerate(names)}
    avg: Dict[str, np.ndarray] = {}
    peak: Dict[str, np.ndarray] = {}
    for layer, blocks in package_layers(package_dir).items():
        cols = []
        areas = []
        for block in blocks:
            if block.name in index:
                cols.append(index[block.name])
                areas.append(max(1e-12, block.width_mm * block.height_mm))
        if not cols:
            continue
        values = temps_c[:, cols]
        area = np.asarray(areas, dtype=float)
        avg[layer] = (values * area[None, :]).sum(axis=1) / area.sum()
        peak[layer] = values.max(axis=1)
    return times, avg, peak


def write_temperature_timeseries(package_dir: Path, out_dir: Path, tag: str) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, avg, peak = layer_temperature_series(package_dir)
    layers = list(avg)
    csv_path = out_dir / f"{tag}_layer_temperature_timeseries.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["time_s", *[f"{layer}_avg_c" for layer in layers], *[f"{layer}_peak_c" for layer in layers]])
        for idx, t_s in enumerate(times):
            writer.writerow([t_s, *[avg[layer][idx] for layer in layers], *[peak[layer][idx] for layer in layers]])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(layers))))
    for idx, layer in enumerate(layers):
        linewidth = 2.4 if layer == "logic" else 1.2
        alpha = 1.0 if layer == "logic" else 0.75
        axes[0].plot(times, avg[layer], label=layer, linewidth=linewidth, alpha=alpha, color=colors[idx % len(colors)])
        axes[1].plot(times, peak[layer], label=layer, linewidth=linewidth, alpha=alpha, color=colors[idx % len(colors)])
    axes[0].set_ylabel("Area-weighted average temp (C)")
    axes[1].set_ylabel("Peak block temp (C)")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=5, fontsize=8, loc="upper left")
    fig.suptitle(tag)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = out_dir / f"{tag}_layer_temperature_timeseries.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    return {
        "csv": str(csv_path),
        "plot": str(png_path),
        "logic_peak_c": float(np.max(peak.get("logic", np.asarray([float("nan")])))),
        "logic_avg_max_c": float(np.max(avg.get("logic", np.asarray([float("nan")])))),
        "package_peak_c": float(max(np.max(values) for values in peak.values())),
    }


def write_temperature_cdfs(package_dir: Path, out_dir: Path, tag: str) -> Dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times, avg, peak = layer_temperature_series(package_dir)
    dt_s = float(times[1] - times[0]) if len(times) > 1 else 0.0
    layers = list(avg)
    artifacts: Dict[str, str] = {}
    for metric, data in (("avg", avg), ("peak", peak)):
        csv_path = out_dir / f"{tag}_layer_{metric}_temperature_cdf.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["layer", "temperature_c", "cumulative_time_s", "cumulative_fraction"])
            for layer in layers:
                values = np.sort(data[layer])
                cumulative_time = np.arange(1, len(values) + 1, dtype=float) * dt_s
                total_time = cumulative_time[-1] if len(cumulative_time) else 0.0
                fractions = cumulative_time / total_time if total_time > 0 else np.zeros_like(cumulative_time)
                for temp_c, ctime_s, frac in zip(values, cumulative_time, fractions):
                    writer.writerow([layer, temp_c, ctime_s, frac])

        fig, ax = plt.subplots(figsize=(9, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(layers))))
        for idx, layer in enumerate(layers):
            values = np.sort(data[layer])
            fractions = np.arange(1, len(values) + 1, dtype=float) / max(1, len(values))
            ax.plot(values, fractions, label=layer, linewidth=2.2 if layer == "logic" else 1.1, color=colors[idx % len(colors)])
        ax.set_xlabel(f"{metric.capitalize()} layer temperature (C)")
        ax.set_ylabel("Cumulative execution time fraction")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        ax.set_title(f"{tag}: {metric} temperature CDF")
        fig.tight_layout()
        png_path = out_dir / f"{tag}_layer_{metric}_temperature_cdf.png"
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
        artifacts[f"{metric}_csv"] = str(csv_path)
        artifacts[f"{metric}_plot"] = str(png_path)
    return artifacts


def write_temperature_gif(package_dir: Path, out_dir: Path, tag: str, task: Dict[str, Any]) -> str | None:
    if bool(task.get("skip_gif")):
        return None
    package_dir = package_dir.resolve()
    gif_path = (out_dir / f"{tag}_temperature_stack.gif").resolve()
    cmd = [
        sys.executable,
        str(REPO_SRC / "benchmark_scripts" / "animate_thermal_stack.py"),
        str(package_dir),
        "--quantity", "temperature",
        "--trace", str(package_dir / "threedice_temperature.ttrace"),
        "--out", str(gif_path),
        "--layers", str(task["gif_layers"]),
        "--samples", str(task["gif_samples"]),
        "--fps", str(task["gif_fps"]),
        "--dpi", str(task["gif_dpi"]),
    ]
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/3dstack-mpl")
    subprocess.run(cmd, cwd=REPO_SRC.parent, env=env, check=True, capture_output=True, text=True)
    return str(gif_path)


def summarize_package(
    task: Dict[str, Any],
    artifact: RunArtifacts,
    package_dir: Path,
    out_dir: Path,
    tag: str,
    bins: int,
    started: float,
    threedice_runtime_s: float | None,
) -> Dict[str, Any]:
    timeseries = write_temperature_timeseries(package_dir, out_dir, tag)
    cdfs = write_temperature_cdfs(package_dir, out_dir, tag)
    gif = write_temperature_gif(package_dir, out_dir, tag, task)
    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    summary = {
        "tag": tag,
        "model": artifact.run_id.model,
        "mode": artifact.run_id.mode,
        "requested_batch_size": int(task["requested_batch_size"]),
        "actual_artifact_batch_size": int(task["actual_batch_size"]),
        "batch_size_note": task.get("batch_size_note", ""),
        "decode_iterations": int(task["decode_iterations"]) if artifact.run_id.mode == "decode" else None,
        "cooling_profile": task["cooling_profile"],
        "r_convec_k_per_w": float(task["r_convec"]),
        "ambient_c": float(task["ambient_c"]),
        "bond_thickness_um": float(task["bond_thickness_um"]),
        "duration_s": float(metadata.get("duration_s") or 0.0),
        "dt_s": float(metadata.get("dt_s") or 0.0),
        "bins": int(bins),
        "threads": int(task["cores_per_job"]),
        "threedice_runtime_s": threedice_runtime_s,
        "elapsed_s": elapsed,
        "package_dir": str(package_dir),
        "temperature_trace": str(package_dir / "threedice_temperature.ttrace"),
        "gif": gif,
        **timeseries,
        **cdfs,
    }
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def load_task_artifact(task: Dict[str, Any]) -> RunArtifacts:
    artifacts = load_artifacts(Path(task["results_dir"]), models=str(task["model"]), modes=str(task["mode"]), impls=str(task["impl"]))
    matches = [
        artifact for artifact in artifacts
        if artifact.run_id.dram_bw == int(task["dram_bw"])
        and artifact.run_id.row == int(task["row"])
        and artifact.run_id.core_group == int(task["core_group"])
        and (task.get("batch_size") is None or artifact.run_id.batch_size == int(task["batch_size"]))
    ]
    if not matches:
        raise RuntimeError(f"no artifact for {task['model']} {task['mode']}")
    return matches[0]


def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["TSIM_THREEDICE_CORES"] = str(int(task["cores_per_job"]))
    os.environ["OMP_NUM_THREADS"] = str(int(task["cores_per_job"]))
    os.environ.setdefault("OMP_PROC_BIND", "close")
    os.environ.setdefault("OMP_PLACES", "cores")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/3dstack-mpl")

    artifact = load_task_artifact(task)

    started = time.perf_counter()
    cfg = trace_config_for_task(task, artifact)
    trace = build_component_power_trace(artifact, cfg)
    tag = str(task["tag"])
    package_dir = Path(task["package_dir"])
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    export_trace_package(artifact, trace, cfg, package_dir, write_visualizations=False)
    result = run_threedice(package_dir, str(task["threedice_bin"]), float(task["timeout_s"]), heatmap_count=0)
    if result.returncode != 0:
        raise RuntimeError(f"3D-ICE failed for {tag}: {result.returncode}\n{result.stderr}")

    return summarize_package(
        task,
        artifact,
        package_dir,
        out_dir,
        tag,
        int(trace.component_power_w.shape[0]),
        started,
        result.runtime_s,
    )


def postprocess_task(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/3dstack-mpl")
    artifact = load_task_artifact(task)
    started = time.perf_counter()
    cfg = trace_config_for_task(task, artifact)
    package_dir = Path(task["package_dir"])
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (package_dir / "threedice_temperature.ttrace").exists():
        raise RuntimeError(f"missing completed temperature trace in {package_dir}")
    threedice_runtime_s = None
    summary_path = package_dir / "threedice_summary.json"
    if summary_path.exists():
        threedice_runtime_s = json.loads(summary_path.read_text(encoding="utf-8")).get("runtime_s")
    return summarize_package(
        task,
        artifact,
        package_dir,
        out_dir,
        str(task["tag"]),
        int(cfg.max_bins),
        started,
        None if threedice_runtime_s is None else float(threedice_runtime_s),
    )


def build_tasks(args: argparse.Namespace, artifacts: Iterable[RunArtifacts]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    selected_profiles = set(split_csv(args.cooling_profiles))
    for artifact in artifacts:
        requested_bs = args.batch_size if args.batch_size is not None else artifact.run_id.batch_size
        actual_bs = artifact.run_id.batch_size
        note = ""
        for profile, r_convec in COOLING_PROFILES:
            if profile not in selected_profiles:
                continue
            tag = "_".join([
                safe_tag(artifact.run_id.model),
                artifact.run_id.mode,
                profile,
                f"uf{int(round(args.bond_thickness_um))}um",
            ])
            package_dir = args.out_dir / "packages" / tag
            tasks.append({
                "results_dir": str(args.results_dir),
                "out_dir": str(args.out_dir),
                "package_dir": str(package_dir),
                "tag": tag,
                "model": artifact.run_id.model,
                "mode": artifact.run_id.mode,
                "impl": args.impl,
                "dram_bw": args.dram_bw,
                "row": args.row,
                "core_group": args.core_group,
                "batch_size": args.batch_size,
                "ambient_c": args.ambient_c,
                "bond_thickness_um": args.bond_thickness_um,
                "decode_iterations": args.decode_iterations,
                "decode_samples_per_iteration": args.decode_samples_per_iteration,
                "prefill_bins": args.prefill_bins,
                "hotspot_grid": args.hotspot_grid,
                "dram_layers": args.dram_layers,
                "cooling_profile": profile,
                "r_convec": r_convec,
                "cores_per_job": args.cores_per_job,
                "timeout_s": args.timeout_s,
                "threedice_bin": args.threedice_bin,
                "gif_samples": args.gif_samples,
                "gif_layers": args.gif_layers,
                "gif_fps": args.gif_fps,
                "gif_dpi": args.gif_dpi,
                "skip_gif": args.skip_gif,
                "noc_power_backend": args.noc_power_backend,
                "noc_power_flit_bits": args.noc_power_flit_bits,
                "noc_power_injection_rate": args.noc_power_injection_rate,
                "noc_power_link_length_mm": args.noc_power_link_length_mm,
                "noc_power_dsent_tech": args.noc_power_dsent_tech,
                "requested_batch_size": requested_bs,
                "actual_batch_size": actual_bs,
                "batch_size_note": note,
            })
    return tasks


def write_matrix_summary(out_dir: Path, rows: List[Dict[str, Any]]) -> Path:
    csv_path = out_dir / "thermal_cooler_matrix_summary.csv"
    if not rows:
        return csv_path
    preferred = [
        "tag", "model", "mode", "cooling_profile", "r_convec_k_per_w",
        "requested_batch_size", "actual_artifact_batch_size", "decode_iterations",
        "duration_s", "bins", "threads", "threedice_runtime_s", "elapsed_s",
        "logic_peak_c", "logic_avg_max_c", "package_peak_c", "gif", "package_dir",
        "batch_size_note",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    with csv_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return csv_path


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = select_artifacts(args)
    if not artifacts:
        raise SystemExit("No matching artifacts found.")
    log("selected artifacts:")
    for artifact in artifacts:
        log(f"  {artifact.run_id.label()}")

    tasks = build_tasks(args, artifacts)
    action = "postprocessing" if args.postprocess_only else "running"
    log(f"{action} {len(tasks)} thermal tasks with jobs={args.jobs}, cores/job={args.cores_per_job}")
    summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    worker = postprocess_task if args.postprocess_only else run_task
    with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as pool:
        futures = {pool.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                summary = future.result()
                summaries.append(summary)
                log(f"done {summary['tag']} peak={summary['package_peak_c']:.2f}C")
            except Exception as exc:  # noqa: BLE001
                message = f"{task['tag']}: {exc}"
                failures.append(message)
                log(f"FAILED {message}")

    summaries.sort(key=lambda row: (row["model"], row["mode"], row["cooling_profile"]))
    csv_path = write_matrix_summary(args.out_dir, summaries)
    manifest = {
        "out_dir": str(args.out_dir),
        "summary_csv": str(csv_path),
        "completed": len(summaries),
        "failed": failures,
        "settings": vars(args),
    }
    (args.out_dir / "thermal_cooler_matrix_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
