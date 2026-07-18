"""Command-line orchestration for TSIM thermal validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from .artifacts import REPO_SRC, load_artifacts
from .backends import (
    ThermalBackendContext,
    ThermalBackendOptions,
    parse_backend_names,
    resolve_backends,
)
from .defaults import DEFAULT_AMBIENT_C
from .models import ThermalConfig
from .reports import write_csv, write_markdown
from .trace import (
    TraceConfig,
    build_component_power_trace,
    estimate_logic_area_mm2,
    export_trace_package,
    resolve_dram_bank_mapping,
    temporal_sampling_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TSIM power traces and compare simple thermal proxy against detailed 3D stack model."
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_SRC / "results" / "logs")
    parser.add_argument("--out-dir", type=Path, default=REPO_SRC / "results" / "thermal_validation")
    parser.add_argument("--models", default="", help="Comma-separated model filter.")
    parser.add_argument("--modes", default="decode,prefill", help="Comma-separated mode filter.")
    parser.add_argument("--impls", default="best", help="Comma-separated implementation filter.")
    parser.add_argument("--dram-bws", default="", help="Comma-separated DRAM bandwidth filter.")
    parser.add_argument("--rows", default="", help="Comma-separated DRAM row/access-count filter.")
    parser.add_argument("--core-groups", default="", help="Comma-separated core-group filter.")
    parser.add_argument("--npu-freq-mhz", type=float, default=1500.0)
    parser.add_argument("--duration-ms", type=float, default=250.0)
    parser.add_argument("--max-bins", type=int, default=25000)
    parser.add_argument("--major-op-samples", type=int, default=4,
                        help="Target samples for most matmul/SA stages. Use 0 to make --max-bins exact.")
    parser.add_argument("--major-op-percentile", type=float, default=25.0,
                        help="Matmul duration percentile used for operator-aware temporal sampling.")
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--dram-layers", type=int, default=1,
                        help="Number of stacked DRAM/HBM dies to export to HotSpot.")
    parser.add_argument("--dram-capacity-gb", type=float, default=192.0,
                        help="Total DRAM/HBM capacity used for HotSpot HBM package floorplanning.")
    parser.add_argument("--hbm-package-capacity-gb", type=float, default=16.0,
                        help="Capacity per HBM package for package-count calculation.")
    parser.add_argument("--hbm-package-area-mm2", type=float, default=87.62745402745404,
                        help="2D footprint area of one HBM package.")
    parser.add_argument("--hbm-package-aspect-ratio", type=float, default=1.0,
                        help="HBM package footprint aspect ratio as width/height.")
    parser.add_argument("--hbm-banks-per-package", type=int, default=16,
                        help="Number of pseudochannel-aligned thermal bank blocks inside each HBM package in bank granularity.")
    parser.add_argument("--hbm-interleave-stripe-bytes", type=int, default=256,
                        help="Stripe size for address_trace and hbm_interleave DRAM bank mapping.")
    parser.add_argument("--dram-floorplan-granularity", choices=("package", "bank"), default="bank",
                        help="DRAM floorplan granularity for HotSpot export.")
    parser.add_argument("--dram-bank-mapping",
                        choices=("address_trace", "hbm_interleave", "hbm-interleave",
                                 "fine_interleave", "fine-interleave", "bank_interleave", "bank-interleave",
                                 "from_impl", "uniform", "interleave_size", "software_aware"),
                        default="address_trace",
                        help="DRAM bank placement policy used for bank-level thermal attribution.")
    parser.add_argument("--dram-bank-mappings", default="",
                        help="Optional comma-separated DRAM bank mappings to sweep.")
    parser.add_argument("--thermal-backends", default="simple",
                        help="Comma-separated thermal backends. Known: simple,hotspot,threedice,adaptive_grid. Default: simple.")
    parser.add_argument("--noc-power-backend", choices=("tsim_simple", "simple", "dsent", "orion"), default="tsim_simple",
                        help="NoC power backend used when constructing thermal power traces. Default: tsim_simple.")
    parser.add_argument("--noc-power-flit-bits", type=int, default=64,
                        help="Flit width passed to external NoC power tools.")
    parser.add_argument("--noc-power-injection-rate", type=float, default=0.3,
                        help="Injection/load point used for DSENT/ORION NoC characterization.")
    parser.add_argument("--noc-power-link-length-mm", type=float, default=1.0,
                        help="Representative router-to-router link length for DSENT/ORION link characterization.")
    parser.add_argument("--noc-power-dsent-tech", default="TG11LVT",
                        help="DSENT technology model basename, e.g. TG11LVT, Bulk22LVT, Bulk32LVT, Bulk45LVT.")
    parser.add_argument("--noc-power-orion-version", type=int, default=2,
                        help="ORION model version selector. The current external wrapper supports ORION 2 routing output.")
    parser.add_argument("--spatial-policy", choices=("uniform", "center_hotspot", "edge_hotspot"),
                        default="center_hotspot")
    parser.add_argument("--spatial-policies", default="",
                        help="Optional comma-separated list of spatial policies to sweep.")
    parser.add_argument(
        "--die-size-mm",
        type=float,
        default=None,
        help=(
            "Optional square logic die side. By default this is derived from "
            f"the simulator's static logic power at 0.061 W/mm^2."
        ),
    )
    parser.add_argument("--logic-floorplan", choices=("intra_core", "coarse_grid"), default="intra_core",
                        help="Logic-layer HotSpot floorplan granularity.")
    parser.add_argument("--hotspot-grid", type=int, default=128,
                        help="HotSpot grid resolution for exported packages. Independent of --grid.")
    parser.add_argument("--ambient-c", type=float, default=DEFAULT_AMBIENT_C)
    parser.add_argument("--throttle-c", type=float, default=85.0)
    parser.add_argument("--simple-r-k-per-w", type=float, default=0.10)
    parser.add_argument("--simple-c-j-per-k", type=float, default=80.0)
    parser.add_argument("--stack-r-sink-k-per-w", type=float, default=1.20)
    parser.add_argument("--stack-r-vertical-k-per-w", type=float, default=0.22)
    parser.add_argument("--stack-r-lateral-k-per-w", type=float, default=1.80)
    parser.add_argument("--logic-c-j-per-k", type=float, default=2.0)
    parser.add_argument("--dram-c-j-per-k", type=float, default=3.0)
    parser.add_argument("--run-hotspot", action="store_true",
                        help="Run HotSpot on each exported package if the binary is available.")
    parser.add_argument("--hotspot-bin", default="hotspot")
    parser.add_argument("--hotspot-timeout-s", type=float, default=300.0)
    parser.add_argument("--threedice-bin", default="3D-ICE",
                        help="3D-ICE executable path for the threedice backend.")
    parser.add_argument("--threedice-timeout-s", type=float, default=300.0)
    parser.add_argument("--threedice-heatmap-count", type=int, default=4,
                        help="Number of time positions visualized by the threedice backend.")
    parser.add_argument("--threedice-heatmap-layers", default="logic,dram0,dram3,dram7",
                        help="Comma-separated layer names visualized by the threedice backend, or empty for all.")
    parser.add_argument("--adaptive-grid-heatmap-count", type=int, default=6,
                        help="Number of time positions visualized by the adaptive_grid backend.")
    parser.add_argument("--operator-hotspot-analysis", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate per-operator HotSpot summaries after successful HotSpot runs.")
    parser.add_argument("--operator-heatmap-count", type=int, default=12,
                        help="Number of high-energy fused operators to visualize with layer heatmaps.")
    parser.add_argument("--power-density-threshold", type=float, default=0.7,
                        help="Existing TSIM/paper simple thermal limit in W/mm^2.")
    parser.add_argument("--max-trace-power-error-pct", type=float, default=2.0)
    parser.add_argument("--max-temp-delta-c", type=float, default=2.0)
    parser.add_argument("--max-slowdown-delta-pct", type=float, default=5.0)
    parser.add_argument("--min-policy-spearman", type=float, default=0.9)
    return parser.parse_args()


def split_int_filter(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def validate_args(args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    positive_fields = [
        "npu_freq_mhz",
        "duration_ms",
        "max_bins",
        "major_op_samples",
        "major_op_percentile",
        "grid",
        "hotspot_grid",
        "dram_layers",
        "dram_capacity_gb",
        "hbm_package_capacity_gb",
        "hbm_package_area_mm2",
        "hbm_package_aspect_ratio",
        "hbm_banks_per_package",
        "throttle_c",
        "simple_r_k_per_w",
        "simple_c_j_per_k",
        "stack_r_sink_k_per_w",
        "stack_r_vertical_k_per_w",
        "stack_r_lateral_k_per_w",
        "logic_c_j_per_k",
        "dram_c_j_per_k",
        "hbm_interleave_stripe_bytes",
        "hotspot_timeout_s",
        "threedice_timeout_s",
        "threedice_heatmap_count",
        "adaptive_grid_heatmap_count",
        "operator_heatmap_count",
        "power_density_threshold",
        "noc_power_flit_bits",
        "noc_power_injection_rate",
        "noc_power_link_length_mm",
    ]
    for field in positive_fields:
        value = getattr(args, field)
        if field in {"major_op_samples", "operator_heatmap_count", "adaptive_grid_heatmap_count", "threedice_heatmap_count"}:
            if value < 0:
                errors.append(f"--{field.replace('_', '-')} must be non-negative")
        elif value <= 0:
            errors.append(f"--{field.replace('_', '-')} must be positive")
    if args.die_size_mm is not None and args.die_size_mm <= 0:
        errors.append("--die-size-mm must be positive")
    if args.ambient_c >= args.throttle_c:
        errors.append("--ambient-c must be below --throttle-c")
    if args.max_trace_power_error_pct < 0 or args.max_temp_delta_c < 0 or args.max_slowdown_delta_pct < 0:
        errors.append("validation tolerances must be non-negative")
    if not 0 <= args.min_policy_spearman <= 1:
        errors.append("--min-policy-spearman must be in [0, 1]")
    if args.noc_power_orion_version != 2:
        errors.append("--noc-power-orion-version currently supports only 2 for the vnoc20/orion3 wrapper")
    for field in ("dram_bws", "rows", "core_groups"):
        try:
            split_int_filter(getattr(args, field))
        except ValueError:
            errors.append(f"--{field.replace('_', '-')} must be a comma-separated list of integers")
    try:
        resolve_backends(parse_backend_names(args.thermal_backends, args.run_hotspot))
    except ValueError as exc:
        errors.append(str(exc))
    allowed_bank_mappings = {
        "address_trace", "hbm_interleave", "hbm-interleave",
        "fine_interleave", "fine-interleave", "bank_interleave", "bank-interleave",
        "from_impl", "uniform", "interleave_size", "software_aware",
    }
    if args.dram_bank_mappings:
        invalid = sorted(
            item.strip()
            for item in args.dram_bank_mappings.split(",")
            if item.strip() and item.strip() not in allowed_bank_mappings
        )
        if invalid:
            errors.append(f"--dram-bank-mappings contains invalid values: {', '.join(invalid)}")
    return errors


def make_trace_config(
    args: argparse.Namespace,
    spatial_policy: str | None = None,
    dram_bank_mapping: str | None = None,
) -> TraceConfig:
    return TraceConfig(
        npu_freq_mhz=args.npu_freq_mhz,
        duration_ms=args.duration_ms,
        max_bins=args.max_bins,
        major_op_samples=args.major_op_samples,
        major_op_percentile=args.major_op_percentile,
        grid=args.grid,
        spatial_policy=spatial_policy or args.spatial_policy,
        die_size_mm=args.die_size_mm,
        ambient_c=args.ambient_c,
        dram_layers=args.dram_layers,
        dram_capacity_mb=int(round(args.dram_capacity_gb * 1024)),
        hbm_package_capacity_mb=int(round(args.hbm_package_capacity_gb * 1024)),
        hbm_package_area_mm2=args.hbm_package_area_mm2,
        hbm_package_aspect_ratio=args.hbm_package_aspect_ratio,
        hbm_banks_per_package=args.hbm_banks_per_package,
        hbm_interleave_stripe_bytes=args.hbm_interleave_stripe_bytes,
        dram_floorplan_granularity=args.dram_floorplan_granularity,
        dram_bank_mapping=dram_bank_mapping or args.dram_bank_mapping,
        logic_floorplan=args.logic_floorplan,
        hotspot_grid=args.hotspot_grid,
        noc_power_backend=args.noc_power_backend,
        noc_power_flit_bits=args.noc_power_flit_bits,
        noc_power_injection_rate=args.noc_power_injection_rate,
        noc_power_link_length_mm=args.noc_power_link_length_mm,
        noc_power_dsent_tech=args.noc_power_dsent_tech,
        noc_power_orion_version=args.noc_power_orion_version,
    )


def make_thermal_config(args: argparse.Namespace) -> ThermalConfig:
    return ThermalConfig(
        ambient_c=args.ambient_c,
        throttle_c=args.throttle_c,
        simple_r_k_per_w=args.simple_r_k_per_w,
        simple_c_j_per_k=args.simple_c_j_per_k,
        stack_r_sink_k_per_w=args.stack_r_sink_k_per_w,
        stack_r_vertical_k_per_w=args.stack_r_vertical_k_per_w,
        stack_r_lateral_k_per_w=args.stack_r_lateral_k_per_w,
        logic_c_j_per_k=args.logic_c_j_per_k,
        dram_c_j_per_k=args.dram_c_j_per_k,
    )


def row_from_result(artifact, trace_cfg, trace, result, package_dir: Path, backend_results,
                    package_exported: bool, args: argparse.Namespace) -> Dict[str, float | str | int | bool]:
    runtime_s = artifact.summary.exec_cycles / (args.npu_freq_mhz * 1e6)
    trace_avg_power_w = float(trace.total_power_w.mean())
    spatial = trace.spatial_attribution or {}
    spatial_components = ",".join(str(component) for component in spatial.get("components", []))
    spatial_events = int(getattr(trace, "spatial_events", 0) or spatial.get("attributed_event_count", 0))
    inferred_events = int(getattr(trace, "inferred_events", 0))
    fallback_events = int(getattr(trace, "fallback_events", 0) or spatial.get("fallback_event_count", 0))
    spatial_trace_used = trace.block_power_w is not None and (spatial_events > 0 or inferred_events > 0 or bool(spatial.get("used", False)))
    temporal = temporal_sampling_summary(artifact, trace_cfg, trace)
    avg_power_error_pct = 0.0
    if artifact.summary.total_power_w:
        avg_power_error_pct = (trace_avg_power_w / artifact.summary.total_power_w - 1.0) * 100.0
    logic_area_mm2 = estimate_logic_area_mm2(artifact)
    power_density = (
        artifact.summary.total_power_w / logic_area_mm2
        if logic_area_mm2 and math.isfinite(logic_area_mm2)
        else float("nan")
    )
    simple_backend = backend_results.get("simple")
    hotspot_backend = backend_results.get("hotspot")
    hotspot_status = hotspot_backend.status if hotspot_backend is not None else "not_requested"
    hotspot_peak_c = None
    if hotspot_backend is not None:
        hotspot_peak_c = hotspot_backend.metrics.get("hotspot_peak_c")
    threedice_backend = backend_results.get("threedice")
    threedice_status = threedice_backend.status if threedice_backend is not None else "not_requested"
    threedice_peak_c = None
    if threedice_backend is not None:
        threedice_peak_c = threedice_backend.metrics.get("threedice_peak_c")
    adaptive_backend = backend_results.get("adaptive_grid")
    adaptive_status = adaptive_backend.status if adaptive_backend is not None else "not_requested"
    adaptive_peak_c = None
    if adaptive_backend is not None:
        adaptive_peak_c = adaptive_backend.metrics.get("adaptive_grid_peak_c")
    backend_status_json = json.dumps(
        {name: backend.status for name, backend in sorted(backend_results.items())},
        sort_keys=True,
    )
    return {
        "label": artifact.run_id.label(),
        "model": artifact.run_id.model,
        "mode": artifact.run_id.mode,
        "impl": artifact.run_id.impl,
        "num_cores": artifact.run_id.num_cores,
        "sa": artifact.run_id.sa,
        "vu": artifact.run_id.vu,
        "sram_kb": artifact.run_id.sram_kb,
        "dram_bw": artifact.run_id.dram_bw,
        "noc_topo": artifact.run_id.noc_topo,
        "noc_bw": artifact.run_id.noc_bw,
        "core_group": artifact.run_id.core_group,
        "row": artifact.run_id.row,
        "spatial_policy": trace_cfg.spatial_policy,
        "grid": trace_cfg.grid,
        "hotspot_grid": trace_cfg.hotspot_grid,
        "dram_layers": trace_cfg.dram_layers,
        "dram_capacity_gb": trace_cfg.dram_capacity_mb / 1024,
        "hbm_package_count": math.ceil(trace_cfg.dram_capacity_mb / trace_cfg.hbm_package_capacity_mb),
        "hbm_package_area_mm2": trace_cfg.hbm_package_area_mm2,
        "dram_floorplan_granularity": trace_cfg.dram_floorplan_granularity,
        "dram_bank_mapping": resolve_dram_bank_mapping(trace_cfg, artifact),
        "hbm_banks_per_package": trace_cfg.hbm_banks_per_package,
        "logic_floorplan": trace_cfg.logic_floorplan,
        "noc_power_backend": trace_cfg.noc_power_backend,
        "noc_power_flit_bits": trace_cfg.noc_power_flit_bits,
        "noc_power_injection_rate": trace_cfg.noc_power_injection_rate,
        "noc_power_link_length_mm": trace_cfg.noc_power_link_length_mm,
        "noc_power_dsent_tech": trace_cfg.noc_power_dsent_tech,
        "thermal_backends": ",".join(sorted(backend_results)),
        "backend_status": backend_status_json,
        "simple_status": simple_backend.status if simple_backend is not None else "not_requested",
        "exec_ms": runtime_s * 1e3,
        "thermal_window_ms": trace.duration_s * 1e3,
        "thermal_repetitions": trace.duration_s / runtime_s if runtime_s > 0 else float("nan"),
        "thermal_bins": temporal["bins"],
        "thermal_dt_us": temporal["dt_s"] * 1e6,
        "major_op_samples_target": temporal["major_op_samples_target"],
        "major_ops_meeting_target_fraction": temporal["major_ops_meeting_target_fraction"],
        "target_major_op_duration_us": temporal["target_major_op_duration_s"] * 1e6,
        "avg_power_w": artifact.summary.total_power_w,
        "dyn_power_w": artifact.summary.dynamic_power_w,
        "trace_avg_power_w": trace_avg_power_w,
        "trace_avg_power_error_pct": avg_power_error_pct,
        "trace_peak_power_w": float(trace.total_power_w.max()),
        "spatial_events": spatial_events,
        "inferred_spatial_events": inferred_events,
        "fallback_events": fallback_events,
        "spatial_trace_used": spatial_trace_used,
        "spatial_components": spatial_components,
        "spatial_ops_with_metadata": int(spatial.get("ops_with_spatial_metadata", 0) or spatial_events),
        "logic_area_mm2": logic_area_mm2,
        "simple_power_density_w_per_mm2": power_density,
        "simple_power_density_margin_w_per_mm2": args.power_density_threshold - power_density,
        "simple_peak_c": result.simple_peak_c,
        "stack_logic_peak_c": result.stack_logic_peak_c,
        "stack_dram_peak_c": result.stack_dram_peak_c,
        "stack_peak_c": result.stack_peak_c,
        "stack_p95_peak_c": result.stack_p95_peak_c,
        "simple_margin_c": args.throttle_c - result.simple_peak_c,
        "stack_margin_c": args.throttle_c - result.stack_peak_c,
        "simple_slowdown": result.simple_slowdown,
        "stack_slowdown": result.stack_slowdown,
        "slowdown_delta_pct": result.slowdown_delta_pct,
        "package_dir": str(package_dir),
        "package_exported": package_exported,
        "hotspot_status": hotspot_status,
        "hotspot_peak_c": hotspot_peak_c if hotspot_peak_c is not None else "",
        "hotspot_delta_vs_stack_c": (hotspot_peak_c - result.stack_peak_c) if hotspot_peak_c is not None else "",
        "hotspot_margin_c": (args.throttle_c - hotspot_peak_c) if hotspot_peak_c is not None else "",
        "threedice_status": threedice_status,
        "threedice_peak_c": threedice_peak_c if threedice_peak_c is not None else "",
        "threedice_delta_vs_stack_c": (threedice_peak_c - result.stack_peak_c) if threedice_peak_c is not None else "",
        "threedice_margin_c": (args.throttle_c - threedice_peak_c) if threedice_peak_c is not None else "",
        "adaptive_grid_status": adaptive_status,
        "adaptive_grid_peak_c": adaptive_peak_c if adaptive_peak_c is not None else "",
        "adaptive_grid_delta_vs_stack_c": (adaptive_peak_c - result.stack_peak_c) if adaptive_peak_c is not None else "",
        "adaptive_grid_margin_c": (args.throttle_c - adaptive_peak_c) if adaptive_peak_c is not None else "",
    }


def validation_failures(rows: List[Dict[str, float | str | int | bool]], args: argparse.Namespace) -> List[str]:
    from .models import spearman

    failures: List[str] = []
    max_power_err = max(abs(float(row["trace_avg_power_error_pct"])) for row in rows)
    max_temp_delta = max(abs(float(row["simple_peak_c"]) - float(row["stack_peak_c"])) for row in rows)
    max_slowdown_delta = max(abs(float(row["slowdown_delta_pct"])) for row in rows)
    if max_power_err > args.max_trace_power_error_pct:
        failures.append(
            f"trace average-power error {max_power_err:.3f}% exceeds {args.max_trace_power_error_pct:.3f}%"
        )
    if max_temp_delta > args.max_temp_delta_c:
        failures.append(f"simple-vs-stack peak delta {max_temp_delta:.3f} C exceeds {args.max_temp_delta_c:.3f} C")
    if max_slowdown_delta > args.max_slowdown_delta_pct:
        failures.append(
            f"slowdown delta {max_slowdown_delta:.3f}% exceeds {args.max_slowdown_delta_pct:.3f}%"
        )
    if any(float(row["simple_margin_c"]) >= 0 and float(row["stack_margin_c"]) < 0 for row in rows):
        failures.append("simple lumped model has a false-safe thermal-limit classification")

    policies = sorted({str(row["spatial_policy"]) for row in rows})
    for policy in policies:
        group = [row for row in rows if str(row["spatial_policy"]) == policy]
        if len(group) < 2:
            continue
        rho = spearman(
            [float(row["simple_peak_c"]) for row in group],
            [float(row["stack_peak_c"]) for row in group],
        )
        if math.isfinite(rho) and rho < args.min_policy_spearman:
            failures.append(
                f"{policy} simple-vs-stack rank correlation {rho:.4f} is below {args.min_policy_spearman:.4f}"
            )
    return failures


def main() -> int:
    args = parse_args()
    arg_errors = validate_args(args)
    if arg_errors:
        for error in arg_errors:
            print(error)
        return 2
    artifacts = load_artifacts(args.results_dir, args.models, args.modes, args.impls, args.npu_freq_mhz)
    dram_bw_filter = split_int_filter(args.dram_bws)
    row_filter = split_int_filter(args.rows)
    core_group_filter = split_int_filter(args.core_groups)
    if dram_bw_filter or row_filter or core_group_filter:
        artifacts = [
            artifact for artifact in artifacts
            if (not dram_bw_filter or artifact.run_id.dram_bw in dram_bw_filter)
            and (not row_filter or artifact.run_id.row in row_filter)
            and (not core_group_filter or artifact.run_id.core_group in core_group_filter)
        ]
    if not artifacts:
        print(f"No analyzable TSIM logs found under {args.results_dir}")
        return 1
    backend_names = parse_backend_names(args.thermal_backends, args.run_hotspot)
    report_backend_names = list(backend_names)
    if "simple" not in backend_names:
        # The current validation/report schema is anchored on TSIM's simple
        # baseline, so compute it even for export-only or HotSpot-only runs.
        backend_names = ["simple", *backend_names]
    backends = resolve_backends(backend_names)

    allowed_policies = {"uniform", "center_hotspot", "edge_hotspot"}
    spatial_policies = [args.spatial_policy]
    if args.spatial_policies:
        spatial_policies = [item.strip() for item in args.spatial_policies.split(",") if item.strip()]
        invalid = sorted(set(spatial_policies) - allowed_policies)
        if invalid:
            print(f"Invalid spatial policies: {', '.join(invalid)}")
            return 2
    dram_bank_mappings = [args.dram_bank_mapping]
    if args.dram_bank_mappings:
        dram_bank_mappings = [item.strip() for item in args.dram_bank_mappings.split(",") if item.strip()]

    thermal_cfg = make_thermal_config(args)
    backend_options = ThermalBackendOptions(
        hotspot_bin=args.hotspot_bin,
        hotspot_timeout_s=args.hotspot_timeout_s,
        threedice_bin=args.threedice_bin,
        threedice_timeout_s=args.threedice_timeout_s,
        threedice_heatmap_count=args.threedice_heatmap_count,
        threedice_heatmap_layers=args.threedice_heatmap_layers,
        adaptive_grid_heatmap_count=args.adaptive_grid_heatmap_count,
        operator_hotspot_analysis=args.operator_hotspot_analysis,
        operator_heatmap_count=args.operator_heatmap_count,
    )
    rows: List[Dict[str, float | str | int | bool]] = []
    packages_dir = args.out_dir / "packages"
    for artifact in artifacts:
        for policy in spatial_policies:
            for bank_mapping in dram_bank_mappings:
                trace_cfg = make_trace_config(args, policy, bank_mapping)
                trace = build_component_power_trace(artifact, trace_cfg)
                safe_name = (
                    f"{artifact.run_id.model}_{artifact.run_id.mode}_{artifact.run_id.impl}_"
                    f"c{artifact.run_id.num_cores}_sa{artifact.run_id.sa}_bw{artifact.run_id.dram_bw}_"
                    f"sram{artifact.run_id.sram_kb}_cg{artifact.run_id.core_group}_"
                    f"row{artifact.run_id.row}_{policy}_drammap{bank_mapping}"
                )
                package_dir = packages_dir / safe_name
                package_exported = any(backend.requires_package for backend in backends)
                if package_exported:
                    export_trace_package(artifact, trace, trace_cfg, package_dir)
                backend_results = {}
                context = ThermalBackendContext(
                    artifact=artifact,
                    trace_cfg=trace_cfg,
                    thermal_cfg=thermal_cfg,
                    trace=trace,
                    package_dir=package_dir,
                    options=backend_options,
                    prior_results=backend_results,
                )
                for backend in backends:
                    backend_results[backend.name] = backend.run(context)
                result = backend_results["simple"].raw
                row = row_from_result(artifact, trace_cfg, trace, result, package_dir, backend_results, package_exported, args)
                row["requested_thermal_backends"] = ",".join(report_backend_names)
                rows.append(row)

    rows.sort(key=lambda row: (
        str(row["model"]), str(row["mode"]), int(row["num_cores"]),
        int(row["dram_bw"]), int(row["sram_kb"]), int(row["core_group"]),
        str(row["spatial_policy"]), str(row.get("dram_bank_mapping", "")),
    ))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "thermal_compare.csv"
    md_path = args.out_dir / "thermal_compare.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    failures = validation_failures(rows, args)
    status_path = args.out_dir / "validation_status.json"
    backend_status = {}
    for row in rows:
        statuses = json.loads(str(row.get("backend_status", "{}")))
        for backend, status in statuses.items():
            backend_status.setdefault(backend, {})
            backend_status[backend][status] = backend_status[backend].get(status, 0) + 1
    status_path.write_text(
        json.dumps({
            "passed": not failures,
            "failures": failures,
            "requested_thermal_backends": report_backend_names,
            "effective_thermal_backends": backend_names,
            "backend_status": backend_status,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {status_path}")
    print(md_path.read_text(encoding="utf-8"))
    if failures:
        print("Validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 3
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
