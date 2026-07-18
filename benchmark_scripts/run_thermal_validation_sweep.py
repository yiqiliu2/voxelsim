#!/usr/bin/env python3
"""Run a small TSIM sweep and then launch thermal validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark_scripts.run_all_tests import make_config


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TSIM points for thermal validation and compare thermal models."
    )
    parser.add_argument("--model", default="llama2-13")
    parser.add_argument("--num-cores", type=int, default=256)
    parser.add_argument("--core-mem-kb", type=int, default=2048)
    parser.add_argument("--dram-bws", default="8192,12288,16384")
    parser.add_argument("--core-group", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--sim-layers", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=SRC_DIR / "results" / "thermal_validation")
    parser.add_argument("--spatial-policies", default="uniform,center_hotspot,edge_hotspot")
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--max-bins", type=int, default=10000)
    parser.add_argument("--major-op-samples", type=int, default=4)
    parser.add_argument("--major-op-percentile", type=float, default=25.0)
    parser.add_argument("--dram-layers", type=int, default=1)
    parser.add_argument("--dram-capacity-gb", type=float, default=192.0)
    parser.add_argument("--hbm-package-capacity-gb", type=float, default=16.0)
    parser.add_argument("--hbm-package-area-mm2", type=float, default=87.62745402745404)
    parser.add_argument("--hbm-package-aspect-ratio", type=float, default=1.0)
    parser.add_argument("--hbm-banks-per-package", type=int, default=16)
    parser.add_argument("--hbm-interleave-stripe-bytes", type=int, default=256)
    parser.add_argument("--dram-floorplan-granularity", choices=("package", "bank"), default="bank")
    parser.add_argument("--dram-bank-mapping",
                        choices=("address_trace", "hbm_interleave", "hbm-interleave",
                                 "fine_interleave", "fine-interleave", "bank_interleave", "bank-interleave",
                                 "from_impl", "uniform", "interleave_size", "software_aware"),
                        default="address_trace")
    parser.add_argument("--dram-bank-mappings", default="")
    parser.add_argument("--thermal-backends", default="simple")
    parser.add_argument("--noc-power-backend", choices=("tsim_simple", "simple", "dsent", "orion"), default="tsim_simple")
    parser.add_argument("--noc-power-flit-bits", type=int, default=64)
    parser.add_argument("--noc-power-injection-rate", type=float, default=0.3)
    parser.add_argument("--noc-power-link-length-mm", type=float, default=1.0)
    parser.add_argument("--noc-power-dsent-tech", default="TG11LVT")
    parser.add_argument("--logic-floorplan", choices=("intra_core", "coarse_grid"), default="intra_core")
    parser.add_argument(
        "--die-size-mm",
        type=float,
        default=None,
        help="Optional square logic die side; omitted by default so thermal export derives it from simulator logic area.",
    )
    parser.add_argument("--hotspot-grid", type=int, default=128)
    parser.add_argument("--run-hotspot", action="store_true")
    parser.add_argument("--hotspot-bin", default="hotspot")
    parser.add_argument("--hotspot-timeout-s", type=float, default=300.0)
    parser.add_argument("--operator-hotspot-analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--operator-heatmap-count", type=int, default=12)
    parser.add_argument("--force", action="store_true", help="Rerun TSIM even if the expected log already exists.")
    return parser.parse_args()


def config_name(dram_bw: int, noc_bw: int = 32) -> str:
    return f"sa_32_vu_32_drambw_{dram_bw}_noc_1_{noc_bw}"


def ensure_hw_config(dram_bw: int, npu_freq_mhz: int = 1500, noc_bw: int = 32) -> str:
    name = config_name(dram_bw, noc_bw)
    sa = 32
    old_cwd = Path.cwd()
    try:
        os.chdir(SRC_DIR)
        make_config(
            ew_pad_len=sa,
            mm_pad_shape=[sa, sa, sa],
            ew_reuse_num=sa,
            mm_reuse_list=[sa, sa**2, sa],
            ew_flopc=sa,
            mm_flopc=2 * sa**2,
            load_store_bw_bytepc=sa * 2,
            byte_per_elem=2,
            mm_init_cycle=sa,
            ew_mm_overlap=True,
            bandwidth_bytepc=noc_bw,
            topology=1,
            default_noc=False,
            CL=14,
            tRCD=14,
            tRP=14,
            bandwidth_GBps=dram_bw,
            num_access_per_row=64,
            npu_freq_MHz=npu_freq_mhz,
            cfg_name=name,
        )
    finally:
        os.chdir(old_cwd)
    return name


def expected_log(args: argparse.Namespace, dram_bw: int) -> Path:
    return (
        SRC_DIR / "results" / "logs" / args.model / f"bs_{args.batch_size}" /
        f"core_{args.num_cores}" / "decode" / "sa_32-vu_32" /
        f"sram_{args.core_mem_kb}-drambw_{dram_bw}_PLACEHOLDER" /
        "topo_1-nocbw32" / "best" / f"output_cg_{args.core_group}_row_64.log"
    )


def run_checked(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def main() -> int:
    args = parse_args()
    dram_bws = parse_int_list(args.dram_bws)
    out_dir = args.out_dir.resolve()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/3dstack-mpl")

    for dram_bw in dram_bws:
        hw_json = ensure_hw_config(dram_bw)
        log_path = expected_log(args, dram_bw)
        if log_path.exists() and not args.force:
            print(f"Skipping existing TSIM result: {log_path}", flush=True)
            continue
        run_checked([
            sys.executable,
            "icbm_launch.py",
            args.model,
            str(args.num_cores),
            "--core_mem_kb", str(args.core_mem_kb),
            "--output_dir", f"results/pickles/outputs_icbm_{args.core_mem_kb}",
            "--layers", str(args.layers),
            "--batch_size", str(args.batch_size),
            "--sequence_length", str(args.sequence_length),
            "--use_pickle",
            "--split_factor", "1.03",
            "--hw_json", hw_json,
            "--core_group", str(args.core_group),
            "--dram_name", "PLACEHOLDER",
            "--dram_bw", str(dram_bw),
            "--sim_layers", str(args.sim_layers),
            "--trace_out_dir_base", "results/logs",
        ], SRC_DIR, env)

    compare_cmd = [
        sys.executable,
        str(SRC_DIR / "benchmark_scripts" / "thermal_compare.py"),
        "--results-dir", str(SRC_DIR / "results" / "logs"),
        "--out-dir", str(out_dir),
        "--models", args.model,
        "--modes", "decode",
        "--impls", "best",
        "--duration-ms", str(args.duration_ms),
        "--max-bins", str(args.max_bins),
        "--major-op-samples", str(args.major_op_samples),
        "--major-op-percentile", str(args.major_op_percentile),
        "--dram-layers", str(args.dram_layers),
        "--dram-capacity-gb", str(args.dram_capacity_gb),
        "--hbm-package-capacity-gb", str(args.hbm_package_capacity_gb),
        "--hbm-package-area-mm2", str(args.hbm_package_area_mm2),
        "--hbm-package-aspect-ratio", str(args.hbm_package_aspect_ratio),
        "--hbm-banks-per-package", str(args.hbm_banks_per_package),
        "--hbm-interleave-stripe-bytes", str(args.hbm_interleave_stripe_bytes),
        "--dram-floorplan-granularity", args.dram_floorplan_granularity,
        "--dram-bank-mapping", args.dram_bank_mapping,
        "--thermal-backends", args.thermal_backends,
        "--noc-power-backend", args.noc_power_backend,
        "--noc-power-flit-bits", str(args.noc_power_flit_bits),
        "--noc-power-injection-rate", str(args.noc_power_injection_rate),
        "--noc-power-link-length-mm", str(args.noc_power_link_length_mm),
        "--noc-power-dsent-tech", args.noc_power_dsent_tech,
        "--logic-floorplan", args.logic_floorplan,
        "--hotspot-grid", str(args.hotspot_grid),
        "--operator-heatmap-count", str(args.operator_heatmap_count),
        "--dram-bws", args.dram_bws,
        "--rows", "64",
        "--core-groups", str(args.core_group),
        "--spatial-policies", args.spatial_policies,
    ]
    if args.die_size_mm is not None:
        compare_cmd.extend(["--die-size-mm", str(args.die_size_mm)])
    if args.run_hotspot:
        compare_cmd.extend([
            "--run-hotspot",
            "--hotspot-bin", args.hotspot_bin,
            "--hotspot-timeout-s", str(args.hotspot_timeout_s),
        ])
    if not args.operator_hotspot_analysis:
        compare_cmd.append("--no-operator-hotspot-analysis")
    if args.dram_bank_mappings:
        compare_cmd.extend(["--dram-bank-mappings", args.dram_bank_mappings])
    run_checked(compare_cmd, SRC_DIR, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
