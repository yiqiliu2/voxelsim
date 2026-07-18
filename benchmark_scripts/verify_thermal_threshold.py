#!/usr/bin/env python3
"""Compute peak power density for each hardware configuration.

This utility scans simulator logs, extracts the maximum total power
observed for every unique hardware design, and reports the associated
chip area and power density.  The design key combines the systolic array
size, vector unit size, core count, DRAM bandwidth, SRAM capacity, and
NoC parameters present in the directory structure.
"""

from __future__ import annotations

import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

# Ensure project root is on sys.path so tsim_components/tsim_simple imports work
# regardless of the working directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tsim_components.comp_util import Compute
from tsim_components.mem import (
    DRAM,
    get_per_cycle_bytes_per_core_from_DRAM_config,
)
from tsim_components.noc import NoC
from tsim_simple import get_area_breakdown

# Regular expressions for parsing log content and directory names.
POWER_RE = re.compile(
    r"Power\s*\(w\)\s*:\s*([-+eE0-9\.]+)\s*,\s*Static\s*:\s*([-+eE0-9\.]+)\s*W"
    r"(?:\s*\(dram:\s*([-+eE0-9\.]+)\s*logic:\s*([-+eE0-9\.]+)\))?\s*,\s*"
    r"Dyn\.\s*:\s*([-+eE0-9\.]+)\s*W",
    re.IGNORECASE,
)

CORE_DIR_RE = re.compile(r"core_(\d+)")
COMP_DIR_RE = re.compile(r"sa_(\d+)-vu_(\d+)")
MEM_DIR_RE = re.compile(r"sram_(\d+)-drambw_(\d+)_")
TOPO_DIR_RE = re.compile(r"topo_(\d+)-nocbw(\d+)")
LOG_NAME_RE = re.compile(r"output_cg_(\d+)_row_(\d+)\.log")

TRACE_OUT = "results/logs"

HBM_COLS = 64
NPU_FREQ_MHZ = 2000
HBM_FREQ_MHZ = 3200
HBM_TIMING = tuple(
    14 / (HBM_FREQ_MHZ / NPU_FREQ_MHZ) for _ in range(3)
)
DRAM_CAPACITY_MB = 64 * 1024


@dataclass(frozen=True)
class DesignKey:
    """Canonical identifier for a hardware design."""

    sa: int
    vu: int
    num_cores: int
    dram_bw: int
    sram_kb: int
    noc_topo: int
    noc_bw: int

    def to_cfg_dict(self) -> Dict[str, int]:
        return {
            "sa": self.sa,
            "vu": self.vu,
            "num_cores": self.num_cores,
            "dram_bw": self.dram_bw,
            "sram_kb": self.sram_kb,
            "noc_topo": self.noc_topo,
            "noc_bw": self.noc_bw,
        }


def repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_power_metrics(log_path: str) -> float | None:
    """Return the total power in Watts recorded in the summary block."""

    with open(log_path, "r", encoding="utf-8") as infile:
        content = infile.read()

    match = POWER_RE.search(content)
    if match is None:
        return None

    total_power = float(match.group(1))
    return total_power


def parse_design_from_path(path_parts: Iterable[str]) -> Tuple[DesignKey, Dict[str, int], str | None, str | None] | None:
    """Extract the hardware configuration and model/stage from a log file path."""

    sa = vu = num_cores = dram_bw = sram_kb = noc_topo = noc_bw = None
    model = None
    stage = None
    after_trace_root = False

    for part in path_parts:
        if not after_trace_root:
            if part == TRACE_OUT:
                after_trace_root = True
            continue

        if model is None and part and not part.startswith("bs_") and not part.startswith("core_"):
            if part not in {"best", "spmd_compiler", "uniform_dram", "seq_noc", "no_fuse"} and not part.startswith("sa_") and not part.startswith("sram_"):
                model = part

        if stage is None and part in {"decode", "prefill"}:
            stage = part

        if sa is None:
            comp_match = COMP_DIR_RE.search(part)
            if comp_match:
                sa = int(comp_match.group(1))
                vu = int(comp_match.group(2))
                continue
        if dram_bw is None:
            mem_match = MEM_DIR_RE.search(part)
            if mem_match:
                sram_kb = int(mem_match.group(1))
                dram_bw = int(mem_match.group(2))
                continue
        if noc_topo is None:
            topo_match = TOPO_DIR_RE.search(part)
            if topo_match:
                noc_topo = int(topo_match.group(1))
                noc_bw = int(topo_match.group(2))
                continue
        if num_cores is None:
            core_match = CORE_DIR_RE.search(part)
            if core_match:
                num_cores = int(core_match.group(1))
                continue

    if sa is None or vu is None or num_cores is None or dram_bw is None or sram_kb is None:
        return None

    if noc_topo is None or noc_bw is None:
        noc_topo = 1
        noc_bw = 16

    design = DesignKey(sa, vu, num_cores, dram_bw, sram_kb, noc_topo, noc_bw)

    cfg = {
        "sa": sa,
        "vu": vu,
        "num_cores": num_cores,
        "dram_bw": dram_bw,
        "sram_kb": sram_kb,
        "noc_topo": noc_topo,
        "noc_bw": noc_bw,
    }
    return design, cfg, model, stage


def estimate_logic_area_mm2(cfg: Dict[str, int]) -> float:
    """Compute total on-die logic area (SA + VU + SRAM + NoC)."""

    num_cores = cfg["num_cores"]
    sa_size = cfg["sa"]
    vu_size = cfg.get("vu", sa_size)
    noc_topo = cfg["noc_topo"]
    noc_bw = cfg["noc_bw"]
    dram_bw = cfg["dram_bw"]
    sram_kb = cfg["sram_kb"]

    mm_shape = [sa_size, sa_size, sa_size]
    comp = Compute(
        vu_size,
        mm_shape,
        vu_size,
        mm_shape,
        vu_size,
        2 * mm_shape[-1] ** 2,
        mm_shape[-1],
        2,
    )

    nodes = list(range(num_cores))
    noc = NoC(noc_bw, noc_topo, nodes)

    bytes_per_cycle = get_per_cycle_bytes_per_core_from_DRAM_config(
        num_cores=num_cores,
        total_bandwidth_GBps=dram_bw,
        npu_freq_MHz=NPU_FREQ_MHZ,
    )
    bytes_per_row = bytes_per_cycle * HBM_COLS
    dram = DRAM(
        HBM_TIMING[0],
        HBM_TIMING[1],
        HBM_TIMING[2],
        int(bytes_per_row),
        int(bytes_per_cycle),
        num_cores=num_cores,
    )

    area, logic_area = get_area_breakdown(
        comp,
        noc,
        dram,
        per_core_sram_size_KB=sram_kb,
        total_dram_size_MB=DRAM_CAPACITY_MB,
        num_cores=num_cores,
        sram_type="SRAM",
        dram_type="HBM",
    )
    return float(logic_area)


def collect_design_metrics() -> List[Tuple[DesignKey, float, float, str | None, str | None]]:
    root = os.path.join(repo_root(), TRACE_OUT)
    results: Dict[DesignKey, Tuple[float, str | None, str | None]] = {}
    area_cache: Dict[DesignKey, float] = {}

    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".log"):
                continue
            log_match = LOG_NAME_RE.match(filename)
            if not log_match:
                continue

            path_parts = dirpath.split(os.sep)
            parsed = parse_design_from_path(path_parts)
            if parsed is None:
                continue
            design, cfg, model, stage = parsed

            log_path = os.path.join(dirpath, filename)
            total_power = parse_power_metrics(log_path)
            if total_power is None or not math.isfinite(total_power):
                continue

            prev_entry = results.get(design)
            prev_max = prev_entry[0] if prev_entry else float("-inf")
            if total_power > prev_max:
                results[design] = (total_power, model, stage)
                if design not in area_cache:
                    area_cache[design] = estimate_logic_area_mm2(cfg)

    aggregated: List[Tuple[DesignKey, float, float, str | None, str | None]] = []
    for design, (max_power, model, stage) in results.items():
        area = area_cache.get(design)
        if area is None:
            area = estimate_logic_area_mm2(design.to_cfg_dict())
        aggregated.append((design, max_power, area, model, stage))

    aggregated.sort(key=lambda item: item[1] / item[2] if item[2] else -math.inf, reverse=True)
    return aggregated


def format_design(design: DesignKey) -> str:
    return (
        f"SA={design.sa:<3d} VU={design.vu:<3d} cores={design.num_cores:<4d} "
        f"DRAM_BW={design.dram_bw:<6d} SRAM_KB={design.sram_kb:<5d} "
        f"NoC(topology={design.noc_topo}, bw={design.noc_bw})"
    )


def main() -> None:
    rows = collect_design_metrics()
    if not rows:
        print("No log files found under", os.path.join(repo_root(), TRACE_OUT))
        return

    header = (
        "Design Configuration",
        "Max Power [W]",
        "Logic Area [mm^2]",
        "Power Density [W/mm^2]",
        "Model",
        "Stage",
    )
    print(
        f"{header[0]:<70s} {header[1]:>14s} {header[2]:>16s} {header[3]:>20s} {header[4]:>12s} {header[5]:>10s}"
    )
    print("-" * 140)

    for design, max_power, area, model, stage in rows:
        density = max_power / area if area else float("nan")
        print(
            f"{format_design(design):<70s} "
            f"{max_power:14.3f} {area:16.3f} {density:20.6f} "
            f"{(model or 'n/a'):>12s} {(stage or 'n/a'):>10s}"
        )


if __name__ == "__main__":
    main()
