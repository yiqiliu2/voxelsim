"""Load TSIM simulation artifacts for thermal validation."""

from __future__ import annotations

import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

OUTPUT_RE = re.compile(r"output_cg_(\d+)_row_(\d+)\.log$")
SUMMARY_RE = re.compile(
    r"EXE time \(total, fused\):\s*(\d+),Energy \(mJ\):\s*([-+eE0-9.]+),"
    r"\s*Static:\s*([-+eE0-9.]+)\s*mJ,\s*Dyn\.:\s*([-+eE0-9.]+)\s*mJ"
)
POWER_RE = re.compile(
    r"Power \(w\):\s*([-+eE0-9.]+),\s*Static:\s*([-+eE0-9.]+)\s*W"
    r"\s*\(dram:\s*([-+eE0-9.]+)\s*logic:\s*([-+eE0-9.]+)\),\s*Dyn\.:\s*([-+eE0-9.]+)\s*W"
)
STATIC_COMP_RE = re.compile(
    r"Static Energy: SA = ([-+eE0-9.]+) mJ, VU = ([-+eE0-9.]+) mJ,"
    r"\s*SRAM= ([-+eE0-9.]+) mJ, Core: ([-+eE0-9.]+) mJ"
)
STATIC_OTHER_RE = re.compile(
    r"Static Energy: DRAM = ([-+eE0-9.]+) mJ, TSV = ([-+eE0-9.]+) mJ,"
    r"\s*NoC: ([-+eE0-9.]+) mJ"
)
DYNAMIC_COMP_RE = re.compile(
    r"Dynamic Energy: SA = ([-+eE0-9.]+) mJ, VU = ([-+eE0-9.]+) mJ,"
    r"\s*SRAM= ([-+eE0-9.]+) mJ, Core: ([-+eE0-9.]+) mJ,"
    r"\s*NoC: ([-+eE0-9.]+) mJ"
)
DYNAMIC_OTHER_RE = re.compile(
    r"Dynamic Energy: DRAM = ([-+eE0-9.]+) mJ, TSV = ([-+eE0-9.]+) mJ"
)
COMP_DIR_RE = re.compile(r"sa_(\d+)-vu_(\d+)")
MEM_DIR_RE = re.compile(r"sram_(\d+)-drambw_(\d+)_")
CORE_DIR_RE = re.compile(r"core_(\d+)")
TOPO_DIR_RE = re.compile(r"topo_(\d+)-nocbw(\d+)")
BATCH_DIR_RE = re.compile(r"bs_(\d+)")

COMPONENTS = ("sa", "vu", "sram", "noc", "dram", "tsv")
LOGIC_COMPONENTS = ("sa", "vu", "sram", "noc", "tsv")


@dataclass(frozen=True)
class RunId:
    model: str
    batch_size: int
    mode: str
    impl: str
    num_cores: int
    sa: int
    vu: int
    sram_kb: int
    dram_bw: int
    noc_topo: int
    noc_bw: int
    core_group: int
    row: int

    def label(self) -> str:
        return (
            f"{self.model}/bs={self.batch_size}/{self.mode}/{self.impl}: cores={self.num_cores}, "
            f"sa={self.sa}, vu={self.vu}, sram={self.sram_kb}KB, dram_bw={self.dram_bw}, "
            f"noc={self.noc_topo}:{self.noc_bw}, cg={self.core_group}, row={self.row}"
        )


@dataclass
class RunSummary:
    exec_cycles: int
    total_energy_mj: float
    static_energy_mj: float
    dynamic_energy_mj: float
    total_power_w: float
    static_power_w: float
    dram_static_power_w: float
    logic_static_power_w: float
    dynamic_power_w: float
    static_component_w: Dict[str, float]
    dynamic_component_w: Dict[str, float]


@dataclass
class RunArtifacts:
    run_id: RunId
    log_path: Path
    pickle_path: Path
    summary: RunSummary
    op_logs: List[object]


def split_filter(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_run_id(log_path: Path, results_dir: Path) -> RunId | None:
    match = OUTPUT_RE.match(log_path.name)
    if not match:
        return None
    core_group = int(match.group(1))
    row = int(match.group(2))

    rel_parts = log_path.relative_to(results_dir).parts
    model = rel_parts[0] if rel_parts else "unknown"
    mode = "unknown"
    impl = log_path.parent.name
    batch_size = None
    num_cores = sa = vu = sram_kb = dram_bw = noc_topo = noc_bw = None

    for part in rel_parts:
        if (m := BATCH_DIR_RE.match(part)):
            batch_size = int(m.group(1))
        if part in {"decode", "prefill"}:
            mode = part
        if (m := CORE_DIR_RE.match(part)):
            num_cores = int(m.group(1))
        if (m := COMP_DIR_RE.match(part)):
            sa = int(m.group(1))
            vu = int(m.group(2))
        if (m := MEM_DIR_RE.match(part)):
            sram_kb = int(m.group(1))
            dram_bw = int(m.group(2))
        if (m := TOPO_DIR_RE.match(part)):
            noc_topo = int(m.group(1))
            noc_bw = int(m.group(2))

    if None in {batch_size, num_cores, sa, vu, sram_kb, dram_bw, noc_topo, noc_bw}:
        return None
    return RunId(model, batch_size, mode, impl, num_cores, sa, vu, sram_kb, dram_bw,
                 noc_topo, noc_bw, core_group, row)


def parse_summary(log_path: Path, npu_freq_mhz: float) -> RunSummary:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    summary_match = next((SUMMARY_RE.search(line) for line in lines if SUMMARY_RE.search(line)), None)
    power_match = next((POWER_RE.search(line) for line in lines if POWER_RE.search(line)), None)
    if not summary_match or not power_match:
        raise ValueError(f"Could not parse summary/power block: {log_path}")

    exec_cycles = int(summary_match.group(1))
    total_energy_mj = float(summary_match.group(2))
    static_energy_mj = float(summary_match.group(3))
    dynamic_energy_mj = float(summary_match.group(4))
    total_power_w = float(power_match.group(1))
    static_power_w = float(power_match.group(2))
    dram_static_w = float(power_match.group(3))
    logic_static_w = float(power_match.group(4))
    dynamic_power_w = float(power_match.group(5))

    static_component_w = {name: 0.0 for name in COMPONENTS}
    dynamic_component_w = {name: 0.0 for name in COMPONENTS}
    runtime_s = exec_cycles / (npu_freq_mhz * 1e6)
    static_comp_match = next((STATIC_COMP_RE.search(line) for line in lines if STATIC_COMP_RE.search(line)), None)
    static_other_match = next((STATIC_OTHER_RE.search(line) for line in lines if STATIC_OTHER_RE.search(line)), None)
    if runtime_s > 0 and static_comp_match and static_other_match:
        static_component_w["sa"] = (float(static_comp_match.group(1)) / 1e3) / runtime_s
        static_component_w["vu"] = (float(static_comp_match.group(2)) / 1e3) / runtime_s
        static_component_w["sram"] = (float(static_comp_match.group(3)) / 1e3) / runtime_s
        static_component_w["dram"] = (float(static_other_match.group(1)) / 1e3) / runtime_s
        static_component_w["tsv"] = (float(static_other_match.group(2)) / 1e3) / runtime_s
        static_component_w["noc"] = (float(static_other_match.group(3)) / 1e3) / runtime_s
    else:
        for component in LOGIC_COMPONENTS:
            static_component_w[component] = logic_static_w / len(LOGIC_COMPONENTS)
        static_component_w["dram"] = dram_static_w

    parsed_static = sum(static_component_w.values())
    if parsed_static > 0:
        scale = static_power_w / parsed_static
        static_component_w = {name: value * scale for name, value in static_component_w.items()}

    dynamic_comp_match = next((DYNAMIC_COMP_RE.search(line) for line in lines if DYNAMIC_COMP_RE.search(line)), None)
    dynamic_other_match = next((DYNAMIC_OTHER_RE.search(line) for line in lines if DYNAMIC_OTHER_RE.search(line)), None)
    if runtime_s > 0 and dynamic_comp_match and dynamic_other_match:
        dynamic_component_w["sa"] = (float(dynamic_comp_match.group(1)) / 1e3) / runtime_s
        dynamic_component_w["vu"] = (float(dynamic_comp_match.group(2)) / 1e3) / runtime_s
        dynamic_component_w["sram"] = (float(dynamic_comp_match.group(3)) / 1e3) / runtime_s
        dynamic_component_w["noc"] = (float(dynamic_comp_match.group(5)) / 1e3) / runtime_s
        dynamic_component_w["dram"] = (float(dynamic_other_match.group(1)) / 1e3) / runtime_s
        dynamic_component_w["tsv"] = (float(dynamic_other_match.group(2)) / 1e3) / runtime_s
        parsed_dynamic = sum(dynamic_component_w.values())
        if parsed_dynamic > 0:
            scale = dynamic_power_w / parsed_dynamic
            dynamic_component_w = {name: value * scale for name, value in dynamic_component_w.items()}

    return RunSummary(exec_cycles, total_energy_mj, static_energy_mj,
                      dynamic_energy_mj, total_power_w, static_power_w,
                      dram_static_w, logic_static_w, dynamic_power_w,
                      static_component_w, dynamic_component_w)


def find_pickle(log_path: Path, core_group: int) -> Path | None:
    candidate = log_path.with_name(f"output_cg_{core_group}.pickle")
    return candidate if candidate.exists() else None


def load_artifacts(
    results_dir: Path,
    models: str = "",
    modes: str = "decode,prefill",
    impls: str = "best",
    npu_freq_mhz: float = 1500.0,
) -> List[RunArtifacts]:
    model_filter = split_filter(models)
    mode_filter = split_filter(modes)
    impl_filter = split_filter(impls)
    artifacts: List[RunArtifacts] = []
    for log_path in sorted(results_dir.rglob("output_cg_*_row_*.log")):
        run_id = parse_run_id(log_path, results_dir)
        if run_id is None:
            continue
        if model_filter and run_id.model not in model_filter:
            continue
        if mode_filter and run_id.mode not in mode_filter:
            continue
        if impl_filter and run_id.impl not in impl_filter:
            continue
        pickle_path = find_pickle(log_path, run_id.core_group)
        if pickle_path is None:
            print(f"Skipping {log_path}: no fused-op pickle found", file=sys.stderr)
            continue
        with open(pickle_path, "rb") as infile:
            op_logs = pickle.load(infile)
        artifacts.append(
            RunArtifacts(
                run_id=run_id,
                log_path=log_path,
                pickle_path=pickle_path,
                summary=parse_summary(log_path, npu_freq_mhz),
                op_logs=op_logs,
            )
        )
    return artifacts
