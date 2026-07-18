"""Paths and constants for the VoxelSim Workbench backend.

Everything is rooted at the project directory (the parent of ``web_profiler``).
No existing project files are modified; runtime state lives in
``web_profiler/runtime/``.
"""

import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WEB_DIR = Path(__file__).resolve().parent.parent          # web_profiler/
PROJECT_ROOT = WEB_DIR.parent                             # repo root
RUNTIME_DIR = WEB_DIR / "runtime"
JOBS_DIR = RUNTIME_DIR / "jobs"
CACHE_DIR = RUNTIME_DIR / "cache"

RESULTS_DIR = PROJECT_ROOT / "results"
HW_CONFIG_DIR = PROJECT_ROOT / "hw_config"
MODELS_DIR = PROJECT_ROOT / "models"
PARSED_MODELS_DIR = MODELS_DIR / "parsed"
ORIGINAL_MODELS_DIR = MODELS_DIR / "original"
NOC_TABLES_DIR = PROJECT_ROOT / "noc_distance_tables"
PICKLES_DIR = RESULTS_DIR / "pickles"
TEST_LOGS_DIR = PROJECT_ROOT / "test_logs"
LLMSERVING_DIR = PROJECT_ROOT / "llmservingsim"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"

for _d in (RUNTIME_DIR, JOBS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Result roots
# ---------------------------------------------------------------------------

def discover_result_roots():
    """Return [{name, path, seq_length}] for every results/logs*-style tree.

    ``results/logs`` is the primary root (seq 2048 default). ``logs_seq<N>``
    roots hold sequence-length sweeps. ``results_pareto_*/logs`` hold DSE runs.
    """
    roots = []

    def _add(name, path, seq):
        if path.is_dir():
            roots.append({"name": name, "path": str(path), "seq_length": seq})

    # Every results/logs*-style tree: logs, logs_seq<N>, logs_matrix_*,
    # logs_web (web-launched runs), thermal_matrix_* leftovers, ...
    for p in sorted(RESULTS_DIR.glob("logs*")):
        if not p.is_dir():
            continue
        m = re.search(r"seq(\d+)", p.name)
        seq = int(m.group(1)) if m else 2048
        _add(p.name, p, seq)
    for p in sorted(PROJECT_ROOT.glob("results_pareto_*")):
        if p.is_dir():
            _add(f"pareto_{p.name.rsplit('_', 1)[-1]}", p / "logs", 2048)
    return roots

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

MODES = ["decode", "prefill"]
NPU_FREQ_MHZ_DEFAULT = 1500
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ_DEFAULT * 1e3)

# Hardware units appearing in overlap logs (mirrors tsim_analysis_lib.HW_UNITS)
HW_UNITS = ["dram_r", "dram_w", "noc", "comp", "comp_sa", "comp_vu",
            "comp_sram_r", "comp_sram_w"]

IMPL_LABELS = {
    "best": "Compute-Shift (optimized)",
    "seq_noc": "Sequential mapping",
    "uniform_dram": "Uniform DRAM map",
    "dataflow": "Dataflow",
    "spmd_compiler": "SPMD",
    "ipu_tsim": "IPU",
}

TOPO_LABELS = {1: "2D Mesh", 2: "2D Torus", 3: "All-to-All"}

# Log file name pattern: output_cg{G}_row_{R}[_trcd_X_trp_Y].log
# (the separator after "cg" is an optional underscore: both output_cg8_*
# and output_cg_8_* occur in the result trees)
LOG_NAME_RE = re.compile(
    r"^output_cg_?(\d+)_row_(\d+)(?:_trcd_(\d+)_trp_(\d+))?\.log$")

# Directory pattern: {model}/bs_{B}/core_{C}/{mode}/sa_{S}-vu_{V}/
#   sram_{KB}-drambw_{BW}_{dram_name}/topo_{T}-nocbw{W}/{impl}/
DIR_PATTERNS = {
    "batch_size": re.compile(r"^bs_(\d+)$"),
    "num_cores": re.compile(r"^core_(\d+)$"),
    "sa": re.compile(r"^sa_(\d+)-vu_(\d+)$"),
    "sram": re.compile(r"^sram_(\d+)-drambw_(\d+)_(.+)$"),
    "noc": re.compile(r"^topo_(\d+)-nocbw(\d+)$"),
}

# Metrics exposed by summary parsing, with display metadata.
# key -> (label, unit, better: 'lower'|'higher')
SUMMARY_METRICS = {
    "total_time":        ("Execution Time", "cycles", "lower"),
    "time_ms":           ("Latency", "ms", "lower"),
    "total_energy_mj":   ("Total Energy", "mJ", "lower"),
    "static_energy_mj":  ("Static Energy", "mJ", "lower"),
    "dynamic_energy_mj": ("Dynamic Energy", "mJ", "lower"),
    "avg_power_w":       ("Average Power", "W", "lower"),
    "static_power_w":    ("Static Power", "W", "lower"),
    "dynamic_power_w":   ("Dynamic Power", "W", "lower"),
    "overall_util":      ("Overall Util", "", "higher"),
    "dram_r_util":       ("DRAM Read Util", "%", "higher"),
    "dram_w_util":       ("DRAM Write Util", "%", "higher"),
    "sa_util":           ("SA Util", "", "higher"),
    "vu_util":           ("VU Util", "", "higher"),
    "noc_util":          ("NoC Util", "", "higher"),
    "mm_gflops":         ("MM Throughput", "GFLOPS", "higher"),
    "vu_gflops":         ("VU Throughput", "GFLOPS", "higher"),
}

SWEEP_PARAMS = {
    "batch_size": "Batch Size",
    "num_cores":  "Number of Cores",
    "sa_size":    "Systolic Array Size",
    "sram_kb":    "Per-core SRAM (KB)",
    "dram_bw":    "DRAM Bandwidth (GB/s)",
    "noc_topo":   "NoC Topology",
    "noc_bw":     "NoC Link BW (B/cycle)",
    "core_group": "Core Group Size",
    "seq_length": "Sequence Length",
}
