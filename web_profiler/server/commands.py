"""Command builders for launching simulations, sweeps, DSE and thermal runs.

These build the exact command lines used by ``benchmark_scripts/run_all_tests.py``
(verified against the source) so web-launched runs produce results that are
indistinguishable from batch-launched ones.  Hardware configs are generated
with the same schema as ``run_all_tests.make_config`` and are only ever
*added* — existing files are never overwritten.
"""

import itertools
import json
import shlex
from pathlib import Path
from typing import Dict, List, Optional

from .config import PROJECT_ROOT, RESULTS_DIR, HW_CONFIG_DIR, VENV_PYTHON

# Model metadata: name -> transformer layer count (mirrors run_all_tests.all_list)
MODEL_INFO = {
    "llama2-13": {"layers": 40, "params_b": 13},
    "llama3-70": {"layers": 80, "params_b": 70},
    "opt-30":    {"layers": 48, "params_b": 30},
    "gemma2":    {"layers": 46, "params_b": 27},
    "dit-xl":    {"layers": 32, "params_b": 0.7},
}

CL = 14
TRCD = 14
TRP_DEFAULT = 14
NPU_FREQ_MHZ = 1500
SPLIT_FACTOR_DECODE = 1.03
SPLIT_FACTOR_PREFILL = 1.1

IMPL_FLAGS = {
    "best": None,
    "uniform_dram": "--uniform_dram_mapping",
    "spmd_compiler": "--spmd_compiler",
    "seq_noc": "--seq_noc",
    "dataflow": "--dataflow",
    "ipu_tsim": "--ipu_tsim",
}

# Parameter tiers offered by the Run page sliders (multi-select enabled).
SWEEP_TIERS = {
    "sa_size":    [16, 23, 32, 45, 64, 90, 128],
    "sram_kb":    [512, 1024, 2048, 4096, 8192, 16384],
    "dram_bw":    [4000, 6000, 8000, 12000, 12288, 16000, 20000, 24000],
    "num_cores":  [32, 64, 128, 256, 512, 1024],
    "noc_bw":     [4, 8, 16, 32, 64],
    "noc_topo":   [1, 2, 3],
    "core_group": [1, 2, 4, 8, 16, 32],
    "row":        [512, 1024, 2048, 4096, 8192],
    "batch_size": [1, 8, 16, 32, 64],
    "sequence_length": [1024, 2048, 4096],
}

DEFAULT_SIM_PARAMS = {
    "model": "llama2-13",
    "mode": "decode",
    "batch_size": 32,
    "sequence_length": 2048,
    "sa_size": 32,
    "sram_kb": 2048,
    "dram_bw": 12288,
    "num_cores": 256,
    "noc_topo": 1,
    "noc_bw": 16,
    "core_group": 8,
    "row": 8192,
    "impl": "best",
    "trcd": TRCD,
    "trp": None,           # None -> default naming, no filename override
    "split_factor": None,  # None -> auto by mode
    "sim_layers": None,    # None -> auto by mode
    "root": "logs_web",    # web runs land in results/logs_web
}

# Sweep dimensions that may receive a list of values (cartesian product).
_SWEEPABLE = ("sa_size", "sram_kb", "dram_bw", "num_cores", "noc_bw",
              "noc_topo", "core_group", "row", "batch_size",
              "sequence_length", "model", "impl")


def make_hw_config_dict(sa: int, noc_topo: int, noc_bw: int, dram_bw: int,
                        row: int, trp: int = TRP_DEFAULT,
                        freq_mhz: int = NPU_FREQ_MHZ) -> Dict:
    """Same schema as run_all_tests.make_config."""
    return {
        "compute": {
            "ew_pad_len": sa,
            "mm_pad_shape": [sa, sa, sa],
            "ew_reuse_num": sa,
            "mm_reuse_list": [sa, sa * sa, sa],
            "ew_flopc": sa,
            "mm_flopc": 2 * sa * sa,
            "load_store_bw_bytepc": sa * 2,
            "byte_per_elem": 2,
            "mm_init_cycle": sa,
            "ew_mm_overlap": True,
        },
        "noc": {
            "bandwidth_bytepc": noc_bw,
            "topology": noc_topo,
            "default_noc": True,
        },
        "dram": {
            "CL": CL,
            "tRCD": TRCD,
            "tRP": trp,
            "bandwidth_GBps": dram_bw,
            "num_access_per_row": row,
            "npu_freq_MHz": freq_mhz,
        },
    }


def hw_config_name(sa: int, noc_topo: int, noc_bw: int, dram_bw: int,
                   trp: int = TRP_DEFAULT) -> str:
    return (f"sa_{sa}_vu_{sa}_drambw_{dram_bw}_noc_{noc_topo}_{noc_bw}"
            f"_trcd_{TRCD}_trp_{trp}")


def ensure_hw_config(sa: int, noc_topo: int, noc_bw: int, dram_bw: int,
                     row: int, trp: int = TRP_DEFAULT) -> str:
    """Return the hw_config name, creating the JSON only if it does not exist.

    Never overwrites: if a file with the same name exists it is reused as-is;
    if it exists with different content a ``web_``-prefixed variant is written.
    """
    name = hw_config_name(sa, noc_topo, noc_bw, dram_bw, trp)
    path = HW_CONFIG_DIR / f"{name}.json"
    want = make_hw_config_dict(sa, noc_topo, noc_bw, dram_bw, row, trp)
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
            if existing == want:
                return name
        except ValueError:
            pass
        # name taken with different content: fall back to a web-specific name
        name = f"web_{name}_row_{row}"
        path = HW_CONFIG_DIR / f"{name}.json"
        if path.is_file():
            try:
                if json.loads(path.read_text()) == want:
                    return name
            except ValueError:
                pass
    path.write_text(json.dumps(want, indent=4))
    return name


def expected_output_path(p: Dict) -> Path:
    """Predict the output log path for a sim (mirrors get_cfg_str/one_pass)."""
    trp = p.get("trp")
    suffix = f"_trcd_{p.get('trcd', TRCD)}_trp_{trp}" if trp is not None else ""
    root = p.get("root") or "logs_web"
    base = RESULTS_DIR / root
    return (base / p["model"] / f"bs_{p['batch_size']}" /
            f"core_{p['num_cores']}" / p["mode"] /
            f"sa_{p['sa_size']}-vu_{p['sa_size']}" /
            f"sram_{p['sram_kb']}-drambw_{p['dram_bw']}_PLACEHOLDER" /
            f"topo_{p['noc_topo']}-nocbw{p['noc_bw']}" / p["impl"] /
            f"output_cg_{p['core_group']}_row_{p['row']}{suffix}.log")


def build_sim_command(params: Dict) -> Dict:
    """Build one icbm_launch.py invocation.

    ``params`` follows DEFAULT_SIM_PARAMS.  Returns argv, a shell display
    string, the predicted output path and the hw_config name (created lazily
    by the caller via :func:`ensure_hw_config` — dry-run/reproduce paths pass
    ``create_config=False``).
    """
    p = {**DEFAULT_SIM_PARAMS, **params}
    mode = p["mode"]
    prefill = mode == "prefill"
    seq = p["sequence_length"]
    layers = MODEL_INFO.get(p["model"], {}).get("layers", 0)
    split_factor = p.get("split_factor")
    if split_factor is None:
        split_factor = SPLIT_FACTOR_PREFILL if prefill else SPLIT_FACTOR_DECODE
    sim_layers = p.get("sim_layers")
    if sim_layers is None:
        sim_layers = 1 if prefill else 0
    trp = p.get("trp") if p.get("trp") is not None else TRP_DEFAULT

    hw_name = hw_config_name(p["sa_size"], p["noc_topo"], p["noc_bw"],
                             p["dram_bw"], trp)
    output_dir = RESULTS_DIR / "pickles" / \
        f"outputs_icbm_{seq}{'_prefill' if prefill else ''}"
    trace_base = RESULTS_DIR / (p.get("root") or "logs_web")

    argv = [str(VENV_PYTHON), "icbm_launch.py", p["model"],
            str(p["num_cores"]),
            "--core_mem_kb", str(p["sram_kb"]),
            "--output_dir", str(output_dir),
            "--layers", str(layers),
            "--batch_size", str(p["batch_size"]),
            "--sequence_length", str(seq),
            "--use_pickle",
            "--split_factor", str(split_factor),
            "--hw_json", hw_name,
            "--core_group", str(p["core_group"]),
            "--dram_name", "PLACEHOLDER",
            "--dram_bw", str(p["dram_bw"]),
            "--sim_layers", str(sim_layers),
            "--trace_out_dir_base", str(trace_base)]
    flag = IMPL_FLAGS.get(p["impl"])
    if flag:
        argv.append(flag)
    if prefill:
        argv.append("--prefill")
    label = (f"{p['model']} {mode} bs{p['batch_size']} c{p['num_cores']} "
             f"sa{p['sa_size']} sram{p['sram_kb']} bw{p['dram_bw']} "
             f"topo{p['noc_topo']}/noc{p['noc_bw']} cg{p['core_group']} "
             f"{p['impl']}")
    return {
        "argv": argv,
        "display": " ".join(shlex.quote(a) for a in argv),
        "cwd": str(PROJECT_ROOT),
        "expected_output": str(expected_output_path(p)),
        "hw_config_name": hw_name,
        "hw_config_args": {"sa": p["sa_size"], "noc_topo": p["noc_topo"],
                           "noc_bw": p["noc_bw"], "dram_bw": p["dram_bw"],
                           "row": p["row"], "trp": trp},
        "label": label,
        "params": p,
    }


def expand_sim_specs(params: Dict) -> List[Dict]:
    """Expand a spec whose sweepable keys may be lists into individual sims."""
    keys = [k for k in _SWEEPABLE
            if isinstance(params.get(k), (list, tuple))]
    if not keys:
        return [params]
    value_lists = [list(params[k]) for k in keys]
    out = []
    for combo in itertools.product(*value_lists):
        spec = {k: v for k, v in params.items() if k not in keys}
        spec.update(dict(zip(keys, combo)))
        out.append(spec)
    return out


def build_dse_command(params: Dict) -> Dict:
    """dse_pareto.py invocation."""
    mode = params.get("mode", "decode")
    argv = [str(VENV_PYTHON), "dse_pareto.py", "--mode", mode]
    if params.get("models"):
        argv += ["--models", ",".join(params["models"])]
    if params.get("num_sweeps"):
        argv += ["--num-sweeps", str(params["num_sweeps"])]
    if params.get("max_cycles"):
        argv += ["--max-cycles", str(params["max_cycles"])]
    if params.get("exhaustive"):
        argv.append("--exhaustive")
    if params.get("core_group"):
        argv += ["--core-group", str(params["core_group"])]
    return {"argv": argv, "display": " ".join(shlex.quote(a) for a in argv),
            "cwd": str(PROJECT_ROOT), "label": f"DSE Pareto ({mode})"}


def build_thermal_command(params: Dict) -> Dict:
    """tsim_thermal.cli invocation (defaults to the built-in 'simple' backend,
    which needs no external solvers)."""
    argv = [str(VENV_PYTHON), "-m", "tsim_thermal.cli",
            "--results-dir", params.get("results_dir", "results/logs"),
            "--out-dir", params.get("out_dir",
                                    "results/thermal_validation_web"),
            "--models", params.get("models", "llama2-13"),
            "--modes", params.get("modes", "decode"),
            "--impls", params.get("impls", "best"),
            "--dram-bws", str(params.get("dram_bws", 12288)),
            "--rows", str(params.get("rows", 8192)),
            "--core-groups", str(params.get("core_groups", 8)),
            "--thermal-backends", params.get("backends", "simple")]
    if params.get("dram_layers"):
        argv += ["--dram-layers", str(params["dram_layers"])]
    return {"argv": argv, "display": " ".join(shlex.quote(a) for a in argv),
            "cwd": str(PROJECT_ROOT),
            "label": f"Thermal validation ({params.get('models', '')})"}
