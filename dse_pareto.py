#!/usr/bin/env python3
"""Pareto frontier: area vs geo-mean latency across DNN models.

Uses multi-level area-constrained coordinate descent to trace the
Pareto-optimal tradeoff curve between chip area and execution time.

Algorithm:
  1. Pre-compute area range from the full search space (instant, O(1) per config).
  2. Choose N area thresholds spanning [min_area, max_area].
  3. At each threshold, run coordinate descent minimizing geo-mean exe_time
     subject to area <= threshold (warm-started from previous level).
  4. Collect ALL evaluated points, extract Pareto front.
  5. Save JSON results + PNG plot.

 Usage:
     python3 dse_pareto.py --mode decode --num-sweeps 15
     python3 dse_pareto.py --mode decode --exhaustive        # full grid
    python3 dse_pareto.py --mode decode --plot-only          # re-plot from JSON
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Search space (same as dse_geomean.py) ──
SEARCH_SPACE = {
    "sa":        [16, 32, 64, 128, 256],
    "sram_kb":   [256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576],
    "num_cores": [32, 48, 64, 96, 128, 192, 256, 384, 512],
    "dram_bw":   [1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000],
}

# ── Fixed parameters ──
FIXED = {
    "noc_topo": 1,
    "noc_bw": 16,
    "core_group": 8,
    "row": 8192,
    "batch_size": 32,
    "seq_length": 2048,
}

# ── Models: name -> (layers, sf_decode, sf_prefill) ──
MODELS = {
    "llama2-13": (40, 1.03, 1.1),
    "llama3-70": (80, 1.03, 1.1),
    "opt-30":    (48, 1.03, 1.1),
    "gemma2":    (46, 1.03, 1.1),
    "dit-xl":  (32, 1.1,  1.1),
}

# ── Hardware constants ──
CL = 14
TRCD = 14
TRP = 14
NPU_FREQ_MHZ = 1500
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis

# Coordinate descent dimension order
PARAM_ORDER = ["sa", "num_cores", "sram_kb", "dram_bw"]

# DRAM bank area constant: 8 DRAM dies × (800 mm² - 9×bw_area_at_default_12288GB/s)
# bw_area_default = 18.4 mm²  →  (800 - 9×18.4) × 8 = 634.4 × 8 = 5075.2 mm²
_BW_AREA_DEFAULT = 18.4  # mm² at dram_bw=12288 GB/s
DRAM_BANK_AREA_CONST = (800 - 9 * _BW_AREA_DEFAULT) * 8  # = 5075.2 mm²

# ── Module-level state ──
TRACE_OUT = None
PICKLE_OUT = None
OUT_DIR = None

_hw_config_cache = {}
_area_cache = {}       # (sa, sram_kb, num_cores, dram_bw) -> (compute_die, area_cost)
_eval_cache = {}       # same key -> (area_cost, geo_mean_exe, per_model_dict)
_sim_log_lock = threading.Lock()


def _sim_workers(num_cores, prefill):
    """Concurrent sim limit by core count level (high→low: decode 112356, prefill 111235)."""
    if prefill:
        # Real prefill is batch-1 (~3GB/proc, not the old 60GB) -> fill the box.
        if num_cores <= 32:  return 8
        if num_cores <= 64:  return 6
        if num_cores <= 128: return 4
        if num_cores <= 256: return 3
        return 2
    else:
        if num_cores <= 32:  return 6
        if num_cores <= 64:  return 5
        if num_cores <= 128: return 3
        if num_cores <= 256: return 2
        return 1


def set_output_dirs(mode):
    """Set module-level output dirs based on mode."""
    global TRACE_OUT, PICKLE_OUT, OUT_DIR
    OUT_DIR = f"results_pareto_{mode}"
    TRACE_OUT = f"{OUT_DIR}/logs"
    PICKLE_OUT = f"{OUT_DIR}/pickles"


# ═══════════════════════════════════════════════════════════════════════
# Utility functions (self-contained, same logic as dse_geomean.py)
# ═══════════════════════════════════════════════════════════════════════

def _cfg_key(cfg):
    """Config dict -> hashable key."""
    return (cfg["sa"], cfg["sram_kb"], cfg["num_cores"], cfg["dram_bw"])


def make_hw_config(sa, dram_bw, noc_topo, noc_bw):
    """Generate hw_config JSON file, return config name string."""
    config_name = f"sa_{sa}_vu_{sa}_drambw_{dram_bw}_noc_{noc_topo}_{noc_bw}"
    if config_name in _hw_config_cache:
        return config_name

    config = {
        "compute": {
            "ew_pad_len": sa, "mm_pad_shape": [sa] * 3,
            "ew_reuse_num": sa, "mm_reuse_list": [sa, sa**2, sa],
            "ew_flopc": sa, "mm_flopc": 2 * sa**2,
            "load_store_bw_bytepc": sa * 2, "byte_per_elem": 2,
            "mm_init_cycle": sa, "ew_mm_overlap": True,
        },
        "noc": {
            "bandwidth_bytepc": noc_bw, "topology": noc_topo, "default_noc": True,
        },
        "dram": {
            "CL": CL, "tRCD": TRCD, "tRP": TRP,
            "bandwidth_GBps": dram_bw,
            "num_access_per_row": FIXED["row"],
            "npu_freq_MHz": NPU_FREQ_MHZ,
        },
    }
    os.makedirs("hw_config", exist_ok=True)
    with open(f"hw_config/{config_name}.json", "w") as f:
        json.dump(config, f, indent=4)
    _hw_config_cache[config_name] = True
    return config_name


def compute_area(cfg):
    """Compute chip area (mm^2). Cached.

    Returns (compute_die, area_cost):
      compute_die: logic die area = compute + noc + sram + 2*bw_area.
      area_cost:   full stack cost = compute + noc + sram + 9*bw_area + DRAM_BANK_AREA_CONST.
                   (73x = 1 compute face + 8 DRAM×9; DRAM_BANK_AREA_CONST=5075.2 = 8×(800-9×18.4))
    """
    key = _cfg_key(cfg)
    if key in _area_cache:
        return _area_cache[key]

    from tsim_components.mem import (get_per_cycle_bytes_per_core_from_DRAM_config,
                                     get_sram_area_from_size)
    from tsim_components.comp_util import Compute
    from tsim_components.noc import NoC

    num_cores = cfg["num_cores"]
    sa = cfg["sa"]
    comp = Compute(sa, [sa] * 3, sa, [sa] * 3, sa, 2 * sa**2, sa, 2)
    noc = NoC(cfg.get("noc_bw", FIXED["noc_bw"]),
              cfg.get("noc_topo", FIXED["noc_topo"]),
              list(range(num_cores)))
    sram_area = get_sram_area_from_size(cfg["sram_kb"], memtype="SRAM")
    logic_area = num_cores * (comp.get_area()[0] + noc.get_switch_area() + sram_area)
    # TSV/DRAM-BW area: 18.4 mm² at reference dram_bw=12288 GB/s, scales linearly.
    bw_area = cfg["dram_bw"] * (_BW_AREA_DEFAULT / 12288)
    # compute_die = logic + 1×bw_area  (compute die: 1 TSV face)
    compute_die = logic_area + bw_area
    # area_cost = logic + 1×bw (compute) + 8×9×bw (DRAM dies) + DRAM_BANK_AREA_CONST
    area_cost = logic_area + 73 * bw_area + DRAM_BANK_AREA_CONST

    _area_cache[key] = (compute_die, area_cost)
    return compute_die, area_cost


def get_log_path(model, num_cores, sa, sram_kb, dram_bw, mode):
    """Construct expected output log path."""
    comp_str = f"sa_{sa}-vu_{sa}"
    mem_str = f"sram_{sram_kb}-drambw_{dram_bw}_PLACEHOLDER"
    noc_str = f"topo_{FIXED['noc_topo']}-nocbw{FIXED['noc_bw']}"
    cg_str = f"cg_{FIXED['core_group']}_row_{FIXED['row']}"
    bs = 1 if mode == "prefill" else FIXED['batch_size']  # prefill = real batch 1
    return os.path.join(
        TRACE_OUT, model, f"bs_{bs}", f"core_{num_cores}",
        mode, comp_str, mem_str, noc_str, "best", f"output_{cg_str}.log"
    )


def check_result_exists(log_path):
    """Check if a valid simulation result exists at log_path.
    Returns False if no result, True if valid result, or the string
    'FAILED' if a previous run already marked this config as infeasible."""
    failed_path = log_path + ".failed"
    if os.path.exists(failed_path):
        return "FAILED"
    if not os.path.exists(log_path):
        return False
    try:
        with open(log_path, 'r') as f:
            return 'Overall Util:' in f.read()
    except Exception:
        return False


def mark_failed(log_path):
    """Write a failure marker so this config is not retried."""
    failed_path = log_path + ".failed"
    os.makedirs(os.path.dirname(failed_path), exist_ok=True)
    with open(failed_path, 'w') as f:
        f.write("1")


def parse_exe_time(log_path):
    """Parse execution time (cycles) from log file. Returns None on failure."""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            if 'Overall Util:' not in content:
                return None
            return float(content.split('\n')[0].split(', ')[1].split(':')[1].split(',')[0])
    except (FileNotFoundError, IndexError, ValueError):
        return None


def run_simulation(model, num_layers, split_factor, cfg, prefill):
    """Launch simulation subprocess and return exe_time. Reuses cached results."""
    mode = "prefill" if prefill else "decode"
    log_path = get_log_path(model, cfg["num_cores"], cfg["sa"],
                            cfg["sram_kb"], cfg["dram_bw"], mode)

    cached = check_result_exists(log_path)
    if cached == "FAILED":
        return None
    if cached:
        return parse_exe_time(log_path)

    hw_json = make_hw_config(cfg["sa"], cfg["dram_bw"],
                             FIXED["noc_topo"], FIXED["noc_bw"])
    seq_length = FIXED["seq_length"]  # real prefill: full seq 2048 at batch 1 (no //8 proxy)
    batch_size = 1 if prefill else FIXED['batch_size']

    output_dir = f"{PICKLE_OUT}/outputs_icbm_{FIXED['seq_length']}"
    if prefill:
        output_dir += "_prefill"

    sim_layers = 1 if prefill else 0

    cmd = (f"python3 icbm_launch.py {model} {cfg['num_cores']}"
           f" --core_mem_kb {cfg['sram_kb']}"
           f" --output_dir {output_dir}"
           f" --layers {num_layers}"
           f" --batch_size {batch_size}"
           f" --sequence_length {seq_length}"
           f" --use_pickle"
           f" --split_factor {split_factor}"
           f" --hw_json {hw_json}"
           f" --core_group {FIXED['core_group']}"
           f" --dram_name PLACEHOLDER"
           f" --dram_bw {cfg['dram_bw']}"
           f" --sim_layers {sim_layers}"
           f" --trace_out_dir_base {TRACE_OUT}")
    if prefill:
        cmd += " --prefill"

    print(f"      SIM: {model} cores={cfg['num_cores']} sa={cfg['sa']} "
          f"sram={cfg['sram_kb']} bw={cfg['dram_bw']} {mode}...", flush=True)

    os.makedirs(os.path.dirname(TRACE_OUT), exist_ok=True)
    sim_log = os.path.join(os.path.dirname(TRACE_OUT), "dse_sim.log")
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=3600)
    with _sim_log_lock:
        with open(sim_log, "ab") as log_f:
            log_f.write(result.stdout)
            log_f.write(result.stderr)

    exe_time = parse_exe_time(log_path)
    if exe_time is None:
        mark_failed(log_path)
        print(f"      WARNING: simulation failed (exit {result.returncode})", flush=True)
    else:
        print(f"      Done: {exe_time:.0f} cycles", flush=True)
    return exe_time


# ═══════════════════════════════════════════════════════════════════════
# Evaluation (geo-mean exe_time across models)
# ═══════════════════════════════════════════════════════════════════════

def evaluate_config(cfg, model_list, prefill):
    """Evaluate one config. Returns (area, geo_mean_exe, per_model_dict).

    geo_mean_exe is None if any model simulation fails.
    per_model_dict maps model_name -> exe_time (or None).
    Model simulations run concurrently; parallelism depends on num_cores.
    """
    key = _cfg_key(cfg)
    if key in _eval_cache:
        return _eval_cache[key]

    full_cfg = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
    _area_display, area_cost = compute_area(full_cfg)

    max_w = min(_sim_workers(cfg["num_cores"], prefill), len(model_list))

    def _run_one(model):
        layers, sf_d, sf_p = MODELS[model]
        sf = sf_p if prefill else sf_d
        return model, run_simulation(model, layers, sf, cfg, prefill)

    per_model = {}
    with ThreadPoolExecutor(max_workers=max_w) as ex:
        for model, exe in ex.map(_run_one, model_list):
            per_model[model] = exe

    log_exes = []
    for model in model_list:
        exe = per_model[model]
        if exe is None or exe <= 0:
            result = (area_cost, None, per_model)
            _eval_cache[key] = result
            return result
        log_exes.append(math.log(exe))

    geo_exe = math.exp(sum(log_exes) / len(log_exes))
    result = (area_cost, geo_exe, per_model)
    _eval_cache[key] = result
    return result


# ═══════════════════════════════════════════════════════════════════════
# Area pre-computation
# ═══════════════════════════════════════════════════════════════════════

def enumerate_all_configs():
    """Generate all configs in the search space."""
    keys = list(SEARCH_SPACE.keys())
    return [dict(zip(keys, vals))
            for vals in itertools.product(*[SEARCH_SPACE[k] for k in keys])]


def _bucket_worker(args):
    """Module-level worker: compute area for a bucket of configs (picklable for ProcessPoolExecutor)."""
    bucket, noc_topo, noc_bw = args
    out = []
    for cfg in bucket:
        full_cfg = {**cfg, "noc_topo": noc_topo, "noc_bw": noc_bw}
        compute_die, area_cost = compute_area(full_cfg)
        out.append((cfg, compute_die, area_cost))
    return out


def precompute_all_areas():
    """Compute area for every config in parallel. Returns sorted list of (cfg, compute_die, area_cost).

    Sorted by area_cost (used for Pareto threshold generation and constraints).
    Configs are split into cpu_count buckets; each process handles one bucket to
    bypass the GIL and get true CPU parallelism.
    """
    from concurrent.futures import ProcessPoolExecutor

    configs = enumerate_all_configs()
    n_workers = min(os.cpu_count() or 4, len(configs))
    buckets = [configs[i::n_workers] for i in range(n_workers)]
    noc_topo = FIXED["noc_topo"]
    noc_bw   = FIXED["noc_bw"]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        bucket_results = list(executor.map(_bucket_worker,
                                           [(b, noc_topo, noc_bw) for b in buckets]))

    results = [item for bucket in bucket_results for item in bucket]
    results.sort(key=lambda x: x[2])  # sort by area_cost
    return results


# ═══════════════════════════════════════════════════════════════════════
# Coordinate descent (minimise geo-mean exe_time, subject to area cap)
# ═══════════════════════════════════════════════════════════════════════

def sweep_one_level(area_cap, model_list, prefill, start_cfg=None, max_cycles=3):
    """Coordinate descent minimising geo_mean(exe_time) with area <= area_cap.

    Returns (best_cfg, area, geo_exe, per_model) or None if infeasible.
    """
    # Find a feasible starting point
    if start_cfg is not None:
        cfg = start_cfg.copy()
        full = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
        _, area_cost = compute_area(full)
        if area_cost > area_cap:
            # Shrink each param until feasible
            cfg = _shrink_to_fit(cfg, area_cap)
            if cfg is None:
                return None
    else:
        cfg = _shrink_to_fit({p: max(SEARCH_SPACE[p]) for p in SEARCH_SPACE}, area_cap)
        if cfg is None:
            return None

    for cycle in range(max_cycles):
        changed = False
        for param in PARAM_ORDER:
            old_val = cfg[param]
            best_val = old_val
            best_geo = float('inf')

            for val in SEARCH_SPACE[param]:
                test_cfg = cfg.copy()
                test_cfg[param] = val
                test_full = {**test_cfg, "noc_topo": FIXED["noc_topo"],
                             "noc_bw": FIXED["noc_bw"]}
                _, test_area_cost = compute_area(test_full)
                if test_area_cost > area_cap:
                    continue

                area, geo_exe, pm = evaluate_config(test_cfg, model_list, prefill)
                if geo_exe is not None and geo_exe < best_geo:
                    best_geo = geo_exe
                    best_val = val

            if best_val != old_val:
                cfg[param] = best_val
                changed = True

        if not changed:
            break

    area, geo_exe, per_model = evaluate_config(cfg, model_list, prefill)
    return cfg, area, geo_exe, per_model


def _shrink_to_fit(cfg, area_cap):
    """Greedily shrink params to fit area budget. Returns cfg or None."""
    cfg = cfg.copy()
    # Try shrinking each param from largest to smallest
    for param in reversed(PARAM_ORDER):
        for val in sorted(SEARCH_SPACE[param], reverse=True):
            cfg[param] = val
            full = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
            _, area_cost = compute_area(full)
            if area_cost <= area_cap:
                break
    # Final check
    full = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
    _, area_cost = compute_area(full)
    if area_cost > area_cap:
        return None
    return cfg


# ═══════════════════════════════════════════════════════════════════════
# Pareto front extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_pareto_front(points):
    """Extract Pareto front from list of (area, exe_time, cfg) tuples.

    Minimises both area and exe_time.
    Returns Pareto-optimal points sorted by area ascending.
    """
    valid = [(a, e, c) for a, e, c in points if e is not None]
    if not valid:
        return []
    # Sort by area asc, then exe_time asc (tiebreak)
    valid.sort(key=lambda x: (x[0], x[1]))

    pareto = []
    best_exe = float('inf')
    for a, e, c in valid:
        if e < best_exe:
            pareto.append((a, e, c))
            best_exe = e
    return pareto


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════

_MODE_COLORS = {
    'decode':  {'scatter': '#80b880',      'pareto': '#145c30',   'default': '#0a2e1a',  'annot_bg': 'lightyellow'},
    'prefill': {'scatter': '#daeaf7',      'pareto': '#74afd4',   'default': '#4a90c0',  'annot_bg': 'lightcyan'},
}


def plot_pareto(datasets, out_path):
    """Plot Pareto fronts for one or more modes on the same axes.

    datasets: list of (all_points, pareto_points, mode)
    Each mode gets its own color; decode=red, prefill=blue.
    Prefill points get letters A,B,C; decode points get D,E.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # DRAM_BANK_AREA is now baked into compute_area's area_cost (via DRAM_BANK_AREA_CONST)
    DRAM_BANK_AREA = 0

    # Latency is parsed in cycles; plot it in ms.  (Knee detection normalizes in
    # log-space, so a constant scale factor leaves the chosen points unchanged.)
    datasets = [([(a, e * CYCLE_TO_MS if e is not None else None, c)
                  for a, e, c in all_pts],
                 [(a, e * CYCLE_TO_MS if e is not None else None, c)
                  for a, e, c in par_pts], mode)
                for all_pts, par_pts, mode in datasets]

    # ── Layout macros ──────────────────────────────────────────────────────────
    _FIG_W, _FIG_H  = 7, 4.25           # figure size (inches)
    _FONT           = 9            # font size for legend and table
    _XLIM_PAD       = 1.05         # multiplicative x-axis padding around pareto range
    _LEG_X, _LEG_Y  = 0.895, 0.915  # legend anchor in axes fraction (upper-right corner)
    _TABLE_GAP      = 0.008      # gap (axes fraction) between legend right and table left
    _LS_SCALE       = 1.29         # linespacing multiplier over measured legend row height
    # ──────────────────────────────────────────────────────────────────────────

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))

    all_par_a_global = []
    star_pts = []   # (mode, cfg, die_area) for default-cfg stars

    # Plot prefill first so its legend entries appear above decode
    datasets_plot = sorted(datasets, key=lambda d: 0 if d[2] == 'prefill' else 1)
    for all_points, pareto_points, mode in datasets_plot:
        clr = _MODE_COLORS.get(mode, {'scatter': 'lightgray', 'pareto': 'green',
                                       'default': 'darkgreen', 'annot_bg': 'lightyellow'})

        valid = [(a + DRAM_BANK_AREA, e) for a, e, _ in all_points if e is not None]
        if not valid:
            continue
        va, ve = zip(*valid)
        par_a = [a + DRAM_BANK_AREA for a, _, __ in pareto_points]
        par_e = [e for _, e, __ in pareto_points]
        all_par_a_global.extend(par_a)

        ax.scatter(va, ve, s=15, facecolors='white', edgecolors=clr['pareto'],
                   linewidths=0.6, alpha=0.8,
                   label=f'Evaluated ({mode})', zorder=1)
        ax.step(par_a, par_e, where='post', color=clr['pareto'],
                linewidth=1.5, alpha=0.5, zorder=2)
        ax.scatter(par_a, par_e, color=clr['pareto'], s=50, zorder=3,
                   label=f'Pareto optimal ({mode})')

        # Default config marker — read geo_exe from existing simulation logs
        _DEFAULT_CFG = {'sa': 32, 'num_cores': 256, 'sram_kb': 2048, 'dram_bw': 12288,
                        'noc_topo': FIXED['noc_topo'], 'noc_bw': FIXED['noc_bw']}
        _sa, _nc, _sk, _bw = (_DEFAULT_CFG['sa'], _DEFAULT_CFG['num_cores'],
                               _DEFAULT_CFG['sram_kb'], _DEFAULT_CFG['dram_bw'])
        _cg = FIXED['core_group']
        _log_exes = []
        _bs = 1 if mode == "prefill" else FIXED['batch_size']  # prefill = real batch 1
        for _model in MODELS:
            _lp = (f"results/logs/{_model}/bs_{_bs}/core_{_nc}/{mode}/"
                   f"sa_{_sa}-vu_{_sa}/sram_{_sk}-drambw_{_bw}_PLACEHOLDER/"
                   f"topo_{FIXED['noc_topo']}-nocbw{FIXED['noc_bw']}/best/"
                   f"output_cg_{_cg}_row_{FIXED['row']}.log")
            _exe = parse_exe_time(_lp)
            if _exe:
                _log_exes.append(math.log(_exe))
        if _log_exes:
            _def_geo_exe = math.exp(sum(_log_exes) / len(_log_exes)) * CYCLE_TO_MS
            # Use stored area_mm2 from simulation data (same formula as labeled points)
            _def_a = next((a for a, e, c in all_points
                           if c.get('sa') == _sa and c.get('num_cores') == _nc
                           and c.get('sram_kb') == _sk and c.get('dram_bw') == _bw), None)
            _def_compute_die, _def_area_cost = compute_area(_DEFAULT_CFG)
            _def_area_cost = _def_a if _def_a is not None else _def_area_cost
            _def_px = _def_area_cost + DRAM_BANK_AREA
            _def_compute_die = max(_def_px / 9, _def_compute_die)
            star_pts.append((mode, _DEFAULT_CFG, _def_compute_die))
            ax.scatter([_def_px], [_def_geo_exe],
                       marker='*', s=600, color=clr['default'], zorder=8,
                       edgecolors='white', linewidths=2.0)  # no label; added via Line2D below
            # Add smaller legend handle here so order matches: Evaluated→Pareto→Default
            ax.plot([], [], linestyle='', marker='*', color=clr['default'],
                    markersize=13, markeredgecolor='white', markeredgewidth=1.0,
                    label=f'Default config ({mode})')

    if not all_par_a_global:
        print("No valid points to plot.")
        return

    ax.set_xlabel('Total Area Cost of All Die Layers (mm²)', fontsize=11)
    ax.set_ylabel('Geo-Mean Latency of All Tested Models (ms)', fontsize=11, labelpad=2, y=0.48)
    mode_str = ' + '.join(m for _, _, m in datasets)
    # ax.set_title(f'Pareto Frontier: Area vs Latency ({mode_str})', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(min(all_par_a_global) / _XLIM_PAD, max(all_par_a_global) * _XLIM_PAD)
    _X_TICKS = [6000, 8000, 10000, 15000, 20000]
    ax.set_xticks(_X_TICKS)
    ax.set_xticklabels([f'{v:,}' for v in _X_TICKS])

    N_LABELS = {'decode': 2, 'prefill': 3}

    def _knee_indices(pts, n):
        """Return n indices of pts with greatest perpendicular deviation (knee/elbow) on the
        log-space Pareto front, using divide-and-conquer, then greedy selection with separation."""
        import math
        if len(pts) <= n:
            return list(range(len(pts)))
        lx = [math.log10(a + DRAM_BANK_AREA) for a, e, _ in pts]
        ly = [math.log10(e) for a, e, _ in pts]
        xmin, xmax = min(lx), max(lx)
        ymin, ymax = min(ly), max(ly)
        def nrm(xi, yi):
            return (xi - xmin) / (xmax - xmin), (yi - ymin) / (ymax - ymin)
        npts = [nrm(xi, yi) for xi, yi in zip(lx, ly)]
        cands = []
        def collect(s, e):
            if e <= s + 1:
                return
            x0, y0 = npts[s]; x1, y1 = npts[e]
            dx, dy = x1 - x0, y1 - y0
            L = math.sqrt(dx * dx + dy * dy)
            if L == 0:
                return
            best_i, best_d = -1, 0
            for k in range(s + 1, e):
                xk, yk = npts[k]
                d = abs(dy * (xk - x0) - dx * (yk - y0)) / L
                if d > best_d:
                    best_d, best_i = d, k
            if best_i >= 0:
                cands.append((best_d, best_i))
                collect(s, best_i)
                collect(best_i, e)
        collect(0, len(pts) - 1)
        cands.sort(reverse=True)
        gap = max(2, len(pts) // (n + 1) // 2)
        selected = []
        for _, idx in cands:
            if len(selected) >= n:
                break
            if all(abs(idx - s) >= gap for s in selected):
                selected.append(idx)
        return sorted(selected)

    # Manual index overrides (bypass _knee_indices for specific modes)
    # decode[4]→top knee; area target 7800→E
    _LABEL_IDX = {
        'decode': [4],
    }
    # Area-target overrides: appended after index overrides (sorted by area ascending)
    _LABEL_AREA = {
        'decode': [7800],   # E: pareto point nearest 7800 mm²
    }

    # Label points: prefill first (A,B,C), then decode (D,E)
    labeled_pts = []   # (letter, mode, clr, px, e, cfg, die_area)
    letter_iter = iter('ABCDE')
    sorted_for_letters = sorted(datasets, key=lambda d: 0 if d[2] == 'prefill' else 1)
    for all_points, pareto_points, mode in sorted_for_letters:
        clr = _MODE_COLORS.get(mode, {'annot_bg': 'lightyellow'})
        if not pareto_points:
            continue
        n_labels = N_LABELS.get(mode, 5)
        if mode in _LABEL_IDX or mode in _LABEL_AREA:
            idxs = list(_LABEL_IDX.get(mode, []))
            for target_a in _LABEL_AREA.get(mode, []):
                best = min(range(len(pareto_points)),
                           key=lambda i: abs(pareto_points[i][0] + DRAM_BANK_AREA - target_a))
                if best not in idxs:
                    idxs.append(best)
            idxs = sorted(idxs)
        else:
            idxs = _knee_indices(pareto_points, n_labels)
        for i in idxs:
            a, e, cfg = pareto_points[i]
            full_cfg = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
            compute_die_area, _ = compute_area(full_cfg)
            px = a + DRAM_BANK_AREA
            compute_die_area = max(px / 9, compute_die_area)
            letter = next(letter_iter)
            labeled_pts.append((letter, mode, clr, px, e, cfg, compute_die_area))
            ax.scatter([px], [e], s=200, color=clr['pareto'], zorder=6,
                       edgecolors='white', lw=0.8)
            _txt_color = 'black' if mode == 'prefill' else 'white'
            ax.text(px, e, letter, fontsize=10, fontweight='bold',
                    ha='center', va='center', color=_txt_color, zorder=7)

    # 2-column legend: col0 = standard handles (Evaluated/Pareto/Default × modes),
    #                  col1 = A-E markers (letters only)
    leg_handles_std, leg_labels_std = ax.get_legend_handles_labels()

    letter_handles = []
    letter_labels_short = []
    for letter, mode, clr, px, e, cfg, die in labeled_pts:
        letter_handles.append(mlines.Line2D([], [], linestyle='', marker='o',
                                             color=clr['pareto'], markersize=7,
                                             markeredgecolor='white', markeredgewidth=0.8))
        letter_labels_short.append(letter)

    # Column-major fill: matplotlib fills col0 first then col1, so concatenate groups.
    # Prepend one invisible entry to col1 so A is one row lower (leaving room for table header).
    invisible = mlines.Line2D([], [], linestyle='', marker='', color='none')
    col1_offset = 1  # number of blank rows above A in col1
    letter_handles   = [invisible] * col1_offset + letter_handles
    letter_labels_short = [''] * col1_offset + letter_labels_short
    n1, n2 = len(leg_handles_std), len(letter_handles)
    nrows = max(n1, n2)
    h1 = leg_handles_std + [invisible] * (nrows - n1)
    l1 = leg_labels_std  + ['']        * (nrows - n1)
    h2 = letter_handles  + [invisible] * (nrows - n2)
    l2 = letter_labels_short + ['']    * (nrows - n2)
    all_h = h1 + h2
    all_l = l1 + l2

    # Config table — build text first (needed for width estimate)
    _s2 = '\u00b2'
    hdr1 = (f'{"SA":^4}  {"Core":>5}  {"SRAM":^7} {"DRAM":>5} {"Bottom":>8}')
    hdr2 = (f'{"Size":^4}  {"Count":>5}  {"MB/Core":>7} {"TB/s":>5} {"Die mm"+_s2:>8}')
    sep  = '\u2500' * max(len(hdr1), len(hdr2))
    rows = [hdr1, hdr2, sep]
    for letter, mode, clr, px, e, cfg, die in labeled_pts:
        bw_tbps  = round(cfg['dram_bw'] / 1000, 1)
        sram_mb  = round(cfg['sram_kb'] / 1024, 2)
        bw_str   = f'{bw_tbps:g}'
        _s = f'{sram_mb:g}'
        sram_str = _s[1:] if _s.startswith('0.') else _s
        rows.append(f'{str(cfg["sa"]):^4} {str(cfg["num_cores"]):>5}    '
                    f' {sram_str:<6}  {bw_str:<4}  {str(int(die)):>5}')
    table_txt = '\n'.join(rows)

    # First pass: place legend at anchor to measure its width
    ax.legend(handles=all_h, labels=all_l, ncol=2, fontsize=_FONT,
              loc='upper right', bbox_to_anchor=(_LEG_X, _LEG_Y),
              bbox_transform=ax.transAxes,
              handletextpad=0.5, columnspacing=0.8,
              edgecolor='black')

    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    abb = ax.get_window_extent(renderer=renderer)
    lbb = ax.get_legend().get_window_extent(renderer=renderer)
    leg_w_ax = (lbb.x1 - lbb.x0) / abb.width

    # Estimate table width (monospace 8pt ≈ 5px/char)
    max_chars = max(len(r) for r in rows)
    table_w_ax = (max_chars * 5.2 + 16) / abb.width  # +16px for bbox padding

    # Reposition legend so legend+gap+table all fit within axes (right-aligned)
    new_leg_right = min(_LEG_X - table_w_ax - _TABLE_GAP, _LEG_X)
    ax.get_legend().set_bbox_to_anchor((new_leg_right, _LEG_Y), transform=ax.transAxes)
    fig.canvas.draw()

    lbb2 = ax.get_legend().get_window_extent(renderer=renderer)
    leg_right_ax = (lbb2.x1 - abb.x0) / abb.width
    leg_top_ax   = (lbb2.y1 - abb.y0) / abb.height

    # Align table so data row A is at same height as legend label A.
    # Column-major fill: col1 starts at index nrows_fill in legend texts.
    nrows_fill = max(n1, n2)
    a_idx = nrows_fill + col1_offset  # index of 'A' in legend texts
    leg_texts = ax.get_legend().get_texts()
    linespacing = 2.2  # default fallback
    table_top_ax = leg_top_ax
    if len(leg_texts) > a_idx + 1:
        a_bb = leg_texts[a_idx    ].get_window_extent(renderer=renderer)
        b_bb = leg_texts[a_idx + 1].get_window_extent(renderer=renderer)
        # Actual legend row height in display pixels (A is above B)
        leg_row_h_px = ((a_bb.y0 + a_bb.y1) - (b_bb.y0 + b_bb.y1)) / 2
        text_h_px    = _FONT * 1.2 * fig.dpi / 72
        linespacing  = max(1.0, leg_row_h_px / text_h_px) * _LS_SCALE
        leg_row_h_ax = leg_row_h_px / abb.height
        a_mid_ax     = ((a_bb.y0 + a_bb.y1) / 2 - abb.y0) / abb.height
        # 3 prefix lines (hdr1, hdr2, sep) above data row A
        table_top_ax = a_mid_ax + (3.2) * leg_row_h_ax

    ax.text(leg_right_ax + _TABLE_GAP, table_top_ax, table_txt,
            transform=ax.transAxes, fontsize=_FONT, family='monospace',
            va='top', ha='left', linespacing=linespacing, clip_on=False,
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='black', alpha=0.90, lw=0.8))

    _DPI = 300
    plt.savefig(out_path, dpi=_DPI, bbox_inches='tight')
    print(f"Plot saved to {out_path}")


def _load_datasets_for_plot(primary_json, primary_mode):
    """Load primary mode results + other mode if available. Returns datasets list."""
    all_points, pareto, loaded_mode = load_results(primary_json)
    print(f"Loaded {loaded_mode}: {len(all_points)} points, {len(pareto)} Pareto-optimal")
    datasets = [(all_points, pareto, loaded_mode)]

    other_mode = 'prefill' if primary_mode == 'decode' else 'decode'
    other_json = f"results_pareto_{other_mode}/pareto_results.json"
    if os.path.exists(other_json):
        try:
            ap2, p2, lm2 = load_results(other_json)
            datasets.append((ap2, p2, lm2))
            print(f"Also loaded {lm2}: {len(ap2)} points, {len(p2)} Pareto-optimal")
        except Exception as e:
            print(f"Could not load {other_json}: {e}")
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# Save / load results
# ═══════════════════════════════════════════════════════════════════════

def _load_eval_cache_from_disk(result_json):
    """Pre-populate _eval_cache from existing results JSON on disk.
    Returns True if cache was loaded."""
    if not os.path.exists(result_json):
        return False
    try:
        all_points, _, _ = load_results(result_json)
        for area, geo_exe, cfg in all_points:
            key = _cfg_key(cfg)
            if key not in _eval_cache:
                _eval_cache[key] = (area, geo_exe, {})
        return True
    except Exception:
        return False
def save_results(new_points, pareto, mode, model_list, elapsed, out_path):
    """Save results to JSON, merging with any existing data on disk."""
    existing = {}
    if os.path.exists(out_path):
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except Exception:
            pass
    existing_all = {}
    for p in existing.get("all_points", []):
        c = p["config"]
        existing_all[(c["sa"], c["sram_kb"], c["num_cores"], c["dram_bw"])] = p
    for a, e, c in new_points:
        key = (c["sa"], c["sram_kb"], c["num_cores"], c["dram_bw"])
        existing_all[key] = {"config": c, "area_mm2": a, "geo_mean_exe": e}

    merged_all = list(existing_all.values())
    merged_pareto = extract_pareto_front(
        [(p["area_mm2"], p["geo_mean_exe"], p["config"]) for p in merged_all])

    output = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "models": model_list,
        "search_space": SEARCH_SPACE,
        "fixed_params": FIXED,
        "eval_time_sec": elapsed,
        "num_evaluated": len(merged_all),
        "num_pareto": len(merged_pareto),
        "all_points": merged_all,
        "pareto_front": [
            {"config": c, "area_mm2": a, "geo_mean_exe": e}
            for a, e, c in merged_pareto
        ],
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path} ({len(merged_all)} total, {len(merged_pareto)} Pareto)")


def load_results(json_path):
    """Load results from JSON. Returns (all_points, pareto_points)."""
    with open(json_path) as f:
        data = json.load(f)
    all_points = [(p["area_mm2"], p["geo_mean_exe"], p["config"])
                  for p in data["all_points"]]
    pareto = [(p["area_mm2"], p["geo_mean_exe"], p["config"])
              for p in data["pareto_front"]]
    return all_points, pareto, data.get("mode", "decode")


# ═══════════════════════════════════════════════════════════════════════
# Sweep mode: multi-level area-constrained descent
# ═══════════════════════════════════════════════════════════════════════

def run_sweep(model_list, prefill, num_sweeps, reverse=False, max_cycles=5):
    """Run multi-level area sweep. Returns (all_points, pareto).

    reverse=False (default): explore area thresholds small→large.
    reverse=True:            explore area thresholds large→small.
    """
    mode = "prefill" if prefill else "decode"

    # Pre-compute all areas
    print("Pre-computing areas for all configs...", flush=True)
    config_areas = precompute_all_areas()
    area_min = config_areas[0][2]
    area_max = config_areas[-1][2]
    print(f"  Area range: {area_min:.1f} — {area_max:.1f} mm² (cost)")
    print(f"  Compute die range: {config_areas[0][1]:.1f} — {config_areas[-1][1]:.1f} mm²")
    print(f"  Total configs in search space: {len(config_areas)}")

    # Build a sorted list of (area_cost, compute_die) for threshold annotation
    cost_to_display = [(r[2], r[1]) for r in config_areas]  # already sorted by area_cost

    # Generate area thresholds (geometrically spaced — constant ratio between levels)
    if num_sweeps <= 1:
        thresholds = [area_max]
    else:
        log_min = math.log(area_min)
        log_max = math.log(area_max)
        log_step = (log_max - log_min) / (num_sweeps - 1)
        thresholds = [math.exp(log_min + i * log_step) for i in range(num_sweeps)]

    if reverse:
        thresholds = list(reversed(thresholds))

    print(f"  Sweep order: {'large→small' if reverse else 'small→large'}")
    print(f"  Sweep thresholds ({len(thresholds)}): "
          f"{', '.join(f'{t:.0f}' for t in thresholds)}\n")

    prev_cfg = None
    sweep_results = []

    for i, thresh in enumerate(thresholds):
        # Approximate per-layer display for this cost threshold: largest display area
        # among configs with area_cost <= thresh
        disp_approx = next(
            (d for c, d in reversed(cost_to_display) if c <= thresh),
            cost_to_display[0][1]
        )
        print(f"{'─'*60}")
        print(f"  Level {i+1}/{len(thresholds)}: area_cap (cost) = {thresh:.1f} mm²  "
              f"(≈ {disp_approx:.1f} mm² per-layer)", flush=True)

        result = sweep_one_level(thresh, model_list, prefill,
                                 start_cfg=prev_cfg, max_cycles=max_cycles)
        if result is None:
            print(f"    No feasible config at this area cap", flush=True)
            continue

        cfg, area, geo_exe, per_model = result
        sweep_results.append((area, geo_exe, cfg))
        prev_cfg = cfg

        if geo_exe is not None:
            best_full = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
            compute_die, _ = compute_area(best_full)
            print(f"    Best: area_cost={area:.1f} mm²  compute_die={compute_die:.1f} mm²  geo_exe={geo_exe:.0f} cycles")
            print(f"    Config: sa={cfg['sa']} sram={cfg['sram_kb']} "
                  f"cores={cfg['num_cores']} bw={cfg['dram_bw']}")
            for m, e in per_model.items():
                exe_str = f"{e:.0f}" if e else "FAIL"
                print(f"      {m}: {exe_str} cycles")
        else:
            print(f"    Best: area={area:.1f} mm²  FAILED")

    # Collect ALL evaluated points from the cache (not just sweep endpoints)
    all_points = [(area, geo, cfg_from_key(key))
                  for key, (area, geo, _) in _eval_cache.items()]

    pareto = extract_pareto_front(all_points)
    return all_points, pareto


def cfg_from_key(key):
    """Reconstruct config dict from cache key tuple."""
    return {"sa": key[0], "sram_kb": key[1], "num_cores": key[2], "dram_bw": key[3]}


# ═══════════════════════════════════════════════════════════════════════
# Exhaustive mode: evaluate every config in the grid
# ═══════════════════════════════════════════════════════════════════════

def run_exhaustive(model_list, prefill, reverse=False):
    """Evaluate all 600 configs. Returns (all_points, pareto).

    reverse=False (default): evaluate configs small area→large area.
    reverse=True:            evaluate configs large area→small area.
    """
    configs = enumerate_all_configs()
    total = len(configs)
    print(f"Exhaustive evaluation: {total} configs × {len(model_list)} models  "
          f"(order: {'large→small' if reverse else 'small→large'})\n")

    # Sort by area_cost so progress moves along the Pareto axis
    configs_with_area = []
    for cfg in configs:
        full_cfg = {**cfg, "noc_topo": FIXED["noc_topo"], "noc_bw": FIXED["noc_bw"]}
        _area_display, area_cost = compute_area(full_cfg)
        configs_with_area.append((area_cost, cfg))
    configs_with_area.sort(key=lambda x: x[0], reverse=reverse)

    all_points = []
    result_json = f"{OUT_DIR}/pareto_results.json"

    for i, (pre_area_cost, cfg) in enumerate(configs_with_area):
        print(f"[{i+1}/{total}] sa={cfg['sa']} sram={cfg['sram_kb']} "
              f"cores={cfg['num_cores']} bw={cfg['dram_bw']}  "
              f"(area_cost~{pre_area_cost:.0f} mm²)", flush=True)

        area, geo_exe, per_model = evaluate_config(cfg, model_list, prefill)

        all_points.append((area, geo_exe, cfg))

        if geo_exe is not None:
            print(f"  -> geo_exe={geo_exe:.0f} cycles", flush=True)
        else:
            print(f"  -> FAILED", flush=True)

        # Intermediate save every 50 configs
        if (i + 1) % 50 == 0:
            pareto = extract_pareto_front(all_points)
            save_results(all_points, pareto, "decode" if not prefill else "prefill",
                         model_list, 0, result_json)
            print(f"  [checkpoint: {len(all_points)} evaluated, "
                  f"{len(pareto)} Pareto points]\n", flush=True)

    pareto = extract_pareto_front(all_points)
    return all_points, pareto


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pareto frontier: area vs geo-mean latency")
    parser.add_argument("--mode", type=str, default="decode",
                        choices=["decode", "prefill"],
                        help="Simulation mode (default: decode)")
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all' (default: all)")
    parser.add_argument("--num-sweeps", type=int, default=15,
                        help="Number of area thresholds for sweep mode (default: 15)")
    parser.add_argument("--max-cycles", type=int, default=5,
                        help="Max coordinate descent cycles per level (default: 5)")
    parser.add_argument("--exhaustive", action="store_true",
                        help="Evaluate all configs in the grid (slow but complete)")
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from existing JSON without running sims")
    parser.add_argument("--forward", action="store_true",
                        help="Explore area buckets small→large (default: large→small)")
    parser.add_argument("--core-group", type=int, default=None,
                        help="Override core-group size (default: use built-in value)")
    args = parser.parse_args()

    # Parse models
    if args.models == "all":
        model_list = list(MODELS.keys())
    else:
        model_list = [m.strip() for m in args.models.split(",")]
        for m in model_list:
            if m not in MODELS:
                print(f"ERROR: Unknown model '{m}'. Available: {list(MODELS.keys())}")
                return 1

    mode = args.mode
    prefill = (mode == "prefill")
    if not prefill:
        model_list = [m for m in model_list if m != "dit-xl"]
    if args.core_group is not None:
        FIXED['core_group'] = args.core_group
    set_output_dirs(mode)

    cg_suffix = f"_cg{FIXED['core_group']}" if args.core_group is not None else ""
    result_json = f"{OUT_DIR}/pareto_results{cg_suffix}.json"
    plot_path = f"figures/pareto_front{cg_suffix}.png"

    # ── Plot-only mode ──
    if args.plot_only:
        if not os.path.exists(result_json):
            print(f"ERROR: {result_json} not found. Run without --plot-only first.")
            return 1
        datasets = _load_datasets_for_plot(result_json, mode)
        plot_pareto(datasets, plot_path)
        print_pareto_table(datasets[0][1])
        return 0

    # ── Run mode ──
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TRACE_OUT, exist_ok=True)
    os.makedirs(PICKLE_OUT, exist_ok=True)

    # Load cached results from previous runs (enables forward+reverse sharing)
    if _load_eval_cache_from_disk(result_json):
        print(f"Loaded cached results from {result_json}")

    print(f"Pareto DSE Configuration:")
    print(f"  Mode: {mode}")
    print(f"  Models: {model_list}")
    print(f"  Strategy: {'exhaustive grid' if args.exhaustive else f'area sweep ({args.num_sweeps} levels, {args.max_cycles} cycles/level)'}")
    reverse = not args.forward
    print(f"  Order:    {'large→small' if reverse else 'small→large (forward)'}")
    if args.forward:
        print(f"  WARNING: forward mode may miss Pareto-optimal points. "
              f"Default reverse (large→small) is recommended for better coverage.")
    print(f"  Output: {OUT_DIR}/\n")

    t_start = time.time()

    if args.exhaustive:
        all_points, pareto = run_exhaustive(model_list, prefill, reverse=reverse)
    else:
        all_points, pareto = run_sweep(model_list, prefill, args.num_sweeps, reverse=reverse, max_cycles=args.max_cycles)

    elapsed = time.time() - t_start

    # ── Results ──
    print(f"\n{'='*70}")
    print(f"  Pareto Front: {len(pareto)} points  "
          f"({len(all_points)} total evaluated)")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"{'='*70}")

    print_pareto_table(pareto)

    save_results(all_points, pareto, mode, model_list, elapsed, result_json)
    datasets = _load_datasets_for_plot(result_json, mode)
    plot_pareto(datasets, plot_path)

    return 0


def print_pareto_table(pareto):
    """Print the Pareto front as a formatted table."""
    if not pareto:
        print("  (empty Pareto front)")
        return
    print(f"\n  {'#':>3}  {'Area (mm²)':>12}  {'Geo Exe (cyc)':>15}  "
          f"{'SA':>4}  {'SRAM':>6}  {'Cores':>6}  {'DRAM BW':>8}")
    print(f"  {'─'*65}")
    for i, (area, exe, cfg) in enumerate(pareto, 1):
        exe_str = f"{exe:.0f}" if exe is not None else "FAIL"
        print(f"  {i:>3}  {area:>12.1f}  {exe_str:>15}  "
              f"{cfg['sa']:>4}  {cfg['sram_kb']:>6}  "
              f"{cfg['num_cores']:>6}  {cfg['dram_bw']:>8}")


if __name__ == "__main__":
    sys.exit(main())
