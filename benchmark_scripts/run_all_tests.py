#!/usr/bin/env python3

import json
import os
import itertools
import subprocess
import time
# from hw_config.config_factory import generate_config
# from figures.fig_common import *

PARALLEL = True
# PARALLEL = True
RUN_ALL= False
DRY_RUN = False  # If True, only inspect output files and collect counts; never launch simulations
REVERSE_ORDER = False  # Reverse mode and config iteration order
MODEL_FILTER = ""  # If non-empty, restrict all_list to this model name
PARALLEL_ALL = False  # If True, parallelize all runs (ignore last-model and large num_cores guards)
EXTREME_PARALLEL = False  # If True, skip _wait_all() between one_pass calls; single barrier at end of mode
MODE = 8 # 1:dense sweep, 2:sweep, 3:sweep pairs, 
         # 5:spmd compiler and spmd compiler,
         # 7:dataflow paradigm, 
         # 8: spmd compiler vs dram bw 
         # 9: default run
         # 11: tRP sweep
PREFILL = False
TRACE_OUT = f"results/logs"

DF_FACTOR = 2
DF_CORE_FACTOR = 2
DF_DRAM_DIV = 1
DF_DRAM_MUL = 1

SPLIT_FACTOR = 1.03
SPLIT_FACTOR_PREFILL = 1.1

CL = 14
TRCD = 14
TRP = 14
trp_sweep_list = [9, 15, 21, 27, 33]
NPU_FREQ_MHZ = 1500
# num_cores = [1472]
# kb = 624
# all_list = [(f"llama2-13", 40, 3), (f"llama3-70", 80, 3), (f"opt-30", 48, 4), (f"gemma2", 46, 2)]

out_location = f"results/pickles/"

# Memory-aware concurrency control
GB_PER_PREFILL_LAUNCH = 60
GB_PER_INFERENCE_LAUNCH = 20
MAX_DECODE_WORKERS = 3  # Max concurrent decode processes to prevent OOM
_active_procs = []
_dry_total = 0
_dry_present = 0
_dry_good = 0
_dry_missing = 0

def _get_available_memory_gb():
    """Get available system memory in GB using /proc/meminfo (fast, no deps)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        return 32.0  # conservative fallback

def _wait_for_slot(gb_needed, max_workers=None):
    """Wait until enough RAM is available and worker slots are free, reaping finished processes.

    Args:
        gb_needed: GB of RAM required for the new process
        max_workers: Maximum number of concurrent workers (None = no limit)
    """
    global _active_procs
    while True:
        _active_procs = [p for p in _active_procs if p.poll() is None]
        # Check both memory and worker count limits
        memory_ok = _get_available_memory_gb() >= gb_needed
        workers_ok = max_workers is None or len(_active_procs) < max_workers
        if memory_ok and workers_ok:
            return
        time.sleep(2)

def _wait_all():
    """Wait for all active subprocesses to finish."""
    global _active_procs
    for p in _active_procs:
        p.wait()
    _active_procs = []


def _record_dry_stat(best_f):
    global _dry_total, _dry_present, _dry_good, _dry_missing
    _dry_total += 1
    if os.path.exists(best_f):
        _dry_present += 1
        try:
            with open(best_f, 'r') as f:
                content = f.read()
                if 'Overall Util:' in content:
                    _dry_good += 1
        except OSError:
            pass
    else:
        _dry_missing += 1

if not os.path.exists("models/TExpr"):
    os.makedirs("models/TExpr")
if not os.path.exists("models/parsed"):
    os.makedirs("models/parsed")



####################################
######### Init. Parameters #########
####################################



# (model_name, num_layers, split_factor, full_mha)
# full_mha=True for standard multi-head attention (FMHA), False for grouped-query.
all_list = [
            (f"llama2-13", 40, SPLIT_FACTOR,          True),
            (f"llama3-70", 80, SPLIT_FACTOR,          False),
            (f"opt-30",    48, SPLIT_FACTOR,          True),
            (f"gemma2",    46, SPLIT_FACTOR,          False),
            (f"dit-xl",  32, SPLIT_FACTOR_PREFILL, True),
            ]
if PREFILL:
    all_list = [
                (f"llama2-13", 40, SPLIT_FACTOR_PREFILL, True),
                (f"llama3-70", 80, SPLIT_FACTOR_PREFILL, False),
                (f"opt-30",    48, SPLIT_FACTOR_PREFILL, True),
                (f"gemma2",    46, SPLIT_FACTOR_PREFILL, False),
                (f"dit-xl", 32, SPLIT_FACTOR_PREFILL, True),
            ]

# Reverse model iteration order if requested
if REVERSE_ORDER:
    all_list = list(reversed(all_list))
if MODEL_FILTER:
    all_list = [(m, l, sf, fmha) for m, l, sf, fmha in all_list if m == MODEL_FILTER]
# all_list = [
#             (f"llama2-13", 40, 1.5),
#             (f"llama3-70", 80, 1.5),
#             (f"opt-30", 48, 1.5),
#             (f"gemma2", 46, 1.5),
#             (f"dit-xl", 32, 4),
#             ]
batch_sizes = [32]
seq_lengths = [2048]

default_params = {
            "sa":32, # Defaults
            "dram_bw":12288,
            "sram_kb":2048,
            "core_group":8,
            "noc_topo": 1,
            "noc_bw": 16,
            "num_cores": 256,
            "row": 8192,
            }

'''
SWEEP PARAMS FOR 3D stacked chip insights
'''

dense_sweep_lists = {k:[v] for k, v in default_params.items()}
# dense_sweep_lists["dram_bw"] = [4096,8192,12288,16384]
# dense_sweep_lists["sram_kb"] = [256, 512, 1024, 2048, 4096, 8192, 16384]
dense_sweep_lists["noc_topo"] = [1, 2, 3]
dense_sweep_lists["noc_bw"] = [8, 16, 32, 64, 128]

sweep_lists = {
                # "sa":[16, 23, 32, 45, 64, 90, 128],
                "sa":[16, 32, 64, 128],
                "dram_bw":[4000, 8000, 12000, 16000, 20000],
                # "dram_bw":[12000],
                "sram_kb":[512, 1024, 2048, 4096, 8192, 16384],
                "core_group":[1, 2, 4, 8, 16, 32],
                "noc_topo": [1, 2, 3],
                "noc_bw": [4, 8, 16, 32, 64],
                # "num_cores": [64],
                "num_cores":[64, 128, 256, 512, 1024],
                "row":[512, 1024, 2048, 4096, 8192],
                }
# sweep_lists = {
#                 # "sa":[16, 23, 32, 45, 64, 90, 128],
#                 "sa":[32],
#                 "dram_bw":[12000],
#                 "num_cores":[256],
#                 }
# sweep_lists = {key:reversed(val) for key, val in sweep_lists.items()}

sweep_pairs = { "sa":[16, 23, 32, 45, 64, 90],
                "sram_kb":[512, 1024, 2048, 4096, 8192, 16384],
                "num_cores":[1024, 512, 256, 128, 64, 32],
                }
# sweep_pairs = { "sram_kb":[512, 1024, 2048, 4096],
#                 "dram_bw":[20000, 16000, 12000, 8000],
#                 }
cg_list = [1, 2, 4, 8, 16]
# cg_list = [default_params["core_group"]]
 
SWEEP_PARAMS = ["sa", "dram_bw", "sram_kb", "core_group", "num_cores", "noc_topo", "noc_bw"]
# SWEEP_PARAMS = ["noc_bw"]
# SWEEP_PARAMS.reverse()

# cg = max(8, round(256*384/bw)): each core group gets 384 GB/s DRAM BW
_BW = [4096, 6144, 8192, 10240, 12288, 14336, 16384]
uniform_map_vs_bw_pairs = {"core_group": [max(8, round(256*384/bw)) for bw in _BW], "dram_bw": _BW}

# Cache for config files to avoid redundant writes
_config_cache = {}

def make_config(
        ew_pad_len,
        mm_pad_shape,
        ew_reuse_num,
        mm_reuse_list,
        ew_flopc,
        mm_flopc,
        load_store_bw_bytepc,
        byte_per_elem,
        mm_init_cycle,
        ew_mm_overlap,
        bandwidth_bytepc,
        topology,
        default_noc,
        CL,
        tRCD,
        tRP,
        bandwidth_GBps,
        num_access_per_row,
        npu_freq_MHz,
        cfg_name
        ) -> None:

    config = {
        "compute": {
            "ew_pad_len": ew_pad_len,
            "mm_pad_shape": mm_pad_shape,
            "ew_reuse_num": ew_reuse_num,
            "mm_reuse_list": mm_reuse_list,
            "ew_flopc": ew_flopc,
            "mm_flopc": mm_flopc,
            "load_store_bw_bytepc": load_store_bw_bytepc,
            "byte_per_elem": byte_per_elem,
            "mm_init_cycle": mm_init_cycle,
            "ew_mm_overlap": ew_mm_overlap,
        },
        "noc": {
            "bandwidth_bytepc": bandwidth_bytepc,
            "topology": topology,
            "default_noc": default_noc,
        },
        "dram": {
            "CL": CL,
            "tRCD": tRCD,
            "tRP": tRP,
            "bandwidth_GBps": bandwidth_GBps,
            "num_access_per_row": num_access_per_row,
            "npu_freq_MHz": npu_freq_MHz,
        },
    }

    # Check if config already exists and is identical
    config_path = f"hw_config/{cfg_name}.json"
    if cfg_name in _config_cache and _config_cache[cfg_name] == config:
        return

    _config_cache[cfg_name] = config
    os.makedirs("hw_config", exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

def get_cfg_str(sa_size, vu_size, sram_size, dram_bw_GBps, noc_topo, noc_bw, core_grp_size, row, trcd=None, trp=None):
    comp_str = f"sa_{sa_size}-vu_{vu_size}"
    mem_str = f"sram_{int(sram_size)}-drambw_{dram_bw_GBps}_PLACEHOLDER"
    noc_str = f"topo_{noc_topo}-nocbw{noc_bw}"
    # Only include tRCD/tRP in filename when explicitly overridden (e.g. MODE 11 tRP sweep).
    # Default modes keep the simple naming for backward compatibility.
    if trp is not None:
        cg_str = f"cg_{core_grp_size}_row_{row}_trcd_{trcd}_trp_{trp}"
    else:
        cg_str = f"cg_{core_grp_size}_row_{row}"
    return comp_str, mem_str, noc_str, cg_str

def one_pass(num_cores:int, sram_KB, core_group, batch_sizes, dram_bw,
             seq_lengths, all_list, hw_json, cfg_strs, prefill=PREFILL,
             trace_out_dir_base=TRACE_OUT,
             uniform_dram_mapping=False,
             spmd_compiler=False, seq_noc=False,
             ipu_tsim=False,
             dataflow=False,
             stdout_file="one_pass.log"):
    '''
    Run tsim via icbm launch on a specific sram, core group, dram bw configuration.
    tsim will iterate through possible execution space sizes for this sram size.
    '''
    print("Onepass")
    if prefill:
        # Real single-request prefill: batch 1 at the full sequence length.
        # (Previously this used batch 32 with seq//8=256 as a cheaper proxy.)
        batch_sizes = [1]

    # Apply reverse order if requested
    if REVERSE_ORDER:
        all_list = list(reversed(all_list))
        batch_sizes = list(reversed(batch_sizes))
        seq_lengths = list(reversed(seq_lengths))

    for model, num_layers, split_factor, _ in all_list:
        dram_type = "PLACEHOLDER" # TOOD FILL THIS OUT LATER
        # for sram_KB, core_group in itertools.product(sram_KBs, core_groups):
        for batch_size in batch_sizes:
            core = num_cores

            for seq_length in seq_lengths:
                output_dir = f"{out_location}outputs_icbm_{seq_length}"
                if prefill:
                    output_dir += "_prefill"

                for i in range(1):
                    if prefill:
                        mode = "prefill"
                    else:
                        mode = "decode"
                    comp_str, mem_str, noc_str, cg_str = cfg_strs
                    if uniform_dram_mapping:
                        best_str = "uniform_dram"
                    elif spmd_compiler:
                        best_str = "spmd_compiler"
                    elif seq_noc:
                        best_str = "seq_noc"
                    elif ipu_tsim:
                        best_str = "ipu_tsim"
                    elif dataflow:
                        best_str = "dataflow"
                    else:
                        best_str = "best"
                    out_dir = os.path.join(TRACE_OUT, model, f"bs_{batch_size}", f"core_{core}",
                                           mode, comp_str, mem_str, noc_str, best_str)
                    best_f = os.path.join(out_dir, f"output_{cg_str}.log")

                    if DRY_RUN:
                        _record_dry_stat(best_f)
                        continue

                    if not RUN_ALL:
                        if os.path.exists(best_f):
                            good = False
                            # Read file once and check all conditions
                            try:
                                with open(best_f, 'r') as f:
                                    content = f.read()
                                    if 'Overall Util:' in content:
                                        good = True
                            except:
                                good = False

                            if good:
                                print("Skipping good output file: ", best_f)
                                continue
                            else:
                                print("Bad output file: ", best_f)
                        else:
                            print("No output file: ", best_f)

                    if prefill:
                        sim_layers = 1
                    else:
                        sim_layers = 0

                    cmd = f"python3 icbm_launch.py {model} {core}"
                    cmd += f" --core_mem_kb {sram_KB}"
                    cmd += f" --output_dir {output_dir}"
                    cmd += f" --layers {num_layers}"
                    cmd += f" --batch_size {batch_size}"
                    cmd += f" --sequence_length {seq_length}"
                    cmd += f" --use_pickle"
                    cmd += f" --split_factor {split_factor}"
                    cmd += f" --hw_json {hw_json}"
                    cmd += f" --core_group {core_group}"
                    cmd += f" --dram_name {dram_type}"
                    cmd += f" --dram_bw {dram_bw}"
                    cmd += f" --sim_layers {sim_layers}"
                    cmd += f" --trace_out_dir_base {trace_out_dir_base}"
                    if uniform_dram_mapping:
                        cmd += f" --uniform_dram_mapping"
                    elif spmd_compiler:
                        cmd += f" --spmd_compiler"
                    elif seq_noc:
                        cmd += f" --seq_noc"
                    elif ipu_tsim:
                        cmd += f" --ipu_tsim"
                    elif dataflow:
                        cmd += f" --dataflow"

                    if prefill:
                        cmd += f" --prefill"
                    cmd += f" > {stdout_file}"
                    # cmd += f" > {output_file}"

                    print(cmd, flush=True)
                    # Memory-aware bounded parallelism
                    if EXTREME_PARALLEL:
                        # No guards, no memory checks — launch everything immediately
                        log_f = open(stdout_file, 'a')
                        p = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                        _active_procs.append(p)
                    elif prefill:
                        if num_cores <= 256:
                            _wait_for_slot(GB_PER_PREFILL_LAUNCH, max_workers=2)
                            log_f = open(stdout_file, 'a')
                            p = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                            _active_procs.append(p)
                        else:
                            _wait_for_slot(GB_PER_PREFILL_LAUNCH)
                            with open(stdout_file, 'a') as log_f:
                                subprocess.run(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                    elif num_cores > 512:
                        # Sequential execution for prefill and large core counts
                        _wait_for_slot(GB_PER_PREFILL_LAUNCH)
                        with open(stdout_file, 'a') as log_f:
                            subprocess.run(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                    elif PARALLEL and (PARALLEL_ALL or (model!=all_list[-1][0] and num_cores<1000)):
                        # Decode mode with worker limit to prevent OOM
                        _wait_for_slot(GB_PER_INFERENCE_LAUNCH, max_workers=MAX_DECODE_WORKERS)
                        log_f = open(stdout_file, 'a')
                        p = subprocess.Popen(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                        _active_procs.append(p)
                    else:
                        with open(stdout_file, 'a') as log_f:
                            subprocess.run(cmd, shell=True, stdout=log_f, stderr=subprocess.STDOUT)
                    print("DONE")
    if not EXTREME_PARALLEL:
        _wait_all()


#Return the relevant parameters for a systolic array with side length sys_arr_size
def get_systolic_array(sys_arr_size):
    mm_pad_shape = [sys_arr_size, sys_arr_size, sys_arr_size]
    mm_reuse_list = [sys_arr_size, sys_arr_size**2, sys_arr_size]
    mm_flopc = 2 * (sys_arr_size ** 2)
    mm_init_cycle = sys_arr_size
    return mm_pad_shape, mm_reuse_list, mm_flopc, mm_init_cycle

# Return the relevant parameters for a systolic array with vector unit width vec_length
def get_vector_unit(vec_length):
    ew_pad_len = vec_length
    ew_reuse_num = vec_length
    ew_flopc= vec_length
    return ew_pad_len, ew_reuse_num, ew_flopc

def one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                    num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list,
                    uniform_dram_mapping=False,
                    spmd_compiler=False,
                    seq_noc=False,
                    ipu_tsim=False,
                    dataflow=False,
                    trp=None,
                    stdout_file="one_pass.log"):
    mm_pad_shape, mm_reuse_list, mm_flopc, mm_init_cycle = get_systolic_array(sys_size)
    ew_pad_len, ew_reuse_num, ew_flopc = get_vector_unit(sys_size)
    tRP_val = trp if trp is not None else TRP
    config_name = f"sa_{sys_size}_vu_{sys_size}_drambw_{dram_bw}_noc_{noc_topo}_{noc_bw}_trcd_{TRCD}_trp_{tRP_val}"
    make_config(
        ew_pad_len=ew_pad_len,
        mm_pad_shape=mm_pad_shape,
        ew_reuse_num=ew_reuse_num,
        mm_reuse_list=mm_reuse_list,
        ew_flopc=ew_flopc,
        mm_flopc=mm_flopc,
        load_store_bw_bytepc=ew_pad_len*2,
        byte_per_elem=2,
        mm_init_cycle=mm_init_cycle,
        ew_mm_overlap=True,
        bandwidth_bytepc=noc_bw,
        topology=noc_topo,
        default_noc=True,
        CL=CL,
        tRCD=TRCD,
        tRP=tRP_val,
        bandwidth_GBps=dram_bw,
        num_access_per_row=row,
        npu_freq_MHz=NPU_FREQ_MHZ,
        cfg_name=config_name
    )
    cfg_strs = get_cfg_str(sys_size, sys_size, kb, dram_bw, noc_topo, noc_bw, cg, row, TRCD, trp)
    one_pass(num_cores, kb, cg, batch_sizes, dram_bw, seq_lengths,
             all_list, config_name, cfg_strs,
             uniform_dram_mapping=uniform_dram_mapping,
             spmd_compiler=spmd_compiler,
             seq_noc=seq_noc,
             ipu_tsim=ipu_tsim,
             dataflow=dataflow,
             stdout_file=stdout_file)

if __name__ == "__main__":
    if MODE == 1:
        print(*[v for k, v in dense_sweep_lists.items()])
        cfgs = itertools.product(*[v for k, v in dense_sweep_lists.items()])
        # print(f"Starting dense sweep. {len(list(cfgs))} configurations {list(cfgs)}")
        for cfg in list(cfgs):
            sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple(cfg)

            print(f"PARAMETERS ({[d for d in dense_sweep_lists.keys()]}): {list(cfg)}")
            one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                            num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list)

    elif MODE == 2:
        for var in SWEEP_PARAMS:
            for val in sweep_lists[var]:
                # params = {"sa":2, # Defaults

                params = default_params.copy()
                params[var] = val # Modify the parameter being swept
                sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])

                print(f"PARAMETERS (variable = {var}): {sys_size, dram_bw, kb, cg}")
                one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                                num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list)
                if var == "noc_topo":
                    # If we are sweeping noc_topo, we need to run the spmd compiler and spmd compiler cases
                    one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                                    num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list,
                                    seq_noc=True)

    elif MODE == 3:
        for cg in cg_list:
            for i in range(len(next(iter(sweep_pairs.values())))):
                params = default_params.copy()
                params["core_group"] = cg
                for var in sweep_pairs.keys():
                    params[var] = sweep_pairs[var][i]
                sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
                print(f"PARAMETERS (variable = {var}): {sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row}")
                one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                                num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list)

    elif MODE == 5:
        params = default_params.copy()
        sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
        one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                        num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list,
                        spmd_compiler=True)
        one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                        num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list,
                        uniform_dram_mapping=True)

    elif MODE == 7:
        # Dataflow paradigm: per-model configs matching draw_sw_diff.py logic
        params = default_params.copy()
        sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
        df_dram_bw = dram_bw // DF_DRAM_DIV * DF_DRAM_MUL
        df_models = [(m, l, sf, fmha) for m, l, sf, fmha in all_list if m in ["llama2-13", "gemma2", "opt-30", "llama3-70"]]
        if not PREFILL:  # Decode
            # DF = floor(K / num_layers).  K = 200·fmha + 2·l·¬fmha
            for m, l, sf, fmha in df_models:
                df = 2 + 2 * fmha  # 4 for FMHA, 2 for GQA
                df_cores = round(num_cores / df / cg) * cg
                df_bs = [round(b * df_cores / num_cores) for b in batch_sizes]
                one_pass_helper(sys_size, df_dram_bw, noc_topo, noc_bw,
                                df_cores, kb, cg, row, df_bs, seq_lengths, [(m, l, sf, fmha)],
                                dataflow=True)
        else:  # Prefill: real batch-1, so micro-batch over the SEQUENCE instead.
            # DF_FACTOR=2: a 2-stage pipeline of seq/2-token chunks on cores/2
            # (opt-30 uses an extra DF_CORE_FACTOR=2).  Written to a dedicated
            # "dataflow" marker dir so it does not collide with the num_cores
            # sweep's "best" points.  (one_pass forces batch_sizes=[1].)
            df_seq = [s // 2 for s in seq_lengths]
            # all 4 models: 2 stages of seq/2-token chunks on cores/2 = 128
            if df_models:
                one_pass_helper(sys_size, df_dram_bw, noc_topo, noc_bw,
                                num_cores // 2, kb, cg, row, batch_sizes, df_seq, df_models,
                                dataflow=True)
        
    elif MODE == 8:
        for i in range(len(next(iter(uniform_map_vs_bw_pairs.values())))):
            params = default_params.copy()
            for var in uniform_map_vs_bw_pairs.keys():
                params[var] = uniform_map_vs_bw_pairs[var][i]
            sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
            print(f"PARAMETERS (variable = {var}): {sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row}")
            one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                            num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list)
            one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                            num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list, uniform_dram_mapping=True)
        if EXTREME_PARALLEL:
            _wait_all()
    elif MODE == 9:
        params = default_params.copy()
        sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
        print(f"DEFAULT PARAMETERS: {sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row}")
        one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                        num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list)
        one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                        num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list,
                        uniform_dram_mapping=True)
    elif MODE == 11:
        # tRP sweep at fixed (default) bandwidth.  Run best + seq_dram for each
        # tRP value so draw_merged_rowchange_overhead.py can plot the right panel.
        params = default_params.copy()
        sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row = tuple([params[v] for v in params.keys()])
        for trp in trp_sweep_list:
            print(f"PARAMETERS (tRP = {trp}): {sys_size, dram_bw, kb, cg, noc_topo, noc_bw, num_cores, row}")
            one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                            num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list, trp=trp)
            one_pass_helper(sys_size, dram_bw, noc_topo, noc_bw,
                            num_cores, kb, cg, row, batch_sizes, seq_lengths, all_list, trp=trp,
                            uniform_dram_mapping=True)
        if EXTREME_PARALLEL:
            _wait_all()

    if DRY_RUN:
        print("DRY RUN SUMMARY")
        print(f"Total expected outputs: {_dry_total}")
        print(f"Existing outputs: {_dry_present}")
        print(f"Good outputs (Overall Util present): {_dry_good}")
        print(f"Missing outputs: {_dry_missing}")
