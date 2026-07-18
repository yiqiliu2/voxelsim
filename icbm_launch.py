"""Main orchestrator for launching hardware simulations of DNN models on
3D-stacked chip configurations.

This module serves as the primary entry point for the ICBM (Intelligent Chip
Benchmark Modeler) simulation framework.  It parses hardware and model
configuration parameters, instantiates the compute / DRAM / NoC hardware
modules, compiles a DNN program (operator graph with per-operator tiling and
data-movement schedules), and sweeps over a range of SRAM execution-space
partitions to find the best-performing configuration.  Simulation tasks are
dispatched in parallel via ``ProcessPoolExecutor`` and final results are
aggregated and written out by ``parse_results``.

Typical usage (CLI)::

    python icbm_launch.py <modelname> <num_cores> --layers <L> \\
        --batch_size <B> --sequence_length <S> --split_factor <F> \\
        --hw_json <config> --core_group <CG> --dram_bw <BW> ...
"""

import argparse
import time
import os
import sys
import math
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor as Pool
import numpy as np

from typing import List, Tuple

from icbm_DNNModels import get_model_from_file, \
    search_optimal_exe_load_config_order_independent_helper, \
    search_optimal_exe_load_config_baseline_independent_helper
import icbm_DNNProgram

import t10_TensorExpression as TE

from t10_utils import IPU_Mk2_cycle_to_ms

from tsim_components.mem import DRAM, get_per_cycle_bytes_per_core_from_DRAM_config
from tsim_components.comp_util import Compute_OP, Compute
from tsim_components.noc import Topo, NoC
from tsim_simple import run_tsim_helper, _init_worker, parse_results

# Maximum edit distance allowed when pruning the operator execution-order
# search space.  A smaller value reduces compile time but may miss better
# orderings.  Overridden to 3 when prefill mode is enabled.
MAX_EDIT_DIST = 4

# Divisor applied to the number of cores for certain ViT models where not
# all cores participate in every layer (set to 2 for ViT variants).
CORE_REDUCE = 1

# Global flag toggling prefill vs. decode (inference) mode.
# Controls model parsing, execution-order search, and output directory naming.
PREFILL = False


def _get_available_memory_gb():
    """Get available system memory in GB using /proc/meminfo (fast, no deps)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        return 32.0  # conservative fallback


def get_hw_modules( hw_config, num_cores,
                    comp_overlap:bool=True,
                    use_ipu_cost:bool=False,
                    use_ipu_pad:bool=False,
                    use_sram:bool=False,
                    precise_dram:bool=False,
                    ultra_precise_dram:bool=False,
                    ultra_precise_backend:str="auto",
                    lock_cores_per_bank:float=0,
                    soft_cores_per_bank:bool=False) -> Tuple[DRAM, NoC, Compute, Compute_OP]:
    """Instantiate and return the four core hardware simulation modules.

    Reads the ``compute``, ``dram``, and ``noc`` sections of *hw_config* and
    creates the corresponding simulation objects, scaling DRAM bandwidth and
    row size on a per-core basis.

    Args:
        hw_config: Dictionary loaded from a hardware-configuration JSON file.
            Must contain ``"compute"``, ``"dram"``, and ``"noc"`` keys.
        num_cores: Total number of processing cores on the simulated chip.
        comp_overlap: If ``True``, element-wise and matrix-multiply compute
            stages are allowed to overlap in time.
        use_ipu_cost: Use IPU-style cost model for compute estimation.
        use_ipu_pad: Apply IPU-style tensor padding rules.
        use_sram: When ``True``, model the system as SRAM-only (no DRAM
            backing store), reflecting an IPU-like architecture.

    Returns:
        A 4-tuple ``(dram, noc, comp, comp_op)`` where:
        - *dram*: :class:`DRAM` module modelling off-chip memory.
        - *noc*: :class:`NoC` module modelling inter-core communication.
        - *comp*: :class:`Compute` module with low-level cycle costs.
        - *comp_op*: :class:`Compute_OP` wrapper exposing operator-level
          compute cost queries.
    """
    compute_info = hw_config["compute"]
    comp = Compute( ew_pad_len=compute_info["ew_pad_len"],
                    mm_pad_shape=compute_info["mm_pad_shape"],
                    ew_reuse_num=compute_info["ew_reuse_num"],
                    mm_reuse_list=compute_info["mm_reuse_list"],
                    ew_flopc=compute_info["ew_flopc"],
                    mm_flopc=compute_info["mm_flopc"],
                    load_store_bw_bytepc=compute_info["load_store_bw_bytepc"],
                    byte_per_elem=compute_info["byte_per_elem"],
                    mm_init_cycle=compute_info["mm_init_cycle"],
                    ew_mm_overlap=comp_overlap,
                    use_ipu_cost=use_ipu_cost,
                    use_ipu_pad=use_ipu_pad)
    comp_op = Compute_OP(comp)

    # --- DRAM module ---
    # Convert the chip-level DRAM bandwidth (GB/s) into per-core, per-cycle
    # byte throughput based on the NPU clock frequency and core count.
    dram_info = hw_config["dram"]
    per_cycle_bytes_per_core = get_per_cycle_bytes_per_core_from_DRAM_config(num_cores,
                                    dram_info["bandwidth_GBps"], dram_info["npu_freq_MHz"])
    # Row size seen by each core = single-cycle bandwidth * accesses per row.
    row_bytes_per_core = per_cycle_bytes_per_core * dram_info["num_access_per_row"]
    # Optional precise-mode timing parameters (fall back to defaults when
    # the hw_config JSON omits them, so existing configs keep working).
    dram_kwargs = {}
    if precise_dram or ultra_precise_dram:
        dram_kwargs["precise"] = True
        for opt_key in ("num_banks_per_channel", "tRRD", "tFAW", "tRFC", "tREFI"):
            if opt_key in dram_info:
                dram_kwargs[opt_key] = dram_info[opt_key]

    # Try to bring up an external cycle-accurate DRAM simulator. On any
    # failure we *do not* silently degrade — a clear warning is printed and
    # we drop back to the pure-Python precise model so the run still
    # produces meaningful (if less authoritative) numbers.
    if ultra_precise_dram:
        try:
            from tsim_components.dram_external import make_backend
            backend = make_backend(ultra_precise_backend)
            dram_kwargs["ultra_precise"] = True
            dram_kwargs["ultra_backend"] = backend
            print(f"Ultra-precise DRAM backend: {backend.name}",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f"WARNING: --ultra_precise_dram requested but no external "
                f"backend is available:\n{e}\n"
                f"Falling back to --precise_dram (pure-Python bank-state "
                f"simulator). Build Ramulator 2.0 or DRAMsim3 and set "
                f"RAMULATOR2_BIN / DRAMSIM3_BIN (and DRAMSIM3_CONFIG) to "
                f"enable the full ultra-precise mode.",
                file=sys.stderr, flush=True)
            dram_kwargs["precise"] = True

    dram = DRAM(CL=dram_info["CL"],
                tRCD=dram_info["tRCD"],
                tRP=dram_info["tRP"],
                bytes_per_row=row_bytes_per_core,
                bytes_per_cycle=per_cycle_bytes_per_core,
                num_cores=num_cores,
                use_sram=use_sram,
                lock_cores_per_bank=lock_cores_per_bank,
                soft_cores_per_bank=soft_cores_per_bank,
                **dram_kwargs)

    # --- NoC (Network-on-Chip) module ---
    noc_info = hw_config["noc"]

    nodes = list(range(num_cores))
    # Placeholder adjacency matrix -- the actual topology is configured inside
    # the NoC constructor based on noc_info["topology"].
    interconnect_graph = [[0]*len(nodes) for node in nodes]

    noc = NoC(bandwidth_bytepc=noc_info["bandwidth_bytepc"],
              topology=noc_info["topology"],
              nodes=nodes,
              use_sram=use_sram)

    return dram, noc, comp, comp_op


def gen_pickle(args:argparse.Namespace, output_dir, layer, order_pickle_filename, pickle_filename, comp:Compute, noc:NoC):
    """Compile a DNN model into an optimised operator program for simulation.

    This function performs the full compilation pipeline:

    1. Invoke ``model_parser.py`` (as a subprocess) to translate the model JSON
       specification into a tensor-expression (TExpr) representation.
    2. Load the TExpr file via :func:`get_model_from_file` to build the
       :class:`DNNProgram` operator graph.
    3. Run intra-operator optimisation (tiling / data-layout search) and
       generate cold/hot memory transfer tables.

    Args:
        args: Parsed CLI arguments (model name, batch size, core config, etc.).
        output_dir: Root directory for intermediate compilation artifacts.
        layer: Number of transformer layers in the model.
        order_pickle_filename: Path where execution-order pickle would be saved
            (currently unused inside this function but kept for interface
            consistency).
        pickle_filename: Path where the compiled program pickle would be saved
            (currently unused inside this function but kept for interface
            consistency).
        comp: :class:`Compute` module used to annotate operators with cycle
            costs.
        noc: :class:`NoC` module used to annotate operators with
            communication costs.

    Returns:
        A fully compiled :class:`DNNProgram` instance ready for simulation.
    """
    print("Generating new pickle...", file=sys.stderr, flush=True)

    total_chip_sram_MB:int = args.core_mem_kb * args.num_cores // 1024

    # Use subprocess.run instead of os.system for better performance and error handling
    if PREFILL:
        subprocess.run([sys.executable, "model_parser.py", f"{args.modelname}.json",
                       str(args.batch_size), str(args.sequence_length),
                       str(total_chip_sram_MB // args.split_factor)],
                      cwd="models", check=True)
    else:
        subprocess.run([sys.executable, "model_parser.py", f"{args.modelname}.json",
                       str(args.batch_size), "1",
                       str(total_chip_sram_MB // args.split_factor),
                       str(args.sequence_length)],
                      cwd="models", check=True)

    prog = get_model_from_file(f"models/TExpr/TExpr_{args.modelname}-b{args.batch_size}.json", name=f"{args.modelname}-b{args.batch_size}",
                            output_dir=f"{output_dir}/{args.num_cores}cores",
                            num_cores=[args.num_cores], tot_mem_size_per_core=args.core_mem_kb*1024)
    prog.update_te_hw(comp, noc, args.spmd_compiler, args.seq_noc, args.ipu_tsim)
    prog.uniform_dram_mapping = args.uniform_dram_mapping
    prog.ipu_no_overlap = args.ipu_tsim
    if args.ipu_tsim:
        prog.ipu_trace_tag = f"{'prefill' if args.prefill else 'decode'}-ipu_tsim"
    if args.ipu_tsim:
        prog.op_init_overhead = 1600

    start = time.perf_counter()

    # Use all available CPUs for compilation; icbm_DNNProgram will
    # internally cap worker count based on available RAM to avoid OOM.
    _compile_threads = os.cpu_count() or 8
    prog.run_intra_op_optimization(num_threads=_compile_threads)
    prog.generate_all_cold_hot_table(num_threads=_compile_threads)
    prog.init_min_cold_size_bytes_per_core()

    end = time.perf_counter()

    print(f"prepare time: {end - start} sec")

    return prog

if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  CLI argument parsing                                                #
    # ------------------------------------------------------------------ #
    parser = argparse.ArgumentParser(prog="ICBM")
    parser.add_argument("modelname", type=str)
    parser.add_argument("num_cores", type=int)
    parser.add_argument("--layers", required=True, type=int)

    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--sequence_length", required=True, type=int)

    parser.add_argument("--split_factor", required=True, type=float)

    parser.add_argument("--core_mem_kb", required=False, type=int, default=624)
    parser.add_argument("--output_dir", required=False, type=str, default="")

    parser.add_argument("--use_pickle", action='store_true')
    parser.add_argument("--update_order_list", action='store_true')


    parser.add_argument("--generate_pickle_only", action='store_true', required=False, default=False)
    parser.add_argument("--generate_order_only", action='store_true', required=False, default=False)

    parser.add_argument("--uniform_dram_mapping", action='store_true', required=False, default=False)
    parser.add_argument("--spmd_compiler", action='store_true', required=False, default=False)
    parser.add_argument("--seq_noc", action='store_true', required=False, default=False)
    parser.add_argument("--dataflow", action='store_true', required=False, default=False)

    parser.add_argument("--ipu_tsim", action='store_true', required=False, default=False)

    parser.add_argument("--hw_json", required=False, type=str, default="1")
    parser.add_argument("--dram_name", required=False, type=str, default="unspecified_mem")
    parser.add_argument("--core_group", required=False, type=int, default=1)
    parser.add_argument("--prefill", "--training", dest="prefill", action='store_true', required=False, default=False)
    parser.add_argument("--dram_bw", required=False, type=int, default=10000)
    parser.add_argument("--sim_layers", required=False, type=int, default=0)
    parser.add_argument("--trace_out_dir_base", required=False, type=str, default="results/logs")
    # Precise (request-level) DRAM mode — see DRAM._precise_num_cycle_of_access
    # in tsim_components/mem.py. Default off; existing analytical fast path
    # is unchanged when this flag is absent.
    parser.add_argument("--precise_dram", action='store_true', required=False, default=False)
    # Ultra-precise mode: defer DRAM latency to an external cycle-accurate
    # simulator (Ramulator 2.0 / DRAMsim3) via subprocess. See
    # tsim_components/dram_external.py for binary discovery (env vars
    # RAMULATOR2_BIN / RAMULATOR2_CONFIG / DRAMSIM3_BIN / DRAMSIM3_CONFIG).
    # When the requested backend is missing, we warn loudly and fall back
    # to --precise_dram so the run still finishes.
    parser.add_argument("--ultra_precise_dram", action='store_true', required=False, default=False)
    parser.add_argument("--ultra_precise_backend", required=False, type=str, default="auto",
                        choices=["auto", "ramulator2", "dramsim3"])
    parser.add_argument("--lock_cores_per_bank", type=float, required=False, default=0,
                        help="Lock DRAM cores_per_bank to this fixed value (e.g. 2). 0 = off (use num_cores/NUM_BANKS).")
    parser.add_argument("--soft_cores_per_bank", action='store_true', required=False, default=True,
                        help="Soft mode: cores_per_bank unchanged when <2, else halved (e.g. 2->1, 4->2, 8->4).")
    args = parser.parse_args()

    # Dataflow mode disables soft cores-per-bank: dataflow splits cores
    # across pipeline stages so the bank-contention heuristic (which
    # assumes all cores compete for DRAM) over-estimates overhead.
    if args.dataflow:
        args.soft_cores_per_bank = False

    # ------------------------------------------------------------------ #
    #  Load hardware configuration and instantiate simulation modules      #
    # ------------------------------------------------------------------ #
    hw_json = f"hw_config/{args.hw_json}.json"
    with open(hw_json, 'r') as f:
        import ujson as json
        hw_config = json.load(f)
        # IPU-style architectures do not overlap compute and memory stages.
        comp_overlap = True
        if args.ipu_tsim:
            comp_overlap = False
        dram, noc, comp, comp_op = \
            get_hw_modules( hw_config, args.num_cores,
                            comp_overlap=comp_overlap,
                            use_ipu_cost=args.ipu_tsim,
                            use_ipu_pad=args.ipu_tsim,
                            use_sram=False,
                            precise_dram=args.precise_dram,
                            ultra_precise_dram=args.ultra_precise_dram,
                            ultra_precise_backend=args.ultra_precise_backend,
                            lock_cores_per_bank=args.lock_cores_per_bank,
                            soft_cores_per_bank=args.soft_cores_per_bank)
    # ------------------------------------------------------------------ #
    #  Pre-compute or load cached NoC shortest-path distance tables        #
    # ------------------------------------------------------------------ #
    if noc.exact_topo:
        noc_pickle_file = f"noc_distance_tables/{noc.topology}_{len(noc.nodes)}"
        os.makedirs("noc_distance_tables", exist_ok=True)
        if noc.topology != Topo.CUSTOM and os.path.exists(f"{noc_pickle_file}.dist") and os.path.exists(f"{noc_pickle_file}.prev"):
            print("Loading NoC distance table from file")
            with open(f"{noc_pickle_file}.dist", 'rb') as f:
                noc.dist = pickle.load(f)
            with open(f"{noc_pickle_file}.prev", 'rb') as f:
                noc.prev = pickle.load(f)
        else:
            print("Generating distance tables for NoC...")
            n = len(noc.nodes)
            for i in range(n):
                noc.dijkstra(i)
            print("DONE")

            with open(f"{noc_pickle_file}.dist", 'wb') as f:
                pickle.dump(noc.dist, f)
            with open(f"{noc_pickle_file}.prev", 'wb') as f:
                pickle.dump(noc.prev, f)

    # ------------------------------------------------------------------ #
    #  Configure global mode flags                                         #
    # ------------------------------------------------------------------ #
    PREFILL = args.prefill
    icbm_DNNProgram.PREFILL = PREFILL
    if PREFILL:
        # Tighter edit-distance bound in prefill mode to limit search space.
        MAX_EDIT_DIST = 3

    if "vit" in args.modelname:
        # ViT models use only half the cores per layer.
        CORE_REDUCE = 2

    # ------------------------------------------------------------------ #
    #  Resolve output directory and pickle paths                           #
    # ------------------------------------------------------------------ #
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = f"results/pickles/outputs_icbm_{args.sequence_length}"
    layer = args.layers

    mem_threshold = TE.MAX_MEM_THRESHOLD

    pickle_filename = f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/program.pickle"
    order_pickle_filename = f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/order.pickle"

    # ------------------------------------------------------------------ #
    #  Load cached program or compile from scratch                         #
    # ------------------------------------------------------------------ #
    if not args.use_pickle or \
       not os.path.exists(f"{output_dir}/{args.num_cores}cores/{args.modelname}-b{args.batch_size}/program.pickle") or \
       not os.path.exists(order_pickle_filename):
        prog = gen_pickle(args, output_dir, layer, order_pickle_filename, pickle_filename, comp, noc)

    else:
        # Attempt to load a previously compiled program from disk.
        load = False
        try:
            with open(pickle_filename, 'rb') as f:
                prog:icbm_DNNProgram.DNNProgram = pickle.load(f)
                load = True
        except Exception:
            print("Error loading pickle file", file=sys.stderr)
            prog = gen_pickle(args, output_dir, layer, order_pickle_filename, pickle_filename, comp, noc)

        # Re-generate and persist the operator execution-order list, then
        # strip heavyweight config data from the program before re-saving
        # (keeps the pickle small for downstream consumption).
        if args.update_order_list and load:
            order_list = None
            if args.reduce_order_list:
                order_list = prog.init_all_order_lists_reduced(layer, max_edit_dist=MAX_EDIT_DIST)
            else:
                order_list = prog.init_all_order_lists(layer)

            with open(order_pickle_filename, 'wb') as f:
                pickle.dump(order_list, f)

            for op in prog.ops:
                op.expr.cold_config_candidates = {}
                op.expr.config_dict = {}
                for cold in op.expr.cold_hot_table:
                    for hot in op.expr.cold_hot_table[cold]:
                        op.expr.cold_hot_table[cold][hot][0][0] = ()
                        op.expr.cold_hot_table[cold][hot][1][0] = ()

            with open(pickle_filename, 'wb') as f:
                pickle.dump(prog, f)

    if args.generate_pickle_only or args.generate_order_only:
        exit(0)

    # ------------------------------------------------------------------ #
    #  Build the execution-space sweep parameter list                      #
    # ------------------------------------------------------------------ #
    # Sweep SRAM execution-space sizes in increments of exe_granularity_KB
    # from the minimum viable size up to the full per-core SRAM capacity.
    exe_granularity_KB = 4
    total_sram_byte_per_core = args.core_mem_kb * 1024

    # Accumulators for per-configuration simulation results.
    hw_cfgs, exec_time_list = [], []
    exec_energy_list, exec_comp_list, exec_sss_list, exec_ssi_list, logs, exec_dram_list = [], [], [], [], [], []
    exec_sa_list, exec_vu_list, exec_noc_list, exec_sram_list, exec_tsv_list = [], [], [], [], []
    sa_flops, vu_flops = [], []
    dram_util = []
    overlap_lists = []

    # Build one parameter tuple per candidate execution-space partition.
    # Each partition allocates `exe_space` bytes for hot (execution) data and
    # the remainder of SRAM for cold (load/store) data.
    params = []
    for i, exe_space in enumerate([i * exe_granularity_KB * 1024 for i in
                      range(1, int(total_sram_byte_per_core/(exe_granularity_KB * 1024)+1))]):
        assert exe_space <= total_sram_byte_per_core, f"exe_space {exe_space} > total_sram_byte_per_core {total_sram_byte_per_core}"
        if exe_space < prog.max_hot_size:
            print(f"exe_space {exe_space} < prog.max_hot_size {prog.max_hot_size}, skipping")
            continue

        params.append((total_sram_byte_per_core, exe_space,
                       args.core_group, args.dram_bw, 
                       hw_config["dram"]["npu_freq_MHz"],
                       args.layers, args.sim_layers, args.dram_name, 
                       args.spmd_compiler, args.seq_noc))

    # Cap the number of configurations to evaluate.  For baseline/IPU modes
    # only a single configuration (min or max exe_space) is meaningful.
    max_exe_space_try = 30
    if args.spmd_compiler:
        # Baseline compiler: use largest exe_space.
        params = [params[-1]]
    elif args.ipu_tsim:
        # IPU-style: all SRAM is execution space (no cold/hot split).
        params = [params[-1]]

    # Down-sample by factor of 2 until within budget.
    while len(params) > max_exe_space_try:
        params = params[::2]

    # ------------------------------------------------------------------ #
    #  Parallel simulation via ProcessPoolExecutor                         #
    # ------------------------------------------------------------------ #
    # Use initializer to pass large objects once per worker, not per task.
    # In prefill mode, limit workers based on available memory to prevent OOM
    if PREFILL:
        GB_PER_PREFILL_WORKER = 60
        available_gb = _get_available_memory_gb()
        max_memory_workers = max(1, int(available_gb // GB_PER_PREFILL_WORKER))
        n_workers = min(len(params), max_memory_workers)
        print(f"Prefill mode: limiting to {n_workers} workers ({available_gb:.1f}GB available, {GB_PER_PREFILL_WORKER}GB per worker)", flush=True, file=sys.stderr)
    else:
        n_workers = min(len(params), os.cpu_count()//2, 20)

    prog.trim_for_simulation()
    with Pool(n_workers, initializer=_init_worker, initargs=(prog, dram, noc, comp_op)) as pool:
        print(f"exec space util: {prog.max_hot_size/total_sram_byte_per_core}", flush=True, file=sys.stderr)
        print(f"Running {len(params)} tasks in parallel with {n_workers} workers...", flush=True, file=sys.stderr)
        results = pool.map(run_tsim_helper, params)

    # ------------------------------------------------------------------ #
    #  Aggregate results from all parallel simulation tasks                #
    # ------------------------------------------------------------------ #
    for result in results:
        hw_cfg, stats, log, overlap_list = result
        hw_cfgs.append(hw_cfg)
        dram_util.append((stats["dram_r_util"], stats["dram_w_util"]))
        exec_time_list.append(stats["exec_time"])
        exec_energy_list.append(stats["exec_energy"])
        exec_sss_list.append(stats["comp_energy"])
        exec_sram_list.append(stats["sram_energy"])
        exec_vu_list.append(stats["vu_energy"])
        exec_sa_list.append(stats["sa_energy"])
        exec_noc_list.append(stats["noc_energy"])
        exec_comp_list.append(stats["sss_energy"])
        exec_dram_list.append(stats["dram_energy"])
        exec_tsv_list.append(stats["tsv_energy"])
        sa_flops.append(stats["sa_flops"])
        vu_flops.append(stats["vu_flops"])
        overlap_lists.append(overlap_list)
        logs.append(log)

    print("Done running tasks", flush=True, file=sys.stderr)

    # ------------------------------------------------------------------ #
    #  Select the best exe_space configuration and write final output      #
    # ------------------------------------------------------------------ #
    # parse_results picks the best-performing exe_space partition and writes
    # timing / energy breakdowns to the output directory.
    mode = "prefill" if args.prefill else "decode"
    out_dir = os.path.join(args.trace_out_dir_base, str(args.modelname), str(f"bs_{args.batch_size}"), str(f"core_{args.num_cores}"), mode)
    parse_results(  hw_cfgs, exec_time_list, exec_energy_list, exec_comp_list, 
                    exec_sss_list, exec_sa_list, exec_vu_list, exec_noc_list,
                    exec_sram_list, exec_dram_list, exec_tsv_list,
                    sa_flops, vu_flops, dram_util, logs, overlap_lists, 
                    hw_config["dram"]["npu_freq_MHz"], comp_op.comp, out_dir,
                    uniform_dram_mapping=args.uniform_dram_mapping,
                    spmd_compiler=args.spmd_compiler,
                    seq_noc=args.seq_noc,
                    ipu_tsim=args.ipu_tsim,
                    dataflow=args.dataflow)
