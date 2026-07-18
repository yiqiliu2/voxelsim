"""Temporal Simulation Engine (TSim) for 3D-Stacked Chip DNN Inference/Training.

This module implements a cycle-level temporal simulation engine that models the
execution of DNN workloads on a 3D-stacked chip architecture. It performs
pipeline overlap analysis to determine hardware unit utilization, execution time,
energy consumption, and DRAM bandwidth usage across multiple hardware
configurations.

Key capabilities:
    - Operator fusion and tiled execution scheduling across systolic arrays (SA),
      vector units (VU), SRAM, DRAM, NoC, and TSV interconnects.
    - Pipeline overlap detection: identifies concurrent hardware unit usage
      intervals to compute utilization and power envelopes.
    - Area and energy breakdown estimation for 3D-stacked chip designs
      (logic die + stacked DRAM layers).
    - Configurable hardware design-space exploration via HardwareConfig, sweeping
      over SRAM sizes, DRAM bandwidths, NoC topologies, core-group sizes, and
      compute array dimensions.
    - Result parsing, logging, and visualization (execution timelines,
      energy-per-operator charts, overlap/utilization plots, DRAM intensity).
"""

import pickle
import os
import sys
import time
import math
import statistics
from typing import Any, Dict, IO, List, Tuple
from concurrent.futures import ThreadPoolExecutor

from icbm_DNNProgram import *
from tsim_components.mem import DRAM, NUM_BANKS, DEFAULT_TRCD, DEFAULT_TRP, get_per_cycle_bytes_per_core_from_DRAM_config
from tsim_components.mem import get_sram_area_from_size, get_dram_area_from_size
from tsim_components.comp_util import Compute_OP, Compute
from tsim_components.noc import Topo, NoC
from tsim_components.tsim_analysis_lib import OverlapInterval, FusedOperatorExecLog, draw_overlap, draw_dram_intensity

# When True, parse_results only fully processes the best-performing configuration.
SKIP_NON_BEST = True

def get_op_ceiling(fused_op_logs: List[FusedOperatorExecLog], op_base: FusedOperatorExecLog,
                            time: int):
    """Return the index of the first operator whose DRAM load starts after op_base finishes.

    Used to upper-bound the scan window in the overlap computation: any operator
    beyond this index cannot overlap with ``op_base``, so it can be safely
    excluded from the per-interval hardware-unit union.

    Args:
        fused_op_logs: Chronologically ordered list of fused-operator execution logs.
        op_base: The reference operator whose finish time defines the cutoff.
        time: Current simulation time (unused in the current implementation but
              retained for interface consistency).

    Returns:
        The 1-based count of operators up to and including the first one that
        starts after ``op_base`` finishes.
    """
    lowest = 0
    for op in fused_op_logs:
        lowest += 1
        if op_base.t_finish < op.t_dram_ld_start: # We can ignore any ops that start after our current operator finishes
            break
    return lowest # Returns lowest op that is not active at given time.

def get_overlap(fused_op_logs: List[FusedOperatorExecLog], max_ops: int=None)->List[OverlapInterval]:
    """Build a timeline of hardware-unit overlap intervals across all operators.

    Sweeps through each operator's execution span and, at each point in time,
    unions the active hardware units from all concurrently executing operators.
    The sweep advances by the shortest remaining phase duration so that every
    change in the active-unit set produces a new ``OverlapInterval``.

    This is the core pipeline-overlap analysis: the resulting interval list
    captures exactly which hardware resources (SA, VU, DRAM, NoC, ...) are
    co-active at every moment, enabling utilization and power-envelope analysis.

    Args:
        fused_op_logs: Chronologically ordered execution logs for each fused
            operator, as produced by ``run_tsim``.
        max_ops: If given, only the first ``max_ops`` operators are analyzed
            (useful for bounding analysis cost on long traces).

    Returns:
        A chronologically ordered list of ``OverlapInterval`` objects covering
        the full simulated execution window.
    """
    curr_time = 0
    intervals = []
    interval_ct = 0
    mem_used = 0
    # Use sets instead of lists for O(1) membership lookup
    entered = set()
    exited = set()
    #For each operator, get a window of relevant ops (TODO, make this a sliding window if better perf needed)
    #We don't need to consider operators below the current one since operators with higher IDs always start
    #later than previous ones. By the time we use an operator as op_base, t_curr will be greater than or equal to the
    #end time of all previous operators, so they will not be executing anymore.
    print("Computing overlap")
    if max_ops:
        fused_op_log_slice = fused_op_logs[:max_ops]
    else:
        fused_op_log_slice = fused_op_logs

    for op_base in fused_op_log_slice:
        if(curr_time >= op_base.t_finish): continue
        # Determine the scan window: [base_idx, ceil_idx) covers all operators
        # that could overlap with op_base's execution span.
        base_idx, ceil_idx = op_base.op_id, get_op_ceiling(fused_op_log_slice, op_base, curr_time)
        # Advance through op_base's span, emitting one interval per HW-unit-set change.
        while curr_time < op_base.t_finish:
            hw_units = set()  # Active hardware units for this interval
            ival_power = 0
            interval_length = -1
            for scan_op in fused_op_log_slice[base_idx : ceil_idx]:
                interval_hw, interval_remaining_time = scan_op.get_active_units_intervals(curr_time)
                old_len = len(hw_units)
                hw_units = hw_units.union(interval_hw)
                new_len = len(hw_units)
                if new_len > old_len: # This operator is contributing to the overlap.
                    ival_power += scan_op.power_W
                #Remaining time becomes the time left on the shortest (valid) operation in the overlap. This ensures we don't miss a combination change
                interval_length = interval_remaining_time if (interval_remaining_time > 0 and \
                                                              (interval_length < 0 or interval_remaining_time < interval_length)) else interval_length
                # Track cumulative on-chip memory (hot data) as operators enter/exit execution.
                if scan_op.t_exec_enter <= curr_time and scan_op not in entered:
                    mem_used += scan_op.hot_size
                    entered.add(scan_op)
                if scan_op.t_finish <= curr_time and scan_op not in exited:
                    mem_used -= scan_op.hot_size
                    exited.add(scan_op)
            assert(interval_length >= 0), f"Remaining time should be non-negative!, opbase: {str(op_base)}, curr_time: {curr_time}, hw_units: {hw_units}"

            intervals.append(OverlapInterval(hw_units, curr_time, curr_time + interval_length, interval_ct, mem_used, ival_power)) # TODO: include op_id -> hw units assoc.
            curr_time = curr_time + interval_length
            interval_ct += 1
    return intervals


class HardwareConfig:
    """Encapsulates a complete hardware configuration for a 3D-stacked NPU.

    Stores the architectural parameters (compute, memory, network-on-chip) that
    define a single design point in the hardware design-space exploration.  Used
    throughout the simulation to pass configuration state and to generate
    human-readable configuration strings for output paths and logging.
    """

    def __init__(self, comp: Compute, noc: NoC, mem: DRAM,
                dram_bw_GBps: int, dram_name: str, core_group_size: int, sram_size: int, exe_size: int, row: int,
                npu_freq_mHz: int, num_cores:int):
        """Initialize a hardware configuration from individual component objects.

        Args:
            comp: Compute unit specification (systolic array + vector unit).
            noc: Network-on-Chip specification (topology, bandwidth).
            mem: DRAM timing/capacity model.
            dram_bw_GBps: Aggregate DRAM bandwidth in GB/s.
            dram_name: Human-readable DRAM identifier (e.g. "HBM", "GDDR5").
            core_group_size: Number of cores in a core group for spatial tiling.
            sram_size: Per-core SRAM capacity in bytes.
            exe_size: Per-core execution scratch-pad size in bytes.
            row: DRAM row buffer size expressed in cycles (bytes_per_row / bytes_per_cycle).
            npu_freq_mHz: NPU clock frequency in MHz.
            num_cores: Total number of cores in the chip.
        """
        self.sram_size = sram_size
        self.dram_bw_GBps = dram_bw_GBps
        self.dram_name = dram_name
        self.cg_size = core_group_size
        self.row = row
        self.npu_freq_mHz = npu_freq_mHz
        self.num_cores = num_cores
        self.set_comp(comp)
        self.set_noc(noc)
        self.set_exe_cg(core_group_size, exe_size)
        self.set_mem(dram_bw_GBps, mem)

    def set_exe_cg(self, core_grp_size:int, exe_size:int):
        """Set core-group size and per-core execution scratch-pad capacity."""
        self.core_grp_size = core_grp_size
        self.exe_sram_size = exe_size

    def set_comp(self, comp:Compute):
        """Extract and store compute-unit dimensions from a Compute object."""
        self.sa_size = comp.mm_pad_shape[-1]  # Assuming square systolic array
        self.vu_size = comp.ew_pad_len
        self.comp=comp

    def set_mem(self, dram_bw_GBps: int, mem:DRAM):
        """Set DRAM bandwidth and timing model."""
        self.dram_bw_GBps = dram_bw_GBps
        self.mem=mem

    def set_noc(self, noc:NoC):
        """Extract and store NoC topology and bandwidth from a NoC object."""
        self.noc_topo = noc.topology
        self.noc_bw = noc.bandwidth_bytepc
        self.noc=noc

    def get_cfg_str(self):
        """Return a tuple of human-readable sub-strings describing this configuration.

        Returns:
            A 5-tuple of strings: (comp_str, mem_str, noc_str, cg_str, exe_str)
            suitable for constructing output directory paths.
        """
        comp_str = f"sa_{self.sa_size}-vu_{self.vu_size}"
        mem_str = f"sram_{int(self.sram_size / 1024)}-drambw_{self.dram_bw_GBps}_{self.dram_name}"
        noc_str = f"topo_{self.noc_topo.value}-nocbw{self.noc_bw}"
        exe_str = f"exe_{int(self.exe_sram_size / 1024)}"
        # tRCD and tRP are appended when they differ from defaults
        # so sweeps over either DRAM timing parameter do not overwrite files.
        cg = f"cg_{self.core_grp_size}_row_{int(self.row)}"
        if int(self.mem.tRCD) != DEFAULT_TRCD or int(self.mem.tRP) != DEFAULT_TRP:
            cg += f"_trcd_{int(self.mem.tRCD)}_trp_{int(self.mem.tRP)}"
        cg_str = cg
        return comp_str, mem_str, noc_str, cg_str, exe_str

_worker_shared = {}  # Module-level dict populated by _init_worker for multiprocessing pool workers.


def _compact_id_ranges(ids) -> Dict[str, Any]:
    """Return a compact, pickle-friendly sorted integer ID set."""
    sorted_ids = sorted(set(int(i) for i in ids))
    if not sorted_ids:
        return {"encoding": "ranges", "ranges": tuple(), "count": 0}
    ranges = []
    start = prev = sorted_ids[0]
    for value in sorted_ids[1:]:
        if value == prev + 1:
            prev = value
        else:
            ranges.append((start, prev + 1))
            start = prev = value
    ranges.append((start, prev + 1))
    return {"encoding": "ranges", "ranges": tuple(ranges), "count": len(sorted_ids)}


def _compact_core_range(num_active_cores: int) -> Dict[str, Any]:
    if num_active_cores <= 0:
        return _compact_id_ranges(())
    return {
        "encoding": "ranges",
        "ranges": ((0, int(num_active_cores)),),
        "count": int(num_active_cores),
    }


def _active_core_count(spatial_partition: List[int], num_cores: int) -> int:
    if not spatial_partition:
        return 1
    return max(1, min(int(num_cores), int(math.prod(spatial_partition))))


def _compact_mod_ids(num_active_cores: int, modulo: int) -> Dict[str, Any]:
    modulo = max(1, int(modulo))
    if num_active_cores >= modulo:
        return _compact_core_range(modulo)
    return _compact_id_ranges(core_id % modulo for core_id in range(num_active_cores))


def _dram_vault_count(dram: DRAM, num_cores: int) -> int:
    banks_per_channel = max(1, int(getattr(dram, "num_banks_per_channel", NUM_BANKS)))
    return max(1, min(int(num_cores), NUM_BANKS // banks_per_channel))


def _compact_noc_link_ids(noc: NoC, num_active_cores: int, num_cores: int) -> Dict[str, Any]:
    if num_active_cores <= 1:
        return _compact_id_ranges(())

    id_space = {
        "id_formula": "logical_directed_link_id = src_core_id * num_cores + dst_core_id",
        "num_cores": int(num_cores),
    }

    if noc.topology == Topo.ALL:
        return {
            "encoding": "complete_directed_no_self",
            "core_ranges": ((0, int(num_active_cores)),),
            "count": int(num_active_cores) * (int(num_active_cores) - 1),
            **id_space,
        }

    graph = getattr(noc, "interconnect_graph", None)
    if graph is None:
        return {"encoding": "unavailable", "count": 0, **id_space}

    active = range(min(num_active_cores, len(graph)))
    link_ids = []
    for src in active:
        row = graph[src]
        for dst in active:
            if src != dst and dst < len(row) and row[dst]:
                link_ids.append(src * num_cores + dst)

    compact = _compact_id_ranges(link_ids)
    compact.update(id_space)
    return compact


def _build_spatial_attribution(spatial_partition: List[int], noc: NoC, dram: DRAM,
                               num_cores: int) -> Dict[str, Any]:
    num_active = _active_core_count(spatial_partition, num_cores)
    active_cores = _compact_core_range(num_active)
    noc_links = _compact_noc_link_ids(noc, num_active, num_cores)
    vault_count = _dram_vault_count(dram, num_cores)
    dram_vaults = _compact_mod_ids(num_active, vault_count)
    tsv_groups = _compact_mod_ids(num_active, vault_count)
    empty_ids = _compact_id_ranges(())

    comp_stage = {
        "active_core_ids": active_cores,
        "noc_logical_link_ids": empty_ids,
        "dram_vault_ids": empty_ids,
        "tsv_group_ids": empty_ids,
    }
    noc_stage = {
        "active_core_ids": active_cores,
        "noc_logical_link_ids": noc_links,
        "dram_vault_ids": empty_ids,
        "tsv_group_ids": empty_ids,
    }
    dram_stage = {
        "active_core_ids": active_cores,
        "noc_logical_link_ids": empty_ids,
        "dram_vault_ids": dram_vaults,
        "tsv_group_ids": tsv_groups,
    }

    return {
        "version": 1,
        "kind": "logical_spatial_attribution",
        "assumptions": (
            "active cores are contiguous logical core IDs [0, prod(spatial_partition))",
            "NoC link IDs use the active-core induced directed graph",
            "DRAM vault IDs are logical DRAM channels derived from NUM_BANKS/num_banks_per_channel",
            "TSV group IDs are aligned 1:1 with logical DRAM vault IDs",
        ),
        "spatial_partition": tuple(int(x) for x in spatial_partition),
        "num_active_cores": num_active,
        "num_dram_vaults": vault_count,
        "num_tsv_groups": vault_count,
        "stages": {
            "dram_r": dram_stage,
            "noc_bcast_sh": noc_stage,
            "dram_w": dram_stage,
            "comp": comp_stage,
            "comp_sram_r": comp_stage,
            "comp_sram_w": comp_stage,
            "comp_vu": comp_stage,
            "comp_sa": comp_stage,
            "noc_reduce": noc_stage,
        },
    }


def _attach_spatial_attribution(fused_op_logs: List[FusedOperatorExecLog],
                                partitions: List[Tuple[List[List[int]], List[int]]],
                                noc: NoC, dram: DRAM, num_cores: int):
    for log, (_, spatial) in zip(fused_op_logs, partitions):
        log.set_spatial_attribution(
            _build_spatial_attribution(spatial, noc, dram, num_cores)
        )


def _init_worker(prog, dram, noc, comp_op):
    """Multiprocessing pool initializer: cache large, read-only objects once per worker.

    Called once when each worker process is spawned so that the DNN program,
    DRAM model, NoC model, and compute-op object are shared across all tasks
    dispatched to that worker, avoiding repeated (de)serialization overhead.

    Args:
        prog: The ``DNNProgram`` workload description.
        dram: The ``DRAM`` timing/capacity model.
        noc: The ``NoC`` topology and bandwidth model.
        comp_op: The ``Compute_OP`` fused-operator compute cost model.
    """
    _worker_shared['prog'] = prog
    _worker_shared['dram'] = dram
    _worker_shared['noc'] = noc
    _worker_shared['comp_op'] = comp_op

def run_tsim_helper(params):
    """Unpack a parameter tuple and dispatch to ``run_tsim``.

    Supports two calling conventions for use with ``multiprocessing.Pool.map``:
        - 10-element tuple (lightweight): large objects are retrieved from the
          module-level ``_worker_shared`` dict set up by ``_init_worker``.
        - 14-element tuple (legacy): all objects are passed inline.

    Args:
        params: A tuple of positional arguments.  See ``run_tsim`` for the
            meaning of each element.

    Returns:
        The return value of ``run_tsim``.
    """
    # 10-param lightweight version (large objects from initializer)
    if len(params) == 10:
        total_sram_byte_per_core, exe_space, core_group_size, dram_bandwidth_GBps, npu_freq_MHz, tot_layers, sim_layers, dram_name, spmd_compiler, seq_noc = params
        prog = _worker_shared['prog']
        dram = _worker_shared['dram']
        noc = _worker_shared['noc']
        comp_op = _worker_shared['comp_op']
    else:
        # Legacy 14-param path
        prog, total_sram_byte_per_core, exe_space, dram, noc, comp_op, core_group_size, dram_bandwidth_GBps, npu_freq_MHz, tot_layers, sim_layers, dram_name, spmd_compiler, seq_noc = params
    return run_tsim(prog=prog,
                    total_sram_byte_per_core=total_sram_byte_per_core,
                    exe_space=exe_space,
                    dram=dram,
                    noc=noc,
                    comp_op=comp_op,
                    core_group_size=core_group_size,
                    dram_bandwidth_GBps=dram_bandwidth_GBps,
                    npu_freq_MHz=npu_freq_MHz,
                    tot_layers=tot_layers,
                    sim_layers=sim_layers,
                    dram_name=dram_name,
                    spmd_compiler=spmd_compiler,
                    seq_noc=seq_noc)

def run_tsim(prog:DNNProgram, total_sram_byte_per_core:int, exe_space:int, dram:DRAM, noc:NoC, comp_op:Compute_OP,
             core_group_size:int, dram_bandwidth_GBps:int, npu_freq_MHz:int, tot_layers:int, sim_layers:int,
             dram_name:str = "unspec_mem", spmd_compiler:bool = False, seq_noc:bool = False) -> \
            Tuple[HardwareConfig, int, int, List[FusedOperatorExecLog], List[OverlapInterval]]:
    """Run a full temporal simulation for one hardware configuration.

    Performs the following stages:
        1. **Operator data gathering** -- For every (unfused) operator, determine
           the best spatial/temporal partition, compute tensor shapes, and
           estimate NoC communication cycles.
        2. **Operator fusion** -- Group unfused operators into fused operators
           using the ``balanced`` fusion heuristic.
        3. **Execution statistics** -- Compute per-operator execution times,
           energy, and DRAM traffic with pipeline overlap between DRAM loads,
           compute, and NoC transfers.
        4. **Overlap analysis** -- Build the hardware-unit overlap timeline
           (see ``get_overlap``).
        5. **Extrapolation** -- If only a subset of layers was simulated
           (``sim_layers > 0``), linearly scale results to the full model.

    Args:
        prog: DNN program (workload) to simulate.
        total_sram_byte_per_core: Per-core SRAM capacity in bytes.
        exe_space: Per-core execution scratch-pad size in bytes.
        dram: DRAM timing model.
        noc: Network-on-Chip model.
        comp_op: Compute-operator cost model (wraps ``Compute``).
        core_group_size: Cores per core-group for spatial tiling.
        dram_bandwidth_GBps: Aggregate DRAM bandwidth in GB/s.
        npu_freq_MHz: NPU clock frequency in MHz.
        tot_layers: Total number of layers in the full model.
        sim_layers: Number of layers to actually simulate (0 = all).
        dram_name: Human-readable DRAM identifier for output paths.
        spmd_compiler: If True, simulate a naive compiler (worse NoC mapping).
        seq_noc: If True, simulate a degraded NoC.

    Returns:
        A 4-tuple of:
            - ``hw_cfg``: The ``HardwareConfig`` for this run.
            - ``stats``: Dict of aggregate statistics (exec_time, energies,
              DRAM bytes, utilizations, FLOPS, etc.).
            - ``fused_op_logs``: Per-operator execution logs.
            - ``overlap_intervals``: Timeline of hardware-unit overlap intervals.
    """
    if sim_layers > 0:  # Non-zero sim_layers means we are only simulating a subset of the layers
        unfused_ops = prog.ops[:len(prog.ops) * (sim_layers) // tot_layers]
    else:
        unfused_ops = prog.ops
    all_tensor_shapes = []
    all_temporal_partitions = []
    all_partitions = []
    all_noc_cycles = []
    print("Gathering op data...")
    op_data_start = time.time()
    for i in range(len(unfused_ops)):
        op = unfused_ops[i]
        # Find the best spatial/temporal partition that fits within the execution scratch-pad.
        spatial, temporal = prog.get_best_config_by_max_mem_size(op, dram, exe_space, core_group_size)
        tensor_shapes = op.expr.get_sub_op_var_sizes(temporal, spatial, False)
        all_temporal_partitions.append(temporal)
        all_partitions.append((temporal, spatial))
        all_tensor_shapes.append(tensor_shapes)
        temporal_var_replicas = op.expr.get_temporal_var_replicas(temporal, spatial)
        spatial_var_replicas = op.expr.get_spatial_var_replicas(temporal, spatial)
        shift_info = op.expr.get_shift_info(temporal, spatial,
                                            output_special_format_for_tsim=True)

        tensor_sizes = op.expr.get_sub_op_var_sizes(temporal, spatial)
        noc_cycles = noc.get_total_cycles_from_expression(tensor_sizes,
                                                          temporal_var_replicas,
                                                          spatial_var_replicas,
                                                          shift_info,
                                                          num_bytes_per_elem=op.num_byte_per_elem,
                                                          spmd_compiler=spmd_compiler,
                                                          seq_noc=seq_noc,
                                                          )
        all_noc_cycles.append(noc_cycles)
    print("Gathering op data took", time.time() - op_data_start, "seconds")
    fuse_start = time.time()
    print("Fusing ops...")
    # Fuse operators using the balanced heuristic to amortize DRAM traffic.
    fused_ops = comp_op.fuse_ops(unfused_ops, exe_space, all_tensor_shapes, method="balanced")
    comp_op_cycles = comp_op.compute_costs([[op] for op in unfused_ops], all_partitions)
    print("Fusing ops took", time.time() - fuse_start, "seconds")
    exec_stat_start = time.time()
    print("Computing execution statistics...")
    # Compute overall execution using UNFUSED ops for memory alloc etc,
    # but fused ops for read/write traffic estimation.
    use_largest_cold = True
    stats, fused_op_logs = prog.get_fused_exec_time(fused_ops=[[op] for op in unfused_ops],
                                                        partitions=all_partitions,
                                                        comp=comp_op.comp,
                                                        dram=dram,
                                                        noc=noc,
                                                        use_largest_cold=use_largest_cold,
                                                        overlap_bcast_read=use_largest_cold,
                                                        compute_times=comp_op_cycles,
                                                        sram_per_core=total_sram_byte_per_core,
                                                        exe_sram_per_core=exe_space,
                                                        core_group_size=core_group_size,
                                                        noc_data=all_noc_cycles,
                                                        dram_bw_GBps=dram_bandwidth_GBps,
                                                        npu_freq_MHz=npu_freq_MHz)
    per_comp_sum =  (stats["sa_energy"] + stats["vu_energy"] + stats["noc_energy"] + stats["dram_energy"] + stats["sram_energy"] + stats["tsv_energy"])
    assert math.isclose(stats["exec_energy"], per_comp_sum, rel_tol=1e-5), \
        f"Energy breakdown does not sum to total energy! total: {stats['exec_energy']} vs {per_comp_sum}"
    dram_bytes = prog.get_fused_dram_bytes_only(fused_ops=fused_ops, partitions=all_partitions, dram=dram,
                            use_largest_cold=use_largest_cold, exe_sram_per_core=exe_space,
                            core_group_size=core_group_size)

    # Overwrite unfused op dram traffic with fused op traffic
    stats["dram_r_bytes"] = sum([dram_op_bytes[0] for dram_op_bytes in dram_bytes])
    stats["dram_w_bytes"] = sum([dram_op_bytes[1] for dram_op_bytes in dram_bytes])

    if(stats["exec_time"] ==-1):
        overlap_intervals =[]
    print("Computing execution statistics took", time.time() - exec_stat_start, "seconds")
    print("Finding overlap...")
    overlap_intervals = get_overlap(fused_op_logs, 250)

    row = dram.bytes_per_row//dram.bytes_per_cycle
    hw_cfg = HardwareConfig(comp_op.comp,
                            noc,
                            dram,
                            dram_bandwidth_GBps,
                            dram_name,
                            core_group_size,
                            total_sram_byte_per_core,
                            exe_space, row, npu_freq_MHz, prog.tot_num_cores)

    # DRAM utilization = actual bytes transferred / theoretical max bytes in exec window.
    # Denominator: exec_time_in_seconds * bandwidth_in_bytes_per_second.
    stats["dram_r_util"] = stats["dram_r_bytes"]/(stats["exec_time"] / (1e6 * npu_freq_MHz) * dram_bandwidth_GBps * 2 ** 30)
    stats["dram_w_util"] = stats["dram_w_bytes"]/(stats["exec_time"] / (1e6 * npu_freq_MHz) * dram_bandwidth_GBps * 2 ** 30)

    # If only a subset of layers was simulated, linearly extrapolate all
    # aggregate metrics to the full model depth.
    if sim_layers:
        stats["dram_r_bytes"] = int(stats["dram_r_bytes"] * tot_layers / sim_layers) # Extrapolate to total layers
        stats["dram_w_bytes"] = int(stats["dram_w_bytes"] * tot_layers / sim_layers) # Extrapolate to total layers
        stats["exec_time"] = int(stats["exec_time"] * tot_layers / sim_layers) #Extrapolate results to all-layer exec time.
        stats["comp_energy"] = int(stats["comp_energy"] * tot_layers / sim_layers)
        stats["sss_energy"] = int(stats["sss_energy"] * tot_layers / sim_layers)

        stats["exec_energy"] = int(stats["exec_energy"] * tot_layers / sim_layers)
        stats["sa_energy"] = int(stats["sa_energy"] * tot_layers / sim_layers)
        stats["vu_energy"] = int(stats["vu_energy"] * tot_layers / sim_layers)
        stats["sram_energy"] = int(stats["sram_energy"] * tot_layers / sim_layers)
        stats["noc_energy"] = int(stats["noc_energy"] * tot_layers / sim_layers)
        stats["dram_energy"] = int(stats["dram_energy"] * tot_layers / sim_layers)
        stats["tsv_energy"] = int(stats["tsv_energy"] * tot_layers / sim_layers)

        stats["sa_flops"] = int(stats["sa_flops"] * tot_layers / sim_layers)  # Per core
        stats["vu_flops"] = int(stats["vu_flops"] * tot_layers / sim_layers) # Per core
    return hw_cfg, stats, fused_op_logs, overlap_intervals

def check_exe_usage(overlap_intervals: List[OverlapInterval], exe_space_size: int):
    """Assert that no overlap interval exceeds the execution scratch-pad capacity.

    Intended as a post-simulation sanity check: if any interval's ``exe_used``
    field exceeds ``exe_space_size``, the tiling or fusion pass has a bug.

    Args:
        overlap_intervals: Timeline of overlap intervals from ``get_overlap``.
        exe_space_size: Per-core execution scratch-pad capacity in bytes.

    Raises:
        AssertionError: If any interval's memory usage exceeds the capacity.
    """
    for i, interval in enumerate(overlap_intervals):
        assert(interval.exe_used <= exe_space_size), f"Interval {i} exceeds exe space size! {interval.exe_used} > {exe_space_size}"

def get_area_breakdown(comp: Compute, noc: NoC, dram: DRAM,
                       per_core_sram_size_KB: int, total_dram_size_MB: int,
                       num_cores: int, sram_type:str, dram_type:str,
                       dram_bw_GBps: float = 0.0,
                       out_dir:str=None):
    """Compute and visualize the area breakdown of a 3D-stacked chip.

    Calculates the silicon area (in mm^2) contributed by each major
    sub-component -- systolic array, vector unit, NoC switch, SRAM, TSV, and
    DRAM -- and generates a pie-chart PDF showing their relative contributions.

    For 3D-DRAM configurations the DRAM footprint is divided by the number of
    stacked layers; for HBM it uses the full die footprint (modeled after
    the A100 HBM2 stack: 8-high, 16 GB/stack).

    Args:
        comp: Compute unit model (provides per-core SA/VU area).
        noc: NoC model (provides per-core switch area).
        dram: DRAM model (provides ``num_layers`` for 3D stacking).
        per_core_sram_size_KB: Per-core SRAM capacity in KiB.
        total_dram_size_MB: Total DRAM capacity across all stacks in MiB.
        num_cores: Total number of cores on the chip.
        sram_type: SRAM technology string (e.g. ``"SRAM"``).
        dram_type: DRAM technology string (e.g. ``"HBM"``, ``"3D-DRAM"``).
        dram_bw_GBps: Aggregate DRAM bandwidth used to size TSV landing area.
        out_dir: Output path for the pie-chart PDF.  If ``None``, a default
            name is generated from the configuration parameters.

    Returns:
        A 3-tuple of:
            - ``area``: Total unstacked area in mm^2.
            - ``logic_area``: Logic-die area (SA + VU + NoC + SRAM + TSV) in mm^2.
            - ``component_areas``: Dict mapping component name to area in mm^2.
    """
    import matplotlib.pyplot as plt
    sram_area_per_core = get_sram_area_from_size(per_core_sram_size_KB, memtype=sram_type)
    dram_area_per_chip = get_dram_area_from_size(total_dram_size_MB, memtype=dram_type)
    dram_footprint = dram_area_per_chip / dram.num_layers if dram_type == "3D-DRAM" else dram_area_per_chip
    tsv_area = max(0.0, float(dram_bw_GBps)) * (90 / (12 * 1024))
    component_areas = {
        "systolic array": num_cores * comp.get_area()[1],
        "vector unit": num_cores * comp.get_area()[2],
        "noc": num_cores * noc.get_switch_area(),
        "sram": num_cores * sram_area_per_core,
        "tsv": tsv_area,
        "dram": dram_footprint
    }
    logic_area = (
        component_areas["systolic array"]
        + component_areas["vector unit"]
        + component_areas["noc"]
        + component_areas["sram"]
        + component_areas["tsv"]
    )
    dram_area = component_areas["dram"]
    component_colors = ["lightgreen", "darkgreen", "red", "lightblue", "orange", "darkblue"]

    area = logic_area + dram_area_per_chip
    fig, ax = plt.subplots(figsize=(10, 15))

    labels = list(component_areas.keys())
    sizes = list(component_areas.values())
    colors = component_colors

    wedges, _ = ax.pie(sizes, colors=colors, startangle=90)

    # Build legend labels with area
    legend_labels = [f"{label}: {size:.1f} mm$^2$" for label, size in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, title="Component Areas")

    text = f"Total (unstacked) area: {area:.1f} mm$^2$"
    ax.text(0, -1.2, text, verticalalignment="top", horizontalalignment="center", fontsize=13)

    if out_dir is None:
        out_dir = f"area_{num_cores}x(sa_{comp.mm_pad_shape[-1]}_vu_{comp.ew_flopc}_sram{per_core_sram_size_KB})_dram_{total_dram_size_MB}.pdf"

    fig.savefig(out_dir, bbox_inches='tight')
    print(f"Saved to {out_dir}")

    return area, logic_area, component_areas

def get_overall_comp_unit_utilization(fused_ops: List[FusedOperatorExecLog], exec_time: int, comp: Compute) -> Tuple[float, float]:
    """Compute aggregate utilization of the systolic array and vector unit.

    Utilization is defined as (actual FLOPS performed) / (peak FLOPS achievable
    in the same execution time), yielding a value in [0, 1].

    Args:
        fused_ops: Per-operator execution logs.
        exec_time: Total execution time in cycles.
        comp: Compute unit model (provides peak FLOP/cycle).

    Returns:
        A 2-tuple ``(sa_util, vu_util)`` with the systolic-array and
        vector-unit utilization ratios, respectively.
    """
    sa_util = 0
    vu_util = 0
    sa_flopc, vu_flopc = comp.get_peak_flopc()
    for op in fused_ops:
        sa_util += op.mm_flop_per_core
        vu_util += op.vu_flop_per_core
    sa_util /= (exec_time * sa_flopc)
    vu_util /= (exec_time * vu_flopc)
    return sa_util, vu_util

def get_noc_util(fused_ops: List[FusedOperatorExecLog], exec_time) -> float:
    """Compute aggregate NoC utilization as a fraction of total execution time.

    NoC utilization is defined as the sum of all broadcast, shift, and reduce
    durations across all operators, divided by the total execution time.

    Args:
        fused_ops: Per-operator execution logs.
        exec_time: Total execution time in cycles.

    Returns:
        NoC utilization ratio (may exceed 1.0 if multiple operators overlap
        their NoC phases).
    """
    noc_util = 0
    for op in fused_ops:
        noc_util += op.bcast_dur + op.shift_dur + op.reduce_dur
    noc_util /= exec_time
    return noc_util


def parse_results(hw_cfg_info: List[HardwareConfig], exec_times: List[int],
                  exec_energy: List[int],
                  comp_energy: List[int],
                  sss_energy: List[int],
                  sa_energy: List[int],
                  vu_energy: List[int],
                  noc_energy: List[int],
                  sram_energy: List[int],
                  dram_energy: List[int],
                  tsv_energy: List[int],
                  sa_flops: List[int],
                  vu_flops: List[int],
                  dram_utils: List[Tuple[float, float]],
                  fused_op_logs: List[List[FusedOperatorExecLog]],
                  overlap_lists: List[List[OverlapInterval]], npu_freq_MHz: int,
                  comp:Compute, out_dir: str = "output",
                  uniform_dram_mapping=False,
                  spmd_compiler=False, seq_noc=False,
                  ipu_tsim=False,
                  dataflow=False):
    """Post-process simulation results: select best configuration, log stats, and generate plots.

    For each hardware configuration in the input lists this function:
        1. Identifies the best-performing configuration (lowest valid execution time).
        2. Computes per-component energy (static + dynamic), power, area breakdown,
           and hardware-unit utilization (SA, VU, NoC, DRAM).
        3. Writes detailed log files, overlap logs, and top-power-interval reports.
        4. Generates multi-panel PDF figures with execution timelines,
           energy-per-operator bar charts, hardware-unit utilization, and
           DRAM intensity plots.
        5. Pickles the fused-operator execution log for later offline analysis.

    When ``SKIP_NON_BEST`` is True (default), only the best configuration is
    fully processed; all others are logged and skipped.

    Args:
        hw_cfg_info: List of ``HardwareConfig`` objects, one per run.
        exec_times: Execution times in cycles (one per run; -1 = invalid).
        exec_energy: Total dynamic energy in pJ (one per run).
        comp_energy: Compute dynamic energy in pJ.
        sss_energy: Spatial-shift-store energy in pJ.
        sa_energy: Systolic-array dynamic energy in pJ.
        vu_energy: Vector-unit dynamic energy in pJ.
        noc_energy: NoC dynamic energy in pJ.
        sram_energy: SRAM dynamic energy in pJ.
        dram_energy: DRAM dynamic energy in pJ.
        tsv_energy: TSV (through-silicon-via) dynamic energy in pJ.
        sa_flops: Systolic-array FLOPS per core.
        vu_flops: Vector-unit FLOPS per core.
        dram_utils: List of (read_util, write_util) tuples.
        fused_op_logs: Per-run lists of ``FusedOperatorExecLog``.
        overlap_lists: Per-run lists of ``OverlapInterval``.
        npu_freq_MHz: NPU clock frequency in MHz.
        comp: Compute model (for utilization and area calculations).
        out_dir: Root output directory for logs and figures.
        uniform_dram_mapping: Flag indicating degraded DRAM mapping baseline.
        spmd_compiler: Flag indicating naive-compiler baseline.
        seq_noc: Flag indicating degraded-NoC baseline.
        ipu_tsim: Flag for IPU-style synthetic comparison mode.
    """
    import matplotlib.pyplot as plt

    print(f"PARSING RESULTS {out_dir=}")
    assert(len(exec_times) == len(fused_op_logs) == len(hw_cfg_info) == len(exec_energy)
           ==len(dram_utils) == len(overlap_lists)), "Each run should have all stats!"
    best_exec = -1
    best_exec_idx = -1
    #Find best config:
    for (i, time) in enumerate(exec_times):
        if time > 0 and (time < best_exec or best_exec == -1):
            print(f"Updating best_time = {time}, idx = {i}")
            best_exec_idx, best_exec = i, time
    print(f"Best exec found at index: {best_exec_idx}")
    assert(best_exec_idx != -1), "No best configuration found!"
    #Iterate through all data and graph only the best-performing config.
    for (i, (exec_time, dyn_energy_pJ, comp_energy_pJ, sss_energy_pJ, sa_energy_pJ, vu_energy_pJ, noc_energy_pJ, sram_energy_pJ, tsv_energy_pJ,
             dram_energy_pJ, sa_flop_per_core, vu_flop_per_core, dram_util, fused_op_log, overlap_list_raw)) in \
        enumerate(zip(exec_times, exec_energy, comp_energy, sss_energy, sa_energy, vu_energy, noc_energy, sram_energy, tsv_energy,
                       dram_energy, sa_flops, vu_flops, dram_utils, fused_op_logs, overlap_lists)):
        print("Processing config index:", i)
        if(exec_time == -1):
            print("Skipping invalid execution time")
            continue
        if SKIP_NON_BEST and i != best_exec_idx:
            print(f"Skipping non-best config {i} with exec time {exec_time}")
            continue

        overlap_list = []

        # Remove extraneous entries in overlap list before graphing
        for overlap in overlap_list_raw:
            if overlap.t_start != overlap.t_end:
                overlap_list.append(overlap)

        # prepare output directory/file
        comp_str, mem_str, noc_str, cg_str, exe_str = hw_cfg_info[i].get_cfg_str()
        out_path = os.path.join(out_dir, comp_str, mem_str, noc_str)
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
        best_path = os.path.join(out_path, best_str)
        best_f = os.path.join(best_path, f"output_{cg_str}.log")
        best_overlap_f = os.path.join(best_path, f"overlap_{cg_str}.log")
        top_power_f = os.path.join(best_path, f"top_power_{cg_str}.log")
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(best_path, exist_ok=True)
        GRAPH_RESULTS = True
        log_str = ""
        num_ops = len(fused_op_log)
        num_graph_ops = int(num_ops/20)
        out_file = os.path.join(out_path, f"output_{cg_str}_{exe_str}.log")
        print("Log open as", out_file)
        sa_util, vu_util = get_overall_comp_unit_utilization(fused_op_log, exec_time, comp)
        noc_util = get_noc_util(fused_op_log, exec_time)
        if noc_util == 0:
            noc_util = 1e-16
            print("NoC utilization is 0, setting to 1e-16", file=sys.stderr)

        check_results = True
        if ipu_tsim:
            check_results = False
        if check_results:
            assert sa_util>0, f"SA Utilization should be > 0, got {sa_util}"
            assert vu_util>0, f"VU Utilization should be > 0, got {vu_util}"
            assert noc_util>0, f"NoC Utilization should be > 0, got {noc_util}"
            assert dram_util[0]>0, f"DRAM Read Utilization should be > 0, got {dram_util[0]}"
            # Geometric mean of utilizations; DRAM read util is squared to weight memory-boundedness.
            overall_util = statistics.geometric_mean([sa_util, vu_util, noc_util, dram_util[0]**2])
        else:
            overall_util = 0.0

        # --- Area breakdown and static/dynamic energy/power estimation ---
        DRAM_SIZE_MB = 192 * 1024  # 192 GB, hardcoded for current HBM assumption
        area, logic_area, per_comp_area = get_area_breakdown(comp=hw_cfg_info[i].comp,
                           noc=hw_cfg_info[i].noc,
                           dram=hw_cfg_info[i].mem,
                           per_core_sram_size_KB=hw_cfg_info[i].sram_size // 1024,
                           total_dram_size_MB=DRAM_SIZE_MB,
                           num_cores=hw_cfg_info[i].num_cores,
                           sram_type="SRAM",
                           dram_type="HBM",
                           dram_bw_GBps=hw_cfg_info[i].dram_bw_GBps,
                           out_dir=os.path.join(best_path, f"area_{cg_str}.pdf"))

        # Static power estimates based on published per-area and per-capacity figures.
        static_power_W_per_sq_mm = 0.061  # W/mm^2 -- estimated from H100
        dram_static_power_W_per_GB = 0.125  # W/GB -- estimated from HBM2 (memsys2018-dramsim paper)
        logic_static_power_W = logic_area * static_power_W_per_sq_mm
        dram_static_power_W = DRAM_SIZE_MB * dram_static_power_W_per_GB / 1024
        static_power_W = logic_static_power_W + dram_static_power_W
        exec_time_sec = (exec_time / (hw_cfg_info[i].npu_freq_mHz * 1e6))
        tsv_static_power_W = per_comp_area.get("tsv", 0.0) * static_power_W_per_sq_mm
        static_energy_J = static_power_W * exec_time_sec + tsv_static_power_W * exec_time_sec

        # Convert per-component dynamic energy from picojoules to joules.
        core_energy_J = (sa_energy_pJ + vu_energy_pJ + sram_energy_pJ) / 1e12
        sa_energy_J = sa_energy_pJ / 1e12
        vu_energy_J = vu_energy_pJ / 1e12
        sram_energy_J = sram_energy_pJ / 1e12
        noc_energy_J = noc_energy_pJ / 1e12
        dram_energy_J = dram_energy_pJ / 1e12
        tsv_energy_J = tsv_energy_pJ / 1e12
        dyn_power_W = (dyn_energy_pJ / 1e12) / (exec_time / (hw_cfg_info[i].npu_freq_mHz * 1e6 ))
        core_static_keys = {"systolic array", "vector unit", "sram"}
        core_static_power_W = sum([per_comp_area[k] * static_power_W_per_sq_mm for k in core_static_keys])
        sa_static_power_W = per_comp_area["systolic array"] * static_power_W_per_sq_mm
        vu_static_power_W = per_comp_area["vector unit"] * static_power_W_per_sq_mm
        sram_static_power_W = per_comp_area["sram"] * static_power_W_per_sq_mm
        noc_static_power_W = per_comp_area["noc"] * static_power_W_per_sq_mm
        dram_static_power_W = dram_static_power_W

        log_str += f"EXE time (total, fused): {exec_time},"
        log_str += f"Energy (mJ): {static_energy_J * 1e3 + dyn_energy_pJ / 1e9}, Static: {static_energy_J * 1e3} mJ, Dyn.: {dyn_energy_pJ / 1e9} mJ\n"
        log_str += f"Static Energy: SA = {sa_static_power_W * exec_time_sec * 1e3 } mJ, VU = {vu_static_power_W * exec_time_sec * 1e3} mJ, SRAM= {sram_static_power_W * exec_time_sec * 1e3} mJ, Core: {core_static_power_W * exec_time_sec * 1e3} mJ\n"
        log_str += f"Static Energy: DRAM = {dram_static_power_W * exec_time_sec * 1e3 } mJ, TSV = {tsv_static_power_W * exec_time_sec * 1e3} mJ, NoC: {noc_static_power_W * exec_time_sec * 1e3} mJ\n"
        log_str += f"Dynamic Energy: SA = {sa_energy_pJ /1e9 } mJ, VU = {vu_energy_pJ / 1e9} mJ, SRAM= {sram_energy_pJ / 1e9} mJ, Core: {core_energy_J * 1e3} mJ, NoC: {noc_energy_pJ / 1e9} mJ\n"
        log_str += f"Dynamic Energy: DRAM = {dram_energy_pJ / 1e9} mJ, TSV = {tsv_energy_pJ / 1e9} mJ \n"
        log_str += f"Power (w): {dyn_power_W + static_power_W}, Static: {static_power_W} W (dram: {dram_static_power_W} logic: {logic_static_power_W}), Dyn.: {dyn_power_W} W\n"
        log_str += f"Overall Util: {overall_util}\n"
        log_str += f"DRAM UTIL (%): {dram_util[0] * 100}/{dram_util[1] * 100} (R/W), SA_UTIL:{sa_util}, VU_UTIL:{vu_util}, NOC: {noc_util}\n"
        log_str += f"FLOPS: {hw_cfg_info[i].num_cores * sa_flop_per_core / (1024 ** 3)} GFLOPS MM, "
        log_str += f"{hw_cfg_info[i].num_cores * vu_flop_per_core / (1024 ** 3)} GFLOPS VU\n"
        for l_idx, log in enumerate(fused_op_log):
            log_str += log.print_stats()
        if(i == best_exec_idx): # Save a copy of the best performing config's log + graph to best path for easy access.
            with open(best_f, "w") as bf:
                print("Writing best log to", best_f)
                bf.write(log_str)
            with open(best_overlap_f, "w") as bof:
                print("Writing best overlap to", best_overlap_f)
                log_overlap(bof, overlap_list)
            with open(top_power_f, "w") as tpf:
                sorted_overlaps = sorted(overlap_list, key=lambda x: x.power_W, reverse=True)
                for overlap in sorted_overlaps[:20]:
                    tpf.write(str(overlap) + "\n")
            if check_results:
                fused_ops_file = os.path.join(best_path, f"output_cg_{hw_cfg_info[i].core_grp_size}.pickle")
                with open(fused_ops_file, "wb") as f:
                    print(f"Pickling fused op log to {fused_ops_file}")
                    pickle.dump(fused_op_log, f)

            fig, axs = plt.subplots(2, 2, width_ratios = [2, 2], layout="constrained") # Plot results and make figure
            exec_by_op_graph = axs[0][0]
            energy_by_op_graph = axs[1][0]
            hw_unit_util_graph = axs[0][1]
            intensity_graph = axs[1][1]
            fig.set_size_inches(30, 20)
            for l_idx, log in enumerate(fused_op_log):
                if l_idx < num_graph_ops and GRAPH_RESULTS:
                    log.draw_execution(exec_by_op_graph)
                    log.draw_energy(energy_by_op_graph)

            draw_overlap(hw_unit_util_graph, hw_unit_util_graph, overlap_list, fused_op_log, hw_cfg_info[i].dram_bw_GBps, npu_freq_MHz, 200)
            draw_dram_intensity(intensity_graph, fused_op_log, hw_cfg_info[i].dram_bw_GBps, npu_freq_MHz, num_graph_ops * 2)
            exec_by_op_graph.set_yticks(range(num_graph_ops), labels=[f"op_{i}" for i in range(num_graph_ops)])
            energy_by_op_graph.set_xticks(range(num_graph_ops), labels=[f"op_{i}" for i in range(num_graph_ops)])
            energy_by_op_graph.set_ylabel("Energy consumed (pJ)")
            energy_by_op_graph.set_xlabel("Fused operator #")
            fig.suptitle(f"Configuration: {hw_cfg_info[i].get_cfg_str()}")
            handles, labels = exec_by_op_graph.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            exec_by_op_graph.legend(by_label.values(), by_label.keys())
            handles, labels = energy_by_op_graph.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            energy_by_op_graph.legend(by_label.values(), by_label.keys())
            plt.savefig(os.path.join(best_path, f"output_{cg_str}_graph.pdf"))
            fig_dir = os.path.join(best_path, f"output_{cg_str}_graph.pdf")
            print(f"Saving best fig with exe_size = {hw_cfg_info[i].exe_sram_size} to {fig_dir}")
            plt.close(fig)

def log_overlap(file: IO, overlap_list: List[OverlapInterval]):
    """Write every overlap interval to *file* as a human-readable text line.

    Args:
        file: An open, writable file handle.
        overlap_list: List of ``OverlapInterval`` objects to serialize.
    """
    for interval in overlap_list:
        file.write(str(interval))

if __name__=="__main__":
    # Stand-alone demo: load a pre-compiled DNN program and sweep execution scratch-pad sizes.
    pickle_filename: str = "results/pickles/outputs_icbm_2048/736cores/llama2-13-b32/program.pickle"
    with open(pickle_filename, 'rb') as f:
        prog: DNNProgram = pickle.load(f)

    # DRAM technology parameters
    GDDR5_col = 128
    HBM_col = 64
    npu_freq_MHz = 2000
    GDDR5_freq_MHz = 7000
    HBM_freq_MHz = 3200
    num_cores = prog.tot_num_cores
    freq_ratio_GDDR5 = GDDR5_freq_MHz / npu_freq_MHz
    freq_ratio_HBM = HBM_freq_MHz/ npu_freq_MHz
    dram_capacity_MB = 1024 * 64
    dram_bandwidth_GBps = 15 * 1024
    sram_capacity_per_core_B= 500 * 1024
    col_size_per_core = get_per_cycle_bytes_per_core_from_DRAM_config(num_cores=num_cores, total_bandwidth_GBps=dram_bandwidth_GBps,  npu_freq_MHz=2000)
    GDDR5_row_size = col_size_per_core * GDDR5_col
    HBM_row_size = col_size_per_core * HBM_col

    GDDR5_timing = (24 / freq_ratio_GDDR5, 22 / freq_ratio_GDDR5, 24 / freq_ratio_GDDR5)
    HBM_timing = (14 / freq_ratio_HBM, 14 / freq_ratio_HBM, 14 / freq_ratio_HBM)
    GDDR5 = DRAM(*GDDR5_timing, GDDR5_row_size, col_size_per_core, num_cores=num_cores)
    HBM = DRAM(*HBM_timing, HBM_row_size, col_size_per_core, num_cores=num_cores)
    dram_names = ["GDDR5", "HBM"]
    drams = [GDDR5, HBM]

    nodes = list(range(num_cores))
    interconnect_graph = [[0]*len(nodes) for node in nodes]

    # Build a simple adjacency matrix for the interconnect graph.
    interconnect_graph[0][2] = 1
    interconnect_graph[2][0] = 1

    interconnect_graph[0][3] = 1
    interconnect_graph[3][0] = 1

    interconnect_graph[1][3] = 1
    interconnect_graph[3][1] = 1

    interconnect_graph[1][4] = 1
    interconnect_graph[4][1] = 1

    noc = NoC(3, Topo.MESH, nodes, interconnect_graph)
    core_group_sizes = [1, 2, 4, 8]
    vector_unit_size = 16
    mm_size = np.array([8, 8, 8])  # 8x8x8 systolic array

    comp = Compute(vector_unit_size, mm_size, vector_unit_size, mm_size, vector_unit_size, 2 * mm_size[-1] ** 2, mm_size[-1], 2)
    comp_op = Compute_OP(comp)
    hw_info = []
    op_logs = [] # List of (exec_t, fused_op_logs)
    exec_times = [] # List of (exec_t, fused_op_logs)
    exec_energy = [] # List of (exec_t, fused_op_logs)
    comp_energy, sss_energy, ssi_energy = [], [], []
    sa_flops, vu_flops = [], []
    overlap_lists = []

    dram_name = "HBM"
    dram= HBM
    core_group_size = 4
    overall_dram_util = []
    exe_space = 500 * 1024
    for exe_space in [i * 50 * 1024 for i in range(6, 11)]:
        print("RUNNING TSIM: EXE_SPACE", exe_space)
        print("COMP FLOPC:", comp.get_peak_flopc())
        hw_cfg, stats, op_log, overlap_list = run_tsim(prog=prog, total_sram_byte_per_core=sram_capacity_per_core_B, exe_space=exe_space,
                    dram=dram, noc=noc, comp_op=comp_op, core_group_size=core_group_size,
                    dram_name=dram_name, npu_freq_MHz=npu_freq_MHz, dram_bandwidth_GBps=dram_bandwidth_GBps,
                    tot_layers=len(prog.ops), sim_layers=1)
        hw_info.append(hw_cfg)
        exec_times.append(stats["exec_time"])
        overall_dram_util.append((stats["dram_r_util"],
                                  stats["dram_w_util"]))
        exec_energy.append(stats["exec_energy"])
        comp_energy.append(stats["comp_energy"])
        sss_energy.append(stats["sss_energy"])
        ssi_energy.append(stats["ssi_energy"])
        sa_flops.append(stats["sa_flops"])
        vu_flops.append(stats["vu_flops"])
        op_logs.append(op_log)
        overlap_lists.append(overlap_list)

    with open("overlap.txt", "w") as o:
        log_overlap(o, overlap_list)
