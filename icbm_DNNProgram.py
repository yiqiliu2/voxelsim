"""Core DNN program representation for 3D-stacked chip simulation.

This module provides the ``TensorOperator`` and ``DNNProgram`` classes that
together form the central data model for mapping deep neural network workloads
onto a 3D-stacked accelerator architecture.  Key responsibilities include:

* **Operator representation** -- wrapping individual tensor operators with
  dimension, partitioning, and memory metadata (``TensorOperator``).
* **Operator fusion** -- grouping consecutive operators so intermediate
  tensors can stay on-chip and DRAM traffic is minimised.
* **DRAM timing** -- computing read / write cycle counts and byte traffic
  for each (fused) operator, accounting for ignore-variable elision and
  unconditional-write semantics.
* **Intra-operator optimisation** -- searching for the best spatial /
  temporal partition of each operator across cores.
* **Inter-operator optimisation** -- selecting cold / hot SRAM allocations
  across a group of operators to minimise end-to-end execution time.
* **Execution planning** -- assembling per-operator timing breakdowns
  (DRAM, NoC broadcast / shift / reduce, compute) and resolving overlaps
  to produce an overall execution schedule.
"""

from functools import lru_cache
import itertools
from concurrent.futures import ProcessPoolExecutor as Pool, as_completed
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import os, sys
import time
import math
import ujson as json

import t10_TensorExpression as TE
from t10_TensorExpression import DATAMOVE_PJ, TensorExpression, perf
from t10_OpPartitionSearch import build_spatial_search_tree, OpSpatialPartitionSearch

from icbm_fast_perms import edit_dist_permutations, reduced_edit_dist_permutations

from tsim_components.noc import NoC
from tsim_components.comp import Compute
from tsim_components.mem import DRAM
from tsim_components.tsim_analysis_lib import OverlapInterval, FusedOperatorExecLog, HardwareExecutionState

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

COLD_THRESHOLD = 1.01
"""Ratio threshold above which a cold partition is considered acceptable."""

SKIP_OP_THRESHOLD = 0
"""Minimum operation size (in bytes) below which an operator is skipped."""

# NoC load thresholds for different scheduling strategies (fraction of peak).
LOAD_NOC_THRESHOLD_NAIVE = 0.7
LOAD_NOC_THRESHOLD_BASE = 0.7
LOAD_NOC_THRESHOLD_ORDER = 0.7
LOAD_NOC_THRESHOLD_REORDER = 0.99

# Whether to delay compute until NoC transfer completes for each strategy.
DELAY_COMP_NAIVE = True
DELAY_COMP_BASE = True
DELAY_COMP_ORDER = False
DELAY_COMP_REORDER = False

GBPS_PER_CORE = 5.5
"""Default DRAM bandwidth per core in GB/s."""

PREFILL = False
"""Flag indicating whether the workload is prefill (True) or decode/inference (False)."""

INNER_GB = 1

def convert_tuple_to_list(t):
    """Recursively convert nested tuples to nested lists."""
    return [convert_tuple_to_list(x) for x in t] if isinstance(t, tuple) else t

def convert_list_to_tuple(l):
    """Recursively convert nested lists to nested tuples (for hashability / caching)."""
    return tuple(convert_list_to_tuple(x) for x in l) if isinstance(l, list) else l

@lru_cache(maxsize=512)
def create_tensor_expression(dim_lengths: Tuple[int, ...],
                             op_type: int,
                             variables: Tuple[Tuple[Tuple[int, ...], ...], ...],
                             num_cores: Tuple[int, ...] = (),
                             name="",
                             num_byte_per_elem: int = 2,
                             max_byte_per_core: int = 250000,
                             ignore_variables: Optional[Tuple[bool]] = None) -> TensorExpression:
    """Create (or return a cached) ``TensorExpression`` from hashable tuple arguments.

    All mutable list arguments are converted to tuples before calling so that
    the result can be memoised via ``@lru_cache``.

    Args:
        dim_lengths: Shape of the operator's iteration space.
        op_type: Operator type enum (e.g. MATMUL, CONV, ELEMWISE).
        variables: Per-variable dimension index lists.
        num_cores: Core counts along each spatial dimension.
        name: Human-readable name (unused by cache key but forwarded).
        num_byte_per_elem: Element width in bytes (default 2 for FP16/BF16).
        max_byte_per_core: Maximum SRAM budget per core in bytes.
        ignore_variables: Per-variable flags; ``True`` means the variable's
            DRAM traffic is elided (e.g. fused intermediate).

    Returns:
        A ``TensorExpression`` instance.
    """
    return TensorExpression(
        op_type,
        convert_tuple_to_list(dim_lengths),
        convert_tuple_to_list(variables),
        convert_tuple_to_list(num_cores),
        name,
        num_byte_per_elem,
        max_byte_per_core,
        convert_tuple_to_list(ignore_variables)
    )

class TensorOperator:
    """A single tensor operator in the DNN program graph.

    Wraps a ``TensorExpression`` with additional metadata required by the
    inter-operator scheduler: dataflow indices, per-core memory footprints,
    and ignore-variable flags that indicate which inputs/outputs can be
    elided from DRAM traffic (e.g. fused intermediates).

    Attributes:
        name: Human-readable identifier (e.g. ``"layer3_matmul"``).
        expr: The underlying ``TensorExpression`` used for partitioning
            and performance modelling.
        output_idx: Index of this operator's output tensor in the global
            tensor table (``None`` if not set).
        input_idx_list: Indices of input tensors consumed by this operator.
        dim_lengths: Shape of the operator's iteration space.
        op_type: Operator type enum value (MATMUL, CONV, ELEMWISE, ...).
        variables: Per-variable dimension index lists describing how each
            tensor maps onto the iteration space.
        num_cores: Core counts along each spatial dimension.
        num_byte_per_elem: Element width in bytes (default 2 = FP16/BF16).
        max_byte_per_core: SRAM budget per core in bytes.
        ignore_variables: Per-variable boolean flags.  ``True`` means the
            variable's DRAM traffic is skipped (fused intermediate).
        ignore_variables_unconfirmed: Tentative copy of ``ignore_variables``
            before fusion decisions are finalised.
        is_unconditional_write: If ``True``, the output *must* be written
            to DRAM regardless of fusion (e.g. KV-cache updates).
        min_hot_size_bytes_per_core: Minimum hot (execution) SRAM footprint
            per core in bytes, or ``None`` if not yet computed.
        min_cold_size_bytes_per_core: Minimum cold (preload) SRAM footprint
            per core in bytes.
    """

    def __init__(self, name: str,
                 op_type: int,
                 dim_lengths: Optional[List[int]] = None,
                 variables: Optional[List[List[List[int]]]] = None,
                 num_cores: List[int] = [1472],
                 num_byte_per_elem: int = 2,
                 max_byte_per_core: int = 624 * 1024,
                 ignore_variables: List[bool] = [],
                 output_idx: Optional[int] = None,
                 input_idx_list: Optional[List[int]] = None,
                ):
        self.name: str = name
        if dim_lengths is None or variables is None:
            return
        self.expr: TensorExpression = create_tensor_expression(
            convert_list_to_tuple(dim_lengths),
            op_type,
            convert_list_to_tuple(variables),
            convert_list_to_tuple(num_cores),
            "",
            num_byte_per_elem,
            max_byte_per_core,
            convert_list_to_tuple(ignore_variables)
        )

        self.output_idx: Optional[int] = output_idx
        self.input_idx_list: Optional[List[int]] = input_idx_list

        self.dim_lengths: List[int] = dim_lengths
        self.op_type = op_type
        self.variables: List[List[List[int]]] = variables
        self.num_cores: List[int] = num_cores
        self.num_byte_per_elem: int = num_byte_per_elem
        self.max_byte_per_core: int = max_byte_per_core
        self.ignore_variables: List[bool] = ignore_variables
        self.ignore_variables_unconfirmed: Optional[List[bool]] = ignore_variables
        self.is_unconditional_write: bool = False
        self.min_hot_size_bytes_per_core: Optional[int] = None
        """Minimum hot size (sum of all variables) in bytes."""
        self.min_cold_size_bytes_per_core: int = 0
        """Minimum cold size (sum of all variables) in bytes."""


# ============================================================================
# Module-level helper functions for parallel compilation (avoid pickling self)
# ============================================================================

def _get_available_memory_gb() -> float:
    """Get available system memory in GB using /proc/meminfo."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        return 32.0  # conservative fallback


def _get_compile_thread_cap() -> int:
    """Return the process-wide compilation thread cap.

    Controlled by the ``ICBM_COMPILE_THREADS`` environment variable.
    Falls back to ``os.cpu_count()`` (or 8) when the variable is unset.
    """
    try:
        return max(1, int(os.environ.get("ICBM_COMPILE_THREADS", "")))
    except ValueError:
        return os.cpu_count() or 8


def _build_spatial_tree(params: Tuple[int, List[int], float, int]) -> OpSpatialPartitionSearch:
    """Worker: build spatial partition search tree for one op (parallelised pre-phase)."""
    from t10_OpPartitionSearch import build_spatial_search_tree
    depth, tot_dim_size, dim_size_TH, num_core = params
    _, tree = build_spatial_search_tree(
        depth=depth, tot_dim_size=tot_dim_size,
        dim_size_TH=dim_size_TH, num_core=num_core,
    )
    return tree

def _precompute_estimates(params: Tuple[TensorExpression, OpSpatialPartitionSearch]) -> TensorExpression:
    """Worker: precompute temporal workload estimates for one heavy op."""
    expr, tree = params
    expr.precompute_estimates(tree)
    return expr

def _compile_single_op_intra(
    params: Union[
        Tuple[TensorExpression, List[Tuple[str, int]], int],
        Tuple[TensorExpression, List[Tuple[str, int]], int, OpSpatialPartitionSearch]
    ]
) -> Tuple[TensorExpression, List[Tuple[str, int]], float]:
    """Worker: run full intra-op optimization for one op.

    Args:
        params: Either (expr, opnames, num_threads) or (expr, opnames, num_threads, spatial_tree)
            - expr: TensorExpression to optimize
            - opnames: List of (name, layer_id) tuples for this operator
            - num_threads: Number of worker threads for parallel search
            - spatial_tree: Optional pre-built spatial partition search tree

    Returns:
        (expr, opnames, elapsed_time): Optimized expression, names, and execution time in seconds
    """
    expr, opnames, num_threads = params[:3]
    spatial_tree = params[3] if len(params) > 3 else None
    start = time.perf_counter()
    expr.search_optimal_config(num_threads, spatial_tree, log_filename=opnames[0][0])
    end = time.perf_counter()
    return expr, opnames, end - start

def _compile_single_op_cold_hot(
    params: Union[
        Tuple[TensorExpression, List[Tuple[str, int]], int, int],
        Tuple[TensorExpression, List[Tuple[str, int]], int, int, OpSpatialPartitionSearch]
    ]
) -> Tuple[TensorExpression, List[Tuple[str, int]], float]:
    """Worker: run cold search + cold-hot table for one op.

    Args:
        params: Either (expr, opnames, num_threads, threshold) or (..., spatial_tree)
            - expr: TensorExpression to optimize
            - opnames: List of (name, layer_id) tuples for this operator
            - num_threads: Number of worker threads for parallel search
            - cold_hot_threshold: Memory threshold for cold-hot table generation
            - spatial_tree: Optional pre-built spatial partition search tree

    Returns:
        (expr, opnames, elapsed_time): Optimized expression, names, and execution time in seconds
    """
    expr, opnames, num_threads, cold_hot_threshold = params[:4]
    spatial_tree = params[4] if len(params) > 4 else None
    start = time.perf_counter()
    expr.search_optimal_config_cold(num_threads, spatial_tree, log_filename=opnames[0][0])
    expr.generate_cold_hot_table(num_threads, cold_hot_threshold, log_filename=opnames[0][0])
    end = time.perf_counter()
    return expr, opnames, end - start


def _classify_and_setup_ops(
    exprs: List[TensorExpression],
    is_intra_mode: bool
) -> Tuple[List[int], List[int], int, int, int, List[Tuple[int, List[int], float, int]], int, int, float]:
    """Classify operators into heavy/light and compute resource allocation.

    Args:
        exprs: List of TensorExpression objects
        is_intra_mode: True for intra-op optimization, False for cold-hot table generation

    Returns:
        Tuple of (heavy_idxs, light_idxs, light_threads, threads_per_heavy,
                  heavy_batch_size, heavy_tree_params, n_heavy, n_light, avail_gb)
    """
    n_cpu = _get_compile_thread_cap()

    # Classify ops using TensorExpression.is_light_op() method
    # Light: elementwise, reduce, relu, slice, pool (or < 3 dims >= 512)
    # Heavy: matmul, conv, gather (or >= 3 dims >= 512)
    light_idxs = [i for i, e in enumerate(exprs) if e.is_light_op()]
    heavy_idxs = [i for i, e in enumerate(exprs) if not e.is_light_op()]

    # Sort heavy ops by estimated spatial config count (largest first) to avoid tail effects
    def _estimate_spatial_configs(idx: int) -> float:
        """Estimate number of spatial configs based on dimensions and core count.

        Spatial config count grows with:
        1. Number of dimensions (more dims = more partition choices)
        2. Dimension sizes (larger dims = more partition options)
        3. Core count (more cores = more ways to distribute)

        Use product of log(min(dim, cores)) to avoid overflow while preserving rank order.
        """
        e = exprs[idx]
        num_core = int(np.prod(e.num_cores))
        # Sum log-space to avoid overflow, convert back for ranking
        log_estimate = 0.0
        for dim in e.dim_lengths[:-1]:  # Exclude last dim (always 0 padding)
            if dim > 1:
                # Each dim contributes log(partition_choices) to total
                partition_choices = min(dim, num_core)
                log_estimate += np.log(partition_choices)
        return log_estimate  # Larger = more spatial configs

    heavy_idxs.sort(key=_estimate_spatial_configs, reverse=True)

    n_heavy, n_light = len(heavy_idxs), len(light_idxs)

    # Light ops: all in parallel, each gets a fair share of CPUs
    light_threads = max(1, n_cpu // max(1, n_light))

    # Heavy ops: run multiple in parallel, each with many threads
    avail_gb   = int(_get_available_memory_gb())
    max_total_threads = min(n_cpu, int(avail_gb // INNER_GB))  # total threads we can spawn
    num_ipu_cores = int(np.prod(exprs[0].num_cores))  # cores per IPU (assume all ops have same core count)
    max_total_threads *= int(np.ceil(513 / num_ipu_cores))
    max_total_threads = max(1, min(n_cpu, max_total_threads))

    # Threads per heavy op differs between modes
    if is_intra_mode:
        threads_per_heavy = max(1, n_cpu // 2)
    else:
        threads_per_heavy = int(max(1, n_cpu // 4, max_total_threads // max(1, n_heavy)))

    # How many heavy ops can run in parallel without exceeding RAM?
    heavy_batch_size = int(max(1, max_total_threads // threads_per_heavy))

    # Build spatial tree parameters for heavy ops
    heavy_tree_params = []
    for i in heavy_idxs:
        e = exprs[i]
        num_core = int(np.prod(e.num_cores))
        tot_dim_size = [min(num_core, d) for d in e.dim_lengths[:-1]]
        heavy_tree_params.append((len(e.dim_lengths) - 1, tot_dim_size,
                                  e.get_util_threshold(), num_core))

    return (heavy_idxs, light_idxs, light_threads, threads_per_heavy,
            heavy_batch_size, heavy_tree_params, n_heavy, n_light, avail_gb)


class DNNProgram:
    """End-to-end DNN workload mapped onto a 3D-stacked accelerator.

    A ``DNNProgram`` owns a sequence of ``TensorOperator`` instances grouped
    into *operator groups* (``op_groups``).  It drives three major compilation
    / analysis phases:

    1. **Intra-operator optimisation** -- for every unique operator expression,
       search for the best spatial / temporal partition across cores
       (``run_intra_op_optimization``).
    2. **Cold / hot table generation** -- for every operator, build a table
       mapping (cold_size, hot_size) pairs to execution times so that the
       inter-operator scheduler can trade SRAM allocation against latency
       (``generate_all_cold_hot_table``).
    3. **Inter-operator scheduling** -- given a memory budget per core,
       select cold / hot allocations across a fused-operator group and
       compute an overlapped execution timeline including DRAM, NoC, and
       compute phases (``compute_exec_time``).

    Attributes:
        name: Identifier for this program / workload configuration.
        op_groups: Operators organised into groups; operators within a
            group may be fused.
        num_cores: Core counts along each spatial dimension.
        tot_num_cores: Total number of cores (product of ``num_cores``).
        tot_mem_size_per_core: Total SRAM capacity per core in bytes.
        op_execution_plan: Serialised execution plans per operator group.
        max_hot_size: Largest hot-size requirement across all operators.
        uniform_dram_mapping: If ``True``, simulate sub-optimal DRAM address
            mapping (used for sensitivity studies).
        ipu_no_overlap: If ``True``, disable pipeline overlap between
            operators (IPU-style sequential execution).
        op_init_overhead: Fixed per-operator startup overhead in cycles.
    """

    def __init__(self, num_cores: Optional[List[int]] = None,
                 tot_mem_size_per_core: int = 624 * 1024,
                 ops: Optional[List[TensorOperator]] = None,
                 name: str = "",
                 output_dir: str = ""):
        """Initialise a DNN program.

        Args:
            num_cores: Core counts along each spatial dimension.
                Defaults to ``[1472]``.
            tot_mem_size_per_core: Total SRAM per core in bytes.
            ops: Initial flat list of operators (placed into a single group).
            name: Human-readable program name.
            output_dir: Root directory for serialised artefacts.
        """
        if ops is None:
            ops = []
        if num_cores is None:
            num_cores = [1472]

        self.name: str = name

        self._ops_cache: Optional[List[TensorOperator]] = None
        self.op_groups: List[List[TensorOperator]] = [ops]
        for op in self.ops:
            op.expr.output_dir = os.path.join(output_dir, name)

        self.num_cores: List[int] = num_cores
        self.tot_num_cores: int = int(np.prod(num_cores))
        self.spatial_search_tree: Dict = {}
        self.tot_mem_size_per_core: int = tot_mem_size_per_core

        # Per-group execution plans: {op_name -> ((cold_plan, (mem, time)),
        #   (hot_plan, (mem, time)), exe_time)}
        self.op_execution_plan: List[Dict[str, List]] = []

        # --- Compilation profiling timers ---
        self.spatial_search_tree_time: Dict[Any, float] = {}
        """Maps ``(depth, tot_dim_size, dim_size_TH)`` to search wall-time (s)."""
        self.intra_op_compile_time: Dict[str, float] = {}
        """Maps operator name to intra-op compile wall-time (s)."""
        self.cold_hot_table_compile_time: Dict[str, float] = {}
        """Maps operator name to cold/hot table generation wall-time (s)."""
        self.inter_op_compile_time: Dict[int, float] = {}
        """Maps operator-group index to inter-op compile wall-time (s)."""

        self.all_order_lists: List[List[int]] = []

        self.output_dir: str = os.path.join(output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.max_hot_size: int = 0

        self.uniform_dram_mapping: bool = False
        self.ipu_no_overlap: bool = False
        self.ipu_trace_tag: str = ""
        self.op_init_overhead: int = 0



    @property
    def ops(self) -> List[TensorOperator]:
        """Flat view of all operators across every group (lazily cached)."""
        if self._ops_cache is None:
            self._ops_cache = list(itertools.chain.from_iterable(self.op_groups))
        return self._ops_cache

    def trim_for_simulation(self):
        """Drop compilation-phase data that run_tsim never reads.

        Call this once, just before creating the worker Pool, to reduce the
        size of the pickled prog passed to each worker via _init_worker.

        Kept:   config_dict, cold_hot_table, hot_cold_table (needed by run_tsim)
        Dropped: spatial_search_tree, profiling dicts, cold_config_candidates,
                 op_execution_plan, all_order_lists, _ops_cache duplicate.
        """
        self.spatial_search_tree = {}
        self.spatial_search_tree_time = {}
        self.intra_op_compile_time = {}
        self.cold_hot_table_compile_time = {}
        self.inter_op_compile_time = {}
        self.op_execution_plan = []
        self.all_order_lists = []
        self._ops_cache = None  # rebuilt lazily from op_groups; avoids double-pickle
        for op in self.ops:
            op.expr.cold_config_candidates = {}

    def get_op_spatial_search_tree(self, op: TensorOperator):
        """Look up the pre-built spatial search tree for *op*."""
        return self.get_op_spatial_search_tree_TExpr(op.expr)

    def get_op_spatial_search_tree_TExpr(self, expr: TensorExpression):
        """Look up the spatial search tree for a ``TensorExpression``.

        The tree is keyed by ``(depth, total_dim_size, dim_size_threshold)``
        and must have been populated by a prior call to
        ``generate_all_spatial_search_trees``.
        """
        depth = len(expr.dim_lengths) - 1
        tot_dim_size = int(np.prod(self.num_cores))
        dim_size_TH = expr.get_dim_size_threshold()
        return self.spatial_search_tree[(depth, tot_dim_size, dim_size_TH)]

    def get_unique_expr_to_opnames_dict(self, intra: bool) \
        -> Dict[TensorExpression, List[Tuple[str, int]]]:
        """Build a mapping from each *unique* ``TensorExpression`` to the
        ``(op_name, op_index)`` pairs that share it.

        Operators that are structurally identical (same dimensions, variables,
        etc.) share a single expression, so optimisation results computed once
        can be reused for all of them.

        Args:
            intra: If ``True``, serialise the mapping to
                ``all_configs_dict.json``; otherwise to
                ``cold_hot_table_dict.json``.

        Returns:
            Dict mapping each unique ``TensorExpression`` to a list of
            ``(operator_name, flat_operator_index)`` tuples.
        """
        expr_to_opnames_dict: Dict[TensorExpression, List[Tuple[str,int]]] = {}
        opnames_to_first_name_dict = {}
        for op_idx, op in enumerate(self.ops):
            if op.expr not in expr_to_opnames_dict:
                expr_to_opnames_dict[op.expr] = []
                op.expr.name = op.name
            expr_to_opnames_dict[op.expr].append((op.name, op_idx))
            opnames_to_first_name_dict[op.name] = op.expr.name

        if intra:
            with open(f"{self.output_dir}/all_configs_dict.json", "w") as f:

                json.dump(opnames_to_first_name_dict, f, indent=4)
        else:
            with open(f"{self.output_dir}/cold_hot_table_dict.json", "w") as f:

                json.dump(opnames_to_first_name_dict, f, indent=4)
        return expr_to_opnames_dict

    def run_intra_op_optimization_for_op(self, params: Tuple[TensorExpression, List[Tuple[str, int]], int]):
        """Run the intra-operator partition search for a single unique expression.

        This is the unit of work dispatched to a ``ProcessPoolExecutor`` by
        ``run_intra_op_optimization``.

        Args:
            params: A 3-tuple of ``(expr, opnames, num_threads)`` where
                *expr* is the ``TensorExpression`` to optimise, *opnames* is
                the list of ``(name, flat_index)`` pairs sharing this
                expression, and *num_threads* controls internal parallelism.

        Returns:
            ``(expr, opnames, compile_time_seconds)`` so the caller can
            propagate results back to all operators that share this expression.
        """
        expr, opnames, num_threads = params
        start = time.perf_counter()
        expr.search_optimal_config(num_threads, None, log_filename=opnames[0][0])
        end = time.perf_counter()
        return expr, opnames, end - start

    def update_ops_and_compile_time(self, results: List[Tuple[TensorExpression, List[Tuple[str, int]], float]]):
        """Propagate optimised expressions back to all operators that share them.

        After parallel intra-op or cold/hot optimisation, each unique
        ``TensorExpression`` has been updated in a worker process.  This
        method copies the result back to every ``TensorOperator`` that
        references the same expression.

        Args:
            results: List of ``(expr, opnames, compile_time)`` tuples
                returned by the worker functions.
        """
        print("# Updating ops and compile time...", flush=True)
        exprs = []
        op_idx_to_expr_idx = {}
        for expr_idx, (expr, opnames, _) in enumerate(results):
            exprs.append(expr)
            for op_name, op_idx in opnames:
                op_idx_to_expr_idx[op_idx] = expr_idx
        print("Done updating op_idx_to_expr_idx.", flush=True)

        for op_idx, op in enumerate(self.ops):
            op.expr = exprs[op_idx_to_expr_idx[op_idx]]

    def run_intra_op_optimization(self, num_threads: int = 1):
        """Search for the optimal spatial/temporal partition for every operator.

        Workers are per-op.  inner_threads for each op is assigned
        proportionally to its tensor size (product of dim_lengths), so large
        ops (e.g. Dot) get more threads and never starve.

        Args:
            num_threads: Total thread budget.
        """
        print("# Running intra-op optimization...", flush=True)

        expr_dict = self.get_unique_expr_to_opnames_dict(intra=True)
        exprs      = list(expr_dict.keys())
        opnames_list = list(expr_dict.values())
        num_ops    = len(exprs)

        # Classify ops and compute resource allocation
        (heavy_idxs, light_idxs, light_threads, threads_per_heavy,
         heavy_batch_size, heavy_tree_params, n_heavy, n_light, avail_gb) = \
            _classify_and_setup_ops(exprs, is_intra_mode=True)

        print(f"  {num_ops} unique ops ({n_heavy} heavy / {n_light} light), "
              f"{avail_gb:.0f} GB avail", flush=True)
        print(f"  light: {n_light} ops × {light_threads} threads in parallel", flush=True)
        print(f"  heavy: {threads_per_heavy} threads/op, {heavy_batch_size} ops/batch", flush=True)

        results = []
        done = 0

        # Phase 0+1: build spatial trees for heavy ops AND compile light ops concurrently
        # Both are lightweight operations, so run them together for better overlap.
        spatial_trees = [None] * n_heavy  # pre-allocate
        tree_idx_map = {}  # future -> index in spatial_trees

        print(f"  running light ops + building {n_heavy} spatial trees + precomputing estimates concurrently...", flush=True)
        with Pool(n_light + n_heavy * 2) as p:
            # Submit tree builds
            for idx, params in enumerate(heavy_tree_params):
                fut = p.submit(_build_spatial_tree, params)
                tree_idx_map[fut] = idx

            # Submit light op compilations
            light_futs = set()
            if light_idxs:
                light_params = [(exprs[i], opnames_list[i], light_threads) for i in light_idxs]
                for par in light_params:
                    light_futs.add(p.submit(_compile_single_op_intra, par))

            # As trees complete, immediately submit estimate precomputation.
            # Track all pending futures and dynamically add estimate futures.
            est_fut_map = {}   # future -> heavy-op index
            pending = set(tree_idx_map.keys()) | light_futs
            while pending:
                # Wait for the next completion
                for fut in as_completed(pending):
                    pending.discard(fut)
                    if fut in tree_idx_map:
                        # Tree build completed — fire off estimate precomputation
                        idx = tree_idx_map[fut]
                        spatial_trees[idx] = fut.result()
                        est_fut = p.submit(_precompute_estimates,
                                           (exprs[heavy_idxs[idx]], spatial_trees[idx]))
                        est_fut_map[est_fut] = idx
                        pending.add(est_fut)
                    elif fut in est_fut_map:
                        # Estimate precomputation completed
                        idx = est_fut_map[fut]
                        est_expr = fut.result()
                        exprs[heavy_idxs[idx]] = est_expr
                    else:
                        # Light op completed
                        res = fut.result()
                        results.append(res)
                        done += 1
                        print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)
                    break  # Re-enter as_completed with updated pending set

        # Re-sort heavy ops by actual spatial config count
        if n_heavy > 0:
            spatial_config_counts = []
            for idx in range(n_heavy):
                expr = exprs[heavy_idxs[idx]]
                num_spatial = len(expr._precomputed_estimates)
                spatial_config_counts.append(num_spatial)
                print(f"    heavy op {idx}: {opnames_list[heavy_idxs[idx]][0][0]} has {num_spatial} spatial configs", flush=True)

            if n_heavy > 1:
                sorted_indices = sorted(range(n_heavy), key=lambda i: spatial_config_counts[i], reverse=True)
                heavy_idxs = [heavy_idxs[i] for i in sorted_indices]
                spatial_trees = [spatial_trees[i] for i in sorted_indices]
                spatial_config_counts = [spatial_config_counts[i] for i in sorted_indices]

                print(f"  sorted order (by spatial config count):", flush=True)
                for idx in range(min(5, n_heavy)):
                    print(f"    {idx+1}. {opnames_list[heavy_idxs[idx]][0][0]}: {spatial_config_counts[idx]} configs", flush=True)

        # Phase 2: heavy ops with continuous sliding window (keep pool fully utilized)
        # NOTE: heavy_idxs now sorted by actual spatial config count (largest first)
        # This ensures ops with most configs start first, reducing tail latency
        if n_heavy == 0:
            pass  # No heavy ops to process
        elif n_heavy == 1:
            # Single heavy op: run directly (no pool overhead)
            i, tree = heavy_idxs[0], spatial_trees[0]
            res = _compile_single_op_intra((exprs[i], opnames_list[i], threads_per_heavy, tree))
            results.append(res)
            done += 1
            print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)
        else:
            # Multiple heavy ops: continuous sliding window (process in size order)
            with Pool(heavy_batch_size) as p:
                pending_futs = {}  # future -> (idx, tree) mapping
                next_op_idx = 0  # Next heavy op to submit (already in size order)

                # Fill initial window with largest ops first
                while next_op_idx < min(heavy_batch_size, n_heavy):
                    i = heavy_idxs[next_op_idx]
                    tree = spatial_trees[next_op_idx]
                    param = (exprs[i], opnames_list[i], threads_per_heavy, tree)
                    fut = p.submit(_compile_single_op_intra, param)
                    pending_futs[fut] = (i, tree)
                    next_op_idx += 1

                # Process completions and submit new ops continuously
                while pending_futs:
                    for fut in as_completed(pending_futs):
                        # Handle completed op
                        res = fut.result()
                        results.append(res)
                        done += 1
                        print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)
                        del pending_futs[fut]

                        # Submit next op if any remain
                        if next_op_idx < n_heavy:
                            i = heavy_idxs[next_op_idx]
                            tree = spatial_trees[next_op_idx]
                            param = (exprs[i], opnames_list[i], threads_per_heavy, tree)
                            new_fut = p.submit(_compile_single_op_intra, param)
                            pending_futs[new_fut] = (i, tree)
                            next_op_idx += 1

                        break  # Only process one completion before checking for new submits

        self.update_ops_and_compile_time(results)
        print("# Done: intra-op optimization.", flush=True)

        with open(f"{self.output_dir}/intra_op_compile_time.json", "w") as f:
            json.dump(self.intra_op_compile_time, f, indent=4)

    def init_min_cold_size_bytes_per_core(self):
        """Initialise each operator's minimum cold-size from its cold/hot table.

        Also records ``self.max_hot_size`` -- the largest minimum hot-size
        across all operators -- and asserts it fits within the per-core SRAM
        budget.  This must be called after cold/hot tables have been generated.
        """
        max_hot_size = 0
        for op in self.ops:
            # Smallest cold size is the first key in the ordered cold_hot_table
            op.min_cold_size_bytes_per_core = next(iter(op.expr.cold_hot_table))
            # Smallest hot size is the first key in the ordered hot_cold_table
            min_hot_size = next(iter(op.expr.hot_cold_table))
            if min_hot_size > max_hot_size:
                max_hot_size = min_hot_size
        self.max_hot_size = max_hot_size
        assert self.max_hot_size <= self.tot_mem_size_per_core, f"max hot size {self.max_hot_size} exceeds total memory size {self.tot_mem_size_per_core}"

    def init_min_cold_size_bytes_per_core_fused(self, fused_hot_cold_table):
        """Initialise minimum cold-sizes for fused operators from a fused hot/cold table."""
        for op in fused_hot_cold_table:
            op.min_cold_size_bytes_per_core = next(iter(op.expr.fused_hot_cold_table)[1])


    def generate_all_cold_hot_table(self, num_threads: int, cold_hot_threshold: float = 1):
        """Build cold-hot Pareto tables for every unique operator expression.

        inner_threads per op is assigned proportionally to tensor size so that
        large ops get more threads and don't bottleneck at the tail.

        Args:
            num_threads: Total thread budget.
            cold_hot_threshold: Performance threshold for pruning the table.
        """
        print("# Generating cold-hot table for all ops...", flush=True)

        expr_dict = self.get_unique_expr_to_opnames_dict(intra=False)
        exprs      = list(expr_dict.keys())
        opnames_list = list(expr_dict.values())
        num_ops    = len(exprs)

        # Classify ops and compute resource allocation
        (heavy_idxs, light_idxs, light_threads, threads_per_heavy,
         heavy_batch_size, heavy_tree_params, n_heavy, n_light, avail_gb) = \
            _classify_and_setup_ops(exprs, is_intra_mode=False)

        print(f"  {num_ops} unique ops ({n_heavy} heavy / {n_light} light), "
              f"{avail_gb:.0f} GB avail", flush=True)
        print(f"  light: {n_light} ops × {light_threads} threads in parallel", flush=True)
        print(f"  heavy: {threads_per_heavy} threads/op, {heavy_batch_size} ops/batch", flush=True)

        results = []
        done = 0

        # Phase 0+1: build spatial trees for heavy ops AND compile light ops concurrently
        spatial_trees = [None] * n_heavy
        tree_idx_map = {}

        print(f"  running light ops + building {n_heavy} spatial trees concurrently...", flush=True)
        with Pool(n_light + n_heavy) as p:
            # Submit tree builds
            for idx, params in enumerate(heavy_tree_params):
                fut = p.submit(_build_spatial_tree, params)
                tree_idx_map[fut] = idx

            # Submit light op compilations
            light_futs = set()
            if light_idxs:
                light_params = [(exprs[i], opnames_list[i], light_threads, cold_hot_threshold)
                                for i in light_idxs]
                for par in light_params:
                    light_futs.add(p.submit(_compile_single_op_cold_hot, par))

            # Collect results as they complete
            all_futs = set(tree_idx_map.keys()) | light_futs
            for fut in as_completed(all_futs):
                if fut in tree_idx_map:
                    idx = tree_idx_map[fut]
                    spatial_trees[idx] = fut.result()
                else:
                    res = fut.result()
                    results.append(res)
                    done += 1
                    print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)

        # Phase 2: heavy ops in batches with pre-built spatial trees
        for batch_start in range(0, n_heavy, heavy_batch_size):
            batch_end = min(batch_start + heavy_batch_size, n_heavy)
            batch_indices = heavy_idxs[batch_start:batch_end]
            batch_trees = spatial_trees[batch_start:batch_end]
            batch_size = len(batch_indices)

            if batch_size == 1:
                i, tree = batch_indices[0], batch_trees[0]
                res = _compile_single_op_cold_hot(
                    (exprs[i], opnames_list[i], threads_per_heavy, cold_hot_threshold, tree))
                results.append(res)
                done += 1
                print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)
            else:
                batch_params = [(exprs[i], opnames_list[i], threads_per_heavy, cold_hot_threshold, tree)
                                for i, tree in zip(batch_indices, batch_trees)]
                with Pool(batch_size) as p:
                    futs = {p.submit(_compile_single_op_cold_hot, par): par for par in batch_params}
                    for fut in as_completed(futs):
                        res = fut.result()
                        results.append(res)
                        done += 1
                        print(f"    [{done}/{num_ops}] Completed {res[1][0][0]}", flush=True)

        self.update_ops_and_compile_time(results)
        print("# Done: cold-hot table generation for all ops.", flush=True)

        with open(f"{self.output_dir}/cold_hot_table_compile_time.json", "w") as f:
            json.dump(self.cold_hot_table_compile_time, f, indent=4)

    def get_best_op_to_advance(self, cur_plan, all_op_next_cold_sizes: List[int], ops: List[TensorOperator]) -> Tuple[int, float]:
        """Select the operator whose cold-size reduction yields the best trade-off.

        For each candidate operator, compute the ratio::

            ratio = (execution_time_increase) / (hot_size_increase)

        where *hot_size_increase* is the freed cold bytes (which become
        available as hot space for other operators).  The operator with the
        smallest ratio gives the most hot-space per unit of extra latency.

        If an operator's execution time does *not* increase (or even
        decreases) when its cold size shrinks, it is chosen immediately
        with ratio 0.

        Args:
            cur_plan: Current inter-op plan tuple (see
                ``search_optimal_global_config_heuristic``).
            all_op_next_cold_sizes: Next candidate cold size for each operator
                (``-1`` if exhausted).
            ops: The operators in this group.

        Returns:
            ``(best_op_index, best_ratio)``
        """
        best_op_idx = -1
        best_ratio = float("inf")
        for i, new_cold_size in enumerate(all_op_next_cold_sizes):
            if new_cold_size == -1:
                continue

            # compute exe time increase of this op
            old_cold_size = cur_plan[0][i][0]
            old_ref_plan = next(reversed(ops[i].expr.cold_hot_table[old_cold_size].values()))
            old_exe_time = old_ref_plan[2]

            new_ref_plan = next(reversed(ops[i].expr.cold_hot_table[new_cold_size].values()))
            new_exe_time = new_ref_plan[2]

            exe_time_increase = new_exe_time - old_exe_time
            if exe_time_increase <= 0:
                # No latency penalty for shrinking cold -- take this free win
                return i, 0
            hot_size_increase = old_cold_size - new_cold_size
            assert hot_size_increase > 0, f"hot size increase should be positive: {hot_size_increase}, old_cold_size: {old_cold_size}, new_cold_size: {new_cold_size}"

            # compute the ratio and update the best op
            ratio = exe_time_increase / hot_size_increase
            if ratio < best_ratio:
                best_ratio = ratio
                best_op_idx = i

        assert best_op_idx >= 0 and best_ratio < float("inf"), f"best_op_idx: {best_op_idx}, best_ratio: {best_ratio}"

        return best_op_idx, best_ratio



    def search_optimal_global_config_heuristic(self, op_groups: List[List[TensorOperator]]):
        """Find the best inter-operator cold/hot SRAM allocation via a greedy heuristic.

        For each operator group the algorithm iteratively *reduces* one
        operator's cold-size per step, choosing the operator with the best
        (smallest) ``(exe_time_increase / hot_size_increase)`` ratio.
        Shrinking cold space frees hot space that can be redistributed to
        all operators, potentially lowering their execution times.

        The search terminates when every operator has reached its minimum
        cold-size.  Among all intermediate plans that fit within the per-core
        SRAM budget, the one with the lowest total execution time is selected.

        Args:
            op_groups: List of operator groups to schedule.

        Returns:
            A list of best plans, one per group.  Each plan is a tuple
            ``(op_plans, cold_size, tot_size, exe_time, comp_cycles,
            shift_cycles, min_cold_size)``.  Returns a partial list if
            any group has no valid plan.
        """
        # Accumulate the best plan for each operator group
        op_group_plans: List[Tuple[List[Tuple[int, int]], int, int, float, float, float, int]] = []

        for op_group_idx, ops in enumerate(op_groups):
            # Candidate plans for this group: each entry is
            # (per-op plans, tot cold, tot size, exe time, comp cycles, shift cycles, min cold)
            ops_cold_hot_plans: List[Tuple[List[Tuple[int, int]], int, int, float, float, float, int]] = []

            # Strategy: iteratively reduce one operator's cold size per step.
            # Shrinking cold size frees hot space for all operators, lowering
            # their execution times but increasing the chosen operator's warm-up.
            # We greedily pick the operator with the best (smallest)
            #   (warm-up time increase) / (hot size freed) ratio.
            # The search continues until every operator has reached its minimum
            # cold size, then the best valid plan is selected.

            # Seed the search with the plan that uses the largest cold size
            # (and therefore the largest hot size) for each operator.
            op_plans = [
                (
                    next(reversed(
                        op.expr.cold_hot_table
                    )),
                    next(reversed(
                        op.expr.cold_hot_table[next(reversed(op.expr.cold_hot_table))]
                    ))
                ) for op in ops
            ]
            cold_size = sum([cold for cold, hot in op_plans])
            tot_size = max([hot - cold for cold, hot in op_plans]) + cold_size
            exe_time = sum([op.expr.cold_hot_table[cold][hot][2] for op, (cold, hot) in zip(ops, op_plans)])
            comp_cycles = sum([op.expr.cold_hot_table[cold][hot][1][1][2] for op, (cold, hot) in zip(ops, op_plans)])
            shift_cycles = sum([op.expr.cold_hot_table[cold][hot][1][1][3] for op, (cold, hot) in zip(ops, op_plans)])
            min_cold_size = sum([op.min_cold_size_bytes_per_core for op in ops if op.min_cold_size_bytes_per_core is not None])
            # (op_plans, cold_size, tot_size, exe_time, comp_cycles, shift_cycles, min_cold_size)
            cur_plan: Tuple[List[Tuple[int, int]], int, int, float, float, float, int] = (op_plans, cold_size, tot_size, exe_time, comp_cycles, shift_cycles, min_cold_size)

            # (cold size, exe time, is valid)
            this_group_search_iter_trace: List[Tuple[int, float, bool]] = []

            # otherwise, we need to iteratively reduce cold size for each op to find the best plan
            if cur_plan[2] <= self.tot_mem_size_per_core:
                # Shallow copy is sufficient - only need to copy the list of tuples, not deep copy
                ops_cold_hot_plans.append((cur_plan[0][:], cur_plan[1], cur_plan[2], cur_plan[3], cur_plan[4], cur_plan[5], cur_plan[6]))
            # init the cold size candidates for each op (the largest size is excluded because it is already in cur_plan)
            # cold_size_candidates[op index] is a list of cold sizes for the op in ascending order
            cold_size_candidates = [
                list(op.expr.cold_hot_table.keys())[:-1] for op in ops
            ]

            # Iterative greedy search
            iter_num = 0
            while True:
                # Record (cold_size, exe_time, is_valid) for this iteration
                this_group_search_iter_trace.append((cur_plan[1], cur_plan[3], cur_plan[2] <= self.tot_mem_size_per_core))

                ### 0. Terminate when every operator has exhausted its candidates
                if all([len(candidates) == 0 for candidates in cold_size_candidates]):
                    break

                ### 1. mutate the execution plan by reducing cold size for the chosen op

                ## 1.1. find the op that has the best (smallest) (warm-up time increase / hot size increase) ratio

                # get the possible next cold size for each op
                all_op_next_cold_sizes = [
                    colds[-1] if len(colds) > 0 else -1
                        for i, colds in enumerate(cold_size_candidates)
                ]

                # find the best op to advance the plan
                best_op_idx, best_ratio = self.get_best_op_to_advance(cur_plan, all_op_next_cold_sizes, ops)

                ## 1.2. update the execution plan

                # update new cold size for the chosen op
                cur_plan[0][best_op_idx] = (all_op_next_cold_sizes[best_op_idx], -1)
                new_max_hot_size = max(0, self.tot_mem_size_per_core - sum([cold for cold, hot in cur_plan[0]]))

                # Recompute hot sizes for *all* ops given the new available space
                for i, op in enumerate(ops):
                    new_cold_size = cur_plan[0][i][0]
                    new_hot_size = op.expr.get_best_hot_size_for_cold(new_cold_size, new_max_hot_size)
                    assert new_hot_size is not None, f"new_hot_size should not be None, new_cold_size: {new_cold_size}, new_max_hot_size: {new_max_hot_size}"
                    cur_plan[0][i] = (new_cold_size, new_hot_size)

                # Rebuild the plan tuple with updated totals
                new_tot_cold_size = sum([cold for cold, hot in cur_plan[0]])
                cur_plan = (
                    cur_plan[0],
                    new_tot_cold_size,
                    new_max_hot_size + new_tot_cold_size,
                    sum([op.expr.cold_hot_table[cold][hot][2] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.expr.cold_hot_table[cold][hot][1][1][2] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.expr.cold_hot_table[cold][hot][1][1][3] for op, (cold, hot) in zip(ops, cur_plan[0])]),
                    sum([op.min_cold_size_bytes_per_core for op in ops if op.min_cold_size_bytes_per_core is not None])
                )

                ### 3. Remove the consumed cold-size candidate for the chosen op
                cold_size_candidates[best_op_idx].pop()

                ### 4. put the new plan into list if this is a valid plan (tot size <= available memory)
                if cur_plan[2] <= self.tot_mem_size_per_core:
                    # Shallow copy is sufficient - only need to copy the list of tuples, not deep copy
                    ops_cold_hot_plans.append((cur_plan[0][:], cur_plan[1], cur_plan[2], cur_plan[3], cur_plan[4], cur_plan[5], cur_plan[6]))

                iter_num += 1

            # Select the plan with the lowest execution time among valid plans
            if len(ops_cold_hot_plans) > 0:
                best_plan = min(ops_cold_hot_plans, key=lambda plan: plan[3])
                op_group_plans.append(best_plan)
            else:
                return op_group_plans

        return op_group_plans


    def update_num_ipus(self, num_ipus: int, new_num_ipus: int):
        """Rename all path / name strings to reflect a different IPU count.

        Performs an in-place string replacement of ``'{num_ipus}ipus'`` with
        ``'{new_num_ipus}ipus'`` in the program name, output directory, and
        every operator expression's name and output directory.

        Args:
            num_ipus: The old IPU count embedded in current names.
            new_num_ipus: The new IPU count to substitute.
        """
        self.name = self.name.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
        self.output_dir = self.output_dir.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
        for op in self.ops:
            op.expr.name = op.expr.name.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")
            op.expr.output_dir = op.expr.output_dir.replace(f"{num_ipus}ipus", f"{new_num_ipus}ipus")

    def baseline_cold_hot(self, op: TensorOperator, exe_byte_per_core: int, use_largest_cold: bool):
        """Select baseline cold/hot sizes for *op* that fit within *exe_byte_per_core*.

        Two strategies are supported:

        * ``use_largest_cold=True`` -- start from the largest cold size whose
          minimum hot size still fits, then pick the largest feasible hot size.
        * ``use_largest_cold=False`` -- use the smallest cold size and pick
          the largest feasible hot size.

        Args:
            op: The operator to query.
            exe_byte_per_core: Available execution SRAM per core in bytes.
            use_largest_cold: Strategy selector (see above).

        Returns:
            ``(cold_size, hot_size)`` tuple in bytes.
        """
        if use_largest_cold:
            best_cold = -1
            for cold_size in reversed(op.expr.cold_hot_table):
                first_hot = next(iter(op.expr.cold_hot_table[cold_size]))
                if first_hot <= exe_byte_per_core:
                    best_cold = cold_size
                    break
            assert best_cold != -1, "exec size too small."
            best_hot = -1
            for hot_size in op.expr.cold_hot_table[best_cold]:
                if hot_size <= exe_byte_per_core:
                    best_hot = hot_size
                else:
                    break
            return best_cold, best_hot
        else:
            min_cold = next(iter(op.expr.cold_hot_table))
            best_hot = -1
            for hot_size in op.expr.cold_hot_table[min_cold]:
                if hot_size <= exe_byte_per_core:
                    best_hot = hot_size
                else:
                    break
            assert best_hot != -1, "exec size too small."
            return min_cold, best_hot

    def get_max_min_cold_and_hot(self, op: TensorOperator, exe_byte_per_core: int, use_max_cold: bool):
        """Return the cold and hot SRAM sizes for *op* given a per-core budget.

        Looks up the fastest configuration that fits in *exe_byte_per_core*,
        then selects either the maximum or minimum cold size from the
        hot-cold table.

        Args:
            op: Target operator.
            exe_byte_per_core: Available SRAM per core in bytes.
            use_max_cold: If ``True`` use the largest cold size; otherwise
                the smallest.

        Returns:
            ``(cold_mem, hot_mem)`` in bytes.
        """
        (hot_spatial, hot_temporal), (hot_mem, hot_exe_time, hot_comp_cycles, hot_shift_cycles) \
            = op.expr.get_fastest_config_by_max_mem_size(exe_byte_per_core)
        cold_dict = op.expr.hot_cold_table[hot_mem]
        if use_max_cold:
            cold_mem = next(reversed(cold_dict))
        else:
            cold_mem = next(iter(cold_dict))
        return cold_mem, hot_mem

    def update_te_hw(self, comp: Compute, noc: NoC,
                     spmd_compiler: bool = False, seq_noc: bool = False,
                     ipu: bool = False):
        """Propagate hardware component models to every operator expression.

        Optionally enables degraded-mode flags for sensitivity analysis:

        * ``spmd_compiler`` -- simulate a sub-optimal compiler.
        * ``seq_noc`` -- simulate a sub-optimal NoC.
        * ``ipu`` -- enable IPU-style execution semantics.

        Args:
            comp: Compute unit model.
            noc: Network-on-chip model.
            spmd_compiler: Enable degraded compiler mode.
            seq_noc: Enable degraded NoC mode.
            ipu: Enable IPU execution mode.
        """
        for op in self.ops:
            op.expr.comp = comp
            op.expr.noc = noc
            if spmd_compiler:
                op.expr.spmd_compiler = True
            if seq_noc:
                op.expr.seq_noc = True
            if ipu:
                op.expr.ipu = True

    def get_best_config_by_max_mem_size(self, op: TensorOperator, dram: DRAM, mem_size: int, core_group: int) -> Tuple[List[int], List[List[int]]]:
        """Select the best (spatial, temporal) config that fits in *mem_size*.

        Iterates over ``op.expr.config_dict`` (ordered by ascending memory)
        and picks the configuration minimising a weighted-power-mean of
        compute execution time and DRAM access time.  The power-mean with
        ``weight=2`` penalises imbalanced compute/DRAM overlap more heavily
        than a simple max.

        Args:
            op: Target operator.
            dram: DRAM model used to estimate access latency.
            mem_size: Maximum per-core SRAM available in bytes.
            core_group: Number of cores in the core group (affects DRAM BW).

        Returns:
            The ``(spatial, temporal)`` configuration tuple with the lowest
            weighted execution cost.
        """
        last_config = ()
        last_time = float("inf")
        weight = 2
        for config, (mem, exe_time, comp_cycles, shift_cycles) in op.expr.config_dict.items():
            if mem > mem_size:
                break
            spatial, temporal = config
            config_it = iter([(temporal, spatial)])
            dram_times = self.get_dram_time([op], dram, core_group, config_it)
            dram_time = dram_times[0]
            # Weighted power-mean: penalises compute/DRAM imbalance
            tot = exe_time**weight + dram_time**weight
            avg_time = (exe_time**(weight+1) + dram_time**(weight+1)) / tot
            if avg_time < last_time:
                last_time = avg_time
                last_config = config

        assert last_config != (), f"op {op.name} has no valid config for mem size {mem_size}"
        return last_config



    # -----------------------------------------------------------------------
    # ICBM Functions -- fused-operator execution timing and scheduling
    #
    # Execution timeline phases for each (fused) operator:
    #   t_dram_load    : start of DRAM read
    #   t_bcast        : start of NoC broadcast
    #   t_comp_shift   : start of compute + shift (NoC + compute overlap)
    #   t_reduce       : start of NoC reduce
    #   t_dram_store   : start of DRAM write
    #   t_finish       : end of execution
    # -----------------------------------------------------------------------

    def get_dram_time(self, fused_op: List[TensorOperator], dram: DRAM, core_group_size, partition_it):
        """Compute DRAM read/write cycles and bytes for a fused operator.

        For each sub-operator in the fused group:
        * Input variables whose ``ignore`` flag is ``False`` contribute to
          **read** traffic.
        * The output variable (index 0 in the access list) contributes to
          **write** traffic.  Intermediate outputs within the fused group
          are only written if the operator is marked ``is_unconditional_write``
          (e.g. KV-cache).  The *last* operator's write is reported separately
          so the caller can decide whether it is needed based on downstream
          fusion.

        Args:
            fused_op: List of sub-operators forming one fused operator.
            dram: DRAM timing model.
            core_group_size: Number of cores sharing the DRAM port.
            partition_it: Iterator yielding ``(temporal, spatial)`` partition
                tuples, one per sub-operator.

        Returns:
            A 9-tuple::

                (dram_r_cycles, dram_w_cycles_intermediate,
                 last_op_dram_w_cycles,
                 dram_r_bytes, dram_w_bytes_intermediate,
                 last_op_dram_w_bytes, last_op_unconditional,
                 dram_access_records, last_op_write_records)

            ``dram_access_records`` contains reads and any immediate writes.
            ``last_op_write_records`` contains the final output write record,
            which the caller includes only when the fused output reaches DRAM.
        """
        dram_r_cycles, dram_w_cycles_intermediate, last_op_dram_w_cycles = (0, 0, 0)
        dram_r_bytes, dram_w_bytes_intermediate, last_op_dram_w_bytes = (0, 0, 0)
        dram_access_records = []
        last_op_write_records = []
        last_idx = len(fused_op) - 1
        for (idx, op) in enumerate(fused_op):
            (temporal, spatial) = next(partition_it)
            # Single call returns cycles, bytes, and access granularity,
            # avoiding duplicate precise/ultra-precise DRAM simulations.
            access_list_cycles, access_list_bytes, access_list_granularity = dram.get_dram_access_list(
                                        op.expr.get_sub_op_var_sizes(temporal, spatial, False),
                                        op.expr.get_temporal_var_replicas(temporal, spatial),
                                        core_group_size,
                                        op.expr.num_byte_per_elem,
                                        return_cycles_bytes_granularity=True,
                                        bad_mapping=self.uniform_dram_mapping,
                                        )
            ignore_list = op.expr.ignore_variables
            # access_list[0] is the output; access_list[1:] are inputs.
            # Only accumulate reads for non-ignored input variables.
            # bytes_per_cycle is per-core, reconstruct chip-level TBps.
            # get_per_cycle_bytes_per_core uses 2^30 (binary GiB) and floor division,
            # so the correct inverse is / 2^40 (not / 1e12). ~1% error from floor div.
            std_time_fac = 1
            for tensor_index, (load_time, load_bytes, granularity, ignore) in enumerate(
                zip(access_list_cycles[1:], access_list_bytes[1:], access_list_granularity[1:], ignore_list[1:]),
                start=1,
            ):
                if not ignore:
                    scaled_load_time = load_time * std_time_fac
                    dram_r_cycles += scaled_load_time
                    dram_r_bytes += load_bytes
                    dram_access_records.append({
                        "version": 1,
                        "source": "tsim_get_dram_access_list",
                        "subop_index": int(idx),
                        "tensor_index": int(tensor_index),
                        "tensor_role": "input",
                        "stage": "read",
                        "bytes_per_core": int(load_bytes),
                        "total_bytes": int(load_bytes * self.tot_num_cores),
                        "access_granularity_bytes": int(granularity),
                        "cycles_per_core": int(load_time),
                        "scheduled_cycles": int(scaled_load_time),
                        "ignored": False,
                    })
            if idx == last_idx:
                # Last sub-op: defer write decision to caller
                last_op_dram_w_cycles = access_list_cycles[0]
                last_op_dram_w_bytes = access_list_bytes[0]
                last_op_unconditional = op.is_unconditional_write
                last_op_write_records.append({
                    "version": 1,
                    "source": "tsim_get_dram_access_list",
                    "subop_index": int(idx),
                    "tensor_index": 0,
                    "tensor_role": "output",
                    "stage": "write",
                    "bytes_per_core": int(access_list_bytes[0]),
                    "total_bytes": int(access_list_bytes[0] * self.tot_num_cores),
                    "access_granularity_bytes": int(access_list_granularity[0]),
                    "cycles_per_core": int(access_list_cycles[0]),
                    "scheduled_cycles": int(access_list_cycles[0]),
                    "ignored": False,
                    "deferred_last_output": True,
                    "unconditional_write": bool(op.is_unconditional_write),
                })
            elif op.is_unconditional_write:
                # Intermediate sub-op that must write (e.g. KV-cache update)
                dram_w_cycles_intermediate += access_list_cycles[0]
                dram_w_bytes_intermediate += access_list_bytes[0]
                dram_access_records.append({
                    "version": 1,
                    "source": "tsim_get_dram_access_list",
                    "subop_index": int(idx),
                    "tensor_index": 0,
                    "tensor_role": "output",
                    "stage": "write",
                    "bytes_per_core": int(access_list_bytes[0]),
                    "total_bytes": int(access_list_bytes[0] * self.tot_num_cores),
                    "access_granularity_bytes": int(access_list_granularity[0]),
                    "cycles_per_core": int(access_list_cycles[0]),
                    "scheduled_cycles": int(access_list_cycles[0]),
                    "ignored": False,
                    "unconditional_write": True,
                })

        return int(dram_r_cycles), int(dram_w_cycles_intermediate), int(last_op_dram_w_cycles), \
                int(dram_r_bytes), int(dram_w_bytes_intermediate), int(last_op_dram_w_bytes), last_op_unconditional, \
                dram_access_records, last_op_write_records
    def get_noc_times(self, fused_op: List[TensorOperator], partition_it, noc_data_it=None, noc: Optional[NoC] = None):
        """Sum the NoC cycle costs of all sub-operators in a fused group.

        When ``noc_data_it`` is provided, pre-computed per-operator
        ``(broadcast, shift, reduce)`` cycle tuples are consumed from the
        iterator.  Otherwise a placeholder cost of 1000 cycles per phase per
        sub-operator is used.

        TODO: replace placeholder cost when NoC data source is finalised.

        Args:
            fused_op: Sub-operators forming one fused operator.
            partition_it: Iterator yielding ``(temporal, spatial)`` per sub-op.
            noc_data_it: Optional iterator of pre-computed NoC cycle tuples.
            noc: NoC model (currently unused when *noc_data_it* is provided).

        Returns:
            ``(noc_bcast_cycles, noc_shift_cycles, noc_reduce_cycles)``
        """
        noc_bcast, noc_shift, noc_reduce = 0, 0, 0

        for op in fused_op:
            (temporal, spatial) = next(partition_it)
            if noc_data_it is None:
                # Placeholder: assume 1000 cycles per NoC phase per sub-op
                noc_bcast += 1000
                noc_shift += 1000
                noc_reduce += 1000
            else:
                op_bcast, op_shift, op_reduce = next(noc_data_it)
                noc_bcast += op_bcast
                noc_shift += op_shift
                noc_reduce += op_reduce

        return int(noc_bcast), int(noc_shift), int(noc_reduce)

    def get_aggregate_hot_cold_sizes(self, fused_op: List[TensorOperator], exe_sram_per_core, use_largest_cold: bool):
        """Sum the hot and cold SRAM sizes across all sub-operators in a fused group.

        TODO: does ignore need to be checked here? -- NOTE: simple summation
        may over-estimate when fused intermediates share memory.

        Args:
            fused_op: Sub-operators forming one fused operator.
            exe_sram_per_core: Available execution SRAM per core in bytes.
            use_largest_cold: If ``True``, pick the largest cold size per
                sub-operator; otherwise the smallest.

        Returns:
            ``(total_cold_size, total_hot_size)`` in bytes.
        """
        (cold_size, hot_size) = (0, 0)
        for op in fused_op:
            op_cold_size, op_hot_size = self.get_max_min_cold_and_hot(op, exe_sram_per_core, use_largest_cold)
            cold_size += op_cold_size
            hot_size += op_hot_size
        return cold_size, hot_size

    def used_by_next_fused_op(self, idx: int, fused_ops: List[List[TensorOperator]]):
        """Check whether fused-op *idx*'s output is consumed by the next fused-op.

        If the next fused operator has any ``True`` entry in its first
        sub-operator's ``ignore_variables[1:]``, that means it receives a
        fused intermediate from the current operator, so no DRAM write is
        needed for the current operator's output.

        Args:
            idx: Index of the current fused operator in *fused_ops*.
            fused_ops: Ordered list of fused-operator groups.

        Returns:
            ``True`` if the output is consumed by the next fused-op (i.e. the
            DRAM write can be elided); ``False`` if this is the last operator
            or the next operator does not consume the output as a fused input.
        """
        if idx == len(fused_ops) - 1:
            return False
        next_op = fused_ops[idx + 1]
        # If any input of the next fused-op's first sub-op is marked ignore,
        # it means that input comes from the current fused-op (fused path).
        if True in next_op[0].expr.ignore_variables[1:]:
            return True

    def get_fused_exec_time(self, fused_ops: List[List[TensorOperator]],
                            partitions: List[Tuple[List[List[int]], List[int]]],
                            comp: Compute, dram: DRAM, noc: NoC,
                            use_largest_cold: bool, overlap_bcast_read: bool,
                            compute_times: List[Tuple[int, int, int, int, int]],
                            sram_per_core: int, exe_sram_per_core: int,
                            core_group_size: int, noc_data: Any,
                            dram_bw_GBps: int, npu_freq_MHz: int):
        """Top-level entry point: compute the full execution timeline for fused operators.

        1. Calls ``get_fused_op_stats`` to gather per-fused-op DRAM, NoC,
           energy, and memory statistics.
        2. Passes those statistics to ``compute_exec_time`` which resolves
           pipeline overlaps and produces the final timeline.

        TODO: consider changing from fixed increment to sub_op_var_size.

        Args:
            fused_ops: List of fused operator groups.
            partitions: Partition configs for each sub-operator.
            comp: Compute unit model.
            dram: DRAM timing model.
            noc: NoC model.
            use_largest_cold: Cold-size selection strategy.
            overlap_bcast_read: Whether to overlap NoC broadcast with DRAM read.
            compute_times: Per-operator compute cycle breakdowns.
            sram_per_core: Total SRAM per core in bytes.
            exe_sram_per_core: Execution SRAM per core in bytes.
            core_group_size: Number of cores sharing a DRAM port.
            noc_data: Pre-computed NoC cycle data.
            dram_bw_GBps: DRAM bandwidth in GB/s.
            npu_freq_MHz: NPU clock frequency in MHz.

        Returns:
            The result of ``compute_exec_time`` -- a ``(stats_dict,
            operator_exec_log)`` tuple.
        """
        dram_bytes, dram_times, noc_times, fused_hot_cold_table, energy, comp_unit_stats, spatial_meta = \
            self.get_fused_op_stats(
                fused_ops=fused_ops, partitions=partitions, comp=comp,
                dram=dram, noc=noc, use_largest_cold=use_largest_cold,
                exe_sram_per_core=exe_sram_per_core,
                core_group_size=core_group_size, noc_data=noc_data)
        return self.compute_exec_time(fused_ops, dram_bytes, dram_times, noc_times,
                                      compute_times, comp_unit_stats, energy,
                                      fused_hot_cold_table, sram_per_core,
                                      exe_sram_per_core, overlap_bcast_read,
                                      dram_bw_GBps, npu_freq_MHz, spatial_meta)

    def get_fused_op_energy_from_scratch(self, fused_op: List[TensorOperator],
                                         part_it, dram_r_traffic: int = 0,
                                         dram_w_traffic: int = 0) -> Tuple[float, float, float, dict]:
        """Compute the total dynamic energy for a fused operator group.

        Sums per-sub-operator on-chip energy (SA, VU, SRAM, NoC) from the
        expression-level energy model, then adds off-chip DRAM access energy
        and TSV (through-silicon via) transfer energy.

        Args:
            fused_op: Sub-operators forming one fused operator.
            part_it: Iterator yielding ``(temporal, spatial)`` per sub-op.
            dram_r_traffic: Total DRAM read bytes (all cores combined).
            dram_w_traffic: Total DRAM write bytes (all cores combined).

        Returns:
            A 4-tuple ``(total_dynamic_energy_pJ, compute_energy_pJ,
            sss_energy_pJ, per_component_energy_dict)``.  The dict keys are
            ``"sa"``, ``"vu"``, ``"sram"``, ``"noc"``, ``"dram"``, ``"tsv"``.
        """
        total_dyn_energy, comp_energy, sss_energy, ssi_energy = 0, 0, 0, 0
        per_component_energy = {"sa": 0, "vu": 0, "sram": 0, "noc": 0}
        for idx, op in enumerate(fused_op):
            temporal, spatial = next(part_it)
            cfg_perf, energy_breakdown = self.get_op_perf(op, temporal, spatial)
            total_dyn_energy += cfg_perf.total_cycles.energy
            comp_energy += cfg_perf.comp_cycles.energy
            sss_energy += cfg_perf.sss_cycles.energy
            assert math.isclose(cfg_perf.total_cycles.energy, sum(energy_breakdown.values()), rel_tol=1e-4), \
                f"Energy breakdown sum {sum(energy_breakdown.values())} does not match total dynamic energy {cfg_perf.total_cycles.energy}"
            for component in per_component_energy:
                per_component_energy[component] += energy_breakdown[component]

        # --- Off-chip energy: DRAM and TSV ---
        # Energy-per-byte references (pJ/byte):
        #   HBM:  7 pJ/bit  => 56 pJ/byte  (https://docs.amd.com/v/u/en-US/wp485-hbm)
        #   HBM2: 3.9 pJ/bit => 31.2 pJ/byte (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8686544)
        #   HBM3: 3.4 pJ/bit => 27.2 pJ/byte (https://passlab.github.io/mchpc/mchpc2019/presentations/MCHPC_Pawlowski_keynote.pdf)
        #   3D-stacked DRAM: 7 pJ/byte (https://dl.acm.org/doi/10.1145/3695794.3695799)
        STACKED_3D_DRAM_PJ_PER_BYTE = 7
        dram_dyn_energy = (dram_r_traffic + dram_w_traffic) * STACKED_3D_DRAM_PJ_PER_BYTE
        # TSV energy: ~0.5 pJ/byte (https://ieeexplore.ieee.org/document/6159032)
        tsv_energy = (dram_r_traffic + dram_w_traffic) * 0.05 * DATAMOVE_PJ
        per_component_energy["dram"] = dram_dyn_energy
        per_component_energy["tsv"] = tsv_energy

        total_dyn_energy += dram_dyn_energy
        total_dyn_energy += tsv_energy
        assert math.isclose(total_dyn_energy, sum(per_component_energy.values()), rel_tol=1e-4), "op-level energy mismatch."
        return total_dyn_energy, comp_energy, sss_energy, per_component_energy

    def get_op_perf(self, op: TensorOperator, temporal, spatial) -> Tuple[perf, dict]:
        """Evaluate a single operator's performance and per-component energy.

        Args:
            op: Target operator.
            temporal: Temporal partition vector.
            spatial: Spatial partition vector.

        Returns:
            ``(perf_object, energy_breakdown_dict)`` where the dict keys are
            hardware component names (e.g. ``"sa"``, ``"vu"``, ``"sram"``,
            ``"noc"``).
        """
        _cfg_id, cfg_perf, energy_breakdown = op.expr.evaluate_energy(tuple([tuple(spatial), tuple(temporal)]))
        return cfg_perf, energy_breakdown

    def get_fused_op_comp(self, fused_op: List[TensorOperator]) -> Tuple[int, int]:
        """Count total floating-point operations for a fused operator group.

        CONV and MATMUL operators contribute to the systolic-array (SA) FLOP
        count (2x for multiply-accumulate), while all other operator types
        contribute to the vector-unit (VU) FLOP count.

        Args:
            fused_op: Sub-operators forming one fused operator.

        Returns:
            ``(sa_flop, vu_flop)`` -- total chip-wide FLOPs split by
            compute unit type.
        """
        sa_flop = 0
        vu_flop = 0
        for op in fused_op:
            dim_prod = np.prod(op.expr.get_dim_lengths())
            if op.expr.op_type == TE.TensorExpression.OP_TYPE_CONV \
                or op.expr.op_type == TE.TensorExpression.OP_TYPE_MATMUL:
                sa_flop += 2 * dim_prod  # multiply-accumulate = 2 FLOPs per element
            else:
                vu_flop += dim_prod

        return sa_flop, vu_flop

    def get_fused_op_stats(self, fused_ops: List[List[TensorOperator]],
                           partitions: List[Tuple[List[List[int]], List[int]]],
                           comp: Compute, dram: DRAM, noc: NoC,
                           use_largest_cold, exe_sram_per_core,
                           core_group_size, noc_data: Any = None) -> \
            Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]],
                  List[Tuple[int, int]], List[Tuple[int, int, int]]]:
        """Gather DRAM, NoC, energy, and memory statistics for all fused operators.

        For each fused operator group this method computes:

        * **DRAM traffic** (bytes and cycles) -- reads for non-ignored inputs;
          writes only for unconditional intermediates or when the output is
          *not* consumed by the next fused operator.
        * **NoC cycles** -- broadcast, shift, and reduce phases.
        * **Energy** -- per-component dynamic energy including DRAM and TSV.
        * **Compute unit statistics** -- per-core SA/VU FLOPs and ideal cycles.
        * **Hot/cold sizes** -- aggregate SRAM footprint.

        Compute-unit *execution* cycles are handled externally via ``comp.py``
        and are not recomputed here.

        Args:
            fused_ops: List of fused operator groups.
            partitions: Partition configs consumed once per sub-operator by
                each of the DRAM, NoC, and energy iterators.
            comp: Compute unit model (used for peak FLOP/cycle rates).
            dram: DRAM timing model.
            noc: NoC model.
            use_largest_cold: Cold-size selection strategy.
            exe_sram_per_core: Execution SRAM per core in bytes.
            core_group_size: Cores sharing a DRAM port.
            noc_data: Pre-computed NoC cycle data (iterable).

        Returns:
            A 7-tuple ``(dram_op_traffic, dram_op_times, noc_op_times,
            agg_hot_cold_table, fused_op_energy, comp_unit_stats, spatial_meta)``.
        """
        # Three independent iterators over the same partition list -- one each
        # for NoC, DRAM, and energy evaluation passes.
        noc_part_it = iter(partitions)
        dram_part_it = iter(partitions)
        energy_part_it = iter(partitions)
        noc_data_it = iter(noc_data)
        agg_hot_cold_table = []
        dram_op_times = []
        dram_op_traffic = []
        noc_op_times = []
        fused_op_energy = []
        comp_unit_stats = []
        spatial_meta = []
        peak_sa_flopc, peak_vu_flopc = comp.get_peak_flopc()
        inv_tot_num_cores = 1.0 / self.tot_num_cores
        for (idx, fused_op) in enumerate(fused_ops):
            # --- 1. NoC timing ---
            noc_bcast_cycles, shift_cycles, reduce_cycles = self.get_noc_times(fused_op, noc_part_it, noc_data_it, noc)
            noc_op_times.append((noc_bcast_cycles, shift_cycles, reduce_cycles))

            # --- 2. SRAM hot/cold footprint ---
            cold_size, hot_size = self.get_aggregate_hot_cold_sizes(fused_op, exe_sram_per_core, use_largest_cold)

            # --- 3. DRAM cycles and byte traffic ---
            read_cycles, write_cycles, last_op_wr_cycles, dram_r_traffic, dram_w_traffic, last_op_wr_traffic, last_op_unconditional, \
                dram_access_records, last_op_write_records \
                = self.get_dram_time(fused_op, dram, core_group_size, dram_part_it)
            # Include the last sub-op's write only if its output must go to DRAM
            # (unconditional write, or consumed by next fused-op via ignore flag)
            used_by_next = last_op_unconditional or self.used_by_next_fused_op(idx, fused_ops)
            write_cycles += last_op_wr_cycles if used_by_next else 0
            dram_w_traffic += last_op_wr_traffic if used_by_next else 0
            if used_by_next:
                dram_access_records.extend(last_op_write_records)

            # --- 4. Compute unit statistics (per-core FLOPs and ideal cycles) ---
            sa_flop, vu_flop = self.get_fused_op_comp(fused_op)
            sa_flop *= inv_tot_num_cores   # chip-wide -> per-core
            vu_flop *= inv_tot_num_cores
            sa_ideal_exe_cyc = sa_flop / peak_sa_flopc
            vu_ideal_exe_cyc = vu_flop / peak_vu_flopc
            comp_unit_stats.append((sa_flop, sa_ideal_exe_cyc, vu_flop, vu_ideal_exe_cyc))

            # --- 5. Accumulate results ---
            agg_hot_cold_table.append((hot_size, cold_size))
            dram_op_times.append((read_cycles, write_cycles))
            # Scale per-core byte traffic to chip-wide totals for energy model
            total_dram_r_traffic = dram_r_traffic * self.tot_num_cores
            total_dram_w_traffic = dram_w_traffic * self.tot_num_cores
            dram_op_traffic.append((total_dram_r_traffic, total_dram_w_traffic))
            fused_op_energy.append(self.get_fused_op_energy_from_scratch(
                fused_op, energy_part_it, total_dram_r_traffic, total_dram_w_traffic))
            spatial_meta.append(self._build_fused_spatial_meta(
                idx, fused_op, partitions, core_group_size, noc,
                total_dram_r_traffic, total_dram_w_traffic,
                dram_access_records))

        return dram_op_traffic, dram_op_times, noc_op_times, agg_hot_cold_table, fused_op_energy, comp_unit_stats, spatial_meta

    @staticmethod
    def _ranges_from_ids(ids):
        ids = sorted(set(int(i) for i in ids))
        if not ids:
            return []
        ranges = []
        start = prev = ids[0]
        for item in ids[1:]:
            if item == prev + 1:
                prev = item
            else:
                ranges.append([start, prev])
                start = prev = item
        ranges.append([start, prev])
        return ranges

    @staticmethod
    def _links_from_path(path):
        links = []
        for u, v in zip(path, path[1:]):
            if u == v:
                continue
            a, b = sorted((int(u), int(v)))
            links.append(f"{a}-{b}")
        return links

    def _build_fused_spatial_meta(self, idx, fused_op, partitions, core_group_size, noc,
                                  total_dram_r_traffic, total_dram_w_traffic,
                                  dram_access_records=None):
        """Build compact local spatial IDs for post-simulation thermal mapping.

        TSIM's current NoC and DRAM models assume contiguous logical core
        mappings. This metadata preserves that assumption explicitly without
        adding per-cycle tracing overhead.
        """
        num_cores = int(self.tot_num_cores)
        active_core_count = 1
        first_part_idx = min(idx, len(partitions) - 1)
        if first_part_idx >= 0:
            _temporal, spatial = partitions[first_part_idx]
            try:
                active_core_count = int(np.prod(spatial))
            except Exception:
                active_core_count = num_cores
        active_core_count = max(1, min(num_cores, active_core_count))
        active_cores = list(range(active_core_count))

        vault_count = max(1, int(math.ceil(num_cores / max(1, core_group_size))))
        vaults = sorted({min(core // max(1, core_group_size), vault_count - 1) for core in active_cores})

        bcast_links = set()
        reduce_links = set()
        shift_links = set()
        if getattr(noc, "exact_topo", False) and active_core_count > 1:
            root = active_cores[0]
            # Sample routes from the root to all active cores. This is compact
            # and follows the same contiguous logical mapping used by NoC timing.
            for core in active_cores[1:]:
                try:
                    _hops, path = noc.get_hops(root, core)
                except Exception:
                    path = [root, core]
                bcast_links.update(self._links_from_path(path))
                reduce_links.update(self._links_from_path(path))
            for u, v in zip(active_cores, active_cores[1:]):
                try:
                    _hops, path = noc.get_hops(u, v)
                except Exception:
                    path = [u, v]
                shift_links.update(self._links_from_path(path))

        return {
            "version": 1,
            "source": "tsim_contiguous_mapping",
            "op_id": int(idx),
            "num_cores": num_cores,
            "active_core_count": active_core_count,
            "core_group_size": int(core_group_size),
            "active_core_ranges": self._ranges_from_ids(active_cores),
            "dram_vault_count": vault_count,
            "dram_vaults": {
                "read": vaults if total_dram_r_traffic > 0 else [],
                "write": vaults if total_dram_w_traffic > 0 else [],
            },
            "dram_access_records": list(dram_access_records or []),
            "tsv_groups": {
                "read": vaults if total_dram_r_traffic > 0 else [],
                "write": vaults if total_dram_w_traffic > 0 else [],
            },
            "noc_links": {
                "bcast": sorted(bcast_links),
                "shift": sorted(shift_links),
                "reduce": sorted(reduce_links),
            },
        }

    @staticmethod
    def _attach_dram_access_timing(meta, read_start_cycle, read_duration_cycle,
                                   write_start_cycle, write_duration_cycle):
        """Attach scheduled cycle windows to tensor-level DRAM access records."""
        if not meta:
            return meta

        out = dict(meta)
        records = [dict(record) for record in meta.get("dram_access_records", [])]

        def _weight(record):
            for key in ("scheduled_cycles", "cycles_per_core", "total_bytes"):
                value = record.get(key)
                if value:
                    return max(0.0, float(value))
            return 1.0

        def _assign_stage(stage, stage_start, stage_duration):
            indices = [i for i, record in enumerate(records) if record.get("stage") == stage]
            if not indices:
                return

            stage_start = int(stage_start)
            stage_duration = max(0, int(stage_duration))
            weights = [_weight(records[i]) for i in indices]
            total_weight = sum(weights)
            cursor = stage_start
            elapsed = 0

            for pos, record_idx in enumerate(indices):
                if stage_duration <= 0 or total_weight <= 0:
                    duration = 0
                elif pos == len(indices) - 1:
                    duration = stage_duration - elapsed
                else:
                    duration = int(round(stage_duration * weights[pos] / total_weight))
                    duration = max(0, min(stage_duration - elapsed, duration))

                records[record_idx]["stage_start_cycle"] = stage_start
                records[record_idx]["stage_duration_cycle"] = stage_duration
                records[record_idx]["access_start_cycle"] = cursor
                records[record_idx]["access_duration_cycle"] = duration

                cursor += duration
                elapsed += duration

        _assign_stage("read", read_start_cycle, read_duration_cycle)
        _assign_stage("write", write_start_cycle, write_duration_cycle)

        out["dram_access_records"] = records
        return out

    def get_fused_dram_bytes_only(self, fused_ops: List[List[TensorOperator]],
                                   partitions: List[Tuple[List[List[int]], List[int]]],
                                   dram: DRAM, use_largest_cold, exe_sram_per_core,
                                   core_group_size) -> List[Tuple[int, int]]:
        """Lightweight variant of ``get_fused_op_stats`` returning only DRAM byte traffic.

        Skips NoC, energy, and compute-unit calculations, making it
        significantly cheaper when only total read/write bytes are needed
        (e.g. for DRAM bandwidth utilisation estimates).

        Args:
            fused_ops: List of fused operator groups.
            partitions: Partition configs for each sub-operator.
            dram: DRAM timing model.
            use_largest_cold: Cold-size selection strategy.
            exe_sram_per_core: Execution SRAM per core in bytes.
            core_group_size: Cores sharing a DRAM port.

        Returns:
            List of ``(total_dram_read_bytes, total_dram_write_bytes)``
            tuples, one per fused operator (chip-wide totals).
        """
        dram_part_it = iter(partitions)
        dram_op_traffic = []
        for idx, fused_op in enumerate(fused_ops):
            read_cycles, write_cycles, last_op_wr_cycles, dram_r_traffic, dram_w_traffic, last_op_wr_traffic, last_op_unconditional, \
                _, _ \
                = self.get_dram_time(fused_op, dram, core_group_size, dram_part_it)
            dram_w_traffic += last_op_wr_traffic if (last_op_unconditional or self.used_by_next_fused_op(idx, fused_ops)) else 0
            total_dram_r_traffic = dram_r_traffic * self.tot_num_cores
            total_dram_w_traffic = dram_w_traffic * self.tot_num_cores
            dram_op_traffic.append((total_dram_r_traffic, total_dram_w_traffic))
        return dram_op_traffic

    def compute_exec_time(self,
                          fused_ops: List[List[TensorOperator]],
                          dram_op_bytes: List[Tuple[int, int]],
                          dram_op_times: List[Tuple[int, int]],
                          noc_op_times: List[Tuple[int, int, int]],
                          comp_op_times: List[Tuple[int, int, int, int, int]],
                          comp_unit_stats: List[Tuple[int, float, int, float]],
                          energy,
                          hot_cold_table: List[Tuple[int, int]],
                          total_sram_per_core_B: int,
                          exe_sram_per_core_B: int,
                          overlap_bcast_read: bool,
                          dram_bw_GBps: int,
                          npu_freq_MHz: int,
                          spatial_meta: List[dict] = None) -> Tuple[int, int, List[FusedOperatorExecLog]]:
        """Assemble the overlapped execution timeline for all fused operators.

        This is the core scheduling function.  For each fused operator it
        determines when DRAM read, NoC broadcast, compute+shift, NoC reduce,
        and DRAM write phases can begin, respecting resource availability and
        SRAM space constraints tracked by ``HardwareExecutionState``.

        Two execution modes are supported:

        * **IPU no-overlap mode** (``self.ipu_no_overlap``): all phases are
          serialised per operator with no inter-operator overlap.
        * **Normal mode**: phases of consecutive operators can overlap as
          long as hardware resources (DRAM port, NoC, execution SRAM) are
          available.

        Args:
            fused_ops: Ordered list of fused operator groups.
            dram_op_bytes: ``(read_bytes, write_bytes)`` per fused operator.
            dram_op_times: ``(read_cycles, write_cycles)`` per fused operator.
            noc_op_times: ``(bcast, shift, reduce)`` cycles per fused operator.
            comp_op_times: ``(comp_cost, mm_cyc, ew_cyc, sram_r_cyc,
                sram_w_cyc)`` per fused operator.
            comp_unit_stats: ``(sa_flop, sa_ideal_cyc, vu_flop,
                vu_ideal_cyc)`` per fused operator.
            energy: Per-fused-operator energy tuples from
                ``get_fused_op_energy_from_scratch``.
            hot_cold_table: ``(hot_size, cold_size)`` per fused operator.
            total_sram_per_core_B: Total SRAM per core in bytes.
            exe_sram_per_core_B: Execution SRAM per core in bytes.
            overlap_bcast_read: Whether to overlap NoC broadcast with DRAM read.
            dram_bw_GBps: DRAM bandwidth in GB/s.
            npu_freq_MHz: NPU clock frequency in MHz.

        Returns:
            ``(stats_dict, operator_exec_log)`` where *stats_dict* aggregates
            performance / energy metrics and *operator_exec_log* contains
            per-operator ``FusedOperatorExecLog`` instances.
        """
        operator_exec_log = []
        # Accumulator for chip-wide performance and energy metrics
        stats = {
            "exec_time": 0,
            "exec_energy": 0,
            "comp_energy": 0,
            "sss_energy": 0,
            "sa_energy": 0,
            "vu_energy": 0,
            "noc_energy": 0,
            "sram_energy": 0,
            "dram_energy": 0,
            "tsv_energy": 0,
            "dram_r_util": 0.0,   # NOTE: initially bytes, converted to util later
            "dram_w_util": 0.0,
            "sa_flops": 0,
            "vu_flops": 0,
            "sa_util": 0.0,
            "vu_util": 0.0,
            "sa_uptime": 0.0,
            "vu_uptime": 0.0,
        }

        # --- IPU no-overlap mode: fully serialised execution ---
        if self.ipu_no_overlap:
            os.makedirs("results/pickles", exist_ok=True)
            trace_name = f"{self.name}-{self.op_init_overhead}.txt"
            if self.ipu_trace_tag:
                trace_name = f"{self.name}-{self.ipu_trace_tag}-{self.op_init_overhead}.txt"
            with open(os.path.join("results/pickles", trace_name), "w") as f:
                for dram_t, noc_t, comp_t in zip(dram_op_times, noc_op_times, comp_op_times):
                    dram_read, dram_write = dram_t
                    noc_bcast, noc_shift, noc_reduce = noc_t
                    comp_cost, mm_cyc, ew_cyc, sram_r_cyc, sram_w_cyc = comp_t
                    stats["exec_time"] += (max(dram_read,noc_bcast) +
                                        comp_cost +
                                        noc_shift +
                                        noc_reduce +
                                        dram_write +
                                        self.op_init_overhead)
                    f.write(f"{dram_read} {noc_bcast} {comp_cost} {noc_shift} {noc_reduce} {dram_write}\n")
            return stats, operator_exec_log

        # --- Normal pipelined execution ---
        exec_state = HardwareExecutionState(total_sram_per_core_B, exe_sram_per_core_B)
        invalid = False
        for (idx, op) in enumerate(fused_ops):
                # Check that the operator's hot-size fits in execution SRAM
                if hot_cold_table[idx][0] > exe_sram_per_core_B:
                    print(f"Unable to execute operators due to insufficient space: operator {idx} needs {hot_cold_table[idx][0]} > {exe_sram_per_core_B} (exe space)")
                    invalid = True
                    break

                # Determine when execution space becomes available for this op
                exec_space_next_avail = exec_state.get_exec_next_avail(hot_cold_table[idx][0])

                # Decide when DRAM read can start based on preload / overlap state
                if idx in exec_state.reserved_preloads:
                    # Preloaded: start as soon as both DRAM port and preload space are free
                    dram_r_start = max(exec_state.dram_r_next_avail_cycle,
                                       exec_state.reserved_preloads[idx].preload_avail_time)
                else:
                    # Not preloaded: must wait for execution space
                    if idx > 0 and exec_state.remaining_execution_space < hot_cold_table[idx][0]:
                        # Insufficient residual space -> cannot overlap with previous op
                        dram_r_start = max(exec_state.noc_next_avail_cycle,
                                           exec_state.dram_w_next_avail_cycle,
                                           exec_space_next_avail)
                    else:
                        # Enough residual space -> overlap DRAM read with previous op's tail
                        dram_r_start = max(exec_state.dram_r_next_avail_cycle,
                                           exec_space_next_avail)

                # Execute the five-phase pipeline for this operator
                op_data = exec_state.perform_op(
                    dram_r_start,
                    dram_r_duration=dram_op_times[idx][0],
                    noc_bcast_duration=noc_op_times[idx][0],
                    comp_shift_duration=max(noc_op_times[idx][1], comp_op_times[idx][0]),
                    noc_reduce_duration=noc_op_times[idx][2],
                    dram_w_duration=dram_op_times[idx][1],
                    op_idx=idx,
                    hot_cold_table=hot_cold_table,
                    exe_next_avail=exec_space_next_avail,
                    overlap_bcast_dram_read=overlap_bcast_read)

                op_spatial_meta = spatial_meta[idx] if spatial_meta and idx < len(spatial_meta) else None
                op_spatial_meta = self._attach_dram_access_timing(
                    op_spatial_meta,
                    read_start_cycle=op_data[0],
                    read_duration_cycle=dram_op_times[idx][0],
                    write_start_cycle=op_data[4],
                    write_duration_cycle=dram_op_times[idx][1])

                op_exec_breakdown = FusedOperatorExecLog(
                    *op_data, idx, dram_op_bytes[idx], dram_op_times[idx],
                    comp_op_times[idx], noc_op_times[idx], energy[idx],
                    comp_unit_stats[idx], hot_cold_table[idx],
                    dram_bw_GBps, npu_freq_MHz,
                    spatial_meta=op_spatial_meta)

                exec_state.sanity_check()

                # Accumulate per-operator stats into chip-wide totals
                stats["sa_flops"] += op_exec_breakdown.mm_flop_per_core
                stats["vu_flops"] += op_exec_breakdown.vu_flop_per_core
                stats["exec_energy"] += op_exec_breakdown.energy_total
                stats["comp_energy"] += op_exec_breakdown.energy_compute
                stats["sss_energy"] += op_exec_breakdown.energy_sss
                stats["sa_energy"] += op_exec_breakdown.energy_sa
                stats["vu_energy"] += op_exec_breakdown.energy_vu
                stats["noc_energy"] += op_exec_breakdown.energy_noc
                stats["sram_energy"] += op_exec_breakdown.energy_sram
                stats["dram_energy"] += op_exec_breakdown.energy_dram
                stats["tsv_energy"] += op_exec_breakdown.energy_tsv
                stats["dram_r_util"] += op_exec_breakdown.dram_r_bytes
                stats["dram_w_util"] += op_exec_breakdown.dram_w_bytes
                operator_exec_log.append(op_exec_breakdown)

        if invalid:
            stats["exec_time"] = -1
        else:
            stats["exec_time"] = operator_exec_log[-1].t_finish
        # NOTE: dram_r/w_util currently hold raw byte counts, not utilisation
        # ratios.  Conversion to utilisation happens after tsim completes.
        assert stats["dram_r_util"] >= 0, "Must read positive bytes!"
        assert stats["dram_w_util"] >= 0, "Must write positive bytes!"
        return stats, operator_exec_log
