"""
Model loading utilities and execution-plan search helpers.

This module bridges the gap between on-disk JSON model descriptions and the
in-memory ``DNNProgram`` representation used by the rest of the simulator.

Key responsibilities:

* **get_model_from_file** -- parse a JSON operator list (produced by the
  model parser) and construct a fully initialised ``DNNProgram``.
* **search_optimal_exe_load_config_order_independent_helper** -- worker
  function suitable for ``multiprocessing.Pool.map`` that evaluates a set
  of operator execution orders for a pickled ``DNNProgram``.
* **search_optimal_exe_load_config_baseline_independent_helper** -- similar
  worker for the baseline (fixed-capacity) search strategy.
"""

from typing import Any, Dict, List, Optional, Tuple
import itertools
import pickle
import time
from functools import lru_cache

from icbm_DNNProgram import DNNProgram, TensorOperator

import t10_TensorExpression as TE

# Cache for JSON file loading to avoid re-parsing the same files
@lru_cache(maxsize=32)
def _load_json_file(filename: str) -> tuple:
    """Load and cache JSON file contents"""
    import ujson as json
    with open(filename, "r") as f:
        ops_arr: List[List] = json.load(f)
    # Convert to tuple for hashability in lru_cache
    return tuple(tuple(op) if isinstance(op, list) else op for op in ops_arr)

def get_model_from_file(filename: str,
                        num_cores: List[int] = [1472],
                        tot_mem_size_per_core: int = 624 * 1024,
                        name: str = "",
                        output_dir: str = "") -> DNNProgram:
    """Load a JSON operator list and build a ``DNNProgram``.

    The JSON file is expected to contain an array of operator descriptors,
    where each descriptor is itself a list with the following positional
    fields::

        [0] name, [1] dim_lengths, [2] variables, [3] ignore_flags,
        [4] op_type, [5] output_idx, [6] input_idx_list

    Args:
        filename:             Path to the JSON file produced by the model parser.
        num_cores:            List of core-count configurations to evaluate.
        tot_mem_size_per_core: SRAM budget per core in bytes.
        name:                 Human-readable model name for logging.
        output_dir:           Directory for intermediate output files.

    Returns:
        A fully initialised ``DNNProgram`` ready for scheduling.
    """
    # Load from cache (lru_cache avoids re-parsing the same JSON)
    ops_arr_tuple = _load_json_file(filename)
    # Convert back to list format for processing
    ops_arr = [list(op) for op in ops_arr_tuple]

    ops = [
        TensorOperator(
            name        = f"Op_{op_idx}_{op_list[0]}",
            op_type     = op_list[4],
            dim_lengths = op_list[1],
            variables   = op_list[2],
            num_cores   = num_cores,
            max_byte_per_core   = tot_mem_size_per_core,
            ignore_variables    = [True] + op_list[3],
            output_idx          = op_list[5],
            input_idx_list      = op_list[6],
        ) for op_idx, op_list in enumerate(ops_arr)
    ]

    print(f"Core Utilization Constraint: {TE.CORE_UTIL_THRESHOLD}")
    print(f"Data Padding Constraint: {TE.DATA_PAD_THRESHOLD}")
    print(f"Num Dimensions Correlation: {TE.NUM_DIMS_CORRELATION}")

    return DNNProgram(num_cores, tot_mem_size_per_core, ops, name, output_dir)


def search_optimal_exe_load_config_order_independent_helper(param: Tuple[str, float, int, List[List[int]]]) \
        -> List[Tuple[  List[Tuple[float,float]],
                        List[Tuple[float,float]],
                        List[Tuple[int  ,int  ]],
                        List[Tuple[float,float]],
                        List[int], List[float]      ]]:
    """Multiprocessing worker: evaluate multiple operator execution orders.

    Deserialises a ``DNNProgram`` from *pickle_filename*, then for each
    execution order in *orders* runs the optimal-config search twice --
    once with delayed-load scheduling and once with delayed-compute
    scheduling.

    Args:
        param: A 4-tuple of (pickle_filename, hbm_bandwidth_GBps,
               num_layers, list_of_orders).

    Returns:
        A list with one ``[delay_load_result, delay_compute_result]`` pair
        per order.
    """
    pickle_filename, hbm_GBps, num_layers, orders = param
    with open(pickle_filename, 'rb') as f:
        prog:DNNProgram = pickle.load(f)

    results = []
    for i, order in enumerate(orders):
        result_delay_load = prog.search_optimal_exe_load_config_order(hbm_GBps, num_layers, order, False)
        result_delay_compute = prog.search_optimal_exe_load_config_order(hbm_GBps, num_layers, order, True)
        results.append([result_delay_load, result_delay_compute])

    return results

def search_optimal_exe_load_config_baseline_independent_helper(param: Tuple[str, float, int, List[int]]) \
        -> List[List[Tuple[ List[Tuple[float,float]],
                            List[Tuple[float,float]],
                            List[Tuple[int  ,int  ]],
                            List[Tuple[float,float]],
                            List[int], List[float]  ]]]:
    """Multiprocessing worker: baseline search over a list of SRAM budgets.

    Similar to ``search_optimal_exe_load_config_order_independent_helper``
    but uses the baseline strategy that sweeps over fixed execution-buffer
    sizes (*exe_kb_list*) rather than operator orderings.

    Args:
        param: A 4-tuple of (pickle_filename, hbm_bandwidth_GBps,
               num_layers, list_of_exe_kb_values).

    Returns:
        A list with one ``[delay_load_result, delay_compute_result]`` pair
        per SRAM budget value.
    """
    pickle_filename, hbm_GBps, num_layers, exe_kb_list = param
    with open(pickle_filename, 'rb') as f:
        prog:DNNProgram = pickle.load(f)

    results: List[List[Tuple[List[Tuple[float,float]],
                             List[Tuple[float,float]],
                             List[Tuple[int,int]],
                             List[Tuple[float,float]],
                             List[int], List[float]]]] = []
    for exe_kb in exe_kb_list:
        result = [prog.baseline_search_optimal_exe_load_config_all(hbm_GBps, num_layers, exe_kb, False),
                    prog.baseline_search_optimal_exe_load_config_all(hbm_GBps, num_layers, exe_kb, True)]
        results.append(result)
    return results
