from functools import lru_cache
import math
import functools
import itertools
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, Set
import numpy as np
from multiprocessing import Pool
import time
import t10_predictor
from tsim_components.comp import Compute
from tsim_components.noc import NoC
from t10_OpPartitionSearch import OpSpatialPartitionSearch, OpTemporalPartitionSearch, build_spatial_search_tree

FUSED_K = 1   # Weight of energy in the fused metric

ICBM_OVERLAP = False

CORE_UTIL_THRESHOLD = 0.97
DATA_PAD_THRESHOLD = 0.99
NUM_DIMS_CORRELATION = 1/17
MATMUL_CORE_RELAX = 32

MAX_MEM_THRESHOLD = 1.0

SYNC_CYCLES = 118
BUFFER_ITERS = 2
BUFFER_SIZE_RATIO = 1
IPU2_NUM_CORES = 1472
INTER_CHIP_OVERHEAD = 1.25

DUMP = False
DUMP_ALL = False
DUMP_ALL_UNIQUE = True
DUMP_ROLLER = False
DUMP_DICT = False

# READ_PJ = 3.6
# WRITE_PJ = 3.9
READ_PJ = 1.67 # https://gwern.net/doc/ai/scaling/hardware/2021-jouppi.pdf, SCALED FROM 7 to 5 nm
WRITE_PJ = READ_PJ #
DATAMOVE_PJ = 12
TSV_PJ_PER_BYTE = DATAMOVE_PJ * 0.05
# Reference: Jouppi et al., "Ten Lessons From Three Generations Shaped Google's TPUv4i"
# (https://www.cs.cmu.edu/~18742/papers/Jouppi2021.pdf)
# TPUv4i measured: MUL_PJ=0.21, ADD_PJ=0.11 (FP16).  We use conservative 5 nm estimates.
# MUL_PJ = 0.34 # 7nm
MUL_PJ = 0.30 # 5nm (est.) - FP16 multiply; cf. Jouppi et al. "Ten Lessons From Three Generations Shaped Google's TPUv4i" (2021) reports 0.21 pJ for FP16 mul on TPUv4i
# ADD_PJ = 0.16 # 7nm
ADD_PJ = 0.14 # 5nm (est.) - FP16 add; cf. Jouppi et al. (2021) reports 0.11 pJ for FP16 add on TPUv4i


SHIFT_INSTR_BYTE_PER_CYCLE = 3
COMP_INSTR_BYTE_PER_CYCLE = 1
IDLE_PJ_PER_CORE_CYCLE = 0
# IDLE_PJ_PER_CORE_CYCLE = 3


def get_mem_available_gb() -> float:
    """Get available system memory in GB using /proc/meminfo."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


class InfoFloat(float):
    time = 0.0
    energy = 0.0

    def __str__(self) -> str:
        return super().__str__() + " " + str(self.time) + " " + str(self.energy)

def create_fused_metric(time: float, energy: float) -> InfoFloat:
    rst = InfoFloat(FUSED_K * energy + (1 - FUSED_K) * time)
    rst.time = time
    rst.energy = energy
    return rst

all_configs_dict: Dict[str, str] = {}
cold_config_candidates_dict: Dict[str, str] = {}
cold_hot_table_dict: Dict[str, str] = {}

te_comp:Compute = None
te_noc:NoC = None

################################ helper functions ################################

# Parallel dict builder for cold config candidates
def _build_cold_dict_chunk(chunk_entries):
    """Worker: build partial cold config dict from a chunk of valid entries."""
    temp_dict = defaultdict(lambda: defaultdict(list))
    for config_id, config, cold_mem_size in chunk_entries:
        temp_dict[cold_mem_size][config[0]].append((config, (cold_mem_size, )))
    return dict(temp_dict)

# handle jagged ndarray
def pad_to_dense(M:List[List[int]]) -> np.ndarray:
    maxlen = max(len(r) for r in M)
    Z = np.zeros((len(M), maxlen),dtype=int)-1  # type: ignore
    for enu, row in enumerate(M):
        Z[enu, :len(row)] = row
    return Z

# convert per var shapes to per var sizes
def shape_to_size(shape:np.ndarray, return_size:bool=True) -> Union[int, np.ndarray]:
    if np.shape(shape)[1]>1:                # e.g., 1*1 kernel will not cause additional input size
        shape[:,1:][shape[:,1:]>0] -= 1
    if return_size:
        return int(np.prod(np.sum(shape,axis=-1)))
    else:
        return np.sum(shape,axis=-1)

# return number of compute iterations
def get_num_comp_iter(temporal_dim_var_parts:List[List[int]]) -> int:
    return np.prod(np.max(temporal_dim_var_parts,axis=-1))

# return all positive factors of @num (in ascending order)
# memoized for speed
@lru_cache(maxsize=4096)
def get_factors(num: int) -> List[int]:
    factors: List[int] = []
    # Optimized: only check up to sqrt(num) instead of num
    sqrt_num = int(num ** 0.5)
    for i in range(1, sqrt_num + 1):
        if num % i == 0:
            factors.append(i)
            if i != num // i:  # Avoid duplicates for perfect squares
                factors.append(num // i)
    # Sort since we added factors out of order
    factors.sort()
    return factors

# assumes @chain is sorted and len(chain) > 1
@lru_cache(maxsize=8192)
def is_chain_divisible_helper(chain: Tuple[int]) -> bool:
    for i in range(1, len(chain)):
        if chain[i] % chain[i - 1] != 0:
            return False
    return True

# return True if @chain forms a chain of divisible numbers
# e.g. if sorted(chain) = [a_1, a_2, ..., a_n], then a_i divides a_{i+1} for all i
def is_chain_divisible(chain: List[int]) -> bool:
    if len(chain) == 1:
        return True
    chain = sorted(chain)
    assert chain[0] > 0, "chain must be positive: %s" % chain
    return is_chain_divisible_helper(tuple(chain))

# @cold/hot_plan: ((spatial, temporal), (mem_size, exe_time))
def is_cold_hot_plan_compatible(cold_plan, hot_plan) -> bool:
    # cold mem size must be no greater than hot mem size
    if cold_plan[1][0] > hot_plan[1][0]:
        return False
    # spatial must be the same
    if cold_plan[0][0] != hot_plan[0][0]:
        return False
    # cold temporal must be divisible by hot temporal at each dimension and each variable
    for cold_var_temporal, hot_var_temporal in zip(cold_plan[0][1], hot_plan[0][1]):
        for c, h in zip(cold_var_temporal, hot_var_temporal):
            if h > c or c % h != 0:
                return False

    return True


# ---------------------------------------------------------------------------
# Module-level worker for temporal config generation (Pool-safe).
# ---------------------------------------------------------------------------
def _gen_temporal_work(args):
    """Generate temporal configs for one work item.

    Two modes selected by the first element of *args*:

      ('bulk', expr, [sc1, sc2, ...])
          Process a list of spatial configs.  Used for normal-sized spatials
          grouped via LPT bucketing.

      ('split', expr, spatial_config, search_space, num_replicas, depth)
          Process one spatial config using a pre-computed dim-0 subset of
          the temporal search space.  Used for mega spatials whose dim-0
          is split across multiple workers.

    Returns [(spatial_tuple, temporal_configs_list), ...].
    """
    tag = args[0]
    if tag == 'bulk':
        _, expr, sc_list = args
        return [(tuple(sc), expr.get_all_temporal_configs(sc)) for sc in sc_list]
    else:  # 'split'
        _, expr, sc, search_space, num_replicas, depth = args
        if not expr.update_spatial_dim_parts_if_valid(sc):
            return [(tuple(sc), [])]
        tree = OpTemporalPartitionSearch(
            depth=depth, search_space=search_space, num_replicas=num_replicas
        )
        tree.generateSearchTree()
        valid = tree.get_all_configs(
            lambda node: len(expr.valid_temporal_dim_var_parts(
                node.getConfig(), sc)) > 0
        )
        return [(tuple(sc), valid)]


################################ tensor expr class ################################

class perf:
    hot_mem_size: int
    cold_mem_size: int
    total_cycles: InfoFloat
    comp_cycles: InfoFloat
    sss_cycles: InfoFloat
    def __init__(self, hot_mem_size:int, cold_mem_size:int,
                 total_cycles:InfoFloat, comp_cycles:InfoFloat,
                 sync_shift_shuffle_cycles:InfoFloat):
        self.hot_mem_size = hot_mem_size
        self.cold_mem_size = cold_mem_size
        self.total_cycles = total_cycles
        self.comp_cycles = comp_cycles
        self.sss_cycles = sync_shift_shuffle_cycles

class TensorExpression:

    # 0: reduce, 1:relu, 2:elementwise, 3:pooling, 4:conv, 5:matmul
    OP_TYPE_REDUCE = 0
    OP_TYPE_RELU = 1
    OP_TYPE_ELEMENT = 2
    OP_TYPE_POOL = 3
    OP_TYPE_CONV = 4
    OP_TYPE_MATMUL = 5
    OP_TYPE_GATHER = 6

    OP_TYPE_SLICE = 8

    # all connected: [num_core]
    # 2d mesh: [num_core_vert, num_core_hori]
    num_cores:List[int]
    num_byte_per_elem:int
    max_byte_per_core:int

    # 0: reduce, 1:relu, 2:elementwise, 3:pooling, 4:conv, 5:matmul
    op_type:int

    #  0:m, 1:n, 2:k
    # [1600, 80, 1600]

    #  0:batches, 1:out_chl, 2:input_chl, 3:out_hei, 4:out_wid, 5:ker_hei, 6:ker_wid
    # [50,        60,        30,        256,       768,       3,         5]
    dim_lengths:np.ndarray
    spatial_dim_parts:np.ndarray            # [num spatial partitions per dim]
    spatial_dim_lengths:np.ndarray          # dim_lengths after spatial partition

    # C = A @ B
    # C [ [[0],[1]], \
    # A   [[0],[2]], \
    # B   [[2],[1]] ]

    # Output = conv(Input, Kernel)
    # O [ [[0],[1],[3  ],[4  ]], \
    # I   [[0],[2],[3,5],[4,6]], \
    # K   [[2],[1],[5  ],[6  ]] ]
    variables:List[np.ndarray]
    spatial_var_shapes:List[np.ndarray]     # replace indexes in variables with lengths
    spatial_var_replicas:List[int]          # [num replicas per var]

    pool_predictor = t10_predictor.pool()
    conv_predictor = t10_predictor.conv()
    ignore_variables:np.ndarray

    is_modified_in_grouping:bool = False
    log_filename_physical:str = ""

    # init TensorExpression using dim_lengths and variables
    def __init__(self, op_type:int, dim_lengths:List[int], variables:List[List[List[int]]], \
                 num_cores:List[int]=[], name="", num_byte_per_elem:int=2, max_byte_per_core:int=250000, ignore_variables:Optional[List[bool]] = None,
                 comp:Compute = te_comp,
                 noc:NoC = te_noc,
                 ) -> None:
        self.name = name
        self.dim_lengths = np.append(dim_lengths,0)                     # type: ignore
        self.spatial_dim_parts = np.zeros(np.shape(self.dim_lengths))
        self.variables = []
        for var in variables:
            self.variables.append(pad_to_dense(var))
        if ignore_variables is None:
            self.ignore_variables = np.array([False for var in variables])
        else:
            self.ignore_variables = np.array(ignore_variables)
        self.ignore_variables[0] = True
        self.num_cores = num_cores
        self.num_byte_per_elem = num_byte_per_elem
        self.max_byte_per_core = max_byte_per_core

        self.cold_hot_table: Dict[int, Dict[int, Any]] = {}
        self.hot_cold_table: Dict[int, Dict[int, Any]] = {}
        '''
        cold mem size -> hot mem size -> (best config: (cold config, hot config, exe time)) mapping
        x_config: ((spatial, temporal), (mem_size, exe_time))
        '''
        self.config_dict = {}
        '''(spatial, temporal) -> (mem_size, exe_time, comp_cycles, shift_cycles) mapping'''
        self.cold_config_candidates: Dict[int, Dict[Tuple, List]] = {}
        '''mem_size -> spatial plan -> [configs: ((spatial, temporal), (mem_size, exe_time))] mapping'''

        self.op_type = op_type

        self.comp = comp
        self.noc = noc

        self.spmd_compiler = False
        self.seq_noc = False
        self.ipu = False

        # Precompute constants for evaluate_config optimization
        self._is_ew = not (op_type == self.OP_TYPE_CONV or op_type == self.OP_TYPE_MATMUL)

    @property
    def _tensor_id_range(self):
        return list(range(len(self.variables)))

    def __getstate__(self):
        state = self.__dict__.copy()
        # Strip fields that workers never need:
        # - Spatial cache: always recomputed by update_spatial_dim_parts_if_valid
        # - _est_cache: transient optimisation cache, not needed across processes
        # NOTE: cold_config_candidates is NOT stripped — it must survive the
        #       round-trip from intra-op workers so that generate_all_cold_hot_table
        #       workers can use it in search_optimal_config_cold.
        for k in ('spatial_dim_parts', 'spatial_dim_lengths',
                  'spatial_var_shapes', 'spatial_var_replicas',
                  '_est_cache'):
            state.pop(k, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Reinitialize to zeros so update_spatial_dim_parts_if_valid triggers
        # a full recompute on first use in the worker.
        self.spatial_dim_parts = np.zeros(np.shape(self.dim_lengths))

    def get_dim_lengths(self) -> np.ndarray:
        return self.dim_lengths[:-1]

    def dump(self) -> bool:
        if DUMP_ALL_UNIQUE:
            return True
        return  self.op_type != self.OP_TYPE_ELEMENT \
            and self.op_type != self.OP_TYPE_RELU \
            and self.op_type != self.OP_TYPE_REDUCE \
            and self.op_type != self.OP_TYPE_SLICE

    def is_light_op(self) -> bool:
        """Classify operator as light (fast eval) or heavy (slow eval).

        Uses same logic as _classify_and_setup_ops in icbm_DNNProgram.
        Light ops: elementwise, reduce, relu, slice, pooling
        Heavy ops: matmul, conv, gather, or ops with >= 3 dimensions >= 512
        """
        # Explicit light ops
        _LIGHT_EXPLICIT = {self.OP_TYPE_ELEMENT, self.OP_TYPE_REDUCE,
                           self.OP_TYPE_RELU, self.OP_TYPE_SLICE, self.OP_TYPE_POOL}
        if self.op_type in _LIGHT_EXPLICIT:
            return True

        # Explicit heavy ops
        _HEAVY_EXPLICIT = {self.OP_TYPE_MATMUL, self.OP_TYPE_CONV, self.OP_TYPE_GATHER}
        if self.op_type in _HEAVY_EXPLICIT:
            return False

        # Unknown op type: classify by dimension complexity
        # Ops with >= 3 dimensions >= 512 are considered heavy
        LONG_DIM = 512
        HEAVY_NLONG = 3
        n_long = sum(1 for d in self.dim_lengths if d >= LONG_DIM)
        return n_long < HEAVY_NLONG

    ################ verification functions ################

    # update spatial_dim_parts [num spatial parts per dim] if it is valid
    #  0:batches, 1:out_chl, 2:input_chl, 3:out_hei, 4:out_wid, 5:ker_hei, 6:ker_wid
    # [5,         15,        3,         8,         8,         1,         1]
    def update_spatial_dim_parts_if_valid(self, spatial_dim_parts:List[int]) -> bool:
        if len(spatial_dim_parts):
            diff = self.spatial_dim_parts[0:-1]-spatial_dim_parts       # type: ignore
            if np.sum(np.abs(diff)):

                if len(self.num_cores):
                    req_cores = np.prod(spatial_dim_parts)
                    if req_cores > np.prod(self.num_cores):          # spatial partitions more than cores
                        return False
                    if req_cores/np.prod(self.num_cores) < \
                        self.get_util_threshold():
                            # print("debug",np.prod(self.dim_lengths[0:-1]))
                        return False                                    # spatial partitions less than util_threshold * num_cores

                spatial_dim_parts = np.append(spatial_dim_parts,1)                                      # type: ignore
                spatial_dim_lengths = np.ceil(self.dim_lengths/spatial_dim_parts)                       # type: ignore
                padded_dim_lengths = spatial_dim_lengths*spatial_dim_parts
                if np.prod(self.dim_lengths[0:-1]/padded_dim_lengths[0:-1]) < \
                    self.get_pad_threshold():
                    return False                                        # spatial partitions cause too much padding per core

                self.spatial_dim_parts = spatial_dim_parts                                              # type: ignore
                self.spatial_dim_lengths = spatial_dim_lengths

                self.spatial_var_shapes = []
                self.spatial_var_replicas = []
                for var in self.variables:
                    spatial_var_shape = self.spatial_dim_lengths[var]
                    padded_var_shape = padded_dim_lengths[var]
                    self.spatial_var_shapes.append(spatial_var_shape)
                    spatial_var_size = np.prod(np.max(spatial_var_shape,axis=-1))
                    padded_var_size = np.prod(np.max(padded_var_shape,axis=-1))
                    self.spatial_var_replicas.append(int(np.ceil(spatial_var_size*np.prod(self.spatial_dim_parts)/padded_var_size))) # type: ignore

        return True

    # validate temporal_dim_var_parts [for each dim [num temporal parts per variable]]
    #   0:batches, 1:out_chl, 2:input_chl, 3:out_hei, 4:out_wid, 5:ker_hei, 6:ker_wid
    # [ [2,1,1],   [1,1,1],   [1,1,1],   [1,1,1],   [1,1,1],   [1,1,1],   [1,1,5] ]
    #    O,I,K
    # return sub_op_shape, see get_sub_op_shape()
    def valid_temporal_dim_var_parts(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> List[int]:
        ##### already handled
        # if self.update_spatial_dim_parts_if_valid(spatial_dim_parts) == False: return []

        ##### already handled by get_factors()
        # if np.sum(self.spatial_var_replicas%temporal_var_parts):  # number of temporal partitions is not a factor of num_replica
        #     return []

        ##### already handled by is_chain_divisible()
        # if np.sum(np.max(temporal_dim_var_parts,axis=-1).reshape(-1,1)%temporal_dim_var_parts):
        #     return []                                             # some temporal parts per dim are not factor pairs
        # if np.sum(temporal_dim_var_parts%np.min(temporal_dim_var_parts,axis=-1).reshape(-1,1)):
        #     return []

        if np.sum(self.spatial_var_replicas % np.prod(temporal_dim_var_parts, axis = 0)) > 0:
            return [] # product of temporal partitions is not a factor of num_replica

        sp_temp_dim_lengths = self.get_sub_op_shape(temporal_dim_var_parts)
        if np.prod(self.spatial_dim_lengths[0:-1]/(sp_temp_dim_lengths*np.max(temporal_dim_var_parts,axis=-1))) < \
            self.get_pad_threshold():
            return []                                               # temporal partitions cause too much padding per iter per core

        # if np.prod(temporal_dim_var_parts,axis=0)[0]!=self.spatial_var_replicas[0]:
        #     return []                                             # drop plans that require additional reduction

        if len(self.num_cores) > 1:                                 # temporal partitions do not fit 2D mesh shape
            if np.max(temporal_dim_var_parts) > np.max(self.num_cores): return []
            if np.sum(temporal_dim_var_parts>1) > len(self.num_cores): return []                        # type: ignore
            if np.sum(temporal_dim_var_parts>1) == len(self.num_cores):                                 # type: ignore
                if np.min(temporal_dim_var_parts[temporal_dim_var_parts>1]) > np.min(self.num_cores):   # type: ignore
                    return []

        return sp_temp_dim_lengths

    # return an nd.array: [num temporal replicas per variable]
    def get_temporal_var_replicas(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> List[int]:
        if self.update_spatial_dim_parts_if_valid(spatial_dim_parts)==False: return []
        return self.spatial_var_replicas/np.prod(temporal_dim_var_parts,axis=0)

    def get_spatial_var_replicas(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> List[int]:
        if self.update_spatial_dim_parts_if_valid(spatial_dim_parts)==False: return []
        return self.spatial_var_replicas

    ################ get info functions ################

    # return threshold ratio based on op shape
    def get_util_threshold(self) -> float:
        prod = np.prod(self.dim_lengths[0:-1])
        util_threshold_ratio = (float) ( (prod) / (prod+2*np.prod(self.num_cores)) )
        maxi:float = max(self.dim_lengths[0:-1]) * len(self.dim_lengths[0:-1])
        util_threshold_ratio = min(util_threshold_ratio, maxi/220)
        if self.op_type==self.OP_TYPE_GATHER:
            return 0.951*util_threshold_ratio
        return CORE_UTIL_THRESHOLD*util_threshold_ratio

    def get_pad_threshold(self) -> float:
        if self.op_type==self.OP_TYPE_GATHER:
            return 0.951
        prod = np.prod(self.dim_lengths[0:-1])
        pad_threshold_ratio = (float) ( (prod) / (prod+np.prod(self.num_cores)) )
        maxi:float = max(self.dim_lengths[0:-1]) * len(self.dim_lengths[0:-1])
        pad_threshold_ratio = min(pad_threshold_ratio, maxi/220)
        pad_threshold_ratio *= (len(self.dim_lengths[0:-1])/7)**NUM_DIMS_CORRELATION
        if self.op_type==self.OP_TYPE_GATHER:
            return 0.96*pad_threshold_ratio
        return DATA_PAD_THRESHOLD*pad_threshold_ratio

    def get_dim_size_threshold(self, tier: int=20) -> float:
        dim_size_TH = self.get_util_threshold()
        dim_size_TH = np.floor(dim_size_TH*tier) / tier
        return dim_size_TH

    # return [sizes of per iter per core variables]
    # Output = conv(Input, Kernel)
    # [92160,      163200, 90]
    def get_sub_op_var_sizes(self, temporal_dim_var_parts:List[List[int]],
                             spatial_dim_parts:List[int]=[],
                             return_size:bool=True) -> Union[List[int], List[np.ndarray]]:
        if self.update_spatial_dim_parts_if_valid(spatial_dim_parts)==False: return []

        temp_dim_var_parts_np = np.concatenate([temporal_dim_var_parts, np.ones([1,len(temporal_dim_var_parts[0])],dtype=int)])

        sp_temp_var_sizes = []
        for var_idx,(var,var_shape) in enumerate(zip(self.variables,self.spatial_var_shapes)):
            divident = temp_dim_var_parts_np[:,var_idx][var]            # type: ignore
            new_shape = np.array(np.ceil(var_shape/divident),dtype=int) # type: ignore
            if return_size:
                sp_temp_var_sizes.append(shape_to_size(new_shape))
            else:
                sp_temp_var_sizes.append(shape_to_size(new_shape,
                                                       return_size=False))

        return sp_temp_var_sizes

    # return [per iter per core sub_OP dimension lengths]
    #  0:batches, 1:out_chl, 2:input_chl, 3:out_hei, 4:out_wid, 5:ker_hei, 6:ker_wid
    # [50,        60,        30,        256,       768,       3,         5]
    def get_sub_op_shape(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> List[int]:
        if self.update_spatial_dim_parts_if_valid(spatial_dim_parts)==False: return []

        sp_temp_dim_lengths = self.spatial_dim_lengths[0:-1]/np.max(temporal_dim_var_parts,axis=-1)
        return np.array(np.ceil(sp_temp_dim_lengths), dtype=int)        # type: ignore

    # return total volumn of shifts for the entire OP, and ordered scheduling info
    # total shift elem per core :   177880
    # index of dimensions       :   [ 0   , 6 ]
    # num shifts per round      :   [ 1   , 4 ]
    # vars to shift per round   :   [[0,1],[2]]
    #         Out-most loop    ->   [ shift [0,1]:[Output,Input] for 1 time along axis 0:batches,
    #       inner-most loop    ->     shift [ 2 ]:[   Kernel   ] for 4 time along axis 6:ker_wid ]
    def get_shift_info(self,
                       temporal_dim_var_parts:List[List[int]],
                       spatial_dim_parts:List[int]=[],
                       output_special_format_for_tsim=False) -> tuple:
        if self.update_spatial_dim_parts_if_valid(spatial_dim_parts)==False: return -1,[],[],[],[]

        shifted_dims = []   # indices of dims to shift
        shifted_iter = []   # number of shift iterations
        shifted_vars = []   # indices of variables to shift
        for dim_idx, dim in enumerate(np.array(temporal_dim_var_parts)):
            while np.sum(dim>1):
                gcd = np.gcd.reduce(dim[dim>1])                         # type: ignore
                if gcd == 1:
                    return -1,[],[],[],[]
                shifted_dims.append(dim_idx)
                shifted_iter.append(gcd-1)
                shifted_vars.append(np.nonzero(dim>1)[0])
                dim[dim>1] = dim[dim>1]/gcd

        if len(shifted_dims)==0:
            return 0,[],[],[],[]                                        # no need to shift
        shifted_vars_np = pad_to_dense(shifted_vars)
        sub_op_var_sizes = self.get_sub_op_var_sizes(temporal_dim_var_parts)
        shifted_sizes_per_tensor = np.append(sub_op_var_sizes,0)[shifted_vars_np]
        shifted_sizes = np.sum(shifted_sizes_per_tensor,axis=-1)
        order = np.argsort(-shifted_sizes)                              # type: ignore

        shifted_dims = np.array(shifted_dims)[order]
        shifted_iter = np.array(shifted_iter)[order]
        shifted_vars = np.array(shifted_vars,dtype=object)[order]
        shifted_sizes = shifted_sizes[order]*shifted_iter               # type: ignore
        shifted_sizes_per_tensor_list = []
        if output_special_format_for_tsim:
            for idx, s_iter in zip(order, shifted_iter):
                tensor_list = [size for size in shifted_sizes_per_tensor[idx] if size>0]
                sorted_tensor_list = np.sort(tensor_list)[::-1]
                shifted_sizes_per_tensor_list.append((sorted_tensor_list, s_iter))

        total_shift_size = 0
        iter_count = 1
        list_of_list_of_size_iter:List[List] = []
        for i in range(len(shifted_sizes)):
            total_shift_size += (iter_count*shifted_sizes[i])
            if output_special_format_for_tsim:
                sorted_tensor_list, s_iter = shifted_sizes_per_tensor_list[i]
                list_of_list_of_size_iter.append([])
                for t_size in sorted_tensor_list:
                    list_of_list_of_size_iter[-1].append((t_size, s_iter, iter_count))
            iter_count *= (shifted_iter[i]+1)

        if output_special_format_for_tsim:
            return total_shift_size, shifted_dims, shifted_iter, shifted_vars, list_of_list_of_size_iter
        return total_shift_size, shifted_dims, shifted_iter, shifted_vars, sub_op_var_sizes

    # return num of shift cycles
    # @shift_size = num of elem to shift
    def get_shift_time(self, shift_size:int) -> float:
        if shift_size:
            return shift_size * self.num_byte_per_elem / 2 * 0.56
        else:
            return 0

    def get_shift_time_energy_no_static(self, shift_size:int) -> Tuple[float, float]:
        if shift_size:
            vol = shift_size*self.num_byte_per_elem/2
            time = vol*0.56
            return (time, vol*DATAMOVE_PJ)
        else:
            return (0, 0)

    # return total num of sync and shift cycles per OP
    # @shift_info = output tuple of get_shift_info
    def get_sync_shift_time(self, shift_info:tuple) -> float:
        total_num_shifts = 0
        iter_acc = 1
        for iter, vars in zip(shift_info[2], shift_info[3]):
            total_num_shifts += (iter_acc*iter*len(vars))
            iter_acc *= (iter+1)
        shift_time = self.get_shift_time(shift_info[0])
        return SYNC_CYCLES*BUFFER_ITERS*total_num_shifts + shift_time

    def get_sync_shift_time_energy(self, shift_info:tuple) -> Tuple[float, float]:
        total_num_shifts = 0
        iter_acc = 1
        for iter, vars in zip(shift_info[2], shift_info[3]):
            total_num_shifts += (iter_acc*iter*len(vars))
            iter_acc *= (iter+1)
        shift_time, shift_energy = self.get_shift_time_energy_no_static(shift_info[0])
        time = SYNC_CYCLES*BUFFER_ITERS*total_num_shifts+shift_time
        # NOC = BUFFER_ITERS*total_num_shifts*DATAMOVE_PJ + shift_energy
        per_comp_breakdown = {
            "noc": BUFFER_ITERS*total_num_shifts*DATAMOVE_PJ + shift_energy,
            "sram": time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ,
            "sa": 0.0,
            "vu": 0.0,
        }
        return time, BUFFER_ITERS*total_num_shifts*DATAMOVE_PJ + shift_energy + time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ + time * IDLE_PJ_PER_CORE_CYCLE, per_comp_breakdown

    # return total inter-OP shuffle time after this OP
    def get_shuffle_time(self, temporal_dim_var_parts:List[List[int]]) -> float:
        if self.op_type in [self.OP_TYPE_ELEMENT, self.OP_TYPE_RELU, self.OP_TYPE_POOL]:
            return SYNC_CYCLES
        sub_output_size = self.get_sub_op_var_sizes(temporal_dim_var_parts)[0]
        num_output_replica = round(self.spatial_var_replicas[0] / np.prod(temporal_dim_var_parts,axis=0)[0])

        sub_op_shape = self.get_sub_op_shape(temporal_dim_var_parts)
        original_output_shape = sub_op_shape[self.variables[0][:,0]]
        original_output_product = int(np.prod(original_output_shape))
        reduced_output_product = original_output_product/num_output_replica

        if num_output_replica==1:
            gather_time = 0
            reduce_time = 0
        elif num_output_replica==2:
            gather_time = 2*SYNC_CYCLES + self.get_shift_time(original_output_product//2)
            if reduced_output_product <= 16432:
                aligned_min = 16434*0.5 + 300
                reduce_time = min(aligned_min, (reduced_output_product*0.75 + 290))
            else:
                reduce_time = (reduced_output_product*0.5 + 300)
        else:
            gather_time = num_output_replica*SYNC_CYCLES + self.get_shift_time(original_output_product)
            batch = num_output_replica
            aligned_output_product = reduced_output_product
            if aligned_output_product>2:
               aligned_output_product = max(4, aligned_output_product)
               if aligned_output_product>4:
                   aligned_output_product = max(8, aligned_output_product)
                   if aligned_output_product>8:
                       aligned_output_product = max(12, aligned_output_product)
                       if aligned_output_product>12:
                           aligned_output_product = max(16, aligned_output_product)
                           if aligned_output_product>16:
                               aligned_output_product = max(24, aligned_output_product)
                               if aligned_output_product>24:
                                   aligned_output_product = np.ceil(aligned_output_product/48)*48
            # 1 2 4 8 12 16 24 48 96 144 192 ...
            if aligned_output_product*batch<=480:
                aligned_min = 481*0.25 + 175
                reduce_time = min(aligned_min, (aligned_output_product*batch*1.5 + 175))
            else:
                reduce_time = (aligned_output_product*batch*0.25 + 175)

        shift_time = self.get_shift_time(sub_output_size)+BUFFER_ITERS*SYNC_CYCLES
        sync_time = SYNC_CYCLES*4
        return shift_time + sync_time + gather_time + reduce_time

    def get_shuffle_time_energy(self, temporal_dim_var_parts:List[List[int]]) -> Tuple[float, float, float]:
        per_comp_breakdown = {
            "noc": 0.0,
            "sram": 0.0,
            "sa": 0.0,
            "vu": 0.0,
        }
        if self.op_type in [self.OP_TYPE_ELEMENT, self.OP_TYPE_RELU, self.OP_TYPE_POOL]:
            per_comp_breakdown["noc"] = DATAMOVE_PJ
            return SYNC_CYCLES, DATAMOVE_PJ, per_comp_breakdown
        sub_output_size = self.get_sub_op_var_sizes(temporal_dim_var_parts)[0]
        num_output_replica = round(self.spatial_var_replicas[0] / np.prod(temporal_dim_var_parts,axis=0)[0])

        sub_op_shape = self.get_sub_op_shape(temporal_dim_var_parts)
        original_output_shape = sub_op_shape[self.variables[0][:,0]]
        original_output_product = int(np.prod(original_output_shape))
        reduced_output_product = original_output_product/num_output_replica

        if num_output_replica==1:
            gather_time = 0
            reduce_time = 0
            gather_energy = 0
            reduce_energy = 0
            reduce_energy_sram = 0
            reduce_energy_vu = 0
        elif num_output_replica==2:
            gather_time, gather_energy = self.get_shift_time_energy_no_static(original_output_product//2)
            gather_time += 2*SYNC_CYCLES
            gather_energy += 2*DATAMOVE_PJ
            if reduced_output_product <= 16432:
                aligned_min = 16434*0.5 + 300
                reduce_time = min(aligned_min, (reduced_output_product*0.75 + 290))
            else:
                reduce_time = (reduced_output_product*0.5 + 300)
            reduce_energy = reduced_output_product*(ADD_PJ+WRITE_PJ+2*READ_PJ)
            reduce_energy_vu = reduced_output_product*(ADD_PJ)
            reduce_energy_sram = reduced_output_product*(WRITE_PJ+2*READ_PJ)
        else:
            gather_time, gather_energy = self.get_shift_time_energy_no_static(original_output_product)
            gather_time += num_output_replica*SYNC_CYCLES
            gather_energy += num_output_replica*DATAMOVE_PJ
            batch = num_output_replica
            aligned_output_product = reduced_output_product
            if aligned_output_product>2:
               aligned_output_product = max(4, aligned_output_product)
               if aligned_output_product>4:
                   aligned_output_product = max(8, aligned_output_product)
                   if aligned_output_product>8:
                       aligned_output_product = max(12, aligned_output_product)
                       if aligned_output_product>12:
                           aligned_output_product = max(16, aligned_output_product)
                           if aligned_output_product>16:
                               aligned_output_product = max(24, aligned_output_product)
                               if aligned_output_product>24:
                                   aligned_output_product = np.ceil(aligned_output_product/48)*48
            # 1 2 4 8 12 16 24 48 96 144 192 ...
            if aligned_output_product*batch<=480:
                aligned_min = 481*0.25 + 175
                reduce_time = min(aligned_min, (aligned_output_product*batch*1.5 + 175))
            else:
                reduce_time = (aligned_output_product*batch*0.25 + 175)
            reduce_energy = aligned_output_product*batch*(READ_PJ+ADD_PJ) + aligned_output_product*WRITE_PJ
            reduce_energy_vu = aligned_output_product*batch*(ADD_PJ)
            reduce_energy_sram = aligned_output_product*batch*(READ_PJ) + aligned_output_product*WRITE_PJ
        shift_time, shift_energy = self.get_shift_time_energy_no_static(sub_output_size)
        shift_time += BUFFER_ITERS*SYNC_CYCLES
        shift_energy += BUFFER_ITERS*DATAMOVE_PJ
        sync_time = SYNC_CYCLES*4
        sync_energy = 2*DATAMOVE_PJ
        # NOC = gather + sync (1/2) + shift
        # core = reduce + shift_instr (1/2)
        time = shift_time + sync_time + gather_time + reduce_time
        per_comp_breakdown["noc"] = gather_energy + shift_energy + sync_energy * 0.5 # For now assume sync to be shared 50/50 between NOC and core
        per_comp_breakdown["sram"] = time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ + reduce_energy_sram + sync_energy * 0.5
        per_comp_breakdown["vu"] = reduce_energy_vu
        return time, shift_energy + sync_energy + gather_energy + reduce_energy + time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ + time * IDLE_PJ_PER_CORE_CYCLE, per_comp_breakdown
    # time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ # + time*IDLE_PJ_PER_CORE_CYCLE,

    # return compute time per iter
    # 0: reduce, 1:relu, 2:elementwise, 3:pooling, 4:conv, 5:matmul
    def get_comp_time_per_iter(self, temporal_dim_var_parts:List[List[int]]) -> float:
        sub_op_shape = self.get_sub_op_shape(temporal_dim_var_parts)
        return self.get_comp_time_per_iter_helper(sub_op_shape)

    def get_comp_time_energy_per_iter(self, temporal_dim_var_parts:List[List[int]]) -> Tuple[float, float]:
        sub_op_shape = self.get_sub_op_shape(temporal_dim_var_parts)
        # return self.get_comp_time_per_energy_iter_helper(sub_op_shape)
        return self.get_comp_time_energy_per_iter_helper(sub_op_shape)

    def get_comp_time_per_iter_helper(self, sub_op_shape) -> float:
        sub_op_product = np.prod(sub_op_shape)

        if self.op_type==self.OP_TYPE_REDUCE:
            output_shape = sub_op_shape[self.variables[0][:,0]]
            output_product = np.prod(output_shape)
            batch = sub_op_product / output_product
            aligned_output_product = output_product
            if aligned_output_product>2:
                aligned_output_product = max(4, aligned_output_product)
                if aligned_output_product>4:
                    aligned_output_product = max(8, aligned_output_product)
                    if aligned_output_product>8:
                        aligned_output_product = max(12, aligned_output_product)
                        if aligned_output_product>12:
                            aligned_output_product = max(16, aligned_output_product)
                            if aligned_output_product>16:
                                aligned_output_product = max(24, aligned_output_product)
                                if aligned_output_product>24:
                                    aligned_output_product = np.ceil(aligned_output_product/48)*48
            # 1 2 4 8 12 16 24 48 96 144 192 ...
            if aligned_output_product*batch<=480:
                aligned_min = 481*0.25 + 175
                return min(aligned_min, (aligned_output_product*batch*1.5 + 175))
            else:
                return (aligned_output_product*batch*0.25 + 175)

        elif self.op_type==self.OP_TYPE_RELU:
            return (sub_op_product*0.5 + 200)

        elif self.op_type==self.OP_TYPE_ELEMENT:
            if sub_op_product <= 16432:
                aligned_min = 16434*0.5 + 300
                return min(aligned_min, (sub_op_product*0.75 + 290))
            else:
                return (sub_op_product*0.5 + 300)

        elif self.op_type==self.OP_TYPE_POOL or self.op_type==self.OP_TYPE_CONV:
            pool_shape = sub_op_shape[-4:]
            kh = pool_shape[2]
            kw = pool_shape[3]
            h = pool_shape[0]
            w = pool_shape[1]

            if self.op_type==self.OP_TYPE_POOL:
                chl = 1
                if len(sub_op_shape)>4:
                    chl = np.prod(sub_op_shape[0:-4])
                    chl = np.ceil(chl/4)*4                          # type: ignore
                px = np.array([chl,h*w,kh*kw]).reshape((1,-1))
                poly, reg = self.pool_predictor.get_poly_reg()
                predict = reg.predict(poly.fit_transform(px))[0]    # type: ignore
                if predict<0:
                    return float("inf")
                return predict

            else:
                convB = 1
                convI = 1
                convO = 4
                if len(sub_op_shape)==5:
                    if len(self.variables[0])>len(self.variables[1]):
                        convO = sub_op_shape[0]
                    elif len(self.variables[0])<len(self.variables[1]):
                        convI = sub_op_shape[0]
                    else:
                        convB = sub_op_shape[0]
                elif len(sub_op_shape)==6:
                    if len(self.variables[0])>len(self.variables[1]):
                        convB = sub_op_shape[0]
                        convO = sub_op_shape[1]
                    elif len(self.variables[0])<len(self.variables[1]):
                        convB = sub_op_shape[0]
                        convI = sub_op_shape[1]
                    else:
                        out_idx = self.variables[0][0]
                        in_idx = 1-out_idx
                        convO = sub_op_shape[out_idx]
                        convI = sub_op_shape[in_idx]
                elif len(sub_op_shape)>6:
                    out_idx = self.variables[0][len(sub_op_shape)-6][0]
                    in_idx = 1-(out_idx-len(sub_op_shape)+6)+len(sub_op_shape)-6
                    convB = np.prod(sub_op_shape[0:-6])
                    convO = sub_op_shape[out_idx]
                    convI = sub_op_shape[in_idx]

                if convI>2:
                    convI = np.ceil(convI/4)*4
                elif convI>1 and convO<8:
                    convI = np.ceil(convI/4)*4
                convO = np.ceil(convO/4)*4

                # print("#### px:", [convB,convI,convO,h,w,kh,kw])
                px = np.array([convB,convI,convO,h*w,kh*kw]).reshape((1,-1))
                poly, reg = self.conv_predictor.get_poly_reg()
                predict = reg.predict(poly.fit_transform(px))[0]    # type: ignore
                if predict<0:
                    predict = float("inf")

                flops = convB * np.ceil(convI/8)*8 * np.ceil(convO/8)*8 * h*w * kh*kw
                predict = min(0.0495*flops+2030, predict)

                return predict

        elif self.op_type==self.OP_TYPE_MATMUL:
            out = self.variables[0].flatten()[-2:]
            inA = self.variables[1].flatten()[-2:]
            inB = self.variables[2].flatten()[-2:]
            kset = np.intersect1d(inA, inB)
            k_idx = np.setdiff1d(kset, out)[0]
            m_idx = np.setdiff1d(inA[-2:], [k_idx])[0]
            n_idx = np.setdiff1d(inB[-2:], [k_idx])[0]
            k = sub_op_shape[k_idx]
            m = sub_op_shape[m_idx]
            n = sub_op_shape[n_idx]
            if n>m:
                m,n = n,m
            if m<=16:
                m,n = n,m
            b = 1
            if len(sub_op_shape>3):
                b = np.prod(sub_op_shape[:-3])

            m_div_6 = np.ceil(m/6)
            k_div_16 = np.ceil(k/(32/self.num_byte_per_elem))
            n_div_16 = np.ceil(n/16)

            per_mm_16 = self.num_byte_per_elem*12*m_div_6*k_div_16*n_div_16 + \
                        514*k_div_16*n_div_16 + 113*n_div_16 + 173
            time_16 = b*per_mm_16
            return time_16

        elif self.op_type==self.OP_TYPE_GATHER:
            num_indices = np.prod(sub_op_shape[self.variables[2]])
            num_total = np.prod(sub_op_shape[:-1])
            new_total_half = np.ceil(num_total/num_indices/2)*num_indices
            return 300 + new_total_half + num_indices

        elif self.op_type==self.OP_TYPE_SLICE:
            output_shape = sub_op_shape[self.variables[0][:,0]]
            output_product = np.prod(output_shape)
            return self.get_shift_time(output_product) + SYNC_CYCLES

        else:
            raise Exception(f"Unsupported op type: {self.op_type}")

    def get_comp_time_energy_per_iter_helper(self, sub_op_shape) -> Tuple[float, float]:
        sub_op_product = np.prod(sub_op_shape)
        output_shape = sub_op_shape[self.variables[0][:,0]]
        output_product = np.prod(output_shape)
        per_component_breakdown ={
            "noc": 0,
            "sram": 0,
            "sa" : 0,
            "vu" : 0
        }
        if self.op_type==self.OP_TYPE_REDUCE:
            batch = sub_op_product / output_product
            aligned_output_product = output_product
            if aligned_output_product>2:
                aligned_output_product = max(4, aligned_output_product)
                if aligned_output_product>4:
                    aligned_output_product = max(8, aligned_output_product)
                    if aligned_output_product>8:
                        aligned_output_product = max(12, aligned_output_product)
                        if aligned_output_product>12:
                            aligned_output_product = max(16, aligned_output_product)
                            if aligned_output_product>16:
                                aligned_output_product = max(24, aligned_output_product)
                                if aligned_output_product>24:
                                    aligned_output_product = np.ceil(aligned_output_product/48)*48
            # 1 2 4 8 12 16 24 48 96 144 192 ...
            if aligned_output_product*batch<=480:
                aligned_min = 481*0.25 + 175
                time = min(aligned_min, (aligned_output_product*batch*1.5 + 175))
            else:
                time = (aligned_output_product*batch*0.25 + 175)

            per_component_breakdown["vu"] = aligned_output_product*batch*(ADD_PJ)
            per_component_breakdown["sram"] = time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + aligned_output_product*WRITE_PJ +  aligned_output_product*batch*(READ_PJ)
            return time, aligned_output_product*batch*(READ_PJ+ADD_PJ) + aligned_output_product*WRITE_PJ + time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + time*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_RELU:
            time = sub_op_product*0.5 + 200
            per_component_breakdown["sram"] = time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + sub_op_product*(READ_PJ+WRITE_PJ)
            per_component_breakdown["vu"] = sub_op_product*(ADD_PJ)
            return time, (sub_op_product*(READ_PJ+WRITE_PJ+ADD_PJ)) + time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + time*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_ELEMENT:
            if sub_op_product <= 16432:
                aligned_min = 16434*0.5 + 300
                time = min(aligned_min, (sub_op_product*0.75 + 290))
            else:
                time = (sub_op_product*0.5 + 300)
            per_component_breakdown["sram"] = time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + 2*sub_op_product*READ_PJ + sub_op_product*(WRITE_PJ)
            per_component_breakdown["vu"] = sub_op_product*(ADD_PJ)
            print(f"[{self.name}] op_type element", flush=True)
            return time, 2*sub_op_product*READ_PJ + sub_op_product*(WRITE_PJ+ADD_PJ) + time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + time*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_POOL or self.op_type==self.OP_TYPE_CONV:
            pool_shape = sub_op_shape[-4:]
            kh = pool_shape[2]
            kw = pool_shape[3]
            h = pool_shape[0]
            w = pool_shape[1]

            if self.op_type==self.OP_TYPE_POOL:
                chl = 1
                if len(sub_op_shape)>4:
                    chl = np.prod(sub_op_shape[0:-4])
                    chl = np.ceil(chl/4)*4                          # type: ignore
                px = np.array([chl,h*w,kh*kw]).reshape((1,-1))
                poly, reg = self.pool_predictor.get_poly_reg()
                predict = reg.predict(poly.fit_transform(px))[0]    # type: ignore
                if predict<0:
                    return float("inf"), float("inf"), per_component_breakdown
                per_component_breakdown["sram"] = predict*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + sub_op_product*(READ_PJ)+ output_product *WRITE_PJ
                per_component_breakdown["vu"] = sub_op_product*(ADD_PJ)
                return predict, sub_op_product*(READ_PJ+ADD_PJ) + output_product*WRITE_PJ + predict*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + predict*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

            else:
                convB = 1
                convI = 1
                convO = 4
                if len(sub_op_shape)==5:
                    if len(self.variables[0])>len(self.variables[1]):
                        convO = sub_op_shape[0]
                    elif len(self.variables[0])<len(self.variables[1]):
                        convI = sub_op_shape[0]
                    else:
                        convB = sub_op_shape[0]
                elif len(sub_op_shape)==6:
                    if len(self.variables[0])>len(self.variables[1]):
                        convB = sub_op_shape[0]
                        convO = sub_op_shape[1]
                    elif len(self.variables[0])<len(self.variables[1]):
                        convB = sub_op_shape[0]
                        convI = sub_op_shape[1]
                    else:
                        out_idx = self.variables[0][0]
                        in_idx = 1-out_idx
                        convO = sub_op_shape[out_idx]
                        convI = sub_op_shape[in_idx]
                elif len(sub_op_shape)>6:
                    out_idx = self.variables[0][len(sub_op_shape)-6][0]
                    in_idx = 1-(out_idx-len(sub_op_shape)+6)+len(sub_op_shape)-6
                    convB = np.prod(sub_op_shape[0:-6])
                    convO = sub_op_shape[out_idx]
                    convI = sub_op_shape[in_idx]

                if convI>2:
                    convI = np.ceil(convI/4)*4
                elif convI>1 and convO<8:
                    convI = np.ceil(convI/4)*4
                convO = np.ceil(convO/4)*4

                # print("#### px:", [convB,convI,convO,h,w,kh,kw])
                # px = np.array([convB,convI,convO,h*w,kh*kw]).reshape((1,-1))
                # poly, reg = self.conv_predictor.get_poly_reg()
                # predict = reg.predict(poly.fit_transform(px))[0]    # type: ignore
                # if predict<0:
                #     predict = float("inf")

                flops = convB * np.ceil(convI/8)*8 * np.ceil(convO/8)*8 * h*w * kh*kw
                # predict = min(0.0495*flops+2030, predict)
                predict = 0.0495*flops+2030

                convO_16 = np.ceil(convO/16)*16
                kwkh_16 = np.ceil(kw*kh/16)*16
                add_mul = h*w*convB*convI*convO_16*kwkh_16*(MUL_PJ+ADD_PJ)
                read_input = convO_16/16*kwkh_16*convB*convI*h*w*READ_PJ
                read_kernel = convO_16*kh*kw*convI*READ_PJ
                read_output = kwkh_16/16*convO_16*h*w*convB*(convI-1)*READ_PJ
                read = read_input + read_kernel + read_output
                write = kwkh_16/16*convO_16*h*w*convB*convI*WRITE_PJ
                per_component_breakdown["sram"] = predict*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + read + write
                per_component_breakdown["sa"] = add_mul
            return predict, add_mul + read + write + predict*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + predict*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_MATMUL:
            out = self.variables[0].flatten()[-2:]
            inA = self.variables[1].flatten()[-2:]
            inB = self.variables[2].flatten()[-2:]
            kset = np.intersect1d(inA, inB)
            k_idx = np.setdiff1d(kset, out)[0]
            m_idx = np.setdiff1d(inA[-2:], [k_idx])[0]
            n_idx = np.setdiff1d(inB[-2:], [k_idx])[0]
            k = sub_op_shape[k_idx]
            m = sub_op_shape[m_idx]
            n = sub_op_shape[n_idx]
            if n>m:
                m,n = n,m
            if m<=16:
                m,n = n,m
            b = 1
            if len(sub_op_shape>3):
                b = np.prod(sub_op_shape[:-3])

            m_div_6 = np.ceil(m/6)
            k_div_16 = np.ceil(k/(32/self.num_byte_per_elem))
            n_div_16 = np.ceil(n/16)

            per_mm_16 = self.num_byte_per_elem*12*m_div_6*k_div_16*n_div_16 + \
                        514*k_div_16*n_div_16 + 113*n_div_16 + 173
            time_16 = b*per_mm_16
            m_6 = m
            if m_div_6>1:
                m_6 = m_div_6*6
            k_16 = k_div_16*16
            n_16 = n_div_16*16
            if np.ceil(n/8)==1:
                n_16 = 8

            add_mul_16 = m_6 * k_16 * n_16 * b * (MUL_PJ+ADD_PJ)
            read_16 = (m_6*k_16*n_div_16 + n_16*k_16) * b * READ_PJ
            write_16 = m_6 * n_16 * k_div_16 * b * WRITE_PJ
            energy_16 = add_mul_16 + read_16 + write_16
            per_component_breakdown["sa"] = add_mul_16
            per_component_breakdown["sram"] = read_16 + write_16 + time_16*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ
            return time_16, energy_16 + time_16*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + time_16*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_GATHER:
            num_indices = np.prod(sub_op_shape[self.variables[2]])
            num_total = np.prod(sub_op_shape[:-1])
            new_total_half = np.ceil(num_total/num_indices/2)*num_indices
            time = 300+new_total_half+num_indices
            per_component_breakdown["sram"] = time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + output_product*(READ_PJ+WRITE_PJ)
            per_component_breakdown["vu"] = output_product*ADD_PJ
            return time, output_product*(READ_PJ+WRITE_PJ+ADD_PJ) + time*COMP_INSTR_BYTE_PER_CYCLE*READ_PJ + time*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown

        elif self.op_type==self.OP_TYPE_SLICE:
            time, energy_no_static = self.get_shift_time_energy_no_static(output_product)
            per_component_breakdown["noc"] = energy_no_static
            per_component_breakdown["sram"] = time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ
            return time, energy_no_static + time*SHIFT_INSTR_BYTE_PER_CYCLE*READ_PJ + time*IDLE_PJ_PER_CORE_CYCLE, per_component_breakdown
        else:
            raise Exception(f"Unsupported op type: {self.op_type}")

    # return num of bytes per core (not including temp vars or buffers)
    def get_byte_per_core_idle(self, temporal_dim_var_parts:List[List[int]],
                               spatial_dim_parts:List[int]=[],
                               # ignore_var:List[bool]=[],
                               config_id:int=-1) -> Tuple[int, int]:
        var_sizes = np.array(self.get_sub_op_var_sizes(temporal_dim_var_parts, spatial_dim_parts))
        # ignore_variables = np.array(ignore_var)
        # if len(ignore_var)==0:
            # ignore_variables = self.ignore_variables
        total_size = int(np.sum(var_sizes[np.invert(self.ignore_variables)])*self.num_byte_per_elem)
        return config_id, total_size

    # return num of bytes per core (not including buffers)
    def get_byte_per_core_no_buffer(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> int:
        var_sizes = self.get_sub_op_var_sizes(temporal_dim_var_parts, spatial_dim_parts)
        return np.sum(var_sizes)*self.num_byte_per_elem

    # return num of bytes per core (including buffers)
    def get_byte_per_core_with_buffer(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> int:
        var_sizes = self.get_sub_op_var_sizes(temporal_dim_var_parts, spatial_dim_parts)
        for var_idx in range(len(var_sizes)):
            if np.sum(np.array(temporal_dim_var_parts)[:,var_idx]>1):
                var_sizes[var_idx] += int(var_sizes[var_idx]/BUFFER_SIZE_RATIO)
        return np.sum(var_sizes)*self.num_byte_per_elem

    def get_hot_cold_bytes_per_core(self, temporal_dim_var_parts:List[List[int]], spatial_dim_parts:List[int]=[]) -> Tuple[int, int]:
        var_sizes = np.array(self.get_sub_op_var_sizes(temporal_dim_var_parts, spatial_dim_parts))
        cold_size = int(np.sum(var_sizes[np.invert(self.ignore_variables)])*self.num_byte_per_elem)
        for var_idx in range(len(var_sizes)):
            if np.sum(np.array(temporal_dim_var_parts)[:,var_idx]>1):
                var_sizes[var_idx] += int(var_sizes[var_idx]/BUFFER_SIZE_RATIO)
        hot_size = np.sum(var_sizes)*self.num_byte_per_elem
        return int(hot_size), int(cold_size)

    def evaluate_config(self, config: Tuple[List[int], List[List[int]]], config_id: int = -1) -> Tuple[int, perf]:
        spatial, temporal = config
        mem_size_hot, mem_size_cold = self.get_hot_cold_bytes_per_core(temporal, spatial)

        # Early termination: skip expensive cycle calculations if config exceeds memory limit
        if mem_size_hot > self.max_byte_per_core:
            config_perf = perf(mem_size_hot, mem_size_cold, 999999999, 999999999, 0)
            return config_id, config_perf

        tensor_sizes = self.get_sub_op_var_sizes(temporal, spatial)
        temporal_var_replicas = self.get_temporal_var_replicas(temporal, spatial)
        shift_info = self.get_shift_info(temporal, output_special_format_for_tsim=True)
        broadcast_cycles, shift_cycles, reduce_cycles = \
            self.noc.get_total_cycles_from_expression(tensor_sizes,
                                                      temporal_var_replicas,
                                                      self.spatial_var_replicas,
                                                      shift_info,
                                                      self.num_byte_per_elem,
                                                      spmd_compiler=self.spmd_compiler,
                                                      seq_noc=self.seq_noc,
                                                      )

        op = self.comp.convert_op_simple(dim_len=self.get_sub_op_shape(temporal, spatial),
                                         variables=self.variables,
                                         is_ew=self._is_ew,
                                         ignore_variables=self.ignore_variables,
                                         tensor_id_list=self._tensor_id_range,
                                        )
        comp_cycles,_ ,_ ,_ ,_ = self.comp.get_total_cycle_for_fused_op([op], [temporal])

        if self.ipu:
            comp_shift = comp_cycles+shift_cycles
            broadcast_reduce = reduce_cycles
        else:
            comp_shift = max(comp_cycles, shift_cycles)
            broadcast_reduce = broadcast_cycles + reduce_cycles
        exe_time = comp_shift + broadcast_reduce
        assert exe_time > 0, f"exe_time = {exe_time}, comp_time = {comp_shift}, sync_shift_time = {broadcast_reduce}, shuffle_time = {reduce_cycles}"
        config_perf = perf(mem_size_hot, mem_size_cold, exe_time, comp_shift, broadcast_reduce)
        return config_id, config_perf

    # return (config id, (hot_mem_size, hot_exe_time, cold_mem_size, comp_cycles, shift_cycles))
    # @config = [spatial, temporal]
    # @config_id: ignore it; this is for parallelel execution
    def evaluate_energy(self, config: Tuple[List[int], List[List[int]]], config_id: int = -1) -> Tuple[int, perf]:
        # hot -> with buffer, cold -> no buffer
        mem_size_hot = int(self.get_byte_per_core_with_buffer(config[1], config[0]))
        mem_size_cold = int(self.get_byte_per_core_no_buffer(config[1], config[0]))
        shift_info = self.get_shift_info(config[1])
        num_cores:float = np.prod(config[0]).astype(float)

        _sync_shift_time, _sync_shift_energy, _ss_per_comp_energy = self.get_sync_shift_time_energy(shift_info)
        # _sync_shift_time, _sync_shift_energy, _sync_shift_instr_energy = self.get_sync_shift_time_energy(shift_info)

        _comp_time, _comp_energy, _comp_per_comp_energy = self.get_comp_time_energy_per_iter(config[1])
        iter = get_num_comp_iter(config[1])
        _comp_time *= iter
        _comp_energy *= iter
        for comp_key in _comp_per_comp_energy:
            _comp_per_comp_energy[comp_key] *= iter
        _shuffle_time, _shuffle_energy, _shuffle_per_comp_energy = self.get_shuffle_time_energy(config[1])

        assert math.isclose(_shuffle_energy, sum(_shuffle_per_comp_energy.values()), rel_tol=1e-4), f"_shuffle_energy={_shuffle_energy}, sum of breakdown={sum(_shuffle_per_comp_energy.values())}, {self.op_type}"
        assert math.isclose(_sync_shift_energy, sum(_ss_per_comp_energy.values()), rel_tol=1e-4), f"_sync_shift_energy={_sync_shift_energy}, sum of breakdown={sum(_ss_per_comp_energy.values())}, {self.op_type}"
        assert math.isclose(_comp_energy, sum(_comp_per_comp_energy.values()), rel_tol=1e-4),  f"_comp_energy={_comp_energy}, sum of breakdown={sum(_comp_per_comp_energy.values())}, {self.op_type}"
        _comp_energy *= num_cores
        _sync_shift_energy *= num_cores # total
        _shuffle_energy *= num_cores # total
        # _shuffle_instr_energy *= num_cores #instr only
        # _sync_shift_instr_energy *= num_cores #instr only
        tot_energy_breakdown: Dict[str, float] = {}
        assert (len(list(_shuffle_per_comp_energy.keys())) == len(list(_ss_per_comp_energy.keys())) and\
             len(list(_shuffle_per_comp_energy.keys())) == len(list(_comp_per_comp_energy.keys()))), \
                  f"components mismatch. {_shuffle_per_comp_energy.keys()} {_ss_per_comp_energy.keys()} {_comp_per_comp_energy.keys()}"
        for comp in _ss_per_comp_energy: # each ss/shuffle/comp breakdown should have the same components.
            tot_energy_breakdown[comp] = _ss_per_comp_energy[comp] + _comp_per_comp_energy[comp] + _shuffle_per_comp_energy[comp]
            tot_energy_breakdown[comp] *= num_cores
        # if np.prod(self.num_cores) > IPU2_NUM_CORES:
        #     raise NotImplementedError("Impossible to reach here")
        #     _sync_shift_time *= INTER_CHIP_OVERHEAD
        #     _shuffle_time *= INTER_CHIP_OVERHEAD

        _exe_time = _comp_time + _sync_shift_time + _shuffle_time
        _exe_energy = _comp_energy + _sync_shift_energy + _shuffle_energy

        _sss_time = _sync_shift_time + _shuffle_time
        _sss_energy = _sync_shift_energy + _shuffle_energy
        assert math.isclose(_exe_energy, sum(tot_energy_breakdown.values()), rel_tol=1e-4), f"total energy mismatch."
        # _ssi_energy = _sync_shift_instr_energy + _shuffle_instr_energy
        exe_rst = create_fused_metric(_exe_time, _exe_energy)
        comp_rst = create_fused_metric(_comp_time, _comp_energy)

        sss_rst = create_fused_metric(_sss_time, _sss_energy)

        # print(f"[{self.name}] Energy (EE): {_exe_energy} {sum(tot_energy_breakdown.values())}", flush=True)

        # ssi_rst = create_fused_metric(_sss_time, _ssi_energy)
        config_perf = perf(mem_size_hot, mem_size_cold, exe_rst, comp_rst, sss_rst)
        return config_id, config_perf, tot_energy_breakdown

    # return num of cycles to warm up
    # @x_temp = temp_partition [[]]
    # @spatial = spatial partition []
    def get_warm_up_time(self, cold_temp:List[List[int]], hot_temp:List[List[int]], spatial:List[int]=[]) -> float:

        if np.sum(np.mod(cold_temp, hot_temp)):     # type: ignore
            return -1       # invalid pair, not factor and multiple

        cold_var_sizes = self.get_sub_op_var_sizes(cold_temp, spatial)
        multiplier = np.divide(cold_temp, hot_temp) # type: ignore
        multiplier = np.prod(multiplier, axis=0)
        multiplier[multiplier>1] += 1
        multiplier -= 1

        total_transfer_elem = np.sum((multiplier*cold_var_sizes)[1:])
        total_transfer_time = self.get_shift_time(total_transfer_elem)
        total_sync_time = np.sum(multiplier)*SYNC_CYCLES

        return total_transfer_time+total_sync_time



################################################################



    def get_trivial_temporal_partition(self) -> List[List[int]]:
        '''return a trivial temporal partition plan (all 1s, i.e. no temporal partitioning at all)'''
        return [[1] * len(self.variables)] * len(self.dim_lengths[:-1])

    def get_trivial_spatial_partition(self) -> List[int]:
        '''return a trivial spatial partition plan (all 1s, i.e. no spatial partitioning at all)'''
        return [1] * len(self.dim_lengths[:-1])

    def is_temp_dim_valid_for_variables(self, dim_idx: int, config: List[int]) -> bool:
        for var, dim in zip(self.variables, config):
            if dim > 1:
                if dim_idx not in var:
                    return False
                if max(config) > self.spatial_dim_lengths[dim_idx]:
                    return False
                if self.spatial_dim_lengths[dim_idx] / \
                    (max(config) * np.ceil(self.spatial_dim_lengths[dim_idx] / max(config))) < \
                        self.get_pad_threshold():
                    return False
        return True

    # return list of all possible temporal configs for a given spatial config
    # temporal_dim_var_parts [for each dim [num temporal parts per variable]]
    def get_all_temporal_configs(self, spatial_config: List[int]) -> List[List[List[int]]]:
        ### 1. check if spatial config is valid; return quickly if not
        if self.update_spatial_dim_parts_if_valid(spatial_config) == False:
            return []

        ### 2. find all factors of the number of replicas of each variable
        # factors are sorted in ascending order
        # e.g., replica_factors[i] = [1, 2, 3, 5, 6, 10, 15, 30] if num_replicas[i] = 30
        replica_factors: List[List[int]] = [
            get_factors(i) for i in self.spatial_var_replicas
        ]

        ### 3. generate temp config search space and prune search space
        ## 3.1. generate all possible [num temporal parts per variable] for each dimension
        # e.g., dim_temp_configs = [[1, 1, 1], [1, 2, 1], [2, 4, 1], ...]
        dim_temp_configs: List[List[int]] = list(itertools.product(*replica_factors)) # type: ignore

        ## 3.2. filter out obviously invalid configs (e.g., [1, 2, 3] where 2 does not divide 3)
        dim_temp_configs = list(filter(is_chain_divisible, dim_temp_configs))

        ## 3.3. filter out configs that are not valid for the variable under consideration for each dimension,
        #       and filter out configs that have temporal parts larger than spatial dims after spatial partitioning
        # Generate temp_config_search_space, which is the per-dim temp config search space
        # e.g. temp_config_search_space = [
        #   [[1, 1, 1], [1, 2, 1], [2, 4, 1]],             # dim 0 valid configs
        #   [[1, 1, 1], [1, 2, 4], [2, 1, 1], [4, 2, 1]],  # dim 1 valid configs
        #   ...
        # ]
        temp_config_search_space: List[List[List[int]]] = [[] for _ in self.dim_lengths[:-1]]
        for dim_idx, dim_space in enumerate(temp_config_search_space):
            # @config: [v1, v2, v3]
            this_dim_temp_configs = list(filter(functools.partial(self.is_temp_dim_valid_for_variables, dim_idx), dim_temp_configs))
            dim_space += this_dim_temp_configs

        ### 4. generate all possible temporal configs
        temp_search_tree = OpTemporalPartitionSearch(
            depth = len(self.dim_lengths) - 1,
            search_space = temp_config_search_space,
            num_replicas = self.spatial_var_replicas
        )
        temp_search_tree.generateSearchTree()

        valid_temp_configs: List[List[List[int]]] = temp_search_tree.get_all_configs(
            lambda node: len(self.valid_temporal_dim_var_parts(node.getConfig(), spatial_config)) > 0
        )

        return valid_temp_configs

    def get_all_spatial_configs(self, spatial_search_tree: Optional[OpSpatialPartitionSearch] = None) -> List[List[int]]:
        if spatial_search_tree is None:
            num_core = int(np.prod(self.num_cores))
            tot_dim_size = [min(num_core, dim_size) for dim_size in self.dim_lengths[:-1]]
            _, spatial_search_tree = build_spatial_search_tree(
                depth = len(self.dim_lengths) - 1,
                tot_dim_size = tot_dim_size,
                dim_size_TH = self.get_util_threshold(),
                num_core = num_core,
            )
        search_tree = spatial_search_tree

        all_spatial_configs = search_tree.get_all_configs()

        # Light ops: no filtering needed
        if self.is_light_op():
            return all_spatial_configs

        # Heavy ops: per-dimension tile-size deduplication.
        # For each dimension, a larger spatial factor that produces the same tile
        # length as a smaller one (due to ceiling) wastes cores on padding.
        # Keep only the smallest factor per unique tile size per dimension.
        num_dims = len(self.dim_lengths) - 1
        num_core = int(np.prod(self.num_cores))
        valid_factors_per_dim = []
        for d in range(num_dims):
            dim_len = int(self.dim_lengths[d])
            max_factor = min(num_core, dim_len)
            seen_tiles = {}  # tile_size -> smallest factor that achieves it
            for f in range(1, max_factor + 1):
                tile = -(-dim_len // f)  # ceil(dim_len / f)
                if tile not in seen_tiles:
                    seen_tiles[tile] = f
            valid_factors_per_dim.append(set(seen_tiles.values()))

        return [config for config in all_spatial_configs
                if all(config[d] in valid_factors_per_dim[d] for d in range(num_dims))]

    def dump_config_dict(self, log_filename: str):
        if log_filename == "":
            log_filename = self.name
        if DUMP:
            with open(f"{self.name}/all_configs_{log_filename}.json", "w") as f:
                import ujson as json
                json.dump(self.config_dict, f, indent=4)

    def dump_cold_config_candidates(self, log_filename: str):
        if log_filename == "":
            log_filename = self.name
        if DUMP:
            with open(f"{self.name}/cold_config_candidates_{log_filename}.json", "w") as f:
                import ujson as json
                json.dump(self.cold_config_candidates, f, indent=4)

    def dump_cold_hot_table(self, log_filename):
        if log_filename == "":
            log_filename = self.name
        if DUMP:
            with open(f"{self.name}/cold_hot_table_{log_filename}.json", "w") as f:
                import ujson as json
                json.dump(self.cold_hot_table, f, indent=4)

    def dump_all_configs_dict(self):
        # with open(f"{self.name}/all_configs_dict.json", "w") as f:
        #     import ujson as json
        #     json.dump(all_configs_dict, f, indent=4)
        pass

    def dump_cold_config_candidates_dict(self):
        # with open(f"{self.name}/cold_config_candidates_dict.json", "w") as f:
        #     import ujson as json
        #     json.dump(cold_config_candidates_dict, f, indent=4)
        pass

    def dump_cold_hot_table_dict(self):
        # with open(f"{self.name}/cold_hot_table_dict.json", "w") as f:
        #     import ujson as json
        #     json.dump(cold_hot_table_dict, f, indent=4)
        pass

    def evaluate_config_helper(self, params):
        return self.evaluate_config(*params)

    def _estimate_temporal_workload(self, spatial_config) -> int:
        """Estimate the number of temporal configs for a given spatial config.

        Builds the per-dimension search space (same as get_all_temporal_configs
        steps 1-3), then counts leaves of the temporal search tree *with*
        pruning (filter_by_dim_size) but without the expensive per-leaf
        valid_temporal_dim_var_parts check.  Results are cached by
        (spatial_var_replicas, spatial_dim_lengths) since many spatial configs
        share the same derived state.
        Returns 0 for invalid spatial configs.
        """
        if not self.update_spatial_dim_parts_if_valid(spatial_config):
            return 0
        depth = len(self.dim_lengths) - 1
        if depth == 0:
            return 1

        # Cache key: replicas + dim_lengths fully determine the search space.
        cache_key = (tuple(int(r) for r in self.spatial_var_replicas),
                     tuple(int(d) for d in self.spatial_dim_lengths[:depth]))
        if not hasattr(self, '_est_cache'):
            self._est_cache = {}
        if cache_key in self._est_cache:
            return self._est_cache[cache_key]

        # Build per-dimension search space (cheap, mirrors get_all_temporal_configs).
        replica_factors = [get_factors(int(r)) for r in self.spatial_var_replicas]
        dim_temp_configs = list(filter(is_chain_divisible,
                                       itertools.product(*replica_factors)))
        search_space = []
        for dim_idx in range(depth):
            search_space.append(list(filter(
                functools.partial(self.is_temp_dim_valid_for_variables, dim_idx),
                dim_temp_configs)))

        num_replicas = [int(r) for r in self.spatial_var_replicas]
        num_vars = len(num_replicas)

        # Convert search_space candidates to tuples for hashing.
        ss_tuples = [tuple(tuple(c) for c in level) for level in search_space]

        # Memoized recursive leaf count with pruning (no Node allocation).
        # Key: (level, parent_agg_tuple) — many paths share the same aggregate.
        from functools import lru_cache
        nr_tuple = tuple(num_replicas)

        @lru_cache(maxsize=None)
        def _count(level, parent_agg):
            if level >= depth:
                return 1
            total = 0
            for cand in ss_tuples[level]:
                if parent_agg is not None:
                    new_agg = tuple(c * pa for c, pa in zip(cand, parent_agg))
                    if any(na > nr for na, nr in zip(new_agg, nr_tuple)):
                        continue
                else:
                    new_agg = cand
                total += _count(level + 1, new_agg)
            return total

        result = _count(0, None)
        self._est_cache[cache_key] = result
        return result

    def precompute_estimates(self, spatial_search_tree: Optional[OpSpatialPartitionSearch] = None) -> None:
        """Pre-compute workload estimates for all spatial configs.

        Stores (estimate, spatial_config) pairs in self._precomputed_estimates
        so that search_optimal_config can skip the expensive estimate loop.
        Intended to be called concurrently during Phase 0+1 (while light ops
        and spatial trees are being built).
        """
        all_spatial_configs = self.get_all_spatial_configs(spatial_search_tree)
        self._precomputed_estimates = [
            (self._estimate_temporal_workload(sc), sc)
            for sc in all_spatial_configs
        ]

    def _get_temporal_configs_chunk(self, spatial_configs_list) -> list:
        """Worker: generate temporal configs for a list of spatial configs.

        Takes a list so we can group lightweight spatials into one task,
        avoiding per-task IPC overhead for tiny workloads.
        Returns [(spatial_tuple, temporal_configs_list), ...].
        """
        return [(tuple(sc), self.get_all_temporal_configs(sc)) for sc in spatial_configs_list]

    # return: best configs of this op with the corresponding scores
    # side effect: generates cold config candidates for this op (self.cold_config_candidates, sorted by mem_size)
    def search_optimal_config(self, num_threads: int = 1, spatial_search_tree: Optional[OpSpatialPartitionSearch] = None, log_filename: str = "") -> Dict[Tuple[Tuple, Tuple[Tuple]], Tuple[int, Any]]:
        _t0 = time.time()
        if log_filename == "":
            log_filename = self.name
        if len(self.config_dict) > 0:
            # dump all config_dict and cold config candidates to file
            all_configs_dict[log_filename] = self.log_filename_physical
            if DUMP_ALL or DUMP_ROLLER:
                self.dump_config_dict(log_filename)

            self.dump_all_configs_dict()
            return self.config_dict

        ##### 1. generate spatial partition search tree
        _t1 = time.time()
        is_light = self.is_light_op()
        all_spatial_configs = self.get_all_spatial_configs(spatial_search_tree)

        spatial_to_cold_to_temporal:Dict[Any,Dict] = {}
        for spatial_config in all_spatial_configs:
            spatial_to_cold_to_temporal[tuple(spatial_config)] = {}

        # debug: dump spatial configs to file
        # with open(f"{log_filename}.spatial_configs.json", "w") as f:
        #     import ujson as json
        #     json.dump(all_spatial_configs, f, indent=4)

        ##### 2-4. Streaming batch evaluation: generate temporal configs, evaluate, and update Pareto frontier
        # NOTE: self.config_dict is NOT written inside the loop so workers always pickle a clean self.
        # Results accumulate in temp_config_dict and are assigned to self at the end.
        _t2 = time.time()

        # Use precomputed estimates if available (from Phase 0+1), else compute now.
        if hasattr(self, '_precomputed_estimates') and self._precomputed_estimates is not None:
            all_estimates = self._precomputed_estimates
            self._precomputed_estimates = None  # free memory
        else:
            all_estimates = [(self._estimate_temporal_workload(sc), sc) for sc in all_spatial_configs]
        total_est = sum(e for e, _ in all_estimates) or 1

        # Determine num_batches from total workload budget (num temporal configs per batch).
        # total_est approximates total temporal configs, so this keeps RAM per batch bounded
        # regardless of how many spatials there are.
        TARGET_BATCH_WORKLOAD = 3e5
        num_batches = max(1, int(250/get_mem_available_gb() * (total_est/TARGET_BATCH_WORKLOAD)**1.1))

        if not is_light:
            print(f"[{self.name}] {len(all_spatial_configs)} spatial configs, depth={len(self.dim_lengths)-1}, cores={np.prod(self.num_cores)}, est={total_est}, batches={num_batches}", flush=True)

        # Build workload-balanced batches: sort heavy-first, greedy fill to target.
        # Each batch element is (estimate, spatial_config) so inner chunking reuses estimates.
        target_per_batch = total_est / num_batches
        batches: list = []          # list of [(est, sc), ...]
        cur_batch, cur_est = [], 0.0
        for est, sc in sorted(all_estimates, key=lambda x: -x[0]):
            cur_batch.append((est, sc))
            cur_est += est
            if cur_est >= target_per_batch and len(batches) < num_batches - 1:
                batches.append(cur_batch)
                cur_batch, cur_est = [], 0.0
        if cur_batch:
            batches.append(cur_batch)

        mem_size_to_exe_time = {}
        temp_config_dict = {}   # accumulate here; never touch self.config_dict inside the loop
        total_configs_evaluated = 0
        use_overlap_metric = ICBM_OVERLAP and self.max_byte_per_core < 600000

        for batch_idx, batch_est_sc in enumerate(batches):
            spatial_batch = [sc for _, sc in batch_est_sc]

            # Stage 2: Generate temporal configs
            _t_gen = time.time()
            if num_threads == 1:
                batch_configs = [
                    (tuple(spatial_config), tuple(temporal_config))
                    for spatial_config in spatial_batch
                    for temporal_config in self.get_all_temporal_configs(spatial_config)
                ]
            else:
                # --- Separate mega vs normal spatials ---
                # A spatial is "mega" if its estimate exceeds the fair share
                # per thread, meaning it would dominate one worker.
                batch_total_est = sum(e for e, _ in batch_est_sc) or 1
                fair_share = max(1, batch_total_est / num_threads)
                MEGA_MULT = 1.5  # threshold multiplier
                mega_items = []
                normal_items = []
                for est, sc in batch_est_sc:  # already sorted heavy-first
                    if est >= fair_share * MEGA_MULT:
                        mega_items.append((est, sc))
                    else:
                        normal_items.append((est, sc))

                work_items = []

                # For mega spatials: compute search space in parent,
                # split dim-0 candidates across workers.
                depth = len(self.dim_lengths) - 1
                for est, sc in mega_items:
                    if not self.update_spatial_dim_parts_if_valid(sc):
                        continue
                    replica_factors = [get_factors(int(r))
                                       for r in self.spatial_var_replicas]
                    dim_temp_configs = list(filter(
                        is_chain_divisible,
                        itertools.product(*replica_factors)))
                    search_space = []
                    for dim_idx in range(depth):
                        search_space.append(list(filter(
                            functools.partial(
                                self.is_temp_dim_valid_for_variables, dim_idx),
                            dim_temp_configs)))

                    dim0_full = search_space[0]
                    num_replicas = [int(r) for r in self.spatial_var_replicas]
                    # Number of splits proportional to workload.
                    n_splits = max(1, min(num_threads, len(dim0_full),
                                         int(math.ceil(est / fair_share))))
                    if n_splits <= 1:
                        # Not worth splitting; treat as normal.
                        normal_items.append((est, sc))
                        continue
                    chunk_size = max(1, -(-len(dim0_full) // n_splits))
                    for i in range(0, len(dim0_full), chunk_size):
                        chunk_ss = [dim0_full[i:i + chunk_size]] + search_space[1:]
                        work_items.append(('split', self, list(sc),
                                           chunk_ss, num_replicas, depth))

                n_mega_splits = len(work_items)

                # For normal spatials: LPT bucketing into chunks.
                if normal_items:
                    n_buckets = min(num_threads, len(normal_items))
                    buckets = [[] for _ in range(n_buckets)]
                    bucket_ests = [0] * n_buckets
                    for est, sc in sorted(normal_items, key=lambda x: -x[0]):
                        b = min(range(n_buckets), key=lambda j: bucket_ests[j])
                        buckets[b].append(sc)
                        bucket_ests[b] += est
                    for bucket in buckets:
                        if bucket:
                            work_items.append(('bulk', self, bucket))

                if n_mega_splits > 0:
                    print(f"  [{self.name}] {len(mega_items)} mega spatial(s) "
                          f"→ {n_mega_splits} dim-0 chunks + "
                          f"{len(work_items) - n_mega_splits} normal chunks",
                          flush=True)

                with Pool(num_threads) as pool:
                    all_results = pool.map(_gen_temporal_work, work_items)
                del work_items
                batch_configs = [
                    (spatial_key, tuple(temporal))
                    for result_list in all_results
                    for spatial_key, temporals in result_list
                    for temporal in temporals
                ]
                del all_results
            _t_gen = time.time() - _t_gen

            # Stage 3: Evaluate
            _t_eval = time.time()
            if num_threads == 1:
                batch_scores = [self.evaluate_config(config, i) for i, config in enumerate(batch_configs)]
            else:
                with Pool(num_threads) as pool:
                    params = [(config, i) for i, config in enumerate(batch_configs)]
                    batch_scores = pool.map(self.evaluate_config_helper, params)
                del params  # batch_scores owns the results
            _t_eval = time.time() - _t_eval

            # Stage 4: Update local frontier (never write to self here)
            for config, (_, score) in zip(batch_configs, batch_scores):
                if score.hot_mem_size > self.max_byte_per_core:
                    continue
                hot_mem_size = score.hot_mem_size
                spatial_to_cold_to_temporal[config[0]][score.cold_mem_size] = config[1]
                exe_time = score.total_cycles
                comp_cycles, shift_cycles = score.comp_cycles, score.sss_cycles
                metric = max(comp_cycles, shift_cycles) if use_overlap_metric else exe_time
                current_best = mem_size_to_exe_time.get(hot_mem_size)
                if current_best is None or metric < current_best:
                    mem_size_to_exe_time[hot_mem_size] = metric
                    temp_config_dict[config] = (hot_mem_size, exe_time, comp_cycles, shift_cycles)
            total_configs_evaluated += len(batch_configs)
            del batch_configs, batch_scores

            if not is_light:
                print(f"[{self.name}] batch [{batch_idx+1}/{num_batches}] "
                      f"{total_configs_evaluated} cfg | "
                      f"gen={_t_gen:.2f}s eval={_t_eval:.2f}s "
                      f"→ {len(temp_config_dict)} hot", flush=True)

        # Assign accumulated results to self once, after all batches
        self.config_dict = temp_config_dict

        total_time = time.time() - _t2
        if not is_light:
            print(f"[{self.name}] [TIME] Eval: {total_time:.2f}s, {total_configs_evaluated} configs → {len(self.config_dict)} hot", flush=True)

        ##### 5. filter out configs with the same mem size but higher execution time (Pareto refinement)
        _t5 = time.time()
        config_sorted = sorted(self.config_dict.items(), key=lambda item: item[1][0])
        if len(config_sorted) == 0:
            print(f"[{self.name}] no hot configs. Exit now.", flush=True)
            raise(ValueError(f"# op {self.name}: no hot configs (SRAM too small).\n {self.dim_lengths}"))
        self.config_dict = {}
        self.config_dict[config_sorted[0][0]] = config_sorted[0][1]
        last_inserted_config = config_sorted[0]

        if ICBM_OVERLAP and self.max_byte_per_core < 600000:
            for config, (mem_size, exe_time, comp_cycles, shift_cycles) in config_sorted[1:]:
                last_mem_size, _, last_comp_cycles, last_shift_cycles = last_inserted_config[1]
                overlap_cycles = max(comp_cycles, shift_cycles)
                last_overlap_cycles = max(last_comp_cycles, last_shift_cycles)
                assert mem_size >= last_mem_size
                if overlap_cycles < last_overlap_cycles:
                    self.config_dict[config] = (mem_size, exe_time, comp_cycles, shift_cycles)
                    last_inserted_config = (config, (mem_size, exe_time, comp_cycles, shift_cycles))
        else:
            for config, (mem_size, exe_time, comp_cycles, shift_cycles) in config_sorted[1:]:
                last_mem_size, last_exe_time, _, __ = last_inserted_config[1]
                assert mem_size >= last_mem_size
                if exe_time < last_exe_time:
                    self.config_dict[config] = (mem_size, exe_time, comp_cycles, shift_cycles)
                    last_inserted_config = (config, (mem_size, exe_time, comp_cycles, shift_cycles))

        # Pareto refinement time usually negligible, only show for heavy ops
        if not is_light and (time.time() - _t5) > 0.1:
            print(f"[{self.name}] [TIME] Pareto refinement: {time.time() - _t5:.2f}s", flush=True)

        # reuse all_configs for cold config search
        _t6 = time.time()
        min_cold_configs = []
        spatial_set = set()
        for config in self.config_dict:
            spatial_set.add(config[0])
        for spatial_config in spatial_set:
            sorted_cold_configs = sorted(spatial_to_cold_to_temporal[spatial_config].items(), key=lambda item: item[0])
            min_cold_configs.append((spatial_config, sorted_cold_configs[0][1]))

        self.search_optimal_config_cold(num_threads, spatial_search_tree, log_filename, min_cold_configs)

        # Only show cold search time for heavy ops (light ops are too fast)
        if not is_light:
            _t6_elapsed = time.time() - _t6
            if _t6_elapsed > 0.5:  # Only if it took meaningful time
                print(f"[{self.name}] [TIME] Cold config search: {_t6_elapsed:.2f}s", flush=True)

        # dump config dict to file
        self.log_filename_physical = log_filename
        all_configs_dict[log_filename] = log_filename
        if DUMP_ALL or DUMP_ROLLER or self.dump():
            self.dump_config_dict(log_filename)

        self.dump_all_configs_dict()
        return self.config_dict

    def get_byte_per_core_idle_helper(self, params):
        return self.get_byte_per_core_idle(*params)

    def get_fastest_config_by_max_mem_size(self, mem_size: int) -> Tuple[Tuple[List[int], List[List[int]]], Tuple[int, int, int, int]]:
        last_config, (last_mem, last_exe_time, last_comp_cycles, last_shift_cycles) = next(iter(self.config_dict.items()))
        for config, (mem, exe_time, comp_cycles, shift_cycles) in self.config_dict.items():
            if mem > mem_size:
                break
            last_config, (last_mem, last_exe_time, last_comp_cycles, last_shift_cycles) = \
                config, (mem, exe_time, comp_cycles, shift_cycles)
        return last_config, (last_mem, last_exe_time, last_comp_cycles, last_shift_cycles)

    # return: best configs of this op with the corresponding scores
    # side effect: generates cold config candidates for this op (self.cold_config_candidates, sorted by mem_size)
    def search_optimal_config_cold(self, num_threads:int = 1,
                                   spatial_search_tree:Optional[OpSpatialPartitionSearch] = None,
                                   log_filename:str = "",
                                   min_cold_configs:list = []) -> Dict[int, Dict[Tuple, List]]:
        if log_filename == "":
            log_filename = self.name
        if len(self.cold_config_candidates) > 0:
            # dump all config_dict and cold config candidates to file
            cold_config_candidates_dict[log_filename] = self.log_filename_physical
            if DUMP_ALL:
                self.dump_cold_config_candidates(log_filename)
            if DUMP_DICT:
                self.dump_cold_config_candidates_dict()
            return self.cold_config_candidates

        assert len(self.config_dict) > 0, "search_optimal_config_cold: config_dict is empty"
        assert len(min_cold_configs) > 0, "search_optimal_config_cold: min_cold_configs is empty"
        all_configs = set()
        for config in self.config_dict:
            all_configs.add(config)
        for config in min_cold_configs:
            all_configs.add(config)
        all_configs = list(all_configs)

        # if len(all_configs) == 0:
        #     ##### 1. generate spatial partition search tree
        #     all_spatial_configs = self.get_all_spatial_configs(spatial_search_tree)

        #     # debug: dump spatial configs to file
        #     # with open(f"{log_filename}.spatial_configs.json", "w") as f:
        #     #     import ujson as json
        #     #     json.dump(all_spatial_configs, f, indent=4)

        #     ##### 2. generate temporal configs for each spatial config
        #     print(f"{log_filename}: generate tempral configs for each spatial config...", flush=True)
        #     if num_threads == 1:
        #         from tqdm import tqdm
        #         for spatial_config in tqdm(all_spatial_configs):
        #             temporal_configs = self.get_all_temporal_configs(spatial_config)
        #             for temporal_config in temporal_configs:
        #                 spatial_temporal_configs = (tuple(spatial_config), tuple(temporal_config))
        #                 all_configs.append(spatial_temporal_configs)
        #     else: # parallelize!
        #         with Pool(num_threads) as p:
        #             all_temp_configs = p.map(self.get_all_temporal_configs, all_spatial_configs)
        #         for spatial, temporal in zip(all_spatial_configs, all_temp_configs):
        #             for temporal_config in temporal:
        #                 spatial_temporal_configs = (tuple(spatial), tuple(temporal_config))
        #                 all_configs.append(spatial_temporal_configs)

        ##### 3. evaluate all configs
        # Only print for heavy ops
        if not self.is_light_op():
            print(f"[{self.name}] cold config search...", flush=True)
        if num_threads == 1:
            cold_configs = []   # [(config_id, cold_mem_size)]
            for config_id, config in enumerate(all_configs):
                cold_configs.append(self.get_byte_per_core_idle(config[1], config[0], config_id))
        else: # parallelize!
            with Pool(num_threads) as p:
                params = [(config[1], config[0], config_id) for config_id, config in enumerate(all_configs)]
                cold_configs = p.map(self.get_byte_per_core_idle_helper, params)
                cold_configs = list(cold_configs)

        ##### 4. generate config dict and cold config candidates for different mem sizes
        # remove all configs with the same mem_size (confict_dict[config][0])
        # and keep the ones with the lowest execution time (confict_dict[config][1])
        print(f"[{self.name}] generating cold config candidates...", flush=True)

        # Fast-path: filter valid configs (parallel-friendly)
        valid_entries = [(config_id, config, cold_configs[config_id][1])
                         for config_id, config in enumerate(all_configs)
                         if cold_configs[config_id][1] <= self.max_byte_per_core]

        # Build nested dict: parallelize if large enough
        if num_threads > 1 and len(valid_entries) > 1000:
            # Chunk entries and build partial dicts in parallel
            chunk_size = max(1, len(valid_entries) // num_threads)
            chunks = [valid_entries[i:i+chunk_size] for i in range(0, len(valid_entries), chunk_size)]

            with Pool(min(num_threads, len(chunks))) as p:
                partial_dicts = p.map(_build_cold_dict_chunk, chunks)

            # Merge partial dicts using defaultdict (eliminates all if-checks)
            merged = defaultdict(lambda: defaultdict(list))
            for partial in partial_dicts:
                for cold_mem_size, spatial_dict in partial.items():
                    for spatial_config, config_list in spatial_dict.items():
                        merged[cold_mem_size][spatial_config].extend(config_list)
            self.cold_config_candidates = dict(merged)
        else:
            # Sequential fallback for small ops
            temp_dict = defaultdict(lambda: defaultdict(list))
            for config_id, config, cold_mem_size in valid_entries:
                temp_dict[cold_mem_size][config[0]].append((config, (cold_mem_size, )))
            self.cold_config_candidates = dict(temp_dict)

        # save cold config candidates sorted by mem size
        cold_config_sorted = sorted(self.cold_config_candidates.items(), key=lambda item: item[0])
        self.cold_config_candidates = dict(cold_config_sorted)

        # dump cold config candidates to file
        self.log_filename_physical = log_filename
        cold_config_candidates_dict[log_filename] = log_filename
        if DUMP_ALL or self.dump():
            self.dump_cold_config_candidates(log_filename)

        self.dump_cold_config_candidates_dict()
        return self.cold_config_candidates

    # ---------------------------------------------------------------
    # Flat-pool helpers: separate config generation from evaluation
    # so the outer orchestrator can submit all configs from all ops
    # into one shared pool (no per-op tails).
    # ---------------------------------------------------------------

    def collect_hot_tasks(self, spatial_search_tree=None, log_filename=""):
        """Phases 1-2 of search_optimal_config: generate all (spatial, temporal)
        config pairs WITHOUT evaluating them.

        Returns:
            (all_configs, spatial_to_cold_to_temporal) where all_configs is a
            list of (spatial_tuple, temporal_tuple) and
            spatial_to_cold_to_temporal is the skeleton dict needed by
            apply_hot_scores.  Returns (None, None) if config_dict is already
            populated (cache hit).
        """
        if log_filename == "":
            log_filename = self.name
        if len(self.config_dict) > 0:
            all_configs_dict[log_filename] = self.log_filename_physical
            return None, None  # already done

        all_spatial_configs = self.get_all_spatial_configs(spatial_search_tree)
        all_configs = []
        spatial_to_cold_to_temporal = {}
        for spatial_config in all_spatial_configs:
            spatial_to_cold_to_temporal[tuple(spatial_config)] = {}
            temporal_configs = self.get_all_temporal_configs(spatial_config)
            for temporal_config in temporal_configs:
                all_configs.append((tuple(spatial_config), tuple(temporal_config)))
        return all_configs, spatial_to_cold_to_temporal

    def apply_hot_scores(self, all_configs, config_scores, spatial_to_cold_to_temporal, log_filename=""):
        """Phases 4-5 of search_optimal_config: build config_dict from
        pre-computed scores, then return min_cold_configs for cold search.

        Args:
            all_configs: list returned by collect_hot_tasks.
            config_scores: list of (config_id, perf) tuples in same order.
            spatial_to_cold_to_temporal: dict returned by collect_hot_tasks.
            log_filename: op name for logging.

        Returns:
            min_cold_configs list (input to collect_cold_tasks).
        """
        if log_filename == "":
            log_filename = self.name

        mem_size_to_exe_time = {}
        for config_id, config in enumerate(all_configs):
            _, perf_obj = config_scores[config_id]
            hot_mem_size = perf_obj.hot_mem_size
            if hot_mem_size > self.max_byte_per_core:
                continue
            cold_mem_size = perf_obj.cold_mem_size
            spatial_to_cold_to_temporal[config[0]][cold_mem_size] = config[1]
            exe_time = perf_obj.total_cycles
            comp_cycles = perf_obj.comp_cycles
            shift_cycles = perf_obj.sss_cycles
            if ICBM_OVERLAP and self.max_byte_per_core < 600000:
                overlap_cycles = max(comp_cycles, shift_cycles)
                if hot_mem_size in mem_size_to_exe_time:
                    if overlap_cycles < mem_size_to_exe_time[hot_mem_size]:
                        mem_size_to_exe_time[hot_mem_size] = overlap_cycles
                        self.config_dict[config] = (hot_mem_size, exe_time, comp_cycles, shift_cycles)
                else:
                    mem_size_to_exe_time[hot_mem_size] = overlap_cycles
                    self.config_dict[config] = (hot_mem_size, exe_time, comp_cycles, shift_cycles)
            else:
                if hot_mem_size in mem_size_to_exe_time:
                    if exe_time < mem_size_to_exe_time[hot_mem_size]:
                        mem_size_to_exe_time[hot_mem_size] = exe_time
                        self.config_dict[config] = (hot_mem_size, exe_time, comp_cycles, shift_cycles)
                else:
                    mem_size_to_exe_time[hot_mem_size] = exe_time
                    self.config_dict[config] = (hot_mem_size, exe_time, comp_cycles, shift_cycles)

        # Filter to Pareto front (ascending mem size, decreasing exe time)
        config_sorted = sorted(self.config_dict.items(), key=lambda item: item[1][0])
        if len(config_sorted) == 0:
            raise ValueError(f"# op {self.name}: no hot configs (SRAM too small).\n {self.dim_lengths}")
        self.config_dict = {}
        self.config_dict[config_sorted[0][0]] = config_sorted[0][1]
        last_inserted_config = config_sorted[0]
        if ICBM_OVERLAP and self.max_byte_per_core < 600000:
            for config, (mem_size, exe_time, comp_cycles, shift_cycles) in config_sorted[1:]:
                last_mem_size, _, last_comp_cycles, last_shift_cycles = last_inserted_config[1]
                overlap_cycles = max(comp_cycles, shift_cycles)
                last_overlap_cycles = max(last_comp_cycles, last_shift_cycles)
                assert mem_size >= last_mem_size
                if overlap_cycles < last_overlap_cycles:
                    self.config_dict[config] = (mem_size, exe_time, comp_cycles, shift_cycles)
                    last_inserted_config = (config, (mem_size, exe_time, comp_cycles, shift_cycles))
        else:
            for config, (mem_size, exe_time, comp_cycles, shift_cycles) in config_sorted[1:]:
                last_mem_size, last_exe_time, _, __ = last_inserted_config[1]
                assert mem_size >= last_mem_size
                if exe_time < last_exe_time:
                    self.config_dict[config] = (mem_size, exe_time, comp_cycles, shift_cycles)
                    last_inserted_config = (config, (mem_size, exe_time, comp_cycles, shift_cycles))

        print(f"[{self.name}] num hot configs: {len(self.config_dict)}", flush=True)

        # Dump
        self.log_filename_physical = log_filename
        all_configs_dict[log_filename] = log_filename
        if DUMP_ALL or DUMP_ROLLER or self.dump():
            self.dump_config_dict(log_filename)
        self.dump_all_configs_dict()

        # Build min_cold_configs for the cold search phase
        min_cold_configs = []
        spatial_set = set()
        for config in self.config_dict:
            spatial_set.add(config[0])
        for spatial_config in spatial_set:
            if spatial_config not in spatial_to_cold_to_temporal:
                continue
            sorted_cold = sorted(spatial_to_cold_to_temporal[spatial_config].items(), key=lambda x: x[0])
            if sorted_cold:
                min_cold_configs.append((spatial_config, sorted_cold[0][1]))
        return min_cold_configs

    def collect_cold_tasks(self, min_cold_configs):
        """Phase 3 preparation of search_optimal_config_cold: collect the
        flat parameter list for get_byte_per_core_idle WITHOUT evaluating.

        Returns:
            (all_configs, flat_params) where flat_params[i] = (temporal, spatial)
            in the same order as all_configs.  Returns (None, None) on cache hit.
        """
        if len(self.cold_config_candidates) > 0:
            return None, None  # already done
        all_configs = list(set(list(self.config_dict.keys()) + list(min_cold_configs)))
        flat_params = [(config[1], config[0]) for config in all_configs]
        return all_configs, flat_params

    def apply_cold_scores(self, all_configs, cold_scores, log_filename=""):
        """Phase 4 of search_optimal_config_cold: build cold_config_candidates
        from pre-computed (config_id, cold_mem_size) scores.
        """
        if log_filename == "":
            log_filename = self.name
        if len(self.cold_config_candidates) > 0:
            return
        self.cold_config_candidates = {}
        for local_id, config in enumerate(all_configs):
            cold_mem_size = cold_scores[local_id]
            if cold_mem_size > self.max_byte_per_core:
                continue
            if cold_mem_size not in self.cold_config_candidates:
                self.cold_config_candidates[cold_mem_size] = {}
            if config[0] not in self.cold_config_candidates[cold_mem_size]:
                self.cold_config_candidates[cold_mem_size][config[0]] = []
            self.cold_config_candidates[cold_mem_size][config[0]].append((config, (cold_mem_size,)))
        self.cold_config_candidates = dict(
            sorted(self.cold_config_candidates.items(), key=lambda item: item[0])
        )
        self.log_filename_physical = log_filename
        cold_config_candidates_dict[log_filename] = log_filename
        if DUMP_ALL or self.dump():
            self.dump_cold_config_candidates(log_filename)
        self.dump_cold_config_candidates_dict()

    # @cold_config, hot_config: ((spatial, temporal), (mem_size, exe_time)) -> exe time
    def evaluate_cold_hot_config(self, cold_config, hot_config) -> Union[int, float]:
        cold_mem_size = cold_config[1][0]
        hot_mem_size = hot_config[1][0]
        assert cold_mem_size <= hot_mem_size
        assert cold_config[0][0] == hot_config[0][0] # spatial config should be the same
        tot_exe_time = hot_config[1][1] + self.get_warm_up_time(cold_config[0][1], hot_config[0][1], cold_config[0][0])
        return tot_exe_time

    # candidates: Dict[spatial plan, configs ((spatial, temporal), (mem_size, exe_time))]
    def find_best_cold_hot_config(self, cold, hot, cold_candidates, hot_candidates):
        min_exe_time = float("inf")
        best_cold_config = None
        best_hot_config = None
        for spatial_plan, cold_configs in cold_candidates.items():
            if spatial_plan not in hot_candidates:
                continue
            for cold_config in cold_configs:
                for hot_config in hot_candidates[spatial_plan]:
                    if not is_cold_hot_plan_compatible(cold_config, hot_config):
                        continue
                    cold_hot_score = self.evaluate_cold_hot_config(cold_config, hot_config)
                    if cold_hot_score < min_exe_time:
                        min_exe_time = cold_hot_score
                        best_cold_config = cold_config
                        best_hot_config = hot_config

        return cold, hot, best_cold_config, best_hot_config, min_exe_time

    def find_best_cold_hot_config_helper(self, params):
        return self.find_best_cold_hot_config(*params)

    def generate_cold_hot_table(self, num_threads: int = 1, threshold: float = 1, log_filename: str = ""):
        if log_filename == "":
            log_filename = self.name
        if len(self.cold_hot_table) > 0:

            cold_hot_table_dict[log_filename] = self.log_filename_physical
            if DUMP_ALL or DUMP_ROLLER:
                self.dump_cold_hot_table(log_filename)
            if DUMP_DICT:
                self.dump_cold_hot_table_dict()
            return self.cold_hot_table

        # 1. generate all configs
        self.search_optimal_config(num_threads)

        # 2. get all possible (cold mem size, hot mem size) pairs and generate hot_config_candidates dict
        #    note that self.config_dict (hot configs) is a subset of self.cold_config_candidates (cold configs)
        #    note also that both config list/dict are sorted by mem size
        print(f"[{self.name}] generating cold-hot table...", flush=True)
        cold_mem_size_list = set(self.cold_config_candidates.keys())
        hot_mem_size_list = {x[0] for x in self.config_dict.values()}

        hot_config_candidates: Dict[Any, Dict[Any, List]] = {}
        for config, (mem_size, exe_time, comp_time, shift_time) in self.config_dict.items():
            if mem_size not in hot_config_candidates:
                hot_config_candidates[mem_size] = {}
            if config[0] not in hot_config_candidates[mem_size]:
                hot_config_candidates[mem_size][config[0]] = []
            hot_config_candidates[mem_size][config[0]].append((config, (mem_size, exe_time, comp_time, shift_time)))

        cold_hot_pairs = {
            (cold, hot) for cold in cold_mem_size_list for hot in hot_mem_size_list if cold <= hot
        }

        print(f"[{self.name}] num of cold mem sizes: {len(cold_mem_size_list)}", flush=True)
        print(f"[{self.name}] num of hot mem sizes: {len(hot_mem_size_list)}", flush=True)
        print(f"[{self.name}] num cold-hot pairs: {len(cold_hot_pairs)}", flush=True)

        # 3. find best config for each (cold mem size -> hot mem size) pair
        print(f"[{self.name}] finding best config for each (cold, hot) pair...", flush=True)
        tot_cold_hot_table_size = 0
        if num_threads == 1:
            for cold, hot in cold_hot_pairs:
                cold_candidates = self.cold_config_candidates[cold]
                hot_candidates = hot_config_candidates[hot]
                _, __, best_cold_config, best_hot_config, min_exe_time = self.find_best_cold_hot_config(cold, hot, cold_candidates, hot_candidates)
                if best_cold_config is not None and best_hot_config is not None:
                    if cold not in self.cold_hot_table:
                        self.cold_hot_table[cold] = {}
                    self.cold_hot_table[cold][hot] = (best_cold_config, best_hot_config, min_exe_time)
                    if hot not in self.hot_cold_table:
                        self.hot_cold_table[hot] = {}
                    self.hot_cold_table[hot][cold] = (best_hot_config, best_cold_config, min_exe_time)
                    tot_cold_hot_table_size += 1
        else: # Parallelize!
            params = [(cold, hot, self.cold_config_candidates[cold], hot_config_candidates[hot]) for cold, hot in cold_hot_pairs]
            # All needed data is now in params. Temporarily clear large attrs so workers
            # receive a lean self (cold_config_candidates already stripped by __getstate__;
            # config_dict must be restored afterwards for simulation workers).
            _saved_config_dict = self.config_dict
            self.config_dict = {}
            with Pool(num_threads) as p:
                cold_hot_configs = p.map(self.find_best_cold_hot_config_helper, params)
            self.config_dict = _saved_config_dict
            print(f"[{self.name}] generated cold_hot_configs", flush=True)
            for cold, hot, best_cold_config, best_hot_config, min_exe_time in cold_hot_configs:
                if best_cold_config is not None and best_hot_config is not None:
                    if cold not in self.cold_hot_table:
                        self.cold_hot_table[cold] = {}
                    self.cold_hot_table[cold][hot] = (best_cold_config, best_hot_config, min_exe_time)
                    if hot not in self.hot_cold_table:
                        self.hot_cold_table[hot] = {}
                    self.hot_cold_table[hot][cold] = (best_cold_config, best_hot_config, min_exe_time)
                    tot_cold_hot_table_size += 1

        # # remove the config if exe time is greater than cold exe time
        # # do not perform this pass if there is only 1 valid config
        # # (otherwise the hardcoded threshold removes every possible configs!)
        # if tot_cold_hot_table_size > 1 and False:
        #     new_cold_hot_table = {}
        #     for cold, cold_dict in self.cold_hot_table.items():
        #         for hot, (cold_config, hot_config, cold_hot_time) in cold_dict.items():
        #             if cold_hot_time > cold_config[1][1]:
        #                 tot_cold_hot_table_size -= 1
        #                 continue
        #             else:
        #                 if cold not in new_cold_hot_table:
        #                     new_cold_hot_table[cold] = {}
        #                 new_cold_hot_table[cold][hot] = (cold_config, hot_config, cold_hot_time)
        #     self.cold_hot_table = new_cold_hot_table

        # sort cold_hot_table by cold size
        self.cold_hot_table = dict(sorted(self.cold_hot_table.items(), key=lambda x: x[0]))
        self.hot_cold_table = dict(sorted(self.hot_cold_table.items(), key=lambda x: x[0]))

        # sort hot configs inside each cold size table
        for cold, cold_dict in self.cold_hot_table.items():
            self.cold_hot_table[cold] = dict(sorted(cold_dict.items(), key=lambda x: x[0]))
        for hot, hot_dict in self.hot_cold_table.items():
            self.hot_cold_table[hot] = dict(sorted(hot_dict.items(), key=lambda x: x[0]))

        # # sort hot configs inside each cold size table
        # cold_min_time_dict = {}
        # for cold, cold_dict in self.cold_hot_table.items():
        #     self.cold_hot_table[cold] = dict(sorted(cold_dict.items(), key=lambda x: -x[1][2]))
        #     cold_min_time_dict[cold] = next(reversed(self.cold_hot_table[cold].values()))[2]

        #  # sort cold_hot_table by cold size
        # self.cold_hot_table = dict(sorted(self.cold_hot_table.items(), key=lambda x: -cold_min_time_dict[x[0]]))

        print(f"[{self.name}] num cold mem sizes: {len(self.cold_hot_table)}", flush=True)
        print(f"[{self.name}] tot cold-hot table size: {tot_cold_hot_table_size}", flush=True)

        self.log_filename_physical = log_filename
        cold_hot_table_dict[log_filename] = log_filename
        if DUMP_ALL or DUMP_ROLLER or self.dump():
            self.dump_cold_hot_table(log_filename)

        self.dump_cold_hot_table_dict()
        return self.cold_hot_table

    def get_best_hot_size_for_cold(self, cold_mem_size, max_hot_mem_size) -> Optional[int]:
        if cold_mem_size not in self.cold_hot_table:
            return None
        smallest_hot_size = 0xFFFFFFFFFFFFFFFF
        for hot_size in reversed(self.cold_hot_table[cold_mem_size]):
            if hot_size < smallest_hot_size:
                smallest_hot_size = hot_size
            if smallest_hot_size <= cold_mem_size + max_hot_mem_size:
                break

        return smallest_hot_size

