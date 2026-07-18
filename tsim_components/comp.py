"""Compute-unit model for a single accelerator core.

This module estimates the cycle cost of executing tensor operations (matrix
multiplications and element-wise operations) on a core that contains:

  * A **systolic array (SA)** for matrix-multiply workloads.
  * A **vector unit (VU)** for element-wise / activation workloads.

Two cost models are provided:

  1. **Generic** -- uses configurable SA/VU dimensions, reuse factors, and
     local SRAM bandwidth to compute a first-order cycle estimate that
     accounts for both compute and load/store costs.
  2. **IPU-specific** -- a regression-based cost model calibrated against
     Graphcore IPU profiling data, activated via ``use_ipu_cost=True``.

The module also supports *fused operator* analysis: when consecutive
operators share intermediate tensors, the store of one operator can be
fused with the load of the next, reducing total memory-traffic cycles.

Key classes:
    Compute   -- Core-level cost model (SA + VU).
    Compute.OP -- Lightweight descriptor for a single tensor operator.
"""

import numpy as np
from . import utils
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from math import ceil

# ---------------------------------------------------------------------------
# Operation type constants (used by the IPU cost model)
# ---------------------------------------------------------------------------
OP_TYPE_REDUCE = 0
OP_TYPE_RELU = 1
OP_TYPE_ELEMENT = 2
OP_TYPE_POOL = 3
OP_TYPE_CONV = 4
OP_TYPE_MATMUL = 5
OP_TYPE_GATHER = 6
OP_TYPE_SLICE = 8

# ---------------------------------------------------------------------------
# IPU-specific padding and throughput constants
# ---------------------------------------------------------------------------
IPU_EW_PAD_LEN = 4               # Minimum element count for EW alignment
IPU_MM_PAD_SHAPE = [6, 16, 16]   # [M, K, N] alignment granularity
IPU_MM_FLOPC = 62                # IPU matmul throughput (FLOPs/cycle)

def get_num_comp_iter(temporal_dim_var_parts: List[List[int]]) -> int:
    """Compute the total number of temporal iterations for a tiled operator.

    Each dimension has one or more partitioning factors; the iteration count
    along that dimension is the maximum factor.  The total iteration count is
    the product across all dimensions.

    Args:
        temporal_dim_var_parts: 2-D list where ``[d][v]`` is the temporal
            partition factor of variable *v* along dimension *d*.

    Returns:
        Total number of compute iterations (>= 1).
    """
    num_iter = np.prod(np.max(temporal_dim_var_parts, axis=-1))
    assert(num_iter >= 1), "num_iter should be greater than 1"
    return num_iter

class Compute:
    """Core-level compute model (systolic array + vector unit).

    Models the cycle cost of executing matrix-multiply (MM) and element-wise
    (EW) operations on a single core, including:

      * **Compute cycles** -- determined by the operation's FLOP count and the
        SA or VU throughput (FLOPs/cycle).
      * **Load/store cycles** -- determined by the tensor sizes, hardware
        reuse factors, and local SRAM bandwidth.
      * **Fused operator analysis** -- when consecutive operators share
        intermediate tensors, redundant store-then-load traffic is eliminated.

    The final cycle estimate for a fused operator group is::

        max(compute_cycles, load_cycles, store_cycles)

    reflecting the assumption that compute and memory access are pipelined.
    """

    class OP:
        """Lightweight descriptor for a single tensor operator.

        Captures the dimension-variable mapping (which indices of the global
        dimension vector each tensor references), the concrete dimension
        lengths after tiling, and metadata used for load/store and fused-op
        analysis.

        Example -- matrix multiply ``C[m,n] = A[m,k] @ B[k,n]``::

            variables  = [[[0],[1]],   # C references dims m, n
                          [[0],[2]],   # A references dims m, k
                          [[2],[1]]]   # B references dims k, n
            dim_lengths = [1600, 80, 1600]  # m=1600, n=80, k=1600

        Example -- 2-D convolution ``O[b,oc,oh,ow] = I[b,ic,oh+kh,ow+kw] * K[oc,ic,kh,kw]``::

            variables  = [[[0],[1],[3],[4]],
                          [[0],[2],[3,5],[4,6]],
                          [[2],[1],[5],[6]]]
            dim_lengths = [50, 60, 30, 256, 768, 3, 5]
        """

        def __init__(self,
                     is_ew: bool,
                     variables: List[List[List[int]]],
                     dim_lengths: List[int],
                     is_ignored_list: List[bool],
                     tensor_id_list: List[int],
                     op_type: int = -1) -> None:
            """
            Args:
                is_ew: True for element-wise ops, False for matmul/conv.
                variables: Per-tensor dimension-index mapping (see class
                    docstring for format).
                dim_lengths: Concrete (possibly padded) length of each
                    dimension in the global index space.
                is_ignored_list: Per-tensor flag; True means the tensor's
                    memory traffic should be ignored (e.g. in-place ops).
                tensor_id_list: Unique tensor ID for each operand, used to
                    track cross-operator data reuse during fusion analysis.
                op_type: Operation type constant (``OP_TYPE_*``), used only
                    by the IPU cost model.  Defaults to -1 (generic).
            """
            assert len(variables) == len(is_ignored_list), \
                "variables and is_ignored_list should have same length"
            assert len(variables) == len(tensor_id_list), \
                "variables and tensor_id_list should have same length"

            self.is_ew = is_ew
            self.variables = variables
            self.dim_lengths = dim_lengths
            self.is_ignored_list = is_ignored_list
            self.tensor_id_list = tensor_id_list
            self.op_type = op_type

    def __init__(self,
                 ew_pad_len: int,
                 mm_pad_shape: np.ndarray,
                 ew_reuse_num: int,
                 mm_reuse_list: list,
                 ew_flopc: int,
                 mm_flopc: int,
                 load_store_bw_bytepc: float,
                 byte_per_elem: int,
                 mm_init_cycle: int = 0,
                 ew_mm_overlap: bool = True,
                 use_ipu_cost: bool = False,
                 use_ipu_pad: bool = False) -> None:
        """
        Args:
            ew_pad_len: Minimum element-wise tile length required by the
                vector unit (tiles smaller than ``2 * ew_pad_len`` are padded).
            mm_pad_shape: 3-element list ``[M, K, N]`` giving the systolic
                array's alignment granularity.  Tile dimensions are rounded up
                to multiples of these values.
            ew_reuse_num: Maximum hardware data-reuse count for EW operations
                (number of element visits per SRAM load).
            mm_reuse_list: Per-operand hardware reuse factors for matmul
                ``[output, inputA, inputB]``.
            ew_flopc: Vector-unit throughput in FLOPs per cycle.
            mm_flopc: Systolic-array throughput in FLOPs per cycle.
            load_store_bw_bytepc: Local SRAM read/write bandwidth in bytes
                per cycle.
            byte_per_elem: Bytes per data element (2 for fp16, 4 for fp32).
            mm_init_cycle: Fixed startup overhead per matmul invocation
                (pipeline fill, etc.).
            ew_mm_overlap: If True, EW and MM compute phases can overlap
                (``max``); otherwise they are serialised (``sum``).
            use_ipu_cost: If True, use the IPU regression cost model instead
                of the generic analytical model.
            use_ipu_pad: If True, apply IPU-specific padding rules to tile
                dimensions (can be used independently of ``use_ipu_cost``).
        """
        self.ew_pad_len = ew_pad_len
        self.mm_pad_shape = mm_pad_shape
        self.ew_reuse_num = ew_reuse_num
        self.mm_reuse_list = mm_reuse_list
        self.ew_flopc = ew_flopc
        self.mm_flopc = mm_flopc
        self.load_store_bw_bytepc = load_store_bw_bytepc
        self.byte_per_elem = byte_per_elem

        self.mm_init_cycle = mm_init_cycle
        self.ew_mm_overlap = ew_mm_overlap
        self.use_ipu_cost = use_ipu_cost
        self.use_ipu_pad = use_ipu_pad

        if use_ipu_pad:
            self.mm_flopc = IPU_MM_FLOPC

    def get_area(self):
        """Estimate the silicon area of this core's compute logic (mm^2).

        Scales reference TSMC N7 areas for a 128x128 systolic array and a
        128x8x2 vector unit linearly according to the configured SA/VU sizes.
        Contributions from control logic and other peripherals are assumed
        negligible.

        Returns:
            Tuple[float, float, float]: ``(total_area, sa_area, vu_area)``
            in square millimetres.
        """
        # Reference areas at TSMC N7 node
        sa_128_128_sq_mm = 65.17333 / 4   # 128x128 systolic array
        vu_128_8_2_sq_mm = 22.3942         # 128x8x2 vector unit (= 2048 ops/cycle)

        # Scale quadratically for SA (area ~ side^2), linearly for VU.
        sa_scale_factor = (self.mm_pad_shape[-1] / 128) ** 2
        vu_scale_factor = self.ew_flopc / 2048

        sa_area = sa_128_128_sq_mm * sa_scale_factor
        vu_area = vu_128_8_2_sq_mm * vu_scale_factor
        return sa_area + vu_area, sa_area, vu_area
    
    def convert_op_ipu(self,
                       dim_len,
                       variables,
                       is_ew,
                       ignore_variables,
                       tensor_id_list,
                       op_type: int = -1,
                       ) -> OP:
        """Create an OP descriptor with IPU-specific dimension padding.

        For element-wise ops, the longest dimension is padded to at least
        ``1.5 * IPU_EW_PAD_LEN`` elements.  For matmul ops, the M/K/N
        dimensions are identified from the variable mapping and padded to
        multiples of ``IPU_MM_PAD_SHAPE`` (6, 16, 16).

        Args:
            dim_len: Raw dimension lengths (modified in-place with padding).
            variables: Per-tensor dimension-index mapping.
            is_ew: True for element-wise, False for matmul.
            ignore_variables: Per-tensor ignored flag.
            tensor_id_list: Per-tensor unique identifiers.
            op_type: Operation type constant (``OP_TYPE_*``).

        Returns:
            A padded :class:`OP` instance.
        """
        original_dim_lengths = dim_len
        padded_dim_lengths = original_dim_lengths
        if is_ew:
            if np.prod(original_dim_lengths) < 2*IPU_EW_PAD_LEN:
                longest_dim = np.argmax(original_dim_lengths)
                longest_dim_len = original_dim_lengths[longest_dim]
                remaining_dim_len = np.prod(original_dim_lengths) // longest_dim_len
                padded_longest_dim_len = ceil(1.5 * IPU_EW_PAD_LEN / remaining_dim_len)
                padded_dim_lengths[longest_dim] = padded_longest_dim_len
        else:
            # Identify M, K, N dimension indices from the variable mapping:
            # K is the contraction dim (shared by both inputs but not the output).
            # M comes from input A; N comes from input B.
            out = variables[0].flatten()[-2:]
            inA = variables[1].flatten()[-2:]
            inB = variables[2].flatten()[-2:]
            kset = np.intersect1d(inA, inB)
            k_idx = np.setdiff1d(kset, out)[0]
            m_idx = np.setdiff1d(inA[-2:], [k_idx])[0]
            n_idx = np.setdiff1d(inB[-2:], [k_idx])[0]
            m = padded_dim_lengths[m_idx]
            n = padded_dim_lengths[n_idx]
            # Ensure M >= N for canonical ordering (unless M <= 16).
            if n > m:
                m, n = n, m
                m_idx, n_idx = n_idx, m_idx
            if m <= 16:
                m, n = n, m
                m_idx, n_idx = n_idx, m_idx

            # Round M, K, N up to IPU alignment boundaries.
            padded_dim_lengths[m_idx] = ceil(padded_dim_lengths[m_idx] / IPU_MM_PAD_SHAPE[0]) * IPU_MM_PAD_SHAPE[0]
            padded_dim_lengths[k_idx] = ceil(padded_dim_lengths[k_idx] / IPU_MM_PAD_SHAPE[1]) * IPU_MM_PAD_SHAPE[1]
            padded_dim_lengths[n_idx] = ceil(padded_dim_lengths[n_idx] / IPU_MM_PAD_SHAPE[2]) * IPU_MM_PAD_SHAPE[2]

        op = self.OP(is_ew = is_ew,
                     variables = variables,
                     dim_lengths = padded_dim_lengths, # get sub op
                     is_ignored_list = ignore_variables,
                     tensor_id_list = tensor_id_list,
                     op_type = op_type,
                     )
        return op
    
    def convert_op_simple(self,
                          dim_len,
                          variables,
                          is_ew,
                          ignore_variables,
                          tensor_id_list,
                          op_type: int = -1,
                          ) -> OP:
        """Create an OP descriptor with generic (non-IPU) dimension padding.

        For element-wise ops, the longest dimension is padded when the total
        element count is below ``2 * ew_pad_len``.  For matmul ops, the last
        three dimensions are rounded up to multiples of ``mm_pad_shape``.

        Falls back to :meth:`convert_op_ipu` when ``use_ipu_pad`` is True
        and ``use_ipu_cost`` is False.

        Returns:
            A padded :class:`OP` instance.
        """
        if self.use_ipu_pad and not self.use_ipu_cost:
            return self.convert_op_ipu(dim_len, variables, is_ew, ignore_variables, tensor_id_list, op_type)

        original_dim_lengths = dim_len
        padded_dim_lengths = original_dim_lengths
        if is_ew:
            if np.prod(original_dim_lengths) < 2*self.ew_pad_len:
                longest_dim = np.argmax(original_dim_lengths)
                longest_dim_len = original_dim_lengths[longest_dim]
                remaining_dim_len = np.prod(original_dim_lengths) // longest_dim_len
                padded_longest_dim_len = ceil(1.5 * self.ew_pad_len / remaining_dim_len)
                padded_dim_lengths[longest_dim] = padded_longest_dim_len
        else:
            # Pad the last 3 dimensions (M, K, N) to SA alignment granularity.
            padded_dim_lengths[-1] = ceil(padded_dim_lengths[-1] / self.mm_pad_shape[-1]) * self.mm_pad_shape[-1]
            padded_dim_lengths[-2] = ceil(padded_dim_lengths[-2] / self.mm_pad_shape[-2]) * self.mm_pad_shape[-2]
            padded_dim_lengths[-3] = ceil(padded_dim_lengths[-3] / self.mm_pad_shape[-3]) * self.mm_pad_shape[-3]

        op = self.OP(is_ew = is_ew,
                     variables = variables,
                     dim_lengths = padded_dim_lengths, # get sub op
                     is_ignored_list = ignore_variables,
                     tensor_id_list = tensor_id_list,
                     op_type = op_type,
                     )
        return op

    def get_ew_load_store_cycles_from_padded_tile(self,
                                                  variables: List[List[List[int]]],
                                                  dim_lengths: List[int]) -> np.ndarray:
        """Compute per-tensor SRAM load/store cycles for an element-wise tile.

        The number of load/store repetitions is determined by the ratio of
        tile-level data reuse to the hardware reuse capacity.  The formula is::

            load_store_times = ceil(tile_reuse / min(hw_reuse, tile_flop))
            cycles_per_tensor = tensor_size * load_store_times * byte_per_elem
                                / load_store_bw_bytepc

        Returns:
            np.ndarray of shape ``(num_tensors,)`` with per-tensor cycle counts.
        """
        # format variables and dim_lengths to np array
        variables_np = []
        for var in variables:
            variables_np.append(utils.pad_to_dense(var))
        dim_lengths_np = np.append(dim_lengths,0)

        # get tensor sizes
        sizes = [utils.shape_to_size(utils.var_to_shape(dim_lengths_np, var)) for var in variables_np]
        sizes = np.array(sizes)

        # get max reuse bounded by tile
        tile_flop = np.prod(dim_lengths_np[:-1])
        tile_reuses = tile_flop//sizes

        # get number of repeated load/stores by applying max hardware reuse
        load_store_times = np.ceil(tile_reuses/min(self.ew_reuse_num,tile_flop))

        # get number of bytes to load/store
        load_store_bytes = sizes*load_store_times*self.byte_per_elem

        # get number of cycles to load/store
        load_store_cycles = load_store_bytes/self.load_store_bw_bytepc
        return load_store_cycles
    
    def get_mm_load_store_cycles_from_padded_tile(self,
                                                  variables: List[List[List[int]]],
                                                  dim_lengths: List[int]) -> np.ndarray:
        """Compute per-tensor SRAM load/store cycles for a matmul tile.

        Converts the dimension-variable representation into ``(B, K, M, N)``
        form and delegates to :meth:`get_mm_load_store_cycles_from_padded_bkmn`.

        Returns:
            np.ndarray of shape ``(3,)`` -- ``[output, inputA, inputB]`` cycles.
        """
        bkmn = utils.dim_var_to_bkmn(dim_lengths, variables)
        return self.get_mm_load_store_cycles_from_padded_bkmn(bkmn)
    
    def get_mm_load_store_cycles_from_padded_bkmn(self,
                                                  bkmn: Tuple[int, int, int, int]) -> np.ndarray:
        """Compute per-tensor SRAM load/store cycles for a matmul in BKMN form.

        The matmul ``C[b,m,n] = A[b,m,k] @ B[b,k,n]`` has operand sizes:
          - Output C: ``b*m*n``  |  tile reuse along K
          - Input  A: ``b*m*k``  |  tile reuse along N
          - Input  B: ``b*k*n``  |  tile reuse along M

        The number of load/store repetitions per operand is::

            ceil(tile_reuse / arch_reuse)

        Args:
            bkmn: Tuple ``(batch, K, M, N)`` after padding.

        Returns:
            np.ndarray of shape ``(3,)`` -- ``[output, inputA, inputB]`` cycles.
        """
        b, k, m, n = bkmn

        # Hardware reuse factors [output, inputA, inputB]; inputA reuse = N.
        # Older configs encode a single MM reuse value, so expand it across
        # output/inputB before applying the inputA special case.
        if len(self.mm_reuse_list) >= 3:
            arch_reuse_np_out_in_in = np.array(self.mm_reuse_list[:3], dtype=float)
        else:
            default_reuse = float(self.mm_reuse_list[0]) if self.mm_reuse_list else 1.0
            arch_reuse_np_out_in_in = np.array([default_reuse, default_reuse, default_reuse], dtype=float)
        arch_reuse_np_out_in_in[1] = n

        # Maximum tile-level reuse along each operand's contraction axis.
        tile_reuse_np_out_in_in = np.array([k, n, m])

        # Number of SRAM round-trips per operand.
        load_store_times = np.ceil(tile_reuse_np_out_in_in / arch_reuse_np_out_in_in)

        # Operand sizes in elements, then bytes.
        sizes = np.array([b*m*n, b*m*k, b*k*n])
        load_store_bytes = sizes * load_store_times * self.byte_per_elem

        # Convert to cycles using local SRAM bandwidth.
        load_store_cycles = load_store_bytes / self.load_store_bw_bytepc
        return load_store_cycles

    def get_peak_flopc(self):
        """Return peak throughput ``(mm_flopc, ew_flopc)`` for SA and VU."""
        return self.mm_flopc, self.ew_flopc

    def get_ew_compute_cycle_from_padded_tile(self,
                                              dim_lengths: List[int]) -> float:
        """Return VU compute cycles for an element-wise tile: ``prod(dims) / ew_flopc``."""
        assert 0 not in dim_lengths, "0 dimension is not allowed"
        return np.prod(dim_lengths) / self.ew_flopc
    
    def get_mm_compute_cycle_from_padded_tile(self,
                                              dim_lengths: List[int]) -> float:
        """Return SA compute cycles for a matmul tile.

        The factor of 2 accounts for the multiply-accumulate (MAC) operation
        contributing 2 FLOPs per element::

            cycles = 2 * prod(dims) / mm_flopc + mm_init_cycle
        """
        return 2 * np.prod(dim_lengths) / self.mm_flopc + self.mm_init_cycle

    def get_ipu_cost(self, op: OP) -> float:
        """Estimate cycles for a single op using the IPU regression cost model.

        The model uses piecewise-linear formulas calibrated against Graphcore
        IPU profiling data.  Each ``op_type`` has its own cost function that
        takes into account alignment padding and data-path characteristics.

        Args:
            op: An :class:`OP` descriptor with ``op_type`` set to one of the
                ``OP_TYPE_*`` constants.

        Returns:
            Estimated cycle count (float) for a single invocation of the op.
        """
        sub_op_shape = np.array(op.dim_lengths)
        sub_op_product = np.prod(sub_op_shape)

        if op.op_type == OP_TYPE_REDUCE:
            output_shape = sub_op_shape[op.variables[0][:, 0]]
            output_product = np.prod(output_shape)
            batch = sub_op_product / output_product
            # IPU reduce alignment ladder: snap output_product to the next
            # allowed alignment boundary: 1, 2, 4, 8, 12, 16, 24, then
            # multiples of 48 (i.e. 48, 96, 144, 192, ...).
            aligned_output_product = output_product
            if aligned_output_product > 2:
                aligned_output_product = max(4, aligned_output_product)
                if aligned_output_product > 4:
                    aligned_output_product = max(8, aligned_output_product)
                    if aligned_output_product > 8:
                        aligned_output_product = max(12, aligned_output_product)
                        if aligned_output_product > 12:
                            aligned_output_product = max(16, aligned_output_product)
                            if aligned_output_product > 16:
                                aligned_output_product = max(24, aligned_output_product)
                                if aligned_output_product > 24:
                                    aligned_output_product = np.ceil(aligned_output_product / 48) * 48
            # Piecewise-linear cost: cheaper coefficient above 480 elements.
            if aligned_output_product * batch <= 480:
                aligned_min = 481*0.25 + 175
                return min(aligned_min, (aligned_output_product*batch*1.5 + 175))
            else:
                return (aligned_output_product*batch*0.25 + 175)
            
        elif op.op_type==OP_TYPE_RELU:
            return (sub_op_product*0.5 + 200)
        
        elif op.op_type==OP_TYPE_ELEMENT:
            if sub_op_product <= 16432:
                aligned_min = 16434*0.5 + 300
                return min(aligned_min, (sub_op_product*0.75 + 290))
            else:
                return (sub_op_product*0.5 + 300)
            
        elif op.op_type==OP_TYPE_POOL or op.op_type==OP_TYPE_CONV:
            pool_shape = sub_op_shape[-4:]
            kh = pool_shape[2]
            kw = pool_shape[3]
            h = pool_shape[0]
            w = pool_shape[1]

            if op.op_type==OP_TYPE_POOL:
                chl = 1
                if len(sub_op_shape)>4:
                    chl = np.prod(sub_op_shape[0:-4])
                    chl = np.ceil(chl/4)*4
                flops = chl*h*w*kh*kw
                return 0.0495*flops+2030
            
            else:
                convB = 1
                convI = 1
                convO = 4
                if len(sub_op_shape)==5:
                    if len(op.variables[0])>len(op.variables[1]):
                        convO = sub_op_shape[0]
                    elif len(op.variables[0])<len(op.variables[1]):
                        convI = sub_op_shape[0]
                    else:
                        convB = sub_op_shape[0]
                elif len(sub_op_shape)==6:
                    if len(op.variables[0])>len(op.variables[1]):
                        convB = sub_op_shape[0]
                        convO = sub_op_shape[1]
                    elif len(op.variables[0])<len(op.variables[1]):
                        convB = sub_op_shape[0]
                        convI = sub_op_shape[1]
                    else:
                        out_idx = op.variables[0][0]
                        in_idx = 1-out_idx
                        convO = sub_op_shape[out_idx]
                        convI = sub_op_shape[in_idx]
                elif len(sub_op_shape)>6:
                    out_idx = op.variables[0][len(sub_op_shape)-6][0]
                    in_idx = 1-(out_idx-len(sub_op_shape)+6)+len(sub_op_shape)-6
                    convB = np.prod(sub_op_shape[0:-6])
                    convO = sub_op_shape[out_idx]
                    convI = sub_op_shape[in_idx]

                if convI>2:
                    convI = np.ceil(convI/4)*4
                elif convI>1 and convO<8:
                    convI = np.ceil(convI/4)*4
                convO = np.ceil(convO/4)*4

                # Total FLOPs with I/O channels rounded to multiples of 8.
                flops = convB * np.ceil(convI/8)*8 * np.ceil(convO/8)*8 * h*w * kh*kw
                predict = 0.0495 * flops + 2030

                return predict

        elif op.op_type == OP_TYPE_MATMUL:
            # Identify M, K, N indices (same logic as convert_op_ipu).
            out = op.variables[0].flatten()[-2:]
            inA = op.variables[1].flatten()[-2:]
            inB = op.variables[2].flatten()[-2:]
            kset = np.intersect1d(inA, inB)
            k_idx = np.setdiff1d(kset, out)[0]
            m_idx = np.setdiff1d(inA[-2:], [k_idx])[0]
            n_idx = np.setdiff1d(inB[-2:], [k_idx])[0]
            k = sub_op_shape[k_idx]
            m = sub_op_shape[m_idx]
            n = sub_op_shape[n_idx]
            if n > m:
                m, n = n, m
            if m <= 16:
                m, n = n, m
            b = 1
            if len(sub_op_shape > 3):
                b = np.prod(sub_op_shape[:-3])

            # IPU matmul regression model, parameterised by tile sizes
            # normalised to the hardware granularity (6, 32/byte_per_elem, 16).
            m_div_6 = m / 6
            k_div_16 = k / (32 / self.byte_per_elem)
            n_div_16 = n / 16

            # Per-matmul cost: data-dependent term + fixed overheads.
            per_mm_16 = (self.byte_per_elem * 12 * m_div_6 * k_div_16 * n_div_16
                         + 514 * k_div_16 * n_div_16
                         + 113 * n_div_16
                         + 173)
            time_16 = b * per_mm_16
            return time_16

        elif op.op_type==OP_TYPE_GATHER:
            num_indices = np.prod(sub_op_shape[op.variables[2]])
            num_total = np.prod(sub_op_shape[:-1])
            new_total_half = np.ceil(num_total/num_indices/2)*num_indices
            return 300 + new_total_half + num_indices
        
        else:
            return (sub_op_product*0.5+200)*2

    def get_total_cycle_for_fused_op(self,
                                     ops: List[OP],
                                     temporal_for_ops) -> Tuple[float, float, float, float, float]:
        """Estimate total cycles for a group of fused operators.

        Computes SA and VU compute cycles independently, then determines
        per-tensor load/store cycles with cross-operator fusion (a store
        followed by a load of the same tensor can be partially or fully
        elided).

        The final bottleneck cycle is::

            max(compute_cycles, total_load_cycles, total_store_cycles)

        Args:
            ops: List of :class:`OP` descriptors to be fused.
            temporal_for_ops: Per-op temporal tiling configuration (used to
                compute the iteration multiplier via
                :func:`get_num_comp_iter`).

        Returns:
            Tuple of five floats:
            ``(bottleneck_cycles, mm_compute, ew_compute, load, store)``.
        """
        if self.use_ipu_cost:
            mm_compute_cycle = 0
            ew_compute_cycle = 0
            for op, temporal in zip(ops, temporal_for_ops):
                if op.is_ew:
                    cycle = self.get_ipu_cost(op) * get_num_comp_iter(temporal)
                    ew_compute_cycle += cycle
                else:
                    cycle = self.get_ipu_cost(op) * get_num_comp_iter(temporal)
                    mm_compute_cycle += cycle
            
            if self.ew_mm_overlap:
                compute_cycle = max(ew_compute_cycle, mm_compute_cycle)
            else:
                compute_cycle = ew_compute_cycle + mm_compute_cycle
            total_load_cycle = compute_cycle
            total_store_cycle = compute_cycle
            
            return compute_cycle, mm_compute_cycle, ew_compute_cycle, total_load_cycle, total_store_cycle

        # --- Generic (non-IPU) cost model ------------------------------------
        assert(len(ops) == len(temporal_for_ops)), "each op should have a corresponding temporal config"

        # Sum EW and MM compute cycles separately (they may overlap).
        ew_compute_cycle = sum(self.get_ew_compute_cycle_from_padded_tile(op.dim_lengths) * get_num_comp_iter(temporal)
                               for temporal, op in zip(temporal_for_ops, ops) if op.is_ew)
        mm_compute_cycle = sum(self.get_mm_compute_cycle_from_padded_tile(op.dim_lengths) * get_num_comp_iter(temporal)
                               for temporal, op in zip(temporal_for_ops, ops) if not op.is_ew)
        if self.ew_mm_overlap:
            compute_cycle = max(ew_compute_cycle, mm_compute_cycle)
        else:
            compute_cycle = ew_compute_cycle + mm_compute_cycle

        # --- Load/store cycle analysis with fusion ----------------------------
        # Build a per-tensor history of (signed_cycle, is_ew) tuples.
        # Convention: negative = load, positive = store.
        tensor_id_to_ldst_cycle_dict: Dict[int, List[Tuple[float, bool]]] = {}
        for op in ops:
            # get load/store cycles for each tensor
            if op.is_ew:
                load_store_cycles = self.get_ew_load_store_cycles_from_padded_tile(op.variables, op.dim_lengths)
            else:
                load_store_cycles = self.get_mm_load_store_cycles_from_padded_tile(op.variables, op.dim_lengths)
            
            # Record each tensor's load/store cycle cost.
            # Index 0 is the output (store); indices 1+ are inputs (load).
            for is_input, (tensor_id, cycle) in enumerate(zip(op.tensor_id_list, load_store_cycles)):
                if tensor_id not in tensor_id_to_ldst_cycle_dict:
                    tensor_id_to_ldst_cycle_dict[tensor_id] = []
                if is_input:
                    tensor_id_to_ldst_cycle_dict[tensor_id].append((-cycle, op.is_ew))
                else:
                    tensor_id_to_ldst_cycle_dict[tensor_id].append((cycle, op.is_ew))

        # Walk through each tensor's access history and apply fusion rules.
        total_load_cycle = 0
        total_store_cycle = 0
        for tensor_id, cycle_list in tensor_id_to_ldst_cycle_dict.items():
            cycle, is_ew = cycle_list[0]
            assert(not np.isnan(cycle)), "cycle should not be nan!"
            last_is_ew = is_ew
            last_cycle = cycle
            # First access of this tensor -- no fusion possible.
            if cycle < 0:
                total_load_cycle += -cycle
            else:
                total_store_cycle += cycle
            # Subsequent accesses: attempt store-load fusion when at least one
            # of the consecutive operators is element-wise (EW ops keep data
            # in register files, enabling bypass).
            for cycle, is_ew in cycle_list[1:]:
                if (not last_is_ew) and (not is_ew):
                    # Both are matmul -- no fusion opportunity.
                    if cycle < 0:
                        total_load_cycle += -cycle
                    else:
                        total_store_cycle += cycle
                else:
                    # At least one EW op: fuse the preceding store with the
                    # current load, subtracting the overlapping traffic.
                    if (last_cycle < 0) and (cycle < 0):
                        # Two consecutive loads -- nothing to fuse.
                        total_load_cycle += -cycle
                    elif (last_cycle > 0) and (cycle < 0):
                        # Store followed by load: cancel the smaller of the
                        # two, and attribute the residual appropriately.
                        total_store_cycle -= min(last_cycle, -cycle)
                        total_load_cycle += abs(last_cycle + cycle)
                    else:
                        total_store_cycle += cycle
                last_cycle = cycle
                last_is_ew = is_ew
        return max(compute_cycle, total_load_cycle, total_store_cycle), mm_compute_cycle, ew_compute_cycle, total_load_cycle, total_store_cycle
