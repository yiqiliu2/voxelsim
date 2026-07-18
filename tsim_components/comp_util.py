"""Operator-fusion utilities and compute-cost wrapper.

This module bridges the gap between high-level DNN operator descriptions
(``TensorOperator`` / ``TensorExpression``) and the low-level cycle model
in ``tsim_components.comp.Compute``.

Key responsibilities
--------------------
* Convert ``TensorOperator`` instances into the internal ``Compute.OP``
  format used by the cycle-accurate simulator.
* Fuse consecutive operators into groups that share on-chip SRAM
  buffers, reducing DRAM traffic.  Two fusion strategies are provided:

  - **simple** -- greedily fuses elementwise ops with adjacent matmuls;
    up to two matmuls are fused when the intermediate tensor exceeds
    SRAM capacity.
  - **balanced** -- extends *simple* by considering the execution-cycle
    balance between matmul and elementwise work in each fused group to
    improve pipeline utilisation.
* Compute per-group execution costs (total, matmul, elementwise, SRAM
  read/write cycles).
"""

import numpy as np
from tsim_components.comp import Compute
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from t10_TensorExpression import TensorExpression
from icbm_DNNProgram import TensorOperator
from math import ceil

class Compute_OP:
    """Wrapper that converts DNN operators into ``Compute.OP`` objects and
    orchestrates operator fusion.

    The class maintains a monotonically increasing tensor-ID counter so
    that every unique intermediate tensor produced during conversion
    receives a distinct identifier.  This is consumed downstream by the
    SRAM reuse analysis.

    Parameters
    ----------
    comp : Compute
        The low-level cycle-accurate compute model used to estimate
        matmul / elementwise / SRAM costs.
    """

    def __init__(self,
                 comp: Compute) -> None:
        self.tensor_id = 0  # Monotonic counter for unique tensor IDs.
        self.comp = comp

    def new_tensor_id(self) -> int:
        """Allocate and return the next unique tensor ID."""
        self.tensor_id += 1
        return self.tensor_id


    def fill_tensor_ids(self, ignored_list: List[bool]) -> List[int]:
        """Assign unique IDs to each tensor variable of an operator.

        Variables flagged in *ignored_list* (e.g. residual / skip
        connections) reuse the most-recently produced output ID instead
        of allocating a fresh one.

        TODO: handle the case where *both* inputs of an op are
        previously-produced tensors (e.g. two-input residual adds).

        Parameters
        ----------
        ignored_list : list of bool
            Per-variable flag; ``True`` means the variable should share
            the previous output's ID.

        Returns
        -------
        list of int
            Tensor IDs aligned with the operator's variable list.
        """
        prev_output = self.tensor_id
        tensor_ids = []
        for ignore in ignored_list:
            if ignore:
                tensor_ids.append(prev_output)
            else:
                tensor_ids.append(self.new_tensor_id())
        return tensor_ids

    def convert_op(self, t_op: Union[TensorOperator, TensorExpression],
                    ts_config: Tuple[List[List[int]], List[int]]) -> Compute.OP:
        """Convert a high-level operator into the simulator's ``Compute.OP``.

        Parameters
        ----------
        t_op : TensorOperator or TensorExpression
            The operator to convert.
        ts_config : tuple of (temporal, spatial)
            Temporal and spatial tiling configuration.  ``temporal`` is a
            list-of-lists describing loop tiling factors; ``spatial`` is
            a list of spatial partition sizes.

        Returns
        -------
        Compute.OP
            Internal operator representation consumable by the cycle model.
        """
        temporal, spatial = ts_config
        if isinstance(t_op, TensorOperator):
            expr: TensorExpression = t_op.expr
        else:
            expr: TensorExpression = t_op

        is_ew = self.is_ew(expr)
        original_dim_lengths = expr.get_sub_op_shape(temporal, spatial)
        op = self.comp.convert_op_simple(dim_len=original_dim_lengths,
                                         variables=expr.variables,
                                         is_ew=is_ew,
                                         ignore_variables=expr.ignore_variables,
                                         tensor_id_list=self.fill_tensor_ids(expr.ignore_variables),
                                         op_type=expr.op_type,
                                        )
        return op
    
    def is_ew(self, t_op: Union[TensorOperator, TensorExpression]) -> bool:
        """Return True if the operator is elementwise (not a matmul or conv).

        op_type mapping: 4 = convolution, 5 = matrix multiply.
        Everything else (activations, norms, etc.) is treated as
        elementwise.
        """
        return not (t_op.op_type == 5 or t_op.op_type == 4)
    
    def compute_costs(self, fused_ops: List[List[Union[TensorOperator, TensorExpression]]],
                      ts_configs: List[Tuple[List[List[int]], List[int]]]) -> List[Tuple[int, int, int, int, int]]:
        """Compute cycle costs for every fused-operator group.

        Each sub-operator inside a fused group is first converted to the
        internal ``Compute.OP`` format, then the group is evaluated by
        ``Compute.get_total_cycle_for_fused_op``.

        Parameters
        ----------
        fused_ops : list of list of TensorOperator/TensorExpression
            Groups of fused operators (output of ``fuse_ops``).
        ts_configs : list of (temporal, spatial) tuples
            One tiling configuration per *individual* sub-operator,
            ordered to match the flattened sequence of all sub-ops
            across all fused groups.

        Returns
        -------
        list of (total, mm_comp_cyc, ew_comp_cyc, sram_r_cyc, sram_w_cyc)
            Integer cycle counts for each fused group.
        """
        compute_costs = []
        config_it = ts_configs.__iter__()

        # Extract temporal configs for the fused-op cycle model, which
        # needs the temporal tiling factors (index 0 of each ts_config).
        temp_cfg = [ts[0] for ts in ts_configs]
        op_base = 0
        for idx, op in enumerate(fused_ops):
            compute_op = [self.convert_op(sub_op, next(config_it)) for sub_op in op]
            comp_cost, mm_cyc, ew_cyc, sram_r_cyc, sram_w_cyc = \
                self.comp.get_total_cycle_for_fused_op(compute_op, temp_cfg[op_base:op_base + len(op)])
            compute_costs.append((int(ceil(comp_cost)), int(ceil(mm_cyc)),
                                  int(ceil(ew_cyc)), int(ceil(sram_r_cyc)),
                                  int(ceil(sram_w_cyc))))
            op_base += len(op)
        return compute_costs
    
    class Fuser:
        """Mutable state tracker for the operator-fusion algorithm.

        Maintains bookkeeping across the fusion loop: the current fused
        group boundaries (``start`` / ``end``), the count of matmuls
        already fused, and a three-state machine (``dot`` -> ``bmm1`` ->
        ``bmm2``) used to detect KV-cache write patterns in attention
        layers.

        Parameters
        ----------
        mm_fuse_limit : int
            Maximum number of matmul operators allowed in one fused group.
        sram_size : int
            Available SRAM capacity (bytes) used to decide whether an
            intermediate tensor must spill to DRAM.
        op_list : list of TensorOperator
            The full, un-fused operator list.
        """

        def __init__(self, mm_fuse_limit: int, sram_size: int, op_list: List[TensorOperator]):
            # State machine for KV-cache unconditional-write detection.
            # Cycles through: "dot" -> "bmm1" -> "bmm2" -> "dot" ...
            self.uncond_write_state = "dot"
            self.end = 0       # Exclusive end index of current fused group.
            self.start = 0     # Inclusive start index of current fused group.
            self.fused_mm = 0  # Number of matmuls in the current group.
            self.mm_fuse_limit = mm_fuse_limit
            self.sram_size = sram_size
            self.unfused_ops = op_list
            self.curr_fused_op = []
        def advance_mark_state(self):
            """Advance the KV-cache write-detection state machine.

            State transitions:  ``dot`` -> ``bmm1`` -> ``bmm2`` -> ``dot``
            (cycles once per attention head's Dot / BMM1 / BMM2 triple).
            """
            if self.uncond_write_state == "dot":
                self.uncond_write_state = "bmm1"
            elif self.uncond_write_state == "bmm1":
                self.uncond_write_state = "bmm2"
            elif self.uncond_write_state == "bmm2":
                self.uncond_write_state = "dot"
            else:
                print("Unrecognized state!")
                exit()
        def mark_unconditional_writes(self, fused_op: List[TensorOperator]):
            """Scan a fused group and mark KV-cache unconditional writes.

            Iterates over each sub-operator in the group and delegates
            to ``mark_unconditional_write`` for the state-machine check.
            """
            op_base = self.start
            for i, op in enumerate(fused_op):
                self.mark_unconditional_write(op, op_base + i)

        def mark_unconditional_write(self, op: TensorOperator, idx: int):
            """Conditionally flag an operator's output for unconditional DRAM write.

            In transformer attention layers the pattern is:
            ``Dot`` (QK^T), ``BatchMatMul`` (softmax * V), ``BatchMatMul``
            (output projection).  The Dot and first BMM produce KV-cache
            values that must be written to DRAM regardless of fusion,
            because they are reused across decoding steps.

            The method uses the ``uncond_write_state`` state machine to
            detect this (possibly non-contiguous) triple and sets
            ``op.is_unconditional_write = True`` for the Dot and first
            BMM.
            """
            target_op = ""
            if self.uncond_write_state == "dot":
                target_op = "Dot"
            elif self.uncond_write_state in ["bmm1", "bmm2"]:
                target_op = "BatchMatMul"

            if op.name.split("_")[-1] == target_op:
                if self.uncond_write_state in ["dot", "bmm1"]:
                    op.is_unconditional_write = True
                    print("MARKING unconditional", idx)
                self.advance_mark_state()

        def add_op_to_fused(self, op: TensorOperator, fused_op: List[TensorOperator], is_ew: bool):
            """Append *op* to the current fused group and update bookkeeping.

            Also checks whether the operator should be flagged as an
            unconditional DRAM write (KV-cache pattern).
            """
            self.mark_unconditional_write(op)
            fused_op.append(op)
            self.end += 1
            if not is_ew:
                self.fused_mm += 1

        def start_new_fused_op(self, is_first_op_ew: bool):
            """Begin a new fused-operator group starting at the current end index."""
            self.start = self.end
            self.curr_fused_op = [self.unfused_ops[self.start]]
            self.end += 1
            self.fused_mm = 1 if not is_first_op_ew else 0
    
    def fuse_ops(self, ops: List[TensorOperator], sram_size: int,
                 intermediate_sizes: List[np.ndarray],
                 method: str = "simple") -> List[List[TensorOperator]]:
        """Partition a flat operator list into groups of fused operators.

        Parameters
        ----------
        ops : list of TensorOperator
            Ordered sequence of operators to fuse.
        sram_size : int
            Available SRAM capacity in bytes.  Used to decide whether
            the intermediate tensor between two matmuls can stay on-chip
            (if it fits, the second matmul need not be fused).
        intermediate_sizes : list of np.ndarray
            Per-operator intermediate tensor shapes.  Element ``i``
            has shape ``[shape_array, ...]``; the product of
            ``intermediate_sizes[i][0]`` times
            ``ops[i].expr.num_byte_per_elem`` gives the byte count.
        method : str, optional
            ``"simple"`` -- greedy fusion: chain elementwise ops onto the
            nearest matmul; fuse up to two matmuls when the intermediate
            tensor overflows SRAM.
            ``"balanced"`` -- extends *simple* by additionally considering
            the execution-cycle ratio of matmul vs. elementwise work,
            fusing trailing elementwise ops only while doing so improves
            pipeline-stage balance.

        Returns
        -------
        list of list of TensorOperator
            Each inner list is one fused-operator group.
        """
        fuse_start = 0
        fuse_end = 0
        mm_fuse_limit = 2
        fused_ops = []
        fused_partitions = []
        fuser = self.Fuser(mm_fuse_limit, sram_size, ops)
        # Pre-compute intermediate size bytes to avoid repeated np.prod in fusion loops
        _int_size_bytes = [np.prod(intermediate_sizes[i][0]) * ops[i].expr.num_byte_per_elem
                           for i in range(len(ops))]
        if method == "simple":
            while fuse_end < len(ops):
                fuser.start = fuse_start
                fused_mm = 1 if not self.is_ew(ops[fuse_end]) else 0
                fused_op = [ops[fuse_start]]

                # Decide whether we need to fuse a second matmul:
                #   - If the leading op is elementwise, we must still
                #     reach at least one matmul (fuse_later_mm = True).
                #   - If the leading matmul's output exceeds SRAM, we
                #     fuse the next matmul so the intermediate stays
                #     on-chip and is consumed immediately.
                fuse_later_mm = (self.is_ew(ops[fuse_end])
                                 or _int_size_bytes[fuse_end] > sram_size)
                fuse_end += 1

                # --- Phase 1: fuse matmul(s) (and interleaved EW ops) ---
                while fuse_later_mm and fused_mm < mm_fuse_limit and fuse_end < len(ops):
                    if not self.is_ew(ops[fuse_end]):
                        # When hitting the first matmul in the chain,
                        # re-evaluate: if its output fits in SRAM, stop
                        # trying to fuse further matmuls.
                        if not fused_mm:
                            if _int_size_bytes[fuse_end] < sram_size:
                                fuse_later_mm = False
                        fused_mm += 1
                    fused_op.append(ops[fuse_end])
                    fuse_end += 1
                assert fused_mm > 0, "At least one matmul should be in our fused operator!"

                # --- Phase 2: absorb trailing elementwise ops ---
                while fuse_end < len(ops) and self.is_ew(ops[fuse_end]):
                    fused_op.append(ops[fuse_end])
                    fuse_end += 1

                # Finalise the fused group.
                fuse_start = fuse_end
                fused_ops.append(fused_op)
                fuser.mark_unconditional_writes(fused_op)
        elif method == "balanced":
            while fuse_end < len(ops):
                fuser.start = fuse_start
                mm_exec_cycles = 0  # Running matmul cycle total for balance check.
                ew_exec_cycles = 0  # Running elementwise cycle total.
                fused_mm = 1 if not self.is_ew(ops[fuse_end]) else 0
                fused_op = [ops[fuse_start]]
                fuse_later_mm = (self.is_ew(ops[fuse_end])
                                 or _int_size_bytes[fuse_end] > sram_size)
                fuse_end += 1

                # --- Phase 1: fuse matmul(s) and interleaved EW ops,
                #     same logic as "simple" but also accumulates cycle
                #     estimates for the balance heuristic. ---
                while fuse_end < len(ops) and fuse_later_mm and fused_mm < mm_fuse_limit:
                    if not self.is_ew(ops[fuse_end]):
                        if not fused_mm:
                            if _int_size_bytes[fuse_end] < sram_size:
                                fuse_later_mm = False
                        fused_mm += 1
                        mm_exec_cycles += self.comp.get_mm_compute_cycle_from_padded_tile(ops[fuse_end].dim_lengths)
                    fused_op.append(ops[fuse_end])
                    if self.is_ew(ops[fuse_end]):
                        ew_exec_cycles += self.comp.get_ew_compute_cycle_from_padded_tile(ops[fuse_end].dim_lengths)
                    fuse_end += 1

                # --- Phase 2: absorb trailing EW ops only while doing
                #     so reduces the gap between matmul and EW cycle
                #     totals (improves pipeline balance). ---
                while (fuse_end < len(ops) and self.is_ew(ops[fuse_end]) and
                       np.abs(mm_exec_cycles - ew_exec_cycles) >
                       np.abs(mm_exec_cycles - (ew_exec_cycles +
                              self.comp.get_ew_compute_cycle_from_padded_tile(ops[fuse_end].dim_lengths)))):
                    ew_exec_cycles += self.comp.get_ew_compute_cycle_from_padded_tile(ops[fuse_end].dim_lengths)
                    fused_op.append(ops[fuse_end])
                    fuse_end += 1

                # Finalise the fused group.
                fuse_start = fuse_end
                fused_ops.append(fused_op)
                fuser.mark_unconditional_writes(fused_op)

        else:
            raise NotImplementedError("Other operator fusion methods not yet implemented!")
        return fused_ops


