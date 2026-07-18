import re
import math
from typing import List, Dict, Set, Tuple, Any, Union
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tsim_components.tsim_analysis_lib import FusedOperatorExecLog, TYPE_HW_DICT



def parse_log_file(log_content: str, dram_bw_GBps: int, npu_freq_MHz: int) -> List[FusedOperatorExecLog]:
    """
    Parses the content of a log file to extract operator execution data and create
    FusedOperatorExecLog objects.

    Args:
        log_content: A string containing the log file data.
        dram_bw_GBps: DRAM bandwidth in GB/s.
        npu_freq_MHz: NPU frequency in MHz.

    Returns:
        A list of FusedOperatorExecLog objects.
    """
    op_logs = []
    lines = log_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        op_match = re.match(r"Operator (\d+)===================", line)
        if op_match:
            op_id = int(op_match.group(1))
            
            # Ensure there are enough lines for a full operator block
            if i + 4 >= len(lines):
                break

            start_times_line = lines[i+1]
            durations_line = lines[i+2]
            comp_util_line = lines[i+3]
            bytes_line = lines[i+4]

            # Regex for START TIMES
            start_times_match = re.search(
                r"START TIMES:\s+Ld:\s+(\d+)\s+->\s+broadcast\s+(\d+)\s+->\s+comp/sh\s+(\d+)\s+->\s+reduce\s+(\d+)(?:\s+->\s+St:\s+(\d+))?\s+->\s+fin\s+(\d+)",
                start_times_line
            )
            
            # Regex for DURATIONS
            durations_match = re.search(
                r"DURATIONS:\s+Ld:\s+(\d+)\s+->\s+broadcast\s+(\d+)\s+->\s+comp/sh\s+(\d+)/(\d+)\s+->\s+reduce\s+(\d+)(?:\s+->\s+St:\s+(\d+))?",
                durations_line
            )

            # Regex for Compute Utilization
            comp_util_match = re.search(
                r"Compute Utilization:\s+([\d.]+)\s+VU Utilization:\s+([\d.]+)",
                comp_util_line
            )

            # Regex for Bytes
            bytes_match = re.search(
                r"Write bytes:\s+(\d+)\s+Read bytes:\s+(\d+)",
                bytes_line
            )

            if not (start_times_match and durations_match and comp_util_match and bytes_match):
                i += 1
                continue

            # Extract start times
            t_dram_ld = int(start_times_match.group(1))
            t_bcast = int(start_times_match.group(2))
            t_comp_shift = int(start_times_match.group(3))
            t_reduce = int(start_times_match.group(4))
            t_finish = int(start_times_match.group(6))
            t_dram_st = int(start_times_match.group(5)) if start_times_match.group(5) else t_finish

            # Extract durations
            dram_ld_dur = int(durations_match.group(1))
            bcast_dur = int(durations_match.group(2))
            comp_sh_dur = int(durations_match.group(3))
            shift_dur = int(durations_match.group(4))
            reduce_dur = int(durations_match.group(5))
            dram_st_dur = int(durations_match.group(6)) if durations_match.group(6) else 0

            # Extract utilizations
            mm_util = float(comp_util_match.group(1))
            vu_util = float(comp_util_match.group(2))

            # Extract bytes
            dram_w_bytes = int(bytes_match.group(1))
            dram_r_bytes = int(bytes_match.group(2))

            # Prepare arguments for FusedOperatorExecLog constructor
            dram_rw_bytes = (dram_r_bytes, dram_w_bytes)
            dram_rw_dur = (dram_ld_dur, dram_st_dur)
            
            # These values are not in the log, so we use placeholders.
            # comp_dur: (comp_sh_dur, mm_dur, ew_dur, sram_r_dur, sram_w_dur)
            # mm_dur and ew_dur can be inferred if we assume peak performance, but it's better to set them from comp_sh_dur
            # as we don't have flop counts.
            comp_dur = (comp_sh_dur, comp_sh_dur, comp_sh_dur, comp_sh_dur, comp_sh_dur) 
            noc_op_dur = (bcast_dur, shift_dur, reduce_dur)
            energy = (0.0, 0.0, 0.0, {"sa": 0.0, "vu": 0.0, "noc": 0.0, "sram": 0.0, "dram": 0.0, "tsv": 0.0}) # Placeholder
            # comp_unit_stats: (mm_flop, mm_util, vu_flop, vu_util)
            # We have util, but not flop counts from the log.
            comp_unit_stats = (0, 0, 0, 0) # Placeholder for flops
            # comp_unit_stats = (0, mm_util, 0, vu_util) # Placeholder for flops
            hot_cold_vals = (0, 0) # Placeholder
            t_exec_enter = t_dram_ld # Assumption: execution space is entered at load time for non-preloaded ops.

            op_log = FusedOperatorExecLog(
                t_dram_ld=t_dram_ld,
                t_bcast=t_bcast,
                t_comp_shift=t_comp_shift,
                t_reduce=t_reduce,
                t_dram_st=t_dram_st,
                t_exec_enter=t_exec_enter,
                t_finish=t_finish,
                op_id=op_id,
                dram_rw_bytes=dram_rw_bytes,
                dram_rw_dur=dram_rw_dur,
                comp_dur=comp_dur,
                noc_op_dur=noc_op_dur,
                energy=energy,
                comp_unit_stats=comp_unit_stats,
                hot_cold_vals=hot_cold_vals,
                dram_bw_GBps=dram_bw_GBps,
                npu_freq_MHz=npu_freq_MHz
            )
            op_logs.append(op_log)
            i += 5 # Move to the next potential operator block
        else:
            i += 1
    return op_logs

def get_hw_unit_intervals(hardware_type: str, op_logs: List[FusedOperatorExecLog]) -> List[Tuple[int, int]]:
    """
    Returns the intervals during which the specified hardware type was active.
    """
    if hardware_type not in TYPE_HW_DICT.keys():
        raise ValueError(f"Unsupported hardware type: {hardware_type}. Choose one from: {TYPE_HW_DICT.keys()}.")

    intervals = []
    for op_log in op_logs:
        for hardware_unit in TYPE_HW_DICT[hardware_type]:
            if op_log.start_times[hardware_unit] < op_log.end_times[hardware_unit]:
                intervals.append((op_log.start_times[hardware_unit], op_log.end_times[hardware_unit]))
    return combine_intervals(intervals)

def combine_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Combines overlapping intervals into a single interval.
    """
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    combined = [intervals[0]]

    for current in intervals[1:]:
        last = combined[-1]
        if current[0] <= last[1]:  # Overlapping intervals
            combined[-1] = (last[0], max(last[1], current[1]))  # Merge intervals
        else:
            combined.append(current)  # No overlap, add new interval
    return combined

def reverse_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Reverses the intervals to represent the time when the hardware unit was NOT active.
    """
    if not intervals:
        return []

    reversed_intervals = []
    start_time = 0

    for interval in intervals:
        if interval[0] > start_time:
            reversed_intervals.append((start_time, interval[0]))
        start_time = interval[1]

    # Add the final interval until the end of the last operator
    reversed_intervals.append((start_time, intervals[-1][1]))

    return reversed_intervals

def intersect_intervals(intervals1: List[Tuple[int, int]], intervals2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Returns the intersection of two lists of intervals.
    """
    intersections = []
    i, j = 0, 0

    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]

        # Find the overlap
        start_overlap = max(start1, start2)
        end_overlap = min(end1, end2)

        if start_overlap < end_overlap:  # There is an overlap
            intersections.append((start_overlap, end_overlap))

        # Move to the next interval in the list that ends first
        if end1 < end2:
            i += 1
        else:
            j += 1

    return intersections

def get_hw_units_exclusive_time(op_logs: List[FusedOperatorExecLog], hardware_types: list["str"], excluded_types: list["str"]=None) -> int:
    """
    Returns the number of clock cycles that the specified hardware types (and ONLY those types) were active.
    """
    all_needed_intervals = []
    for hw_type in hardware_types:
        if hw_type not in TYPE_HW_DICT.keys():
            raise ValueError(f"Unsupported hardware type: {hw_type}. Choose one from: {TYPE_HW_DICT.keys()}.")
        intervals = get_hw_unit_intervals(hw_type, op_logs)
        all_needed_intervals += intervals
    all_needed_intervals = combine_intervals(all_needed_intervals)

    all_unneeded_intervals = []
    if excluded_types is None:
        excluded_types = [hw_type for hw_type in TYPE_HW_DICT.keys() if hw_type not in hardware_types]
    for hw_type in excluded_types:
        intervals = get_hw_unit_intervals(hw_type, op_logs)
        all_unneeded_intervals += intervals
    all_unneeded_intervals = combine_intervals(all_unneeded_intervals)
    all_unneeded_intervals_reversed = reverse_intervals(all_unneeded_intervals)

    exclusive_intervals = intersect_intervals(all_needed_intervals, all_unneeded_intervals_reversed)
    # Calculate exclusive time by subtracting unneeded time from needed time
    exclusive_time = sum(end - start for start, end in exclusive_intervals)
    return exclusive_time

def get_noc_time_simple(op_logs: List[FusedOperatorExecLog]) -> int:
    """
    Returns the total time that the NOC was active across all operators.
    """
    noc_time = 0
    for op_log in op_logs:
        noc_time += op_log.bcast_dur + op_log.reduce_dur + max(0, op_log.shift_dur - op_log.comp_dur)
    return noc_time

def get_total_time(op_logs: List[FusedOperatorExecLog]) -> int:
    """
    Returns the total time taken by all operators in the log.
    """
    return op_logs[-1].t_finish

def get_exposed_row_conflict_time(op_logs: List[FusedOperatorExecLog],
                                   dram_bw_GBps: int, npu_freq_MHz: int) -> float:
    """
    Returns the number of clock cycles attributable to DRAM row-buffer conflict
    overhead that were exposed (not hidden behind compute or NoC activity).

    For each exposed DRAM interval (cycles where DRAM was the only active
    hardware unit), the interval is intersected with each operator's DRAM
    read/write window.  The overlapping portion is weighted by that operator's
    conflict fraction:

        conflict_frac = max(0, dram_dur - burst_floor) / dram_dur

    where burst_floor = ceil(chip_total_bytes / chip_bytes_per_cycle) is the
    minimum cycles needed at peak bandwidth with no row conflicts.

    dram_r_bytes / dram_w_bytes in the log are chip-wide totals; dram_ld_dur /
    dram_st_dur are per-core cycles.  chip_bytes_per_cycle = dram_bw_GBps *
    2^30 / (npu_freq_MHz * 1e6) converts both to the same scale.
    """
    chip_bytes_per_cycle = dram_bw_GBps * (2 ** 30) / (npu_freq_MHz * 1e6)

    # Global exposed DRAM intervals
    dram_intervals = get_hw_unit_intervals("dram", op_logs)
    other_intervals = []
    for hw_type in [t for t in TYPE_HW_DICT if t != "dram"]:
        other_intervals += get_hw_unit_intervals(hw_type, op_logs)
    other_intervals = combine_intervals(other_intervals)
    other_complement = reverse_intervals(other_intervals)
    exposed_intervals = intersect_intervals(dram_intervals, other_complement)

    if not exposed_intervals:
        return 0.0

    # Per-operator DRAM segments with their conflict fraction
    op_segs = []  # (seg_start, seg_end, conflict_frac)
    for op in op_logs:
        if op.dram_ld_dur > 0:
            burst_r = math.ceil(op.dram_r_bytes / chip_bytes_per_cycle)
            frac = max(0.0, (op.dram_ld_dur - burst_r) / op.dram_ld_dur)
            op_segs.append((op.t_dram_ld_start,
                            op.t_dram_ld_start + op.dram_ld_dur, frac))
        if op.dram_st_dur > 0:
            burst_w = math.ceil(op.dram_w_bytes / chip_bytes_per_cycle)
            frac = max(0.0, (op.dram_st_dur - burst_w) / op.dram_st_dur)
            op_segs.append((op.t_dram_st_start,
                            op.t_dram_st_start + op.dram_st_dur, frac))

    # Weight each exposed interval by the conflict fraction of its owning segment
    exposed_conflict = 0.0
    for exp_start, exp_end in exposed_intervals:
        for seg_start, seg_end, frac in op_segs:
            overlap = min(exp_end, seg_end) - max(exp_start, seg_start)
            if overlap > 0:
                exposed_conflict += overlap * frac

    return exposed_conflict
