from typing import List, Dict, Set, Tuple, Any, Union
import math
HW_UNITS = ["dram_r", "dram_w", "comp_sram_r", "comp_sram_w", "comp_sa", "comp_vu", "comp", "noc"]
HW_COLORS = {
    "dram_r": "viridis",
    "dram_w" : "inferno",
    "comp_sram_r" : "teal",
    "comp_sram_w" : "lime",
    "comp_sa" : "cividis",
    "comp_vu" : "plasma",
    "comp" : "lightcoral",
    "noc" : "darkblue",
}
HW_TYPE_DICT = {
    "dram_r" : "dram",
    "noc_bcast_sh" : "noc",
    "dram_w" : "dram",
    "comp" : "comp",
    "comp_sram_r" : "comp",
    "comp_sram_w" : "comp",
    "comp_vu" : "comp",
    "comp_sa" : "comp",
    "noc_reduce" : "noc"
}
TYPE_HW_DICT = {
    "dram" : [hw for hw, type in HW_TYPE_DICT.items() if type == "dram"],
    "noc" : [hw for hw, type in HW_TYPE_DICT.items() if type == "noc"],
    "comp" : [hw for hw, type in HW_TYPE_DICT.items() if type == "comp"]
}

class OverlapInterval:
    '''
    hw units:

    dram_r, dram_w, noc, comp (TODO: comp_sa, comp_vu, comp_sram)
    '''

    def __init__(self, active_units: Union[set[str], list[str]], t_start: int, t_end: int, op_id: int = -1, exe_used: int = -1, power_W: float = 0.0):
        self.op_id = op_id
        self.active_units = set(active_units)
        self.t_end = t_end
        self.t_start = t_start
        self.exe_used = exe_used # exe space occupied during this interval (B). -1 means invalid/haven't computed yet.
        self.power_W = power_W # Average dynamic power during this interval (W). 0.0 means invalid/haven't computed yet.
        assert(isinstance(t_end, int)), "t_end should be int."
        assert(isinstance(t_start, int)), "t_start should be int. "

    def __str__(self):
        info = [f"Interval: cycles =[{self.t_start}, {self.t_end}] units = {self.active_units}",
                f"Dynamic Power (W): {self.power_W}\n",]
        return ("\n").join(info)

    def intersect(self, other_interval):
        '''
        Intersect self with another OverlapInterval. NOT USED RN
        '''

        op_id = min(self.op_id, other_interval.op_id)
        active_units = self.active_units.union(other_interval.active_units)
        t_end = min(self.t_end, other_interval.t_end)
        t_start = max(self.t_start, other_interval.t_start)
        intersect_interval = OverlapInterval(active_units, t_start, t_end, op_id)
        return intersect_interval

    def sub(self, interval_to_sub):
        '''
        2 cases:
        selfstart < other start:
        self    ->[xxxxxx]
        other   ->    [xxxxxxx]

        selfstart >= other start
        self    ->        [xxxxxxxx]
        other   ->[xxxxxxxxxx]
        '''
        #case 1
        if(self.t_start < interval_to_sub.t_start):
            self.t_end = interval_to_sub.t_start
        #case 2
        else:
            self.t_start = interval_to_sub.t_end

    def is_empty(self):
        return self.t_end == self.t_start
from collections import deque
class HardwareExecutionState:

    class Preload:
        def __init__(self, preload_size, preload_avail_time, preload_free_time):
            '''
            Fields:
                Avail_time: The time at which preload space will be available for this preload
                Free_time: The time at which the preload space will be freed (by the operator moving to execution space)
            '''
            self.preload_size = preload_size
            self.preload_avail_time = preload_avail_time
            self.preload_free_time = preload_free_time
        def __str__(self):
            return  f"Size = {self.preload_size} avail_time = {self.preload_avail_time} free_time = {self.preload_free_time}"
        def set_preload_leave(self, preload_free_time):
            self.preload_free_time = preload_free_time

    class ExecOp:
        def __init__(self, op_size, op_avail_time, op_free_time, op_idx):
            '''
            Fields: (unused for now)
                Avail_time: The time at which execution space will be available for this operator
                Free_time: The time at which the execution space will be freed (by the operator finishing)
            '''
            self.exe_size = op_size
            self.exe_avail_time = op_avail_time
            self.exe_free_time = op_free_time
            self.op_idx = op_idx
        def __str__(self):
            return f"OP_{self.op_idx}: Size = {self.exe_size} avail_time = {self.exe_avail_time} free_time = {self.exe_free_time}\n"
        def set_leave_time(self, exec_free_time):
            self.exe_free_time = exec_free_time

    def __init__(self, sram_per_core, exe_sram_per_core):
        # Hardware constants
        self.total_sram_per_core_B = sram_per_core
        self.exe_sram_per_core_B = exe_sram_per_core
        self.preload_space_per_core_B = sram_per_core - exe_sram_per_core

        # Counters. Track when availability of each hardware unit.
        self.dram_r_next_avail_cycle = 0
        self.dram_w_next_avail_cycle = 0
        self.noc_next_avail_cycle = 0
        self.next_preload_candidate = 1
        self.remaining_preload_space: int = self.preload_space_per_core_B
        self.remaining_execution_space: int = self.exe_sram_per_core_B
        self.reserved_preloads: Dict[int, HardwareExecutionState.Preload] = {}
        # Current list of preloads w/ reserved SRAM: maps {index: preload info}.
        self.exe_space_ops: deque = deque()
        # Current operators w/ reserved SRAM: maps {index: preload info}.

    def print_dicts(self):
        preload_str = (",").join([f"{idx}: {str(preload)}" for idx, preload in self.reserved_preloads.items()])
        exe_str = (",").join([f"{str(exe)}" for exe in self.exe_space_ops])
        occupation_str = f"\nPreload space: {self.remaining_preload_space}/{self.preload_space_per_core_B} Execution space rem: {self.remaining_execution_space}/{self.exe_sram_per_core_B}\n"
        return f"Ops in Preload space: {preload_str}\n\
        Ops in Exe space: {exe_str}\n{occupation_str}"

    def __str__(self):
        intro_str = f"HardwareExecutionState counters:\n"
        counter_str = f"    DRAM read next available: {self.dram_r_next_avail_cycle}\n\
    DRAM write next available: {self.dram_w_next_avail_cycle}\n\
    NoC next available: {self.noc_next_avail_cycle}\n\
    Remaining preload space: {self.remaining_preload_space}\n\
    Remaining execution space: {self.remaining_execution_space}\n\
    {self.print_dicts()}"
        return (intro_str + counter_str)

    def summary(self):
        space = f"Remaining preload: {self.remaining_preload_space} exe: {self.remaining_execution_space}\n"
        queue_contents = f"Preload queue: {[i for i, p in self.reserved_preloads.items()]} \nExecution queue: {[e.op_idx for e in self.exe_space_ops]}\n"
        return (space + queue_contents)

    def sanity_check(self):
        '''
        Verify consistency of counters with preload/exe queues. May extend in future.
        '''
        assert(self.remaining_preload_space == self.preload_space_per_core_B - sum([p.preload_size for p in self.reserved_preloads.values()])), \
        f"Preload space counter inconsistent with preload queue!: {self.remaining_preload_space} != {self.preload_space_per_core_B - sum([p for p in self.reserved_preloads.values()])}"
        assert(self.remaining_execution_space == self.exe_sram_per_core_B - sum([op.exe_size for op in self.exe_space_ops])), \
        f"Exe space counter inconsistent with exe queue!: {self.remaining_execution_space} != {self.exe_sram_per_core_B - sum([op.exe_size for op in self.exe_space_ops])}"

    def add_preload(self, hot_cold_table, preload_time):
        '''
        Helper function -- adds single preload
        '''
        assert(self.remaining_preload_space >= hot_cold_table[self.next_preload_candidate][1]), "Cold size does not fit in preload!"
        self.reserved_preloads[self.next_preload_candidate] = self.Preload(hot_cold_table[self.next_preload_candidate][1], preload_time, 0)
        # print("Preload table: ", self.reserved_preloads.items() )
        self.remaining_preload_space -= hot_cold_table[self.next_preload_candidate][1]
        self.next_preload_candidate += 1

    def reserve_execution(self, time, hot_cold_table, op_idx):
        # Reserve execution space for the given operator.
        assert(self.remaining_execution_space >= hot_cold_table[op_idx][0]), \
        f"We must have enough space in execution before we can reserve! {self.remaining_execution_space} >= {hot_cold_table[op_idx][0]}"
        self.exe_space_ops.append(self.ExecOp(hot_cold_table[op_idx][0], time, 0, op_idx))
        self.remaining_execution_space -= hot_cold_table[op_idx][0]

    def set_execution_finish(self, time, op_idx):
        # Set the finish time for this operator. We cannot directly free it because of the strange way time is implemented in
        # the simulation -- overlapping of operators means that time can go backwards when we move to the next operator.
        assert(self.exe_space_ops[-1].op_idx == op_idx), f"unexpected op_idx {op_idx} in exe space! {self.exe_space_ops[-1].op_idx}"
        self.exe_space_ops[-1].set_leave_time(time)

    # def free_executions(self, time):
    #     # Clean up operators once they're finished.
    #     # assert(op_idx in self.exe_space_ops), "Operator should be in execution space before it is freed!"
    #     print("Checking free exe space at time ", time)
    #     ops_to_free = []
    #     for idx, op_exe in self.exe_space_ops.items():
    #         if(op_exe.exe_free_time and op_exe.exe_free_time <= time):
    #             print(f"Freeing execution space for {idx} at time {time}")
    #             ops_to_free.append(idx)

        # Use a separate loop to avoid mutating data structure while iterating through it.
        # for op_idx in ops_to_free:
        #     self.remaining_execution_space += self.exe_space_ops[op_idx].exe_size
        #     del self.exe_space_ops[op_idx]

    def get_exec_next_avail(self, exe_amt):
        # Get earliest time at which exe_amt will be available.
        # Use remaining exe space and exe_ops dict to figure this out.
        # Return 0 if remaining space is already sufficient.
        required_space = exe_amt - self.remaining_execution_space
        time_avail = 0
        prev_time = -1
        prev_op_idx = -1
        # print("exe space ops = ", self.exe_space_ops)
        while(required_space > 0):
            op_exe = self.exe_space_ops.popleft()
            required_space -= op_exe.exe_size
            self.remaining_execution_space += op_exe.exe_size
            time_avail = max(op_exe.exe_free_time, time_avail)
            # prev_op_idx = op_idx
            # prev_time = op_exe.exe_free_time
        # for op_idx, op_exe in self.exe_space_ops.items():
        #     assert(prev_op_idx < op_idx), "Operators should be listed in strictly ascending order!"
        #     assert(prev_time < op_exe.exe_free_time), "Operators should be listed in strictly ascending order!"
        #     if(required_space <= 0):
        #         break
        #     assert(op_exe.exe_free_time), f"Ops should have free times set after perform_op! {op_idx} {op_exe.exe_free_time}"
        #     required_space -= op_exe.exe_size
        #     time_avail = op_exe.exe_free_time
        #     prev_op_idx = op_idx
        #     prev_time = op_exe.exe_free_time

        # assert(required_space <= 0), "We should definitely have enough space once everything is freed!"
        # print(f"Exe next avail time = {time_avail} for {exe_amt} bytes")
        return time_avail

    def fill_preload_buffer(self, curr_op_idx, hot_cold_table, preload_time):
        '''
        Helper function -- fills preload space to capacity.
        '''
        self.next_preload_candidate = max(self.next_preload_candidate, curr_op_idx + 1)
        # print(f"Filling preload_buffer with {self.remaining_preload_space} preload space.")
        # if self.next_preload_candidate < len(hot_cold_table):
        #     print(f"First preload candidate = {self.next_preload_candidate} with cold size {hot_cold_table[self.next_preload_candidate][1]}")
        while (self.next_preload_candidate < len(hot_cold_table) and
                self.remaining_preload_space >= hot_cold_table[self.next_preload_candidate][1]):
            # put as many candidates in preload space as possible!
            # print(f"Adding preload: {self.next_preload_candidate}")
            self.add_preload(hot_cold_table, preload_time)


    def update_preloads(self, check_time: int, hot_cold_table: Dict[int, Tuple[int, int]], op_idx: int):
        '''
        Update preload space when operators move to execution space and add more preloads when possible.
        '''
        to_free = []
        for (idx, preload) in self.reserved_preloads.items(): # Free any preloads that have entered execution space.
            # print(f"Free time {idx}, {preload.preload_free_time}")
            if(preload.preload_free_time and preload.preload_free_time <= check_time):
                to_free.append(idx)
        # print(f"Update at time {check_time}: Freeing preloads: {to_free}")
        for free_idx in to_free:
            del self.reserved_preloads[free_idx]
            self.remaining_preload_space += hot_cold_table[free_idx][1]
        self.fill_preload_buffer(op_idx, hot_cold_table, check_time) # Add more preloads

    def perform_op(self, dram_r_start: int, dram_r_duration: int, noc_bcast_duration: int, comp_shift_duration: int,
                    noc_reduce_duration: int, dram_w_duration: int, op_idx: int, hot_cold_table: List[Tuple[int, int]],
                    exe_next_avail: int, overlap_bcast_dram_read: bool = False) -> Tuple[int, int, int, int, int, int]:
        '''
        Log the effect of performing an operation with the given hardware unit occupancy duration and start time (dram_r_start).
        Computing the next available cycle of dram_r availability can be done by adding the start and duration of dram read.

        Assumption: Operators are loaded and executed strictly in-order.

        However, the DRAM read has knock-on effects for Noc availability since the noc can only start broadcasting once the data is loaded in some cases.
        We take the max of the DRAM read's finish time and the Noc's availability to account for this.
        A similar situation is true for the DRAM write -- it can only happen once the NoC has finished reducing the results. Thus, we take
        another max here to account for the possibility.

        We don't need to directly account for compute since compute time is always < noc time and compute can be overlapped with noc.

        At post-read time and noc start time, we check for new preload opportunities and free up unused preload space.
        Operations each must proceed in this order: DRAM read (to SRAM), Noc broadcast, Compute and shift, Noc reduce, DRAM write (if no immediate dependency)
        At most one operation can use the DRAM read, Noc, or DRAM write hardware at any give time.

        3 Main stages:

        DRAM read: read hardware must be free and
        we must have space in either preload or execution space (if this op is next in line for exec). If there is no space, we must wait for it to open up

        Noc broadcast, comp shift, noc reduce: we must have space in execution memory to begin the broadcast, DRAM read must be complete, and Noc must be available.
        For ops that don't write to DRAM, the exection space is free after reduce. No special consideration is needed since this coincides with noc_next_avail.

        DRAM write: write hardware must be free. Execution space is only free after write is finished (for writing ops). In this case, we must consider the possibility
        of the next op entering NoC while the current op finishes its write.

        TODO: current assumption: NOC is reserved for entire compute-shift operation, even if compute is longer than shift.

        dram_r_start should be chosen to comply with all rules.
        '''
        # print(f"Execution: op {op_idx}")
        # self.free_executions(dram_r_start) # Free finished ops before executing new one.
        self.dram_r_next_avail_cycle = dram_r_start + dram_r_duration # Now reflects end of DRAM read for op(op_idx)
        self.update_preloads(self.dram_r_next_avail_cycle, hot_cold_table, op_idx) # If preload space is available, we can start preloading as soon as dram is ready
        if overlap_bcast_dram_read:
            # broadcast prerequisites: noc available and must end concurrently with dram read at the earliest
            noc_bcast_start = max(self.noc_next_avail_cycle, exe_next_avail, dram_r_start + dram_r_duration - noc_bcast_duration)
        else:
            # broadcast prerequisites: noc bcast can start once noc is available and dram read is finished.
            noc_bcast_start = max(self.noc_next_avail_cycle, exe_next_avail, self.dram_r_next_avail_cycle)


        # If the previous operation wrote to DRAM, consider possibility of exec space contention with
        # prev. op(op_idx-1) during op_idx-1 DRAM write, op_idx NOC
        # if(self.dram_w_next_avail_cycle > self.noc_next_avail_cycle and
        #    self.remaining_execution_space - hot_cold_table[op_idx - 1][0] < hot_cold_table[op_idx][0]):
        #     #If there is insufficient space, we can't overlap and must wait until the previous operator leaves exec space (end of dram write).
        #     noc_bcast_start = max(noc_bcast_start, self.dram_w_next_avail_cycle)

        # self.free_executions(self.dram_r_next_avail_cycle) # Earliest time the space will be needed for next op is once dram is available again.

        if op_idx in self.reserved_preloads:
            self.reserved_preloads[op_idx].set_preload_leave(noc_bcast_start)

        if(self.dram_r_next_avail_cycle != noc_bcast_start):
            self.update_preloads(noc_bcast_start, hot_cold_table, op_idx) # If preload space was not available earlier, it may become available once current op enters exe space
            # self.free_executions(noc_bcast_start)
        if(op_idx in self.reserved_preloads): # Preloaded ops only need to move to exe space once computation starts
            t_exec_space_enter = noc_bcast_start
        else: # Non-preloaded ops need exe space before they can start loading data.
            t_exec_space_enter = dram_r_start
        self.reserve_execution(t_exec_space_enter, hot_cold_table, op_idx) # Reserve execution space for the current op
        comp_shift_start = noc_bcast_start + noc_bcast_duration
        noc_reduce_start = comp_shift_start + comp_shift_duration

        self.noc_next_avail_cycle = noc_bcast_start + sum([noc_bcast_duration, comp_shift_duration, noc_reduce_duration]) #Now reflects end of noc time op(op_idx)
        dram_w_start = max(self.dram_w_next_avail_cycle, self.noc_next_avail_cycle)
        if(dram_w_duration > 0):
            finish_time = dram_w_start + dram_w_duration # exe space freed here
            self.dram_w_next_avail_cycle = dram_w_start + dram_w_duration # Now reflects end of DRAM write for op(op_idx)
        else:
            finish_time = noc_reduce_start + noc_reduce_duration

        self.set_execution_finish(finish_time, op_idx)
        # Now that we know when the noc broadcast starts, we know when the operator will leave preload space, if it was preloaded.
        return (dram_r_start, noc_bcast_start, comp_shift_start, noc_reduce_start, dram_w_start, t_exec_space_enter, finish_time)

class FusedOperatorExecLog:
    '''
    params:
        flops: sa, vu flops in the form of (sa_flops, vu_flops)

    Fields: -- All times are in cycles!
        xxx_dur: The duration of xxx in cycles.
        t_bcast_start: Start time of noc broadcast.
        t_comp_shift_start: Start time of comp + shift operations
        t_reduce_start: Start time of noc reduction
        t_dram_st_start: Start time of writeback to DRAM (when applicable)
        t_finish: Time at which the operator is completely finished and execution space is freed.
        op_id: int corresponding to operator id
    '''
    def __init__(self, t_dram_ld: int, t_bcast: int, t_comp_shift: int, t_reduce: int, t_dram_st: int, t_exec_enter:int, t_finish: int, op_id: int,
                    dram_rw_bytes: Tuple[int, int], dram_rw_dur: Tuple[int, int],
                    comp_dur: Tuple[int, int, int, int, int], noc_op_dur: Tuple[int, int, int],
                    energy: Tuple[float, float, float], comp_unit_stats: Tuple[int, float, int, float],
                    hot_cold_vals: Tuple[int, int], dram_bw_GBps: int, npu_freq_MHz: int,
                    spatial_meta: Dict[str, Any] = None):
        self.t_dram_ld_start    = int(t_dram_ld) #NOTE: This value is used as the execution start for ops in many places. Keep this in mind before changing.
        self.dram_ld_dur        = int(dram_rw_dur[0])
        self.t_bcast_start      = int(t_bcast)
        self.bcast_dur          = int(noc_op_dur[0])
        self.t_comp_shift_start = int(t_comp_shift)
        self.mm_dur             = int(comp_dur[1])
        self.ew_dur             = int(comp_dur[2])
        self.sram_r_dur         = int(comp_dur[3])
        self.sram_w_dur         = int(comp_dur[4])
        self.shift_dur          = int(noc_op_dur[1])
        self.comp_dur           = int(comp_dur[0])
        self.comp_sh_dur        = int(max(comp_dur[0], noc_op_dur[1]))
        self.reduce_dur         = int(noc_op_dur[2])
        self.t_reduce_start     = int(t_reduce)
        self.t_dram_st_start    = int(t_dram_st)
        self.dram_st_dur        = int(dram_rw_dur[1])
        self.t_finish           = int(t_finish)
        self.op_id              = int(op_id)
        self.exec_dur           = self.t_finish - self.t_dram_ld_start
        self.dram_w_bytes       = dram_rw_bytes[1] # Total bytes over all cores -- access list produces bytes per core, so it gets multiplied by num cores later.
        self.dram_r_bytes       = dram_rw_bytes[0]
        self.mm_flop_per_core   = comp_unit_stats[0] # per core.
        self.vu_flop_per_core   = comp_unit_stats[2] # TODO: check if this is correct. This is the number of flops per core.

        self.power_W            = (energy[0] / 1e12) / (self.exec_dur / (npu_freq_MHz * 1e6)) # Average power in Watts during the execution of this op.
        self.mm_util            = comp_unit_stats[1] / self.mm_dur if self.mm_dur else 0# Util = Ideal exec / Actual exec = num_ops / (dur * peak flops)
        self.vu_util            = comp_unit_stats[3] / self.comp_dur if self.ew_dur else 0
        self.energy_total       = energy[0] # Dynamic energy numbers (including below ) in pJ
        self.energy_compute     = energy[1]
        self.energy_sss         = energy[2]
        self.energy_sa          = energy[3]["sa"]
        self.energy_vu          = energy[3]["vu"]
        self.energy_noc         = energy[3]["noc"]
        self.energy_sram        = energy[3]["sram"]
        self.energy_dram        = energy[3]["dram"]
        self.energy_tsv         = energy[3]["tsv"]
        # self.energy_ssi         = energy[3]
        # self.energy_dram        = energy[4]
        self.dram_util          = self.get_dram_util_intensity(dram_bw_GBps, npu_freq_MHz)
        assert math.isclose(self.energy_total, (self.energy_sa + self.energy_vu + self.energy_noc + self.energy_sram + self.energy_dram + self.energy_tsv), rel_tol=1e-5), \
            f"Energy breakdown does not sum to total energy! total: {self.energy_total}"
        self.t_exec_enter = t_exec_enter
        self.hot_size = hot_cold_vals[0]    # Size of the hot data in bytes
        self.cold_size = hot_cold_vals[1]   # Size of the cold data in bytes
        # Optional TSIM-side logical spatial attribution.  New runs populate
        # this with compact per-stage ID sets; old pickles simply lack data.
        self.spatial_attribution = {}
        self.spatial_meta = spatial_meta or {}

        # Start and end times contain the times over which individual steps are running,
        # self.intervals contains intervals such that over each interval, the interval's active units are all running.
        # A new interval is created when the active units change.
        self.start_times = {
            "dram_r" : self.t_dram_ld_start,
            "noc_bcast_sh" : self.t_bcast_start,
            "dram_w" : self.t_dram_st_start,
            "comp" : self.t_comp_shift_start,
            "comp_sram_r" : self.t_comp_shift_start, #TODO
            "comp_sram_w" : self.t_comp_shift_start, #TODO Verify when W can start (THIS IS TMP!)
            "comp_vu" : self.t_comp_shift_start, #TODO
            "comp_sa" : self.t_comp_shift_start, #TODO
            "noc_reduce" : self.t_reduce_start
        }

        self.end_times = {
            "dram_r" : self.t_dram_ld_start + self.dram_ld_dur,
            "noc_bcast_sh" : self.t_bcast_start + self.bcast_dur + self.shift_dur,
            "dram_w" : self.t_dram_st_start + self.dram_st_dur,
            "comp" : self.t_comp_shift_start + self.comp_sh_dur,
            "comp_sram_r" : self.start_times["comp_sram_r"] + self.sram_r_dur,
            "comp_sram_w" : self.start_times["comp_sram_w"] + self.sram_w_dur,
            "comp_vu" : self.start_times["comp_vu"] + self.ew_dur,
            "comp_sa" : self.start_times["comp_sa"] + self.mm_dur,
            "noc_reduce" : self.t_reduce_start + self.reduce_dur
        }

        #Translates pipeline stage to the active hw unit(s) during that stage
        self.stage_units = {
            "dram_r" : ["dram_r"],
            "noc_bcast_sh" : ["noc"],
            "dram_w" : ["dram_w"],
            "comp" : ["comp"],
            "comp_sram_r" :["comp_sram_r"],
            "comp_sram_w" : ["comp_sram_w"],
            "comp_vu" : ["comp_vu"],
            "comp_sa" : ["comp_sa"],
            "noc_reduce" : ["noc"]
        }

        self.intervals = self.get_intervals()
        self.sanity_check()

    def set_spatial_attribution(self, attribution: Dict[str, Any]):
        self.spatial_attribution = attribution

    def get_dram_util_intensity(self, dram_bw_GBps, npu_freq_MHz) -> Tuple[float, float]:
        # Compute the dram bandwidth utilization as a fraction of peak bw for r, w. TODO check cycle computation to make sure this aligns.
        assert(isinstance(dram_bw_GBps, int))
        assert(isinstance(npu_freq_MHz, int))
        assert(dram_bw_GBps > 0), "DRAM bandwidth should be positive!"
        assert(npu_freq_MHz > 0), "NPU must be running!"
        # print(f"W bytes: {dram_bw_GBps * 2 ** 30 * self.dram_st_dur / (npu_freq_MHz * 1e6)}")
        if(self.dram_ld_dur):
            read_intensity = self.dram_r_bytes / (dram_bw_GBps * 2 ** 30 * self.dram_ld_dur / (npu_freq_MHz * 1e6))
        else:
            read_intensity = 0
        if(self.dram_st_dur):
            write_intensity = self.dram_w_bytes / (dram_bw_GBps * 2 ** 30 * self.dram_st_dur / (npu_freq_MHz * 1e6))
        else:
            write_intensity = 0
        # assert(read_intensity >= 0 and read_intensity <= 1), "Read intensity should be a float between 0 and 1!"
        # assert(write_intensity >= 0 and write_intensity <= 1), "Write intensity should be a float between 0 and 1!"
        assert(read_intensity >= 0), "Read intensity should be a float between 0 and 1!"
        assert(write_intensity >= 0), "Write intensity should be a float between 0 and 1!"
        return read_intensity, write_intensity

    def sanity_check(self):
        if self.dram_st_dur > 0:
            assert(self.t_dram_st_start + self.dram_st_dur == self.t_finish), f"Sanity check: finish time equal to end of write for writing ops."
        else:
            assert(self.dram_w_bytes == 0), f"We should not write bytes if dram write takes no time!"
            assert(self.t_reduce_start + self.reduce_dur == self.t_finish), \
            f"Sanity check: Finish time equal to end of reduce for non-writing ops: {self.reduce_dur + self.reduce_dur} vs {self.t_finish}"
        assert(self.mm_util >= 0 and self.mm_util <= 1.2), f"MM utilization should be a float between 0 and 1.2! {self.mm_util} flop={self.mm_flop_per_core} dur={self.mm_dur} op={str(self)}"
        assert(self.vu_util >= 0 and self.vu_util <= 1.2), f"VU utilization should be a float between 0 and 1.2! {self.vu_util} flop={self.vu_flop_per_core} dur={self.ew_dur} op={str(self)}"
    def __str__(self):
        '''
        Returns a string representation of the operator execution log.
        '''
        dram_w_string = f" -> St: {self.t_dram_st_start}" if  (self.t_dram_st_start != self.t_finish) else ""
        dram_w_string_dur = f" -> St: {self.dram_st_dur}" if (self.t_dram_st_start != self.t_finish) else ""
        op_str = f"Operator {self.op_id}===================\n"
        event_str = f"START TIMES:  Ld: {self.t_dram_ld_start} -> broadcast {self.t_bcast_start} -> comp/sh {self.t_comp_shift_start} -> reduce {self.t_reduce_start}{dram_w_string} -> fin {self.t_finish}\n"
        dur_str = f"DURATIONS:    Ld: {self.dram_ld_dur} -> broadcast {self.bcast_dur} -> comp/sh {self.comp_sh_dur}/{self.shift_dur} -> reduce {self.reduce_dur}{dram_w_string_dur}\n"
        traffic_str = f"Write bytes: {self.dram_w_bytes} Read bytes: {self.dram_r_bytes}\n"
        comp_util_str= f"Compute Utilization: {self.mm_util} VU Utilization: {self.vu_util}\n"
        power_str = f"Average Power (W): {self.power_W}\n"
        interval_str = ""
        # interval_str = (" ").join([str(ival) for ival in self.intervals])
        return (op_str + event_str + dur_str + interval_str + comp_util_str + traffic_str + power_str)

    def __repr__(self):
        '''
        Returns a detailed string representation of the operator execution log. TODO: make it more detailed.
        '''
        dram_w_string = f" -> St: {self.t_dram_st_start}" if  (self.t_dram_st_start != self.t_finish) else ""
        dram_w_string_dur = f" -> St: {self.dram_st_dur}" if (self.t_dram_st_start != self.t_finish) else ""
        op_str = f"Operator {self.op_id}===================\n"
        energy_str = f"Energy: Total: {self.energy_total}, Compute: {self.energy_compute}, SyncShiftShuffle: {self.energy_sss}\n"
        event_str = f"START TIMES:  Ld: {self.t_dram_ld_start} -> broadcast {self.t_bcast_start} -> comp/sh {self.t_comp_shift_start} -> reduce {self.t_reduce_start}{dram_w_string} -> fin {self.t_finish}\n"
        dur_str = f"DURATIONS:    Ld: {self.dram_ld_dur} -> broadcast {self.bcast_dur} -> comp/sh {self.comp_sh_dur}/{self.shift_dur} -> reduce {self.reduce_dur}{dram_w_string_dur}\n"
        return (op_str + energy_str + event_str + dur_str)

    def get_intervals(self) -> List[OverlapInterval]:

        '''
        Get the time intervals for each hw unit during which the operator is alive
        |DRAM(R)xxxx|   |NoCxx|Noc+Compute|NoCx|        |DRAM(W)xxx|
        Returns a list of OverlapInterval objects with op_id = exec_log.op_id. Empty intervals
        are included too -- the corresponding OverlapInterval just has empty list/set as the hw_unit.

        Returns a list of OverlapInterval objects.

        Note: intervals must be sorted by start time, so append individual intervals accordingly
        or develop a sorting function.

        All the relationships between hardware units has been handled in perform_op in DNNProgram,
        so we can just use the start and end time of each stage to determine the active hardware units
        in each interval.

        (OUTDATED) Considerations/assumptions:
        * DRAM read and broadcast may overlap OR there may be a gap between them
        * Compute/sh always starts immediately after broadcast is finished, and reduce immediately follows compute/sh.
        * Compute and shift durations may be different, but compute and shift will both start at the same time.
        * DRAM write follows reduce, but there may be a gap.
        * Intervals with length 0 will be removed -- the function will only return those with positive duration
        '''
        intervals_raw = []
        intervals = []
        curr_time = self.t_dram_ld_start
        while curr_time < self.t_finish:
            hw_units = set() # Reset interval length and active hw units for each new interval.
            interval_hw, interval_length = self.get_active_units_start_end(curr_time)
            # print(f"union {hw_units} with {interval_hw}")
            hw_units = hw_units.union(interval_hw)
            #Remaining time becomes the time left on the shortest operation in the overlap. This ensures we don't miss a combination change
            assert(interval_length >= 0), "Remaining time should be non-negative!"
            intervals_raw.append(OverlapInterval(hw_units, curr_time, curr_time + interval_length, self.op_id, -1)) # TODO: include op_id -> hw units assoc.
            curr_time = curr_time + interval_length

        #Filter out extraneous intervals (those with length 0)
        for interval in intervals_raw:
            if interval.t_end - interval.t_start > 0:
                intervals.append(interval)
        return intervals
        # return intervals_raw

    def get_active_units_start_end(self, time_cycle: int) -> Tuple[set, int]:
        '''
        using the start and end member dictionaries instead of its intervals -- this is for computing the
        intervals in get_intervals(), before the intervals field has been initialized.

        Returns:
        unit_names: Set of names of active units.
        remaining time: Time before the active units change (in cycles)
        '''

        unit_names = set()
        remaining_time = -1
        prev_key = None
        epsilon = 0 # For numerical stability given float clock counts.
        assert self.t_finish > time_cycle, ("We should never encounter an op that has already finished executing here!")
        if self.t_dram_ld_start > time_cycle: # Before an operator starts, return no units active and the time remaining before it begins exec.
            remaining_time = self.t_dram_ld_start - time_cycle
        else:
            for stage in self.start_times.keys():
                hw_units, start, end = self.stage_units[stage], self.start_times[stage], self.end_times[stage]
                if time_cycle < end + epsilon and time_cycle >= start - epsilon: # Find the interval that is running at the given time with the shortest remaining time.
                    unit_names = unit_names.union(hw_units)
                    # print("adding", unit_names)
                    remaining_time = end - time_cycle if (remaining_time == -1 or ((end - time_cycle) < remaining_time)) else remaining_time
                elif(prev_key and time_cycle < start and time_cycle >= self.end_times[prev_key]): # We are in an empty interval (no units active)
                    remaining_time = start - time_cycle
                prev_key = stage

        assert(remaining_time > -1), f"We should never have negative time! This should be replaced before returning! Curr time = {time_cycle} \
                                    {self.__str__()}"
        # print(f"AU for op {self.op_id}: {unit_names}, time left = {remaining_time}")
        return unit_names, remaining_time


    def get_active_units_intervals(self, time_cycle: int) -> Tuple[set, int]:
        '''
        For a given time = time_cycle and for a given operator log, figure out which operations/intervals are
        currently running and how much longer before it will stop.
        '''
        unit_names = set()
        remaining_time = -1
        # Asserting that self.t_finish > time_cycle doesn't work when earlier ops finish after later ones.
        # assert self.t_finish > time_cycle, (f"We should never encounter an op that has already finished executing here!, Analysis time: {time_cycle}, Operator: {self.__str__()}")
        if self.t_finish <= time_cycle: #Skip finished ops.
            # print("early exit AU for op ", self.op_id)
            return unit_names, 0
        if self.t_dram_ld_start > time_cycle: # Before an operator starts, return no units active and the time remaining before it begins exec.
            remaining_time = self.t_dram_ld_start - time_cycle
        else:
            for interval in self.intervals:
                hw_units, start, end = interval.active_units, interval.t_start, interval.t_end
                if time_cycle < end and time_cycle >= start: # Find the interval that is running at the given time with the shortest remaining time.
                    unit_names = unit_names.union(hw_units)
                    remaining_time = end - time_cycle if (remaining_time == -1 or ((end - time_cycle) < remaining_time)) else remaining_time

        assert(remaining_time > -1), f"We should never have negative time on valid ops! This should be replaced before returning!\
              analysis time = {time_cycle}, op: {str(self)}"
        # print(f"AU for op {self.op_id}: {unit_names}, time left = {remaining_time}")
        return unit_names, remaining_time


    def print_stats(self):
        return self.__str__()

    def draw_execution(self, ax, colors: Any = ["orange", "cyan", "purple"], operator_spacing: float = 1.0, bar_width: float = 0.9):
        '''
        Produce brokenbarh objects that show the stages of this operator's execution over time.
        '''
        reduce_bar  = [(self.t_reduce_start, self.reduce_dur)]
        read_bar    = [(self.t_dram_ld_start, self.dram_ld_dur)]
        bcast_bar   = [(self.t_bcast_start, self.bcast_dur)]
        comp_bar    = [(self.t_comp_shift_start, self.comp_sh_dur)]
        mm_bar      = [(self.t_comp_shift_start, self.mm_dur)]
        ew_bar      = [(self.t_comp_shift_start, self.ew_dur)]
        sh_bar      = [(self.t_comp_shift_start, self.shift_dur)]
        bar_ymin    = self.op_id * operator_spacing - bar_width

        if self.dram_st_dur != 0:
            write_bar = [(self.t_dram_st_start, self.dram_st_dur)]
            ax.broken_barh(write_bar, (bar_ymin, bar_width), color = "cyan", label = "DRAM write")
        if self.op_id == 0:
            ax.broken_barh(read_bar, (bar_ymin, bar_width/2), color="green", label="DRAM read")
            ax.broken_barh(bcast_bar, (bar_ymin + bar_width/2, bar_width/2), color="blue", label = "NoC Broadcast")
            ax.broken_barh(reduce_bar, (bar_ymin, bar_width), color = "orange", label = "NoC Reduce")
            ax.broken_barh(ew_bar, (bar_ymin, bar_width/6), color = "greenyellow", label = "EW")
            ax.broken_barh(mm_bar, (bar_ymin + (bar_width / 6), bar_width/6), color = "fuchsia", label = "MM")
            ax.broken_barh(comp_bar, (bar_ymin + (2 * bar_width / 6), bar_width/6), color = "lightcoral", label = "Overall Compute")
            ax.broken_barh(sh_bar, (bar_ymin + bar_width/2, bar_width/2), color = "maroon", label = "Shift")
        else:
            ax.broken_barh(read_bar, (bar_ymin, bar_width/2), color="green")
            ax.broken_barh(bcast_bar, (bar_ymin + bar_width/2, bar_width/2), color="blue")
            ax.broken_barh(reduce_bar, (bar_ymin, bar_width), color = "orange")
            ax.broken_barh(ew_bar, (bar_ymin, bar_width/6), color = "greenyellow")
            ax.broken_barh(mm_bar, (bar_ymin + (bar_width / 6), bar_width/6), color = "fuchsia")
            ax.broken_barh(comp_bar, (bar_ymin + (2 * bar_width / 6), bar_width/6), color = "lightcoral")
            ax.broken_barh(sh_bar, (bar_ymin + bar_width/2, bar_width/2), color = "maroon")

    def draw_energy(self, ax, colors = ["red", "green", "blue"]):
        '''
        Produce a bar whose height represents energy consumed by the operator. Energy is broken down into
        sync, shift, and shuffle (sss) and compute. The energy consumed by the operator is the sum of these two.
        '''
        bar_spacing = 1.0
        bar_width = 0.9
        ax.bar(x=self.op_id * bar_spacing, height=self.energy_sss, width=bar_width, bottom = 0, color=colors[0], label="sss")
        ax.bar(x=self.op_id * bar_spacing, height=self.energy_compute, width=bar_width, bottom = self.energy_sss, color = colors[1], label="compute")

    def draw_read_intensity(self, ax, dram_bw_GBps, npu_freq_MHz, x_base):
        '''
        Produce a bar whose width represents the dram load duration and height represents the dram read intensity.
        '''
        # bar_spacing = 1.0
        r_int, w_int = self.get_dram_util_intensity(dram_bw_GBps, npu_freq_MHz)
        print(f"drawing bar of h = {r_int}, at x = {x_base}, width = {self.dram_ld_dur}")
        ax.bar(x = x_base, height = r_int, width = self.dram_ld_dur, color = "lightgreen", align="edge")
        return x_base + self.dram_ld_dur

# def get_bars(op_list: List[FusedOperatorExecLog], hw_unit:str, capture_len:int):
def get_bars(interval_list: List[OverlapInterval], hw_unit:str, capture_len:int):
    '''
    Add bars to construct broken bar plot for the given hardware unit.
    '''

    bars = []
    for i, interval in enumerate(interval_list):
        if i >= capture_len: break
        if hw_unit in interval.active_units:
            assert(interval.t_end - interval.t_start > 0), f"Interval length should be positive! {interval}"
            bars.append((interval.t_start, interval.t_end - interval.t_start))
    return bars

def color_bars_intensity(interval_list: List[OverlapInterval], fused_op_log: List[FusedOperatorExecLog],
                         dram_bw_GBps: int, npu_freq_MHz: int, capture_len: int) -> Dict[str, List[float]]:
    '''
    For units with non-binary utilization, use color to represent intensity.
    '''
    interval_idx = 0
    intensities = {"dram_r": [],
              "dram_w" : [],
              "comp_sa" : [],
              "comp_vu" : [],}
    # At each op, get the dram intensity
    for op in fused_op_log:
        r_intensity, w_intensity = op.get_dram_util_intensity(dram_bw_GBps, npu_freq_MHz)
        # Each interval should fit squarely into an op. Use the op's dram intensity to color the interval.
        while interval_idx < capture_len and interval_idx < len(interval_list) and interval_list[interval_idx].t_end <= op.t_finish:
            curr_ival = interval_list[interval_idx]
            print(f"interval_idx: {curr_ival.active_units, curr_ival.t_start, curr_ival.t_end}\n")
            # Only append intensity for intervals that use the hardware unit
            if("dram_r" in curr_ival.active_units):
                intensities["dram_r"].append((r_intensity))

            if("dram_w" in curr_ival.active_units):
                assert(w_intensity != 0), "If writing, we should have > 0 intensity!"
                intensities["dram_w"].append((w_intensity))

            if("comp_sa" in curr_ival.active_units):
                intensities["comp_sa"].append((op.mm_util))

            if("comp_vu" in curr_ival.active_units):
                intensities["comp_vu"].append((op.vu_util))

            interval_idx += 1

    return intensities

def draw_dram_intensity(ax, fused_op_log: List[FusedOperatorExecLog], dram_bw_GBps: int, npu_freq_MHz: int, capture_len: int):
    '''
    Produces a distribution of DRAM read utilization/intensity over time (cycles)
    '''
    x_base = 0
    edges = [0]
    heights = []
    for i, op in enumerate(fused_op_log):
        if i > capture_len:
            break
        r_int, w_int = op.get_dram_util_intensity(dram_bw_GBps, npu_freq_MHz)
        heights.append(r_int),
        x_base += op.dram_ld_dur
        edges.append(x_base)

    ax.stairs(heights, edges, fill=True, color="darkgreen")

def normalize(arr):
    ''' Normalize bar color intensity to [0, 1] range. '''
    max_val = max(arr)
    min_val = min(arr)
    if max_val == min_val:
        return [1 for i in arr]
    else:
        new_arr = [(elem - min_val) / (max_val - min_val) for elem in arr]
        return new_arr
def draw_overlap(ax, ax_lgd, interval_list:List[OverlapInterval], fused_op_log: List[FusedOperatorExecLog],
                 dram_bw_GBps: int, npu_freq_MHz: int, capture_len: int):
    '''
    Draws a series of brokenbarh objects that represent the activity of each hardware unit/function over time.
    '''
    import matplotlib as mpl
    bar_spacing = 2.0
    bar_width = 1.0
    y_min = (bar_width / 2) * -1
    ax.set_yticks([y_min + bar_spacing * i for i in range(len(HW_UNITS))], labels = HW_UNITS)
    raw_intensities = color_bars_intensity(interval_list, fused_op_log, dram_bw_GBps, npu_freq_MHz, capture_len)
    for hw_unit in HW_UNITS:
        bars = get_bars(interval_list, hw_unit, capture_len)
        if len(bars) == 0:
            continue
        # bars = get_bars(fused_op_log, hw_unit, capture_len)
        print("BARS: ", hw_unit, bars)
        for bar in bars: # Sanity check to avoid extraneous bars
            assert(bar[1] > 0), f"Bar length should be positive! {bar}"

        # Compute intensity and corresonding colors for each interval.
        if (hw_unit in ["dram_r", "dram_w", "comp_sa", "comp_vu"]):
            # Get dram (read or write) intensity for each interval (scale = 0-1)
            hw_unit_intensity = raw_intensities[hw_unit]
            # Choose the color map, and scale the range of intensities to the range of the color map.
            cmap_name = HW_COLORS[hw_unit]
            cmap = mpl.colormaps[cmap_name]
            norm = mpl.colors.Normalize(min(hw_unit_intensity), max(hw_unit_intensity))
            color_legend = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
            # Add a legend, using our scalarmappable to provide the scale.
            mpl.pyplot.colorbar(color_legend, ax=ax_lgd, label = f"{hw_unit} intensity")
            color = [cmap(color) for color in hw_unit_intensity]
            assert(len(color) == len(bars)), f"bars should each be assigned a color. {len(color)} vs {len(bars)}"
        else:
            color = HW_COLORS[hw_unit]
        # print(f"overlap dia {hw_unit}: {bar_color}")
        ax.broken_barh(bars, (y_min, bar_width), label = hw_unit, color= color)
        y_min += bar_spacing
