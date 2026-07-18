"""Memory subsystem models for DRAM and SRAM.

Provides cycle-accurate access-cost estimation for DRAM (with row-open /
row-conflict timing) and SRAM, as well as silicon-area estimation helpers
for 3D-stacked and conventional memory technologies.

Key abstractions
----------------
* ``DRAM``  -- models row-based DRAM timing (CAS latency, tRCD, tRP),
               bank contention across cores, and per-core bandwidth.
* ``SRAM``  -- simple bandwidth-limited SRAM model.
* ``get_per_cycle_bytes_per_core_from_DRAM_config`` -- converts chip-level
  DRAM bandwidth into per-core, per-cycle byte count.
* ``get_sram_area_from_size`` / ``get_dram_area_from_size`` -- area
  estimators for technology-exploration sweeps.
"""

import sys
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from math import ceil

# Total number of DRAM banks shared across all cores.
# Used to compute per-bank contention (cores_per_bank = num_cores / NUM_BANKS).
NUM_BANKS = 128

# Default DRAM timing parameters shared across the codebase.
# These match the values in hw_config/*.json and run_all_tests.py.
DEFAULT_CL   = 14
DEFAULT_TRCD = 14
DEFAULT_TRP  = 14

HBM_PACKAGE_CAPACITY_MB = 16 * 1024
"""Default HBM package capacity used by the TSIM area model."""

HBM_PACKAGE_AREA_MM2 = 87.62745402745404
"""Default HBM package footprint area in mm^2 from the TSIM area model."""

def get_hbm_package_count(
    dram_size_MB: int,
    package_capacity_MB: int = HBM_PACKAGE_CAPACITY_MB,
) -> int:
    """Return the number of HBM packages needed for a capacity."""
    return max(1, int(ceil(int(dram_size_MB) / max(1, int(package_capacity_MB)))))


def get_hbm_package_footprint_mm(
    package_area_mm2: float = HBM_PACKAGE_AREA_MM2,
    aspect_ratio: float = 1.0,
) -> Tuple[float, float]:
    """Return ``(width_mm, height_mm)`` for one HBM package footprint.

    ``aspect_ratio`` is width / height.  The area is preserved exactly.
    """
    aspect_ratio = max(1e-9, float(aspect_ratio))
    width_mm = (float(package_area_mm2) * aspect_ratio) ** 0.5
    height_mm = float(package_area_mm2) / width_mm
    return width_mm, height_mm


def get_per_cycle_bytes_per_core_from_DRAM_config(num_cores: int,
                                                  total_bandwidth_GBps: int,
                                                  npu_freq_MHz: int,
                                                  ) -> int:
    """Convert chip-level DRAM bandwidth to per-core, per-cycle byte count.

    Parameters
    ----------
    num_cores : int
        Number of cores sharing the total DRAM bandwidth.
    total_bandwidth_GBps : int
        Aggregate off-chip bandwidth in GiB/s (binary gigabytes).
    npu_freq_MHz : int
        Core clock frequency in MHz.

    Returns
    -------
    int
        Bytes each core can transfer in a single clock cycle
        (floor-divided evenly across cores).
    """
    # Convert GiB/s to bytes/s, then divide by frequency (cycles/s) to get bytes/cycle.
    num_byte_per_cycle = total_bandwidth_GBps * (2**30) / npu_freq_MHz / (10**6)
    # Divide equally across all cores (floor division).
    num_byte_per_core_per_cycle = num_byte_per_cycle // num_cores
    return num_byte_per_core_per_cycle

def get_sram_area_from_size(sram_size_KB: int, memtype="3D-SRAM") -> int:
    """Estimate SRAM silicon area (mm^2) for a given capacity.

    Parameters
    ----------
    sram_size_KB : int
        Desired SRAM capacity in KiB.
    memtype : str, optional
        Technology variant.  ``"3D-SRAM"`` uses DRAM-density numbers from
        Meta's AR/VR 3D chip paper.  ``"SRAM"`` uses a piecewise-linear
        regression fitted to McPAT area sweeps (4 KB -- 1 MB); a slope
        discontinuity exists around 640 KB due to McPAT's internal bank
        restructuring.

    Returns
    -------
    float
        Estimated area in mm^2.
    """
    area_sq_mm = -1
    if memtype == "3D-SRAM":
        sram_MB_per_sq_mm = 4  # density from Meta AR/VR 3D chip paper
        area_sq_mm = sram_size_KB / 1024 * sram_MB_per_sq_mm
    elif memtype == "SRAM":
        # Piecewise-linear fit from McPAT area sweep.
        # Slope change at 640 KB is caused by McPAT's internal bank
        # restructuring at that capacity boundary.
        if sram_size_KB <= 640:
            area_sq_mm = 5.74e-04 * sram_size_KB + 0.133
        else:
            area_sq_mm = 6.7e-04 * sram_size_KB + 0.322
    else:
        print( "area was not set! Invalid memtype!")
        exit(-1)
    return area_sq_mm

def get_dram_area_from_size(dram_size_MB: int, memtype="3D-DRAM") -> int:
    """Estimate DRAM silicon area (mm^2) for a given capacity.

    Parameters
    ----------
    dram_size_MB : int
        Desired DRAM capacity in MiB.
    memtype : str, optional
        ``"3D-DRAM"`` -- uses density of 8.4 MB/mm^2 from
        `<https://openreview.net/pdf?id=P4LViaB8g0>`_.
        ``"HBM"`` -- each HBM die is 16 GB; die area (~87.6 mm^2) was
        measured from an A100 die photo (GPU die 826 mm^2, HBM die
        proportionally scaled from pixel measurements).

    Returns
    -------
    float
        Estimated area in mm^2.  For HBM, the result is rounded up to
        the next whole die.
    """
    area_sq_mm = -1
    if memtype == "3D-DRAM":
        MB_per_sq_mm = 8.4  # 3D-stacked DRAM density
        area_sq_mm = dram_size_MB / MB_per_sq_mm
    elif memtype == "HBM":
        mem_per_die_MB = 16 * 1024  # 16 GiB per HBM die
        area_per_die_sq_mm = 87.62745402745404  # from A100 die-photo measurement
        # Round up to whole HBM dies.
        area_sq_mm = area_per_die_sq_mm * ceil(dram_size_MB / mem_per_die_MB)
    else:
        print( "area was not set! Invalid memtype!")
        exit(-1)
    return area_sq_mm

class SRAM:
    """Simple bandwidth-limited SRAM model.

    Access latency is computed purely from the data volume divided by the
    sustained bandwidth (bytes per cycle).  No row/bank modelling.

    Parameters
    ----------
    bandwidth_bytepc : float
        Sustained read/write bandwidth in bytes per cycle.
    num_layers : int, optional
        Number of 3D-stacked SRAM layers (default 4).
    """

    def __init__(self,
                 bandwidth_bytepc: float,
                 num_layers: int = 4) -> None:
        self.bandwidth_bytepc = bandwidth_bytepc
        self.num_layers = num_layers

    def num_cycle_of_access(self, num_bytes: int) -> float:
        """Return the number of cycles to transfer *num_bytes* at the
        configured bandwidth."""
        return num_bytes / self.bandwidth_bytepc

class DRAM:
    """Cycle-level DRAM access-cost model with row-open / row-conflict timing.

    The model accounts for three key DRAM timing parameters:

    * **CL** (CAS Latency) -- column-access strobe to data.
    * **tRCD** (RAS-to-CAS Delay) -- row activation to column command.
    * **tRP** (Row Precharge) -- minimum time to close a row before
      opening a new one.

    A *row reopen* costs ``CL + tRCD + tRP`` cycles.  The number of
    reopens is determined by both the access granularity (how data is
    tiled across cores) and the row size.  Bank contention is modelled
    by scaling reopen cost by ``num_cores / NUM_BANKS``.

    Parameters
    ----------
    CL, tRCD, tRP : int
        DRAM timing parameters in core clock cycles.
    bytes_per_row : int
        Size of one DRAM row (row buffer) in bytes.
    bytes_per_cycle : int
        Data-bus width in bytes (burst transfer per cycle).
    num_cores : int
        Total number of cores sharing the DRAM.
    num_layers : int, optional
        Number of 3D-stacked DRAM layers (default 8).
    use_sram : bool, optional
        If True, bypass DRAM timing and return a simple fixed-rate
        estimate (used for SRAM-only design-point sweeps).
    """

    def __init__(self,
                 CL: int,
                 tRCD: int,
                 tRP: int,
                 bytes_per_row: int,
                 bytes_per_cycle: int,
                 num_cores: int,
                 num_layers: int = 8,
                 use_sram: bool = False,
                 precise: bool = False,
                 num_banks_per_channel: int = 16,
                 tRRD: int = 4,
                 tFAW: int = 20,
                 tRFC: int = 350,
                 tREFI: int = 12480,
                 precise_cache_size: int = 4096,
                 ultra_precise: bool = False,
                 ultra_backend: Optional[Any] = None,
                 ultra_cache_size: int = 8192,
                 lock_cores_per_bank: float = 0,
                 soft_cores_per_bank: bool = True,
                ) -> None:
        self.CL: int = CL
        self.tRCD: int = tRCD
        self.tRP: int = tRP
        # Full row-reopen penalty: activate + column-access + precharge.
        self.reopen: int = CL + tRCD + tRP
        self.bytes_per_row: int = bytes_per_row
        self.bytes_per_cycle: int = bytes_per_cycle
        self.num_cores: int = num_cores
        self.num_layers: int = num_layers
        self.use_sram: bool = use_sram
        # Switch: when non-zero, lock cores_per_bank to this fixed value
        # (e.g. 2 = pin to the default 256-core / 128-bank ratio) instead of
        # the actual num_cores/NUM_BANKS. 0 (default) = off; any model can opt
        # in via the constructor / CLI flag.
        self.lock_cores_per_bank: float = lock_cores_per_bank
        # Soft mode: compress cores_per_bank above 2 via sqrt(2*raw).
        self.soft_cores_per_bank: bool = soft_cores_per_bank
        # Pre-compute: immutable per DRAM instance, called in hot paths.
        self._cpb: float = self._compute_cores_per_bank()

        # --- Precise (request-level) DRAM mode ---
        # When ``precise=True``, ``num_cycle_of_access`` synthesizes a stream
        # of burst-sized requests and drives a per-bank state machine that
        # tracks open rows, tRRD/tFAW activation throttling, and amortized
        # tRFC refresh penalty. Results
        # are coalesced via a (num_bytes, granularity, need_init) cache so
        # structurally equivalent access patterns reuse a prior simulation.
        # Defaults are off — when ``precise=False`` the original analytical
        # fast path is used unchanged.
        self.precise: bool = precise
        self.num_banks_per_channel: int = num_banks_per_channel
        self.tRRD: int = tRRD
        self.tFAW: int = tFAW
        self.tRFC: int = tRFC
        self.tREFI: int = tREFI
        self._precise_cache_size: int = precise_cache_size
        self._precise_cache: Dict[Tuple[int, int, bool], int] = {}

        # --- Ultra-precise (external simulator) DRAM mode ---
        # Defers latency estimation to a real cycle-accurate DRAM simulator
        # (Ramulator 2.0 or DRAMsim3) via subprocess. The backend object is
        # discovered/instantiated by the caller (icbm_launch.get_hw_modules)
        # and dropped here; if it failed to instantiate we silently leave
        # ``ultra_precise=False`` and the caller may downgrade to
        # ``precise=True``. Same coalescing key as the precise path: the
        # synthesized stream is fully determined by (num_bytes, granularity,
        # need_init), so per-key caching avoids re-simulating
        # signature.
        self.ultra_precise: bool = ultra_precise and (ultra_backend is not None)
        self._ultra_backend: Optional[Any] = ultra_backend if self.ultra_precise else None
        self._ultra_cache_size: int = ultra_cache_size
        self._ultra_cache: Dict[Tuple[int, int, bool], int] = {}
        if self.ultra_precise:
            self._populate_from_dram_cache()

    def _populate_from_dram_cache(self) -> None:
        """Populate ultra cache from pre-computed dram_cache files (``*.dcache``).

        Only called when ultra_precise is active with a valid backend
        (no fallback).  Each cached trace is converted to a cache entry:
        (num_bytes, granularity, need_init) -> total_cycles.
        """
        try:
            from tsim_components.dram_external import load_dram_cache
            cache_data = load_dram_cache()
        except Exception:
            return
        for trace in cache_data:
            if not trace:
                continue
            n = len(trace)
            # need_init: first access is read (rw=0) -> need_init=True
            need_init = (trace[0][1] == 0)
            # granularity: XOR of adjacent address deltas
            prev = trace[0][0]
            gran = 0
            for i in range(1, min(n, 64)):  # sample first 64 entries
                gran |= (trace[i][0] ^ prev)
                prev = trace[i][0]
            if gran == 0:
                gran = 1
            # total cycles = sum of latencies
            cycles = sum(e[2] for e in trace)
            key = (n, gran, need_init)
            self._ultra_cache[key] = cycles

    def _compute_cores_per_bank(self) -> float:
        """Effective cores-per-bank used for bank-contention row scaling.

        Computed once in ``__init__`` and cached as ``self._cpb``.

        Modes (priority order):
          * ``lock_cores_per_bank`` (non-zero) -> pin to that fixed value.
        """
        if self.lock_cores_per_bank:
            return float(self.lock_cores_per_bank)
        if self.soft_cores_per_bank:
            return (self.num_cores/NUM_BANKS)**.8
        return 2

    def _cores_per_bank(self) -> float:
        """Return the cached cores-per-bank value (computed in __init__).

        Kept for backward compatibility; prefer ``self._cpb`` directly.
        """
        return self._cpb

    def num_cycle_of_access(self, num_bytes: int,
                            access_granularity_bytes: int,
                            need_init: bool = False) -> float:
        """Estimate the total cycle cost for a single DRAM access.

        The cost has two components:

        1. **Row-reopen overhead** -- each non-contiguous access to a new
           DRAM row incurs a full reopen penalty (``CL + tRCD + tRP``).
           The number of reopens is the *maximum* of the reopens implied
           by the access granularity and by the physical row size, then
           scaled by bank contention (``num_cores / NUM_BANKS``).
        2. **Column transfer time** -- the raw number of bus cycles to
           move the data (``ceil(num_bytes / bytes_per_cycle)``).

        Parameters
        ----------
        num_bytes : int
            Total bytes to transfer for this access.
        access_granularity_bytes : int
            Contiguous chunk size per access (determined by tensor tiling).
        need_init : bool, optional
            Whether the first access requires a fresh row activation
            (True) or can piggy-back on an already-open row (False).

        Returns
        -------
        int
            Estimated access latency in clock cycles (minimum 1).
        """
        if num_bytes == 0:
            return 0
        if self.use_sram:
            # Bypass DRAM model: use a simple fixed-rate estimate for
            # SRAM-only design points (empirical ~3.57 bytes/cycle).
            return int(num_bytes / 3.57)
        if self.ultra_precise:
            return self._ultra_precise_num_cycle_of_access(num_bytes,
                                                           access_granularity_bytes,
                                                           need_init)
        if self.precise:
            return self._precise_num_cycle_of_access(num_bytes,
                                                     access_granularity_bytes,
                                                     need_init)

        cycle = 0

        # --- Row-reopen count estimation ---
        # Reopens from access granularity: each granularity-sized chunk
        # may land in a different DRAM row.
        num_reopen_granularity = (num_bytes + access_granularity_bytes - 1) // access_granularity_bytes
        # Reopens from row size: data spanning multiple rows forces reopens.
        num_reopen_row_limit = (num_bytes + self.bytes_per_row - 1) // self.bytes_per_row
        # The binding constraint determines the actual reopen count.
        num_reopen = max(num_reopen_granularity, num_reopen_row_limit)

        if need_init == False and num_reopen <= 1:
            # Continuing from an already-open row: the access fits within
            # one row, so compute a fractional reopen cost proportional to
            # the fraction of the row/granularity actually used.
            max_access_granularity = min(access_granularity_bytes, self.bytes_per_row)
            num_reopen = 1 / (max_access_granularity // num_bytes)

        cycle += num_reopen * self.reopen

        # --- Bank contention scaling ---
        # Multiple cores sharing the same bank serialize their row activations.
        cycle *= self._cpb
        # Subtract one tRP: the very last access does not need to precharge
        # for a subsequent row (pipeline overlap with next command).
        cycle -= self.tRP

        # --- Column (burst) transfer time ---
        num_cols = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
        cycle += num_cols

        return int(max(1, cycle))

    def _precise_num_cycle_of_access(self,
                                     num_bytes: int,
                                     access_granularity_bytes: int,
                                     need_init: bool) -> int:
        """Request-level DRAM latency estimator with bank-state simulation.

        Synthesizes a burst-sized request stream from
        ``(num_bytes, access_granularity_bytes)`` and runs it against a
        finite-state model of ``num_banks_per_channel`` banks. Captures:

        * Row-buffer hits vs. row-conflicts (CL / tRCD / tRP).
        * Bank-level parallelism — independent banks can pipeline
          activations and data transfers.
        * tRRD (minimum interval between activations on the same bank)
          and tFAW (max 4 activations within a sliding window).
        * Amortized refresh penalty — the long-run fraction of cycles
          spent in tRFC stalls (tRFC / tREFI).

        Bank contention from multiple cores sharing the same bank is
        applied on top, mirroring the heuristic of the fast path: only
        the row-overhead component scales by ``num_cores / NUM_BANKS``,
        the burst component is bandwidth-bound and does not.

        Results are cached by ``(num_bytes, granularity, need_init)`` —
        the synthetic stream is fully determined by these parameters, so
        the cache key plays the same role as the paper's match-key
        signature for repeated traces.
        """
        ag = access_granularity_bytes
        if ag <= 0:
            ag = self.bytes_per_cycle

        cache_key = (int(num_bytes), int(ag), bool(need_init))
        cached = self._precise_cache.get(cache_key)
        if cached is not None:
            cycles = cached
        else:
            cycles = self._simulate_request_stream(num_bytes, ag, need_init)
            if len(self._precise_cache) < self._precise_cache_size:
                self._precise_cache[cache_key] = cycles

        # Bank contention (mirrors fast path): when more cores than banks,
        # row-related work serializes; bus throughput does not.
        cores_per_bank = self._cpb
        if cores_per_bank > 1:
            burst_floor = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
            row_overhead = max(0, cycles - burst_floor)
            cycles = burst_floor + row_overhead * cores_per_bank

        return int(max(1, cycles))

    def _simulate_request_stream(self,
                                 num_bytes: int,
                                 access_granularity: int,
                                 need_init: bool) -> int:
        """Drive the per-bank state machine over a synthetic burst stream.

        Address layout assumption: each ``access_granularity``-sized chunk
        opens a fresh row, and chunks round-robin across the banks of a
        single channel. This captures both the worst case of granularity
        forcing reopens and the best case of spreading over independent
        banks.
        """
        # Cast to int up front: bytes_per_cycle / bytes_per_row may be float
        # when derived from `total_GBps * 2^30 / freq / 1e6 // num_cores`
        # (`//` on floats returns float).
        bpc = max(1, int(self.bytes_per_cycle))
        bpr = max(bpc, int(self.bytes_per_row))
        # Each chunk maps to one row. Don't let granularity exceed a row
        # (would imply spanning multiple rows in one chunk — not modelled).
        ag_eff = max(bpc, min(int(access_granularity), bpr))
        bursts_per_chunk = max(1, ag_eff // bpc)
        num_bursts = max(1, (int(num_bytes) + bpc - 1) // bpc)
        num_chunks = (num_bursts + bursts_per_chunk - 1) // bursts_per_chunk

        NB = max(1, self.num_banks_per_channel)
        # Bank state: open row id (-1 = closed) and earliest free time.
        bank_open_row: List[int] = [-1] * NB
        bank_free_time: List[int] = [0] * NB
        last_act_per_bank: List[int] = [-self.tRRD] * NB
        # Sliding window of last activations for tFAW (length <= 4).
        act_window: List[int] = []
        bus_free_time: int = 0
        cur_time: int = 0

        # If continuing from a previously open row, pre-open bank 0 row 0
        # so the first chunk hits without paying the activation penalty.
        if not need_init:
            bank_open_row[0] = 0

        for chunk_idx in range(num_chunks):
            bank = chunk_idx % NB
            row = (chunk_idx // NB) + 1  # synthetic row id (>= 1)
            bursts_remaining = num_bursts - chunk_idx * bursts_per_chunk
            bursts_this = min(bursts_per_chunk, bursts_remaining)

            if bank_open_row[bank] != row:
                # Need (precharge if open) + activate + tRCD before CAS.
                t_ready = max(cur_time, bank_free_time[bank])
                if bank_open_row[bank] != -1:
                    t_pre_done = t_ready + self.tRP
                else:
                    t_pre_done = t_ready
                # tRRD: min interval to previous activation on this bank.
                t_act = max(t_pre_done, last_act_per_bank[bank] + self.tRRD)
                # tFAW: at most 4 activations within tFAW cycles globally.
                if len(act_window) >= 4:
                    t_act = max(t_act, act_window[-4] + self.tFAW)
                act_window.append(t_act)
                if len(act_window) > 4:
                    # Keep only the trailing four — that's all tFAW needs.
                    act_window = act_window[-4:]
                last_act_per_bank[bank] = t_act
                bank_open_row[bank] = row
                t_cas_cmd = t_act + self.tRCD
            else:
                # Row hit — go straight to CAS.
                t_cas_cmd = max(cur_time, bank_free_time[bank])

            # First data beat appears tCL after CAS; subsequent beats are
            # one per cycle. Bus serializes across banks.
            t_data_first = max(t_cas_cmd + self.CL, bus_free_time)
            t_data_last = t_data_first + bursts_this  # exclusive end
            bus_free_time = t_data_last
            bank_free_time[bank] = t_data_last
            cur_time = t_data_last

        # Amortized refresh: the fraction of time spent in tRFC stalls.
        # A real DRAM stalls one rank for tRFC cycles every tREFI cycles;
        # over a long run that's a tRFC/tREFI multiplicative slowdown.
        # For short accesses this rounds to ~zero overhead.
        if self.tREFI > 0:
            cur_time += (cur_time * self.tRFC) // self.tREFI

        return cur_time

    def _ultra_precise_num_cycle_of_access(self,
                                           num_bytes: int,
                                           access_granularity_bytes: int,
                                           need_init: bool) -> int:
        """Defer to the configured external DRAM simulator backend.

        Each unique ``(num_bytes, granularity, need_init)`` tuple is fed to
        the backend exactly once; results are cached for subsequent calls.
        On any subprocess failure we fall back to the pure-Python precise
        simulator and warn once per process so users notice.
        """
        ag = access_granularity_bytes
        if ag <= 0:
            ag = self.bytes_per_cycle

        cache_key = (int(num_bytes), int(ag), bool(need_init))
        cached = self._ultra_cache.get(cache_key)
        if cached is not None:
            cycles = cached
        else:
            try:
                cycles = self._ultra_backend.simulate_stream(
                    num_bytes=num_bytes,
                    access_granularity=ag,
                    need_init=need_init,
                    bytes_per_cycle=self.bytes_per_cycle,
                    bytes_per_row=self.bytes_per_row,
                    num_banks_per_channel=self.num_banks_per_channel,
                )
            except Exception as e:
                if not getattr(self, "_ultra_failed_once", False):
                    print(
                        f"WARNING: ultra-precise DRAM backend "
                        f"{getattr(self._ultra_backend, 'name', '?')} "
                        f"failed ({e}); falling back to the pure-Python "
                        f"precise simulator for this and subsequent calls.",
                        file=sys.stderr, flush=True)
                    self._ultra_failed_once = True
                # Permanent downgrade to avoid hammering a broken subprocess.
                self.ultra_precise = False
                if not self.precise:
                    self.precise = True
                return self._precise_num_cycle_of_access(num_bytes, ag, need_init)
            if len(self._ultra_cache) < self._ultra_cache_size:
                self._ultra_cache[cache_key] = cycles

        # Same bank-contention scaling as precise/fast: row-overhead serializes
        # across cores sharing a bank, burst transfer is bandwidth-bound.
        cores_per_bank = self._cpb
        if cores_per_bank > 1:
            burst_floor = (num_bytes + self.bytes_per_cycle - 1) // self.bytes_per_cycle
            row_overhead = max(0, cycles - burst_floor)
            cycles = burst_floor + row_overhead * cores_per_bank
        return int(max(1, cycles))

    def get_dram_access_list(self, tensor_shapes: List[np.ndarray],
                             temporal_var_replicas: List[int],
                             core_group_size: int,
                             num_byte_per_elem: int,
                             return_granularity: bool = False,
                             return_tot_bytes_per_core: bool = False,
                             return_cycles_and_bytes: bool = False,
                             return_cycles_bytes_granularity: bool = False,
                             opti_intra_mapping: bool = False,
                             bad_mapping: bool = False):
        """Compute per-tensor DRAM access costs for a list of tensors.

        For each tensor the method calculates:

        * **total bytes per core** -- element count (after replica
          division) times ``num_byte_per_elem``.
        * **access granularity** -- contiguous byte chunk each core
          touches, derived from the tensor's largest dimension, the
          replica count, and the core-group size.

        What is returned depends on the ``return_*`` flags (exactly one
        should be True, or all False for the default cycle-count mode):

        * ``return_granularity`` -- list of access granularities (bytes).
        * ``return_tot_bytes_per_core`` -- list of total bytes per core.
        * ``return_cycles_and_bytes`` -- tuple of (cycles_list, bytes_list).
        * ``return_cycles_bytes_granularity`` -- tuple of (cycles_list,
          bytes_list, granularity_list).
        * *(default)* -- list of access cycle counts.

        Parameters
        ----------
        tensor_shapes : list of np.ndarray
            Shape arrays for [output, input0, input1, ...].
        temporal_var_replicas : list of int
            Temporal reuse factor per tensor (divides total bytes).
        core_group_size : int
            Number of cores cooperating on the same tile.
        num_byte_per_elem : int
            Element width (e.g. 2 for FP16).
        opti_intra_mapping : bool, optional
            If True, assume an optimised intra-core mapping where the
            full core-group accesses data contiguously (no row reopens
            on first access).
        bad_mapping : bool, optional
            If True, simulate a sub-optimal mapping where input tensors
            use only half their last dimension for granularity.
        """
        dram_access_list: List[float] = []
        dram_bytes_list: List[int] = [] if (return_cycles_and_bytes or return_cycles_bytes_granularity) else None
        dram_granularity_list: List[int] = [] if return_cycles_bytes_granularity else None

        for i, (shape, replica) in enumerate(zip(tensor_shapes, temporal_var_replicas)):
            # Collapse singleton dimensions so they don't inflate the
            # granularity calculation.
            shape1 = shape[shape > 1]
            if len(shape1) == 0:
                shape1 = np.array([1])

            # Total bytes this core must load/store for the tensor.
            tot_access_bytes: int = ceil(shape1.prod() * num_byte_per_elem / replica)

            if opti_intra_mapping:
                # Optimised mapping: all cores in the group access a
                # single contiguous region -- granularity equals the
                # entire per-core block times the group size.
                dram_access_granularity_byte = tot_access_bytes * core_group_size
            else:
                # Standard mapping: granularity is based on the tensor's
                # largest dimension, adjusted for temporal reuse and the
                # core-group cooperative factor.
                if bad_mapping and i:
                    # Sub-optimal: input tensors divide last dim by cpb.
                    last_dim = max(1, shape1[-1] // self._cpb)
                else:
                    last_dim = max(shape1)

                remaining_dims = shape.prod() / last_dim
                # Compute how much the replica factor shrinks the
                # contiguous last-dimension chunk.
                last_dim_div = 2 * replica / remaining_dims
                last_dim_div = max(1, last_dim_div)
                last_dim = last_dim // last_dim_div
                # Scale by core-group: cooperating cores access
                # adjacent elements, widening the contiguous region.
                last_dim = last_dim * core_group_size
                dram_access_granularity_byte = last_dim * num_byte_per_elem
                # Ensure at least one element width.
                dram_access_granularity_byte = max(num_byte_per_elem, dram_access_granularity_byte)
            if return_granularity:
                dram_access_list.append(dram_access_granularity_byte)
            elif return_tot_bytes_per_core:
                dram_access_list.append(tot_access_bytes)
            elif return_cycles_and_bytes:
                need_init = not opti_intra_mapping
                dram_access_cycles = self.num_cycle_of_access(tot_access_bytes,
                                                              dram_access_granularity_byte,
                                                              need_init)
                dram_access_list.append(dram_access_cycles)
                dram_bytes_list.append(tot_access_bytes)
            elif return_cycles_bytes_granularity:
                need_init = not opti_intra_mapping
                dram_access_cycles = self.num_cycle_of_access(tot_access_bytes,
                                                              dram_access_granularity_byte,
                                                              need_init)
                dram_access_list.append(dram_access_cycles)
                dram_bytes_list.append(tot_access_bytes)
                dram_granularity_list.append(int(dram_access_granularity_byte))
            else:
                need_init = not opti_intra_mapping
                dram_access_cycles = self.num_cycle_of_access(tot_access_bytes,
                                                              dram_access_granularity_byte,
                                                              need_init)
                dram_access_list.append(dram_access_cycles)
        if return_cycles_and_bytes:
            return dram_access_list, dram_bytes_list
        if return_cycles_bytes_granularity:
            return dram_access_list, dram_bytes_list, dram_granularity_list
        return dram_access_list