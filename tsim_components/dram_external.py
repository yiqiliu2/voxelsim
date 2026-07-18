"""External cycle-accurate DRAM backends for the ultra-precise mode.

This module wraps two production DRAM simulators that the project does not
ship with — they must be built and made discoverable via PATH or env vars
before use:

* **Ramulator 2.0** -- https://github.com/CMU-SAFARI/ramulator2
  Build: ``git clone && cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && make -j``.
  The driver produces a ``ramulator2`` binary that takes a YAML config
  (``-f cfg.yaml``); we drive its ``LoadStoreTrace`` frontend with a
  trace file of ``LD 0xADDR`` / ``ST 0xADDR`` lines and parse
  ``memory_system_cycles`` plus ``avg_read_latency`` from stdout.
  We avoid the sibling ``ReadWriteTrace`` frontend because its
  ``is_finished()`` always returns true (see
  ``src/frontend/impl/memory_trace/readwrite_trace.cpp`` line ~95), so
  Ramulator exits after a single tick and reports ``cycles=0``.

  **Caveats on the resulting cycle count:**

  - ``memory_system_cycles`` ticks while the frontend is sending requests;
    the LoadStoreTrace frontend stops the moment the last request is
    *sent*, not completed. We add ``avg_read_latency`` to approximate
    the tail. For traces below ~50 requests Ramulator reports
    ``avg_read_latency=0`` and our result floors at the request count.
  - The cycle units are DRAM ticks under whichever ``timing.preset`` you
    pick — they are *not* in NPU clock cycles. The bundled DDR4_2400R
    preset is a generic stand-in. Point ``RAMULATOR2_CONFIG`` at your
    own YAML to match the target technology and clock_ratio.

* **DRAMsim3** -- https://github.com/umd-memsys/DRAMsim3
  Build: ``cmake .. && make -j4``. The ``dramsim3main`` binary takes
  ``configs/*.ini -c <max_cycles> -t trace.txt`` where each trace line is
  ``0xADDR READ <cycle>`` / ``0xADDR WRITE <cycle>``. A ``dramsim3.json``
  is written to the working directory.

Both backends synthesize a request stream from
``(num_bytes, access_granularity, need_init, bytes_per_cycle, bytes_per_row,
num_banks_per_channel)``, hand it to the external simulator, and parse total
cycles back. The caller (``DRAM._ultra_precise_num_cycle_of_access``) caches
on the same key — this avoids re-running
coalescing: identical synthesized streams reuse a prior simulator run instead
of re-invoking the expensive subprocess.

Discovery order
---------------
``RAMULATOR2_BIN``, ``RAMULATOR2_CONFIG`` env vars (or ``ramulator2`` on
``$PATH``); ``DRAMSIM3_BIN``, ``DRAMSIM3_CONFIG`` env vars (or
``dramsim3main`` on ``$PATH``). When ``DRAMSIM3_CONFIG`` is unset DRAMsim3
is unusable because its built-in presets live under the source tree, not
the binary's CWD.
"""

from __future__ import annotations

import json
import os
import pickle as _pickle
import re
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional


def load_dram_cache() -> List[List]:
    """Load all dram_cache files (``*.dcache``) into a nested list.

    Returns a list of traces, where each trace is a list of
    (address, is_write, latency) tuples.
    """
    import glob
    cache_dir = os.path.join(os.path.dirname(__file__), "dram_cache")
    files = sorted(glob.glob(os.path.join(cache_dir, "dram_cache_*.dcache")))
    cache_data = []
    for f in files:
        with open(f, "rb") as fh:
            batch = _pickle.load(fh)
        cache_data.extend(batch)
    return cache_data


class ExternalDRAMUnavailable(RuntimeError):
    """Raised when an external DRAM backend cannot be located or invoked."""


# --------------------------------------------------------------------------- #
#  Stream synthesis: shared helper for both backends                          #
# --------------------------------------------------------------------------- #

def _synthesize_addresses(num_bytes: int,
                          access_granularity: int,
                          bytes_per_cycle: int,
                          bytes_per_row: int,
                          num_banks_per_channel: int) -> List[int]:
    """Generate burst-sized request addresses that mimic the DRAM access.

    Bursts within the same ``access_granularity``-sized chunk are placed on
    contiguous addresses (row hits after the first burst); successive
    chunks land on a different bank in round-robin and on a fresh row, so
    the simulator sees the worst-case mix of bank-level parallelism plus
    row-conflict pressure that the analytical fast path tries to model.
    """
    # Cast to int — DRAM.bytes_per_cycle / bytes_per_row arrive as floats
    # from the chip-bandwidth conversion in icbm_launch.get_hw_modules.
    bpc = max(1, int(bytes_per_cycle))
    bpr = max(bpc, int(bytes_per_row))
    ag_eff = max(bpc, min(int(access_granularity), bpr))
    bursts_per_chunk = max(1, ag_eff // bpc)
    num_bursts = max(1, (int(num_bytes) + bpc - 1) // bpc)
    NB = max(1, num_banks_per_channel)

    # Synthetic mapping: bank stride = bpr (rows are bpr bytes); each chunk
    # advances by bpr * NB so adjacent chunks land in different banks AND
    # fresh rows. Burst stride within a chunk = bpc.
    addrs: List[int] = []
    for chunk_idx in range((num_bursts + bursts_per_chunk - 1) // bursts_per_chunk):
        bank = chunk_idx % NB
        row_in_bank = chunk_idx // NB
        chunk_base = bank * bpr + row_in_bank * (bpr * NB)
        bursts_remaining = num_bursts - chunk_idx * bursts_per_chunk
        bursts_this = min(bursts_per_chunk, bursts_remaining)
        for b in range(bursts_this):
            addrs.append(chunk_base + b * bpc)
    return addrs


# --------------------------------------------------------------------------- #
#  Backend interface                                                           #
# --------------------------------------------------------------------------- #

class ExternalDRAMBackend:
    """Common interface for cycle-accurate DRAM backends."""

    name: str = "abstract"

    def simulate_stream(self,
                        num_bytes: int,
                        access_granularity: int,
                        need_init: bool,
                        bytes_per_cycle: int,
                        bytes_per_row: int,
                        num_banks_per_channel: int) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  Ramulator 2.0 backend                                                       #
# --------------------------------------------------------------------------- #

# Minimal LoadStoreTrace YAML. The DDR4 preset is a stand-in — point
# ``RAMULATOR2_CONFIG`` at a custom YAML to override technology / timing.
# We use LoadStoreTrace (LD/ST single-address format) rather than
# ReadWriteTrace because the latter's ``is_finished()`` always returns
# true upstream (src/frontend/impl/memory_trace/readwrite_trace.cpp), so
# Ramulator exits after a single request and reports zero cycles.
_RAMULATOR2_DEFAULT_YAML = """\
Frontend:
  impl: LoadStoreTrace
  clock_ratio: 1
  path: {trace_path}
MemorySystem:
  impl: GenericDRAM
  clock_ratio: 1
  DRAM:
    impl: DDR4
    org:
      preset: DDR4_8Gb_x8
      channel: 1
      rank: 1
    timing:
      preset: DDR4_2400R
  Controller:
    impl: Generic
    Scheduler:
      impl: FRFCFS
    RefreshManager:
      impl: AllBank
    RowPolicy:
      impl: OpenRowPolicy
  AddrMapper:
    impl: RoBaRaCoCh
"""

# Look for "memory_system_cycles: <int>" in the hierarchical text dump.
_RAMULATOR2_CYCLES_RE = re.compile(r"memory_system_cycles\s*[:=]\s*(\d+)")
# avg_read_latency_<channel>: <float>. We add the avg-tail-latency to
# memory_system_cycles to approximate wall-clock time, because
# LoadStoreTrace.is_finished() returns true the moment the last request
# is *sent* — in-flight requests after that don't extend memory_system_cycles.
_RAMULATOR2_AVGLAT_RE = re.compile(r"avg_read_latency_\d+\s*[:=]\s*([\d.]+)")


class RamulatorBackend(ExternalDRAMBackend):
    """Subprocess driver for the Ramulator 2.0 ``ReadWriteTrace`` frontend."""

    name = "ramulator2"

    def __init__(self,
                 binary_path: Optional[str] = None,
                 config_template: Optional[str] = None,
                 timeout_sec: int = 60) -> None:
        bp = (binary_path
              or os.environ.get("RAMULATOR2_BIN")
              or shutil.which("ramulator2"))
        if not bp or not os.path.isfile(bp):
            raise ExternalDRAMUnavailable(
                "ramulator2 binary not found. Build per "
                "https://github.com/CMU-SAFARI/ramulator2 then set "
                "RAMULATOR2_BIN=/path/to/ramulator2 or place it on PATH.")
        self.binary = bp
        self.timeout_sec = timeout_sec

        # Optional user-supplied YAML; we splice {trace_path} into it before
        # each invocation. If absent we fall back to the bundled DDR4 preset.
        cfg_path = config_template or os.environ.get("RAMULATOR2_CONFIG")
        if cfg_path and os.path.isfile(cfg_path):
            with open(cfg_path, "r") as f:
                self._yaml_template = f.read()
            if "{trace_path}" not in self._yaml_template:
                raise ExternalDRAMUnavailable(
                    f"RAMULATOR2_CONFIG={cfg_path} must contain the literal "
                    "placeholder '{trace_path}' so the trace file path can "
                    "be substituted at runtime.")
        else:
            self._yaml_template = _RAMULATOR2_DEFAULT_YAML

    def simulate_stream(self,
                        num_bytes: int,
                        access_granularity: int,
                        need_init: bool,
                        bytes_per_cycle: int,
                        bytes_per_row: int,
                        num_banks_per_channel: int) -> int:
        addrs = _synthesize_addresses(num_bytes, access_granularity,
                                      bytes_per_cycle, bytes_per_row,
                                      num_banks_per_channel)
        with tempfile.TemporaryDirectory(prefix="ramulator2_") as tmp:
            trace_path = os.path.join(tmp, "trace.txt")
            with open(trace_path, "w") as tf:
                # LoadStoreTrace expects: "LD 0xADDR" or "ST 0xADDR".
                for a in addrs:
                    tf.write(f"LD 0x{a:x}\n")

            yaml_path = os.path.join(tmp, "config.yaml")
            with open(yaml_path, "w") as yf:
                yf.write(self._yaml_template.replace("{trace_path}", trace_path))

            try:
                result = subprocess.run(
                    [self.binary, "-f", yaml_path],
                    capture_output=True, text=True,
                    timeout=self.timeout_sec, check=False)
            except subprocess.TimeoutExpired as e:
                raise ExternalDRAMUnavailable(
                    f"ramulator2 timed out after {self.timeout_sec}s "
                    f"({len(addrs)} requests)") from e

        m = _RAMULATOR2_CYCLES_RE.search(result.stdout or "")
        if not m:
            head = (result.stdout or "")[-400:].strip()
            err = (result.stderr or "")[-200:].strip()
            raise ExternalDRAMUnavailable(
                f"ramulator2 stdout missing 'memory_system_cycles': "
                f"...{head!r} stderr={err!r}")
        cycles = int(m.group(1))
        if cycles <= 0:
            # Ramulator returned cycles=0 — almost always means the frontend
            # exited before the trace drained (e.g. ReadWriteTrace's broken
            # is_finished()). Surface this as a backend failure so the
            # caller can fall back rather than silently using a junk value.
            raise ExternalDRAMUnavailable(
                "ramulator2 reported memory_system_cycles=0 — the trace did "
                "not drain. Check the Frontend impl in your YAML "
                "(use LoadStoreTrace, not ReadWriteTrace).")
        # Add the average tail latency: LoadStoreTrace.is_finished returns
        # true when the *last request is sent*, not when it completes, so
        # memory_system_cycles undercounts by ~1 request latency. Channels
        # are stacked sequentially in the dump; the max across them is the
        # right number for our single-channel synthesized stream.
        tail = max((float(x) for x in _RAMULATOR2_AVGLAT_RE.findall(result.stdout or "")),
                   default=0.0)
        return cycles + int(tail + 0.5)


# --------------------------------------------------------------------------- #
#  DRAMsim3 backend                                                            #
# --------------------------------------------------------------------------- #

class DRAMsim3Backend(ExternalDRAMBackend):
    """Subprocess driver for ``dramsim3main -t trace.txt``."""

    name = "dramsim3"

    def __init__(self,
                 binary_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 timeout_sec: int = 60,
                 max_cycles: int = 100_000_000) -> None:
        bp = (binary_path
              or os.environ.get("DRAMSIM3_BIN")
              or shutil.which("dramsim3main"))
        if not bp or not os.path.isfile(bp):
            raise ExternalDRAMUnavailable(
                "dramsim3main binary not found. Build per "
                "https://github.com/umd-memsys/DRAMsim3 then set "
                "DRAMSIM3_BIN=/path/to/dramsim3main or place it on PATH.")
        self.binary = bp

        cfg = config_path or os.environ.get("DRAMSIM3_CONFIG")
        if not cfg or not os.path.isfile(cfg):
            raise ExternalDRAMUnavailable(
                "DRAMsim3 needs an .ini config file (its presets live in "
                "the source tree, not next to the binary). Set "
                "DRAMSIM3_CONFIG=/path/to/configs/HBM2_4Gb_x128.ini "
                "(or any other preset that ships with DRAMsim3).")
        self.config = cfg
        self.timeout_sec = timeout_sec
        self.max_cycles = max_cycles

    def simulate_stream(self,
                        num_bytes: int,
                        access_granularity: int,
                        need_init: bool,
                        bytes_per_cycle: int,
                        bytes_per_row: int,
                        num_banks_per_channel: int) -> int:
        addrs = _synthesize_addresses(num_bytes, access_granularity,
                                      bytes_per_cycle, bytes_per_row,
                                      num_banks_per_channel)
        # DRAMsim3 trace format: "0xADDR READ <cycle>". Issue requests one
        # per cycle to the controller's input queue — the simulator handles
        # internal scheduling.
        with tempfile.TemporaryDirectory(prefix="dramsim3_") as tmp:
            trace_path = os.path.join(tmp, "trace.txt")
            with open(trace_path, "w") as tf:
                for cycle, a in enumerate(addrs):
                    tf.write(f"0x{a:x} READ {cycle}\n")

            try:
                # dramsim3main writes dramsim3.json into its CWD.
                result = subprocess.run(
                    [self.binary, self.config, "-c", str(self.max_cycles),
                     "-t", trace_path],
                    cwd=tmp, capture_output=True, text=True,
                    timeout=self.timeout_sec, check=False)
            except subprocess.TimeoutExpired as e:
                raise ExternalDRAMUnavailable(
                    f"dramsim3main timed out after {self.timeout_sec}s "
                    f"({len(addrs)} requests)") from e

            json_path = os.path.join(tmp, "dramsim3.json")
            if not os.path.isfile(json_path):
                head = (result.stdout or "")[-400:].strip()
                err = (result.stderr or "")[-200:].strip()
                raise ExternalDRAMUnavailable(
                    f"dramsim3main produced no dramsim3.json: "
                    f"stdout=...{head!r} stderr={err!r}")
            with open(json_path, "r") as jf:
                stats = json.load(jf)

        # DRAMsim3 reports per-channel stats; pick the max ``num_cycles`` so
        # the result reflects the slowest channel (i.e. the wall clock).
        # Falls back to total/avg-latency variants when the schema differs.
        return int(_extract_dramsim3_cycles(stats, fallback=len(addrs)))


def _extract_dramsim3_cycles(stats: dict, fallback: int) -> int:
    """Best-effort extraction of total cycles from a dramsim3.json blob."""
    candidate_keys = ("num_cycles", "total_num_cycles", "num_total_cycles")
    # Newer schema: top-level keyed by channel index ("0", "1", ...) or
    # by a single "global" / "summary" entry.
    if isinstance(stats, dict):
        max_cycles = 0
        for v in stats.values():
            if isinstance(v, dict):
                for k in candidate_keys:
                    if k in v:
                        try:
                            max_cycles = max(max_cycles, int(v[k]))
                        except (TypeError, ValueError):
                            pass
            elif isinstance(v, (int, float)):
                # Top-level scalar (older schema).
                pass
        if max_cycles > 0:
            return max_cycles
        for k in candidate_keys:
            if k in stats:
                try:
                    return int(stats[k])
                except (TypeError, ValueError):
                    pass
    return fallback


# --------------------------------------------------------------------------- #
#  Backend factory                                                             #
# --------------------------------------------------------------------------- #

_BACKENDS = {
    "ramulator2": RamulatorBackend,
    "dramsim3": DRAMsim3Backend,
}


def make_backend(choice: str = "auto") -> ExternalDRAMBackend:
    """Return the first available backend, or raise with a combined error.

    ``choice`` may be ``"auto"``, ``"ramulator2"``, or ``"dramsim3"``.
    With ``"auto"`` Ramulator 2.0 is tried first,
    then DRAMsim3.
    """
    if choice == "auto":
        order = ("ramulator2", "dramsim3")
    elif choice in _BACKENDS:
        order = (choice,)
    else:
        raise ExternalDRAMUnavailable(
            f"Unknown ultra_precise_dram backend: {choice!r}. "
            f"Valid: auto, {', '.join(_BACKENDS)}.")

    errors = []
    for name in order:
        try:
            return _BACKENDS[name]()
        except ExternalDRAMUnavailable as e:
            errors.append(f"  {name}: {e}")
    raise ExternalDRAMUnavailable(
        "no external DRAM backend available:\n" + "\n".join(errors))
