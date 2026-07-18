#!/usr/bin/env python3
"""Generate op-breakdown sweep data over (sequence_length, batch_size).

Grid for the figure:
  rows = seq in {1024, 2048, 4096}
  cols = decode b16 / decode b32 / decode b64 / prefill b1
  each cell = 4 model stacked op-breakdown bars

The trace log path does NOT encode sequence length, so each seq is written to
its own trace_out_dir_base (results/logs_seq{S}) to avoid collisions.  Pickle
dirs are already seq-keyed (outputs_icbm_{S}[_prefill]).
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fixed hardware (= default_params).
CORES   = 256
SA      = 32
SRAM_KB = 2048
DRAM_BW = 12288
NOC_TOPO = 1
NOC_BW   = 16
CG      = 8
ROW     = 8192
HW_JSON = f"sa_{SA}_vu_{SA}_drambw_{DRAM_BW}_noc_{NOC_TOPO}_{NOC_BW}_trcd_14_trp_14"

SF_DECODE  = 1.03
SF_PREFILL = 1.1

MODELS = {"llama3-70": 80}

SEQS         = [1024, 2048, 4096]

# Only the (seq, batch) combos actually drawn by Fig 20's four panels.
# Panel 1: seq sweep, decode bs=32          → (1024,32) (2048,32) (4096,32)
# Panel 2: batch sweep, decode seq=2048     → (2048,16) (2048,32)       (2048,64)
# Panel 3: seq sweep, prefill bs=1          → (1024, 1) (2048, 1) (4096, 1)
# Panel 4: batch sweep, prefill seq=2048    → (2048, 1)             (2048, 2) (2048, 4)
DECODE_JOBS  = [(1024, 32), (2048, 16), (2048, 32), (2048, 64), (4096, 32)]
PREFILL_JOBS = [(1024, 1),  (2048, 1),  (2048, 2),  (2048, 4),  (4096, 1)]

MAX_PARALLEL = 4   # gentle: runs alongside the smile rerun


def trace_base(seq):
    return f"results/logs_seq{seq}"


def out_log(model, seq, batch, mode):
    return os.path.join(
        trace_base(seq), model, f"bs_{batch}", f"core_{CORES}", mode,
        f"sa_{SA}-vu_{SA}", f"sram_{SRAM_KB}-drambw_{DRAM_BW}_PLACEHOLDER",
        f"topo_{NOC_TOPO}-nocbw{NOC_BW}", "best",
        f"output_cg_{CG}_row_{ROW}.log")


def is_done(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return "Overall Util:" in f.read()
    except Exception:
        return False


def build_jobs():
    jobs = []
    for model, layers in MODELS.items():
        for seq, b in DECODE_JOBS:
            jobs.append(dict(model=model, layers=layers, seq=seq, batch=b,
                             prefill=False))
        for seq, b in PREFILL_JOBS:
            jobs.append(dict(model=model, layers=layers, seq=seq,
                             batch=b, prefill=True))
    return jobs


def run_job(job):
    model, layers = job["model"], job["layers"]
    seq, batch, prefill = job["seq"], job["batch"], job["prefill"]
    mode = "prefill" if prefill else "decode"
    log = out_log(model, seq, batch, mode)
    tag = f"{model} seq{seq} b{batch} {mode}"
    if is_done(log):
        return (tag, "cached")

    pickle_dir = f"results/pickles/outputs_icbm_{seq}"
    if prefill:
        pickle_dir += "_prefill"
    sf = SF_PREFILL if prefill else SF_DECODE
    sim_layers = 1 if prefill else 0

    cmd = (f"python3 icbm_launch.py {model} {CORES}"
           f" --core_mem_kb {SRAM_KB}"
           f" --output_dir {pickle_dir}"
           f" --layers {layers}"
           f" --batch_size {batch}"
           f" --sequence_length {seq}"
           f" --use_pickle"
           f" --split_factor {sf}"
           f" --hw_json {HW_JSON}"
           f" --core_group {CG}"
           f" --dram_name PLACEHOLDER"
           f" --dram_bw {DRAM_BW}"
           f" --sim_layers {sim_layers}"
           f" --trace_out_dir_base {trace_base(seq)}")
    if prefill:
        cmd += " --prefill"

    r = subprocess.run(cmd, shell=True, capture_output=True, timeout=1800)
    if is_done(log):
        return (tag, "done")
    err = r.stderr.decode()[-400:] if r.stderr else ""
    if r.returncode != 0:
        return (tag, f"FAILED (exit {r.returncode}): {err}")


def main():
    jobs = build_jobs()
    print(f"{len(jobs)} jobs, {MAX_PARALLEL}-way parallel", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futs = {ex.submit(run_job, j): j for j in jobs}
        for fut in as_completed(futs):
            tag, status = fut.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {tag}: {status}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
