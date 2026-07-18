#!/usr/bin/env python3
"""Generate a new agentic-session JSONL trace with Poisson-sampled arrivals.

Reads a source JSONL workload (agentic-sessions format), assigns new arrival
times drawn from a Poisson process at the requested rate, and writes the result
to an output file.  If --num-sessions exceeds the number of sessions in the
source file the session list is wrapped around cyclically; repeated copies get
a suffix appended to their session_id to keep IDs unique.

Usage:
    python gen_poisson_trace.py swe-bench-faster.jsonl \\
        --rate 0.5 --num-sessions 200 --output out.jsonl

    python gen_poisson_trace.py swe-bench-faster.jsonl \\
        --rate 2.0 --output out.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_sessions(path: Path) -> list[dict]:
    sessions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    if not sessions:
        sys.exit(f"No records found in {path}")
    return sessions


def poisson_arrivals(n: int, rate_per_s: float, rng: np.random.Generator) -> np.ndarray:
    """Return n arrival times (ns) sampled from a Poisson process at rate_per_s."""
    inter_arrival_s = rng.exponential(1.0 / rate_per_s, size=n)
    arrival_s = np.concatenate([[0.0], np.cumsum(inter_arrival_s[:-1])])
    return (arrival_s * 1e9).astype(np.int64)


def build_output(
    source: list[dict],
    n_sessions: int,
    arrivals_ns: np.ndarray,
) -> list[dict]:
    n_src = len(source)
    out = []
    for i in range(n_sessions):
        rep, idx = divmod(i, n_src)
        sess = dict(source[idx])  # shallow copy — sub_requests list is not mutated
        sess["arrival_time_ns"] = int(arrivals_ns[i])
        if rep > 0:
            sess["session_id"] = f"{source[idx]['session_id']}_r{rep}"
        out.append(sess)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Source JSONL workload file")
    parser.add_argument(
        "--rate", "-r", type=float, required=True,
        help="Mean arrival rate (sessions per second)",
    )
    parser.add_argument(
        "--num-sessions", "-n", type=int, default=None,
        help="Number of sessions to emit (default: same as source file)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    if args.rate <= 0:
        sys.exit("--rate must be positive")

    src_path = args.input.expanduser().resolve()
    if not src_path.exists():
        sys.exit(f"File not found: {src_path}")

    source = load_sessions(src_path)
    n_src = len(source)
    n_sessions = args.num_sessions if args.num_sessions is not None else n_src

    if n_sessions <= 0:
        sys.exit("--num-sessions must be positive")

    print(f"Source: {src_path} ({n_src} sessions)")
    print(f"Emitting {n_sessions} sessions at rate {args.rate} sessions/s")
    if n_sessions > n_src:
        print(f"  Wrapping source ({n_sessions // n_src} full passes + {n_sessions % n_src} remainder)")

    rng = np.random.default_rng(args.seed)
    arrivals_ns = poisson_arrivals(n_sessions, args.rate, rng)

    sessions_out = build_output(source, n_sessions, arrivals_ns)

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for sess in sessions_out:
            f.write(json.dumps(sess) + "\n")

    duration_s = arrivals_ns[-1] / 1e9 if n_sessions > 1 else 0.0
    print(f"Saved: {out_path}")
    print(f"  Trace spans {duration_s:.1f}s  |  mean inter-arrival {1/args.rate:.3f}s")


if __name__ == "__main__":
    main()
