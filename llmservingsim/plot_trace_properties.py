#!/usr/bin/env python3
"""Plot distributions of trace properties from a JSONL workload file.

Each record is a session with fields:
    session_id, arrival_time_ns, sub_requests

Each sub_request has:
    input_toks, output_toks, tool_duration_ns

Plots (2×2):
  1. Input token length distribution
  2. Output token length distribution
  3. Tool-call duration distribution (non-zero entries only)
  4. Inter-arrival time distribution

Usage:
    python plot_trace_properties.py workloads/trace.jsonl
    python plot_trace_properties.py workloads/trace.jsonl -o my_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_trace(path: Path) -> list[dict]:
    sessions = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                sessions.append(json.loads(line))
    if not sessions:
        sys.exit(f"No records found in {path}")
    return sessions


def extract_stats(
    sessions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_toks, output_toks, tool_dur_ms = [], [], []
    for s in sessions:
        for sr in s["sub_requests"]:
            input_toks.append(sr["input_toks"])
            output_toks.append(sr["output_toks"])
            if sr["tool_duration_ns"] > 0:
                tool_dur_ms.append(sr["tool_duration_ns"] / 1e6)

    arrivals_s = np.array(sorted(s["arrival_time_ns"] / 1e9 for s in sessions))
    inter_arrival_s = np.diff(arrivals_s)

    return (
        np.array(input_toks, dtype=np.float64),
        np.array(output_toks, dtype=np.float64),
        np.array(tool_dur_ms, dtype=np.float64),
        inter_arrival_s,
    )


def plot(
    input_toks: np.ndarray,
    output_toks: np.ndarray,
    tool_dur_ms: np.ndarray,
    inter_arrival_s: np.ndarray,
    out_stem: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
    from benchmark_scripts import fig_common  # applies rcParams (serif font, pdf.fonttype=42)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    FS = 23  # axis label fontsize
    LOC = MaxNLocator(nbins=4, min_n_ticks=3)

    fig, ((ax_in, ax_out), (ax_tool, ax_arr)) = plt.subplots(2, 2, figsize=(14, 7))

    panels = [
        (ax_in,   input_toks,      "Input tokens",            fig_common.colors[0]),
        (ax_out,  output_toks,     "Output tokens",           fig_common.colors[1]),
        (ax_tool, tool_dur_ms,     "Tool-call duration (ms)", fig_common.colors[2]),
        (ax_arr,  inter_arrival_s, "Inter-arrival time (s)",  fig_common.colors[4]),
    ]

    for ax, data, xlabel, color in panels:
        ax.hist(data, bins=40, color=color, edgecolor="none")
        ax.set_xlabel(xlabel, fontsize=FS)
        ax.set_ylabel("Count", fontsize=FS)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("jsonl", type=Path, help="Path to workload JSONL file")
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output path stem (default: <jsonl stem>_props alongside the file)",
    )
    args = parser.parse_args()

    jsonl_path = args.jsonl.expanduser().resolve()
    if not jsonl_path.exists():
        sys.exit(f"File not found: {jsonl_path}")

    out_stem = args.output or jsonl_path.with_name(jsonl_path.stem + "_props")

    print(f"Loading {jsonl_path} ...")
    sessions = load_trace(jsonl_path)
    print(f"  {len(sessions)} sessions")

    input_toks, output_toks, tool_dur_ms, inter_arrival_s = extract_stats(sessions)
    print(f"  {len(input_toks)} sub-requests, {len(tool_dur_ms)} with tool calls")

    plot(input_toks, output_toks, tool_dur_ms, inter_arrival_s, out_stem)


if __name__ == "__main__":
    main()
