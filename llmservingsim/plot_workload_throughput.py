#!/usr/bin/env python3
"""Plot prefill and decode token throughput over time from a workload CSV.

The CSV must have columns:
    input, output, arrival, end_time, TTFT

All time columns are in nanoseconds.
  prefill phase: [arrival, arrival + TTFT)
  decode phase:  [arrival + TTFT, end_time)

Usage:
    python plot_workload_throughput.py 3D-workloads/sharegpt.csv
    python plot_workload_throughput.py 3D-workloads/sharegpt.csv --tick 2 --output plot.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


def load_requests(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrivals, ttfts, end_times, inputs, outputs = [], [], [], [], []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                arrivals.append(int(row["arrival"]))
                ttfts.append(int(row["TTFT"]))
                end_times.append(int(row["end_time"]))
                inputs.append(int(row["input"]))
                outputs.append(int(row["output"]))
            except (KeyError, ValueError) as exc:
                sys.exit(f"Bad row in {path}: {exc}\nRow: {dict(row)}")
    if not arrivals:
        sys.exit(f"No rows found in {path}")
    return (
        np.array(arrivals, dtype=np.float64),
        np.array(ttfts, dtype=np.float64),
        np.array(end_times, dtype=np.float64),
        np.array(inputs, dtype=np.float64),
        np.array(outputs, dtype=np.float64),
    )


def compute_throughput(
    arrivals: np.ndarray,
    ttfts: np.ndarray,
    end_times: np.ndarray,
    inputs: np.ndarray,
    outputs: np.ndarray,
    tick_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tick_midpoints_s, prefill_tput_tok_per_s, decode_tput_tok_per_s)."""
    prefill_ends = arrivals + ttfts  # ns
    t0 = arrivals.min()

    prefill_ends_s = (prefill_ends - t0) / 1e9
    end_times_s = (end_times - t0) / 1e9

    n_ticks = int(np.ceil(end_times_s.max() / tick_s))
    midpoints = (np.arange(n_ticks) + 0.5) * tick_s

    # --- prefill: distribute input tokens proportionally over [arrival, prefill_end) ---
    arrivals_s = (arrivals - t0) / 1e9
    prefill_dur_s = prefill_ends_s - arrivals_s
    prefill_rates = np.where(prefill_dur_s > 0, inputs / prefill_dur_s, 0.0)  # tok/s

    prefill_tput = np.zeros(n_ticks)
    for b in range(n_ticks):
        t_lo = b * tick_s
        t_hi = (b + 1) * tick_s
        overlap = np.maximum(0.0, np.minimum(prefill_ends_s, t_hi) - np.maximum(arrivals_s, t_lo))
        prefill_tput[b] = np.dot(prefill_rates, overlap)
    prefill_tput /= tick_s  # tokens in bucket -> tokens/s

    # --- decode: distribute output tokens proportionally over decode window ---
    decode_dur_s = end_times_s - prefill_ends_s
    decode_rates = np.where(decode_dur_s > 0, outputs / decode_dur_s, 0.0)  # tok/s

    decode_tput = np.zeros(n_ticks)
    for b in range(n_ticks):
        t_lo = b * tick_s
        t_hi = (b + 1) * tick_s
        overlap = np.maximum(0.0, np.minimum(end_times_s, t_hi) - np.maximum(prefill_ends_s, t_lo))
        decode_tput[b] = np.dot(decode_rates, overlap)
    decode_tput /= tick_s  # tokens in bucket -> tokens/s

    return midpoints, prefill_tput, decode_tput


def plot(
    midpoints: np.ndarray,
    prefill_tput: np.ndarray,
    decode_tput: np.ndarray,
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

    fig, (ax_pre, ax_dec) = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)

    ax_pre.plot(midpoints, prefill_tput, color=fig_common.colors[0], linewidth=2)
    ax_pre.set_ylabel("Prefill\n(tokens/s)", fontsize=23)
    ax_pre.set_ylim(bottom=0)
    ax_pre.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax_pre.grid(True, alpha=0.3)

    ax_dec.plot(midpoints, decode_tput, color=fig_common.colors[1], linewidth=2)
    ax_dec.set_ylabel("Decode\n(tokens/s)", fontsize=23)
    ax_dec.set_xlabel("Time (s)", fontsize=23)
    ax_dec.set_ylim(bottom=0)
    ax_dec.set_xlim(left=0)
    ax_dec.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax_dec.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax_dec.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_stem.with_suffix(f".{ext}")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="Path to workload CSV")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output PNG path (default: <csv stem>_throughput.png alongside the CSV)")
    parser.add_argument("--tick", type=float, default=1.0,
                        help="Bucket width in seconds (default: 1.0)")
    args = parser.parse_args()

    csv_path: Path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")

    out_path: Path = args.output or csv_path.with_name(csv_path.stem + "_throughput")

    print(f"Loading {csv_path} …")
    arrivals, ttfts, end_times, inputs, outputs = load_requests(csv_path)
    print(f"  {len(arrivals)} requests, tick={args.tick}s")

    midpoints, prefill_tput, decode_tput = compute_throughput(
        arrivals, ttfts, end_times, inputs, outputs, tick_s=args.tick
    )

    plot(midpoints, prefill_tput, decode_tput, out_stem=out_path)


if __name__ == "__main__":
    main()
