#!/usr/bin/env python3
"""Analyze a llmservingsim workload trace to determine:

  (1) the average number of requests batched together during decode, and
  (2) whether that decode batch size is limited by prefill throughput
      or by the request arrival rate.

Timing model recovered from the trace (all times share one global clock):
    prefill_start = arrival + queuing_delay
    decode_start  = arrival + queuing_delay + TTFT   (== first token emitted)
    decode_end    = end_time
    (identity: arrival + queuing_delay + TTFT + sum(ITL) == end_time)

So each request occupies:
    prefill interval [prefill_start, decode_start)   duration = TTFT
    decode  interval [decode_start , decode_end )    duration = sum(ITL)

Usage:
    python3 analyze_decode_batch.py [path/to/workload.csv]
"""
import sys
import csv
import ast


def load(path):
    reqs = []
    with open(path) as f:
        for r in csv.DictReader(f):
            arrival = int(r["arrival"])
            queuing = int(r["queuing_delay"])
            ttft = int(r["TTFT"])
            end = int(r["end_time"])
            prefill_start = arrival + queuing
            decode_start = prefill_start + ttft
            reqs.append({
                "arrival": arrival,
                "queuing": queuing,
                "ttft": ttft,
                "end": end,
                "input": int(r["input"]),
                "output": int(r["output"]),
                "prefill_start": prefill_start,
                "decode_start": decode_start,
                "decode_end": end,
            })
    return reqs


def time_weighted_concurrency(intervals):
    """Given [(start, end), ...], return (avg concurrency over the union span,
    avg concurrency while >=1 active, max concurrency, union duration)."""
    events = []
    for s, e in intervals:
        if e <= s:
            continue
        events.append((s, +1))
        events.append((e, -1))
    events.sort()

    active = 0
    prev_t = None
    span_start = events[0][0]
    span_end = events[-1][0]
    area = 0           # integral of concurrency dt
    busy_time = 0      # time with active >= 1
    max_active = 0
    for t, delta in events:
        if prev_t is not None and t > prev_t:
            dt = t - prev_t
            area += active * dt
            if active >= 1:
                busy_time += dt
        active += delta
        max_active = max(max_active, active)
        prev_t = t

    total_span = span_end - span_start
    avg_over_span = area / total_span if total_span else 0
    avg_over_busy = area / busy_time if busy_time else 0
    return avg_over_span, avg_over_busy, max_active, total_span, busy_time


def union_duration(intervals):
    """Total wall-clock time during which at least one interval is active."""
    xs = sorted((s, e) for s, e in intervals if e > s)
    total = 0
    cur_s, cur_e = None, None
    for s, e in xs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "3D-workloads/swebench_heavy.csv"
    reqs = load(path)
    n = len(reqs)

    decode_iv = [(r["decode_start"], r["decode_end"]) for r in reqs]
    prefill_iv = [(r["prefill_start"], r["decode_start"]) for r in reqs]

    # overall experiment span
    first_arrival = min(r["arrival"] for r in reqs)
    last_end = max(r["end"] for r in reqs)
    total_span = last_end - first_arrival

    # (1) decode batch size --------------------------------------------------
    d_avg_span, d_avg_busy, d_max, d_span, d_busy = time_weighted_concurrency(decode_iv)

    # (2) prefill throughput vs request rate ---------------------------------
    p_avg_span, p_avg_busy, p_max, p_span, p_busy = time_weighted_concurrency(prefill_iv)
    prefill_union = union_duration(prefill_iv)
    decode_union = union_duration(decode_iv)

    total_prefill_work = sum(r["ttft"] for r in reqs)        # sum of prefill durations
    total_decode_work = sum(r["end"] - r["decode_start"] for r in reqs)

    # request rate
    arrivals = sorted(r["arrival"] for r in reqs)
    span_arr = arrivals[-1] - arrivals[0]
    mean_interarrival = span_arr / (n - 1) if n > 1 else 0
    mean_prefill = total_prefill_work / n
    mean_queuing = sum(r["queuing"] for r in reqs) / n
    mean_ttft = total_prefill_work / n
    mean_decode = total_decode_work / n

    # prefill utilization: fraction of time at least one prefill is running
    prefill_busy_frac = prefill_union / total_span
    # is there backlog? compare queuing delay to prefill time
    queue_to_prefill = mean_queuing / mean_prefill if mean_prefill else 0

    def fmt(x):
        return f"{x:,.3f}" if isinstance(x, float) else f"{x:,}"

    print("=" * 70)
    print(f"Workload: {path}")
    print(f"Requests: {n}")
    print("=" * 70)

    print("\n(1) DECODE BATCHING")
    print("-" * 70)
    print(f"  Avg # concurrently-decoding requests")
    print(f"    over whole experiment span      : {d_avg_span:8.2f}")
    print(f"    over time decode is active (>=1): {d_avg_busy:8.2f}   <-- effective decode batch")
    print(f"  Max concurrent decode requests    : {d_max}")
    print(f"  Decode active span / total span   : {decode_union/total_span:6.1%}")

    print("\n(2) WHAT LIMITS THE DECODE BATCH SIZE?")
    print("-" * 70)
    print(f"  Mean inter-arrival time           : {mean_interarrival:14,.0f}")
    print(f"  Mean prefill time (TTFT)          : {mean_prefill:14,.0f}")
    print(f"  Mean queuing delay (pre-prefill)  : {mean_queuing:14,.0f}")
    print(f"  Mean decode time per req          : {mean_decode:14,.0f}")
    print()
    print(f"  Prefill busy fraction of timeline : {prefill_busy_frac:6.1%}")
    print(f"  Avg concurrent prefills (when busy): {p_avg_busy:7.2f}")
    print(f"  Queuing delay / prefill time ratio : {queue_to_prefill:7.2f}")
    print(f"  Prefill offered load (prefill/interarrival): {mean_prefill/mean_interarrival:6.3f}")

    # Little's law: if decode keeps up with arrivals, the average decode batch
    # equals (arrival rate) x (decode duration) = mean_decode / mean_interarrival.
    littles_batch = mean_decode / mean_interarrival if mean_interarrival else 0
    batch_ratio = d_avg_busy / littles_batch if littles_batch else 0

    # Stability: is the queue backlog growing? Compare mean queuing delay in the
    # first vs second half of arrivals. Growth => prefill can't keep up.
    by_arrival = sorted(reqs, key=lambda r: r["arrival"])
    half = n // 2
    q_first = sum(r["queuing"] for r in by_arrival[:half]) / half
    q_second = sum(r["queuing"] for r in by_arrival[half:]) / (n - half)
    q_growth = q_second / q_first if q_first else float("inf")

    print("\n  LITTLE'S LAW CHECK")
    print(f"    Expected decode batch from request rate = decode_time / interarrival")
    print(f"      = {mean_decode:,.0f} / {mean_interarrival:,.0f} = {littles_batch:6.2f}")
    print(f"    Measured decode batch                    = {d_avg_busy:6.2f}")
    print(f"    Measured / expected ratio                = {batch_ratio:6.2f}")
    print(f"    Mean queuing delay  1st half / 2nd half  = "
          f"{q_first:,.0f} / {q_second:,.0f}  (growth {q_growth:.2f}x)")

    print("\n  DIAGNOSIS")
    print("  " + "-" * 66)
    # If the measured batch matches the request-rate prediction and the queue is
    # not growing, decode is keeping up with arrivals => request-rate limited.
    keeps_up = batch_ratio >= 0.85
    stable_queue = q_growth < 1.5
    if keeps_up and stable_queue:
        verdict = "REQUEST-RATE LIMITED"
        why = ("the measured decode batch matches the request-rate prediction "
               "(Little's law) and the queue backlog is stable, so decode keeps up "
               "with arrivals. Prefill has spare capacity "
               f"({prefill_busy_frac:.0%} busy, {p_avg_busy:.1f} concurrent) and is "
               "NOT the bottleneck. The batch is small simply because requests do not "
               "arrive fast enough to fill it.")
    else:
        verdict = "PREFILL-THROUGHPUT LIMITED"
        why = ("the measured decode batch falls short of the request-rate prediction "
               "and/or the queue backlog grows over time, so requests pile up waiting "
               "for prefill before they can join the decode batch.")
    print(f"  => {verdict}")
    print(f"     {why}")
    print()


if __name__ == "__main__":
    main()
