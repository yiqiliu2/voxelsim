"""CSV and Markdown report writers for thermal validation."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from .models import spearman


def write_csv(rows: List[Dict[str, float | str | int | bool]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, float | str | int | bool]], out_path: Path) -> None:
    simple = [float(row["simple_peak_c"]) for row in rows]
    stack = [float(row["stack_peak_c"]) for row in rows]
    simple_slow = [float(row["simple_slowdown"]) for row in rows]
    stack_slow = [float(row["stack_slowdown"]) for row in rows]
    rho_temp = spearman(simple, stack)
    rho_slow = spearman(simple_slow, stack_slow)
    max_abs_temp_delta = max(abs(s - d) for s, d in zip(simple, stack))
    max_slowdown_delta = max(abs(float(row["slowdown_delta_pct"])) for row in rows)
    max_avg_power_error = max(abs(float(row["trace_avg_power_error_pct"])) for row in rows)
    max_power_density = max(float(row["simple_power_density_w_per_mm2"]) for row in rows)
    hotspot_peaks = [float(row["hotspot_peak_c"]) for row in rows if row.get("hotspot_peak_c") not in ("", None)]
    backend_sets = sorted({str(row.get("thermal_backends", "simple")) for row in rows})
    requested_backend_sets = sorted({str(row.get("requested_thermal_backends", row.get("thermal_backends", "simple"))) for row in rows})
    spatial_rows = sum(1 for row in rows if bool(row.get("spatial_trace_used")))
    attributed_events = sum(int(row.get("spatial_events", 0)) for row in rows)
    inferred_events = sum(int(row.get("inferred_spatial_events", 0)) for row in rows)
    fallback_events = sum(int(row.get("fallback_events", 0)) for row in rows)
    false_negative = any(float(row["simple_margin_c"]) >= 0 and float(row["stack_margin_c"]) < 0 for row in rows)
    lines = [
        "# Thermal Validation Summary",
        "",
        f"Runs analyzed: {len(rows)}",
        f"Spearman(simple peak, stack peak): {rho_temp:.4f}" if math.isfinite(rho_temp) else "Spearman(simple peak, stack peak): n/a",
        f"Spearman(simple slowdown, stack slowdown): {rho_slow:.4f}" if math.isfinite(rho_slow) else "Spearman(simple slowdown, stack slowdown): n/a",
        f"Max absolute peak-temperature delta: {max_abs_temp_delta:.3f} C",
        f"Max slowdown delta: {max_slowdown_delta:.3f}%",
        f"Max trace average-power error: {max_avg_power_error:.3f}%",
        f"Max existing TSIM simple power density: {max_power_density:.4f} W/mm^2",
        f"Max HotSpot peak temperature: {max(hotspot_peaks):.3f} C" if hotspot_peaks else "Max HotSpot peak temperature: n/a",
        f"Requested thermal backends: {', '.join(requested_backend_sets)}",
        f"Effective thermal backends: {', '.join(backend_sets)}",
        f"Rows using spatial IDs: {spatial_rows}/{len(rows)}",
        f"TSIM-attached spatial op records: {attributed_events}",
        f"Inferred spatial op records: {inferred_events}",
        f"Fallback dynamic events: {fallback_events}",
        f"Simple-model false negative: {'yes' if false_negative else 'no'}",
        "",
        "## Per-Policy Rank Check",
        "",
        "| Policy | Runs | Spearman(simple peak, stack peak) |",
        "| --- | ---: | ---: |",
    ]
    for policy in sorted({str(row["spatial_policy"]) for row in rows}):
        group = [row for row in rows if str(row["spatial_policy"]) == policy]
        rho = spearman(
            [float(row["simple_peak_c"]) for row in group],
            [float(row["stack_peak_c"]) for row in group],
        )
        rho_text = f"{rho:.4f}" if math.isfinite(rho) else "n/a"
        lines.append(f"| {policy} | {len(group)} | {rho_text} |")

    lines.extend([
        "",
        "## Runs",
        "",
        "| Run | Policy | DRAM map | DRAM granularity | Bins | dt us | Major-op sample hit | Backends | Simple | Spatial ops | Inferred ops | Fallback events | TSIM avg W | Trace avg err % | Power density W/mm^2 | Simple peak C | Stack peak C | HotSpot peak C | HotSpot-stack delta C | Stack margin C | Slowdown delta % | HotSpot |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in rows:
        hotspot_peak = row.get("hotspot_peak_c")
        hotspot_delta = row.get("hotspot_delta_vs_stack_c")
        hotspot_peak_text = f"{float(hotspot_peak):.2f}" if hotspot_peak not in ("", None) else "n/a"
        hotspot_delta_text = f"{float(hotspot_delta):.2f}" if hotspot_delta not in ("", None) else "n/a"
        lines.append(
            f"| {row['label']} | {row['spatial_policy']} | {row.get('dram_bank_mapping', '')} | "
            f"{row.get('dram_floorplan_granularity', '')} | {int(row.get('thermal_bins', 0))} | "
            f"{float(row.get('thermal_dt_us', 0.0)):.3f} | "
            f"{float(row.get('major_ops_meeting_target_fraction', 0.0)):.3f} | "
            f"{row.get('thermal_backends', 'simple')} | "
            f"{row.get('simple_status', 'ok')} | {int(row['spatial_events'])} | "
            f"{int(row.get('inferred_spatial_events', 0))} | {int(row['fallback_events'])} | "
            f"{float(row['avg_power_w']):.2f} | "
            f"{float(row['trace_avg_power_error_pct']):.3f} | "
            f"{float(row['simple_power_density_w_per_mm2']):.4f} | "
            f"{float(row['simple_peak_c']):.2f} | {float(row['stack_peak_c']):.2f} | "
            f"{hotspot_peak_text} | {hotspot_delta_text} | "
            f"{float(row['stack_margin_c']):.2f} | {float(row['slowdown_delta_pct']):.3f} | "
            f"{row['hotspot_status']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
