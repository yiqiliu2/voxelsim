"""Thermal models used for side-by-side validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .defaults import DEFAULT_AMBIENT_C
from .trace import PowerTrace, TraceConfig, tile_weights


@dataclass(frozen=True)
class ThermalConfig:
    ambient_c: float = DEFAULT_AMBIENT_C
    throttle_c: float = 85.0
    simple_r_k_per_w: float = 0.10
    simple_c_j_per_k: float = 80.0
    stack_r_sink_k_per_w: float = 1.20
    stack_r_vertical_k_per_w: float = 0.22
    stack_r_lateral_k_per_w: float = 1.80
    logic_c_j_per_k: float = 2.0
    dram_c_j_per_k: float = 3.0


@dataclass
class ThermalResult:
    simple_peak_c: float
    stack_logic_peak_c: float
    stack_dram_peak_c: float
    stack_peak_c: float
    stack_p95_peak_c: float
    simple_slowdown: float
    stack_slowdown: float
    slowdown_delta_pct: float


def simulate_lumped(trace: PowerTrace, cfg: ThermalConfig) -> np.ndarray:
    temp = np.empty(trace.total_power_w.shape[0], dtype=float)
    t = cfg.ambient_c
    for i, power in enumerate(trace.total_power_w):
        dtemp = (power - (t - cfg.ambient_c) / cfg.simple_r_k_per_w) / cfg.simple_c_j_per_k * trace.dt_s
        t += dtemp
        temp[i] = t
    return temp


def simulate_stack(trace: PowerTrace, trace_cfg: TraceConfig, cfg: ThermalConfig) -> Tuple[np.ndarray, np.ndarray]:
    grid = trace_cfg.grid
    n_tiles = grid * grid
    dram_layers = max(1, int(getattr(trace_cfg, "dram_layers", 1)))
    weights = tile_weights(grid, trace_cfg.spatial_policy).reshape(n_tiles)
    uniform = np.full(n_tiles, 1.0 / n_tiles)
    logic_temp = np.full(n_tiles, cfg.ambient_c, dtype=float)
    dram_temp = np.full(n_tiles, cfg.ambient_c, dtype=float)
    logic_peak = np.empty(trace.component_power_w.shape[0], dtype=float)
    dram_peak = np.empty(trace.component_power_w.shape[0], dtype=float)

    component_index = {name: idx for idx, name in enumerate(trace.component_names)}
    lateral_edges: List[Tuple[int, int]] = []
    for y in range(grid):
        for x in range(grid):
            idx = y * grid + x
            if x + 1 < grid:
                lateral_edges.append((idx, idx + 1))
            if y + 1 < grid:
                lateral_edges.append((idx, idx + grid))

    g_sink = 1.0 / cfg.stack_r_sink_k_per_w
    g_vert = 1.0 / cfg.stack_r_vertical_k_per_w
    g_lat = 1.0 / cfg.stack_r_lateral_k_per_w

    for i, row in enumerate(trace.component_power_w):
        if trace.block_power_w is not None:
            p_logic = trace.block_power_w[i, :n_tiles]
            dram_blocks = trace.block_power_w[i, n_tiles:n_tiles * (1 + dram_layers)]
            if dram_blocks.size >= n_tiles * dram_layers:
                p_dram = dram_blocks.reshape(dram_layers, n_tiles).sum(axis=0)
            else:
                p_dram = dram_blocks[:n_tiles]
        else:
            comp = {component: row[idx] for component, idx in component_index.items()}
            logic_power = comp["sa"] + comp["vu"] + comp["sram"] + comp["noc"] + comp["tsv"]
            dram_power = comp["dram"]
            p_logic = logic_power * weights
            p_dram = dram_power * uniform

        q_logic = p_logic + (dram_temp - logic_temp) * g_vert
        q_dram = p_dram + (logic_temp - dram_temp) * g_vert + (cfg.ambient_c - dram_temp) * g_sink
        for a, b in lateral_edges:
            flow = (logic_temp[b] - logic_temp[a]) * g_lat
            q_logic[a] += flow
            q_logic[b] -= flow
            flow = (dram_temp[b] - dram_temp[a]) * g_lat
            q_dram[a] += flow
            q_dram[b] -= flow

        logic_temp += (q_logic / cfg.logic_c_j_per_k) * trace.dt_s
        dram_temp += (q_dram / cfg.dram_c_j_per_k) * trace.dt_s
        logic_peak[i] = float(logic_temp.max())
        dram_peak[i] = float(dram_temp.max())
    return logic_peak, dram_peak


def throttle_slowdown(peak_c: float, threshold_c: float) -> float:
    excess = max(0.0, peak_c - threshold_c)
    return 1.0 + 0.02 * excess


def analyze(trace: PowerTrace, trace_cfg: TraceConfig, thermal_cfg: ThermalConfig) -> ThermalResult:
    simple_temp = simulate_lumped(trace, thermal_cfg)
    logic_temp, dram_temp = simulate_stack(trace, trace_cfg, thermal_cfg)
    simple_peak = float(simple_temp.max())
    logic_peak = float(logic_temp.max())
    dram_peak = float(dram_temp.max())
    stack_peak = max(logic_peak, dram_peak)
    stack_p95 = max(float(np.percentile(logic_temp, 95)), float(np.percentile(dram_temp, 95)))
    simple_slowdown = throttle_slowdown(simple_peak, thermal_cfg.throttle_c)
    stack_slowdown = throttle_slowdown(stack_peak, thermal_cfg.throttle_c)
    return ThermalResult(
        simple_peak_c=simple_peak,
        stack_logic_peak_c=logic_peak,
        stack_dram_peak_c=dram_peak,
        stack_peak_c=stack_peak,
        stack_p95_peak_c=stack_p95,
        simple_slowdown=simple_slowdown,
        stack_slowdown=stack_slowdown,
        slowdown_delta_pct=(stack_slowdown / simple_slowdown - 1.0) * 100.0,
    )


def rankdata(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])
