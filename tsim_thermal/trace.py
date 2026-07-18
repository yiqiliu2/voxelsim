"""Power trace construction and HotSpot-compatible export."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from tsim_components.mem import (
    HBM_PACKAGE_AREA_MM2,
    HBM_PACKAGE_CAPACITY_MB,
    get_hbm_package_count,
    get_hbm_package_footprint_mm,
    get_per_cycle_bytes_per_core_from_DRAM_config,
)
from tsim_components.noc_power import (
    NoCPowerConfig,
    describe as describe_noc_power,
    scale_energy_pj as scale_noc_energy_pj,
)

from .artifacts import COMPONENTS, LOGIC_COMPONENTS, RunArtifacts
from .defaults import (
    DEFAULT_AMBIENT_C,
    DEFAULT_BOND_THICKNESS_UM,
    DEFAULT_C_CONVEC_J_PER_K,
    DEFAULT_R_CONVEC_K_PER_W,
    LOGIC_STATIC_POWER_DENSITY_W_PER_MM2,
)


TSV_AREA_MM2_AT_12_TBPS = 90.0
TSV_AREA_REFERENCE_GBPS = 12 * 1024


@dataclass(frozen=True)
class TraceConfig:
    npu_freq_mhz: float = 1500.0
    duration_ms: float = 250.0
    max_bins: int = 25000
    major_op_samples: int = 4
    major_op_percentile: float = 25.0
    grid: int = 4
    spatial_policy: str = "center_hotspot"
    die_size_mm: float | None = None
    ambient_c: float = DEFAULT_AMBIENT_C
    dram_layers: int = 1
    dram_capacity_mb: int = 192 * 1024
    hbm_package_capacity_mb: int = HBM_PACKAGE_CAPACITY_MB
    hbm_package_area_mm2: float = HBM_PACKAGE_AREA_MM2
    hbm_package_aspect_ratio: float = 1.0
    hbm_banks_per_package: int = 16
    hbm_interleave_stripe_bytes: int = 256
    dram_floorplan_granularity: str = "bank"
    dram_bank_mapping: str = "address_trace"
    logic_floorplan: str = "intra_core"
    hotspot_grid: int = 128
    noc_power_backend: str = "tsim_simple"
    noc_power_flit_bits: int = 64
    noc_power_injection_rate: float = 0.3
    noc_power_link_length_mm: float = 1.0
    noc_power_dsent_tech: str = "TG11LVT"
    noc_power_orion_version: int = 2
    r_convec_k_per_w: float = DEFAULT_R_CONVEC_K_PER_W
    c_convec_j_per_k: float = DEFAULT_C_CONVEC_J_PER_K
    cooling_profile: str = "conservative_h100_pcie_dual_slot_air"
    logic_direct_to_heatsink: bool = False
    bond_thickness_um: float = DEFAULT_BOND_THICKNESS_UM
    tsv_region_area_mm2_at_12tbps: float = TSV_AREA_MM2_AT_12_TBPS


@dataclass
class PowerTrace:
    component_power_w: np.ndarray
    dt_s: float
    component_names: Tuple[str, ...] = COMPONENTS
    block_power_w: np.ndarray | None = None
    block_names: Tuple[str, ...] | None = None
    spatial_attribution: Dict[str, Any] | None = None
    spatial_events: int = 0
    inferred_events: int = 0
    fallback_events: int = 0

    @property
    def total_power_w(self) -> np.ndarray:
        return self.component_power_w.sum(axis=1)

    @property
    def duration_s(self) -> float:
        return self.component_power_w.shape[0] * self.dt_s


def estimate_logic_area_mm2(artifact: RunArtifacts) -> float:
    if artifact.summary.logic_static_power_w <= 0:
        return float("nan")
    return artifact.summary.logic_static_power_w / LOGIC_STATIC_POWER_DENSITY_W_PER_MM2


def effective_die_size_mm(artifact: RunArtifacts, cfg: TraceConfig) -> float:
    if cfg.die_size_mm is not None:
        return float(cfg.die_size_mm)
    if cfg.logic_floorplan == "intra_core":
        return math.sqrt(intracore_total_logic_area_mm2(artifact, cfg))
    area_mm2 = estimate_logic_area_mm2(artifact)
    if not math.isfinite(area_mm2) or area_mm2 <= 0:
        raise ValueError(
            "cannot derive logic die size: simulator logic static power is missing or non-positive; "
            "pass --die-size-mm explicitly"
        )
    return math.sqrt(area_mm2)


def interval_to_bins(start_s: float, end_s: float, dt_s: float, n_bins: int) -> Tuple[int, int]:
    if end_s <= start_s or dt_s <= 0 or n_bins <= 0:
        return 0, 0
    trace_end_s = n_bins * dt_s
    if end_s <= 0 or start_s >= trace_end_s:
        return 0, 0
    clipped_start_s = max(0.0, min(start_s, trace_end_s))
    clipped_end_s = max(0.0, min(end_s, trace_end_s))
    if clipped_end_s <= clipped_start_s:
        return 0, 0
    start = max(0, min(n_bins - 1, int(math.floor(clipped_start_s / dt_s))))
    end = max(start + 1, min(n_bins, int(math.ceil(clipped_end_s / dt_s))))
    return start, end


def noc_power_config(cfg: TraceConfig) -> NoCPowerConfig:
    return NoCPowerConfig(
        backend=cfg.noc_power_backend,
        frequency_hz=cfg.npu_freq_mhz * 1e6,
        flit_bits=cfg.noc_power_flit_bits,
        injection_rate=cfg.noc_power_injection_rate,
        link_length_mm=cfg.noc_power_link_length_mm,
        dsent_tech=cfg.noc_power_dsent_tech,
        orion_version=cfg.noc_power_orion_version,
    )


def modeled_noc_energy_pj(base_energy_pj: float, cfg: TraceConfig) -> float:
    return scale_noc_energy_pj(base_energy_pj, noc_power_config(cfg))


def noc_power_metadata(cfg: TraceConfig) -> Dict[str, Any]:
    return dict(describe_noc_power(noc_power_config(cfg)))


def _oplog_dynamic_energy_pj_by_component(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, float]:
    totals = {component: 0.0 for component in COMPONENTS}
    for op in artifact.op_logs:
        totals["sa"] += float(getattr(op, "energy_sa", 0.0) or 0.0)
        totals["vu"] += float(getattr(op, "energy_vu", 0.0) or 0.0)
        totals["sram"] += float(getattr(op, "energy_sram", 0.0) or 0.0)
        totals["noc"] += modeled_noc_energy_pj(float(getattr(op, "energy_noc", 0.0) or 0.0), cfg)
        totals["dram"] += float(getattr(op, "energy_dram", 0.0) or 0.0)
        totals["tsv"] += float(getattr(op, "energy_tsv", 0.0) or 0.0)
    return totals


def _summary_dynamic_energy_pj_by_component(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, float]:
    runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    dynamic_component_w = getattr(artifact.summary, "dynamic_component_w", {}) or {}
    totals: Dict[str, float] = {}
    for component in COMPONENTS:
        energy_pj = float(dynamic_component_w.get(component, 0.0) or 0.0) * runtime_s * 1.0e12
        if component == "noc":
            energy_pj = modeled_noc_energy_pj(energy_pj, cfg)
        totals[component] = energy_pj
    return totals


def dynamic_energy_scales(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, float]:
    """Scale fused-op template energies to the run-level TSIM energy summary.

    Prefill pickles can contain a compact per-layer operator template while the
    log summary reports the full model pass. Reconciliation at component level
    keeps each event's timing/spatial placement while matching TSIM's audited
    run-level dynamic energy.
    """
    observed = _oplog_dynamic_energy_pj_by_component(artifact, cfg)
    target = _summary_dynamic_energy_pj_by_component(artifact, cfg)
    scales: Dict[str, float] = {}
    for component in COMPONENTS:
        observed_energy = observed.get(component, 0.0)
        target_energy = target.get(component, 0.0)
        if observed_energy > 0.0 and target_energy > 0.0:
            scales[component] = target_energy / observed_energy
        else:
            scales[component] = 1.0
    return scales


def template_repeat_count(artifact: RunArtifacts, cfg: TraceConfig) -> int:
    if artifact.run_id.mode != "prefill":
        return 1
    scales = dynamic_energy_scales(artifact, cfg)
    active_scales = [
        value for value in scales.values()
        if math.isfinite(value) and value > 1.5
    ]
    if not active_scales:
        return 1
    median = float(np.median(np.asarray(active_scales, dtype=float)))
    repeat = max(1, int(round(median)))
    if repeat <= 1:
        return 1
    max_rel_err = max(abs(value - repeat) / repeat for value in active_scales)
    return repeat if max_rel_err <= 0.05 else 1


def template_cycle_offsets(artifact: RunArtifacts, cfg: TraceConfig) -> List[int]:
    repeat = template_repeat_count(artifact, cfg)
    if repeat <= 1:
        return [0]
    stride = max(1, int(round(artifact.summary.exec_cycles / repeat)))
    return [idx * stride for idx in range(repeat)]


def event_energy_scales(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, float]:
    if template_repeat_count(artifact, cfg) > 1:
        return {component: 1.0 for component in COMPONENTS}
    return dynamic_energy_scales(artifact, cfg)


def dynamic_reconciliation_summary(artifact: RunArtifacts, cfg: TraceConfig, trace: PowerTrace | None = None) -> Dict[str, Any]:
    runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    observed = _oplog_dynamic_energy_pj_by_component(artifact, cfg)
    target = _summary_dynamic_energy_pj_by_component(artifact, cfg)
    scales = dynamic_energy_scales(artifact, cfg)
    summary: Dict[str, Any] = {
        "source": "tsim_log_dynamic_energy_by_component",
        "runtime_s": runtime_s,
        "target_dynamic_power_w": artifact.summary.dynamic_power_w,
        "oplog_dynamic_power_w_before_scaling": {
            component: observed.get(component, 0.0) / 1.0e12 / runtime_s if runtime_s > 0 else 0.0
            for component in COMPONENTS
        },
        "target_dynamic_power_w_by_component": {
            component: target.get(component, 0.0) / 1.0e12 / runtime_s if runtime_s > 0 else 0.0
            for component in COMPONENTS
        },
        "scale_by_component": scales,
        "template_repeat_count": template_repeat_count(artifact, cfg),
    }
    if trace is not None:
        static_w = sum(artifact.summary.static_component_w.get(component, 0.0) for component in COMPONENTS)
        mean_power = trace.component_power_w.mean(axis=0)
        summary["trace_mean_total_power_w"] = float(mean_power.sum())
        summary["trace_mean_dynamic_power_w"] = float(mean_power.sum() - static_w)
        summary["trace_mean_power_w_by_component"] = {
            component: float(mean_power[idx]) for idx, component in enumerate(COMPONENTS)
        }
    return summary


def normalize_dynamic_trace_to_summary(
    component_dynamic_w: np.ndarray,
    block_dynamic_w: np.ndarray,
    artifact: RunArtifacts,
    cfg: TraceConfig,
    dt_s: float,
) -> None:
    target_pj = _summary_dynamic_energy_pj_by_component(artifact, cfg)
    observed_total_j = 0.0
    target_total_j = 0.0
    for idx, component in enumerate(COMPONENTS):
        observed_j = float(component_dynamic_w[:, idx].sum() * dt_s)
        target_j = target_pj.get(component, 0.0) / 1.0e12
        if observed_j > 0.0 and target_j > 0.0:
            component_dynamic_w[:, idx] *= target_j / observed_j
        observed_total_j += observed_j
        target_total_j += target_j
    if block_dynamic_w.size and observed_total_j > 0.0 and target_total_j > 0.0:
        block_dynamic_w *= target_total_j / observed_total_j


def spread_repeated_template_dynamic(dynamic_w: np.ndarray, artifact: RunArtifacts, cfg: TraceConfig, dt_s: float) -> None:
    repeat = template_repeat_count(artifact, cfg)
    if repeat <= 1 or dynamic_w.size == 0:
        return
    n_bins = dynamic_w.shape[0]
    stride_s = (artifact.summary.exec_cycles / repeat) / (cfg.npu_freq_mhz * 1e6)
    stride_bins = max(1, min(n_bins, int(math.ceil(stride_s / max(dt_s, 1e-15)))))
    template = dynamic_w[:stride_bins].copy() / repeat
    dynamic_w[:, :] = 0.0
    for idx in range(repeat):
        start = int(round((idx * stride_s) / max(dt_s, 1e-15)))
        if start >= n_bins:
            break
        end = min(n_bins, start + stride_bins)
        dynamic_w[start:end] += template[:end - start]


def normalize_block_dynamic_to_summary(block_dynamic_w: np.ndarray, artifact: RunArtifacts, cfg: TraceConfig, dt_s: float) -> None:
    target_dynamic_j = sum(_summary_dynamic_energy_pj_by_component(artifact, cfg).values()) / 1.0e12
    observed_dynamic_j = float(block_dynamic_w.sum() * dt_s)
    if observed_dynamic_j > 0.0 and target_dynamic_j > 0.0:
        block_dynamic_w *= target_dynamic_j / observed_dynamic_j


def _target_duration_s(artifact: RunArtifacts, cfg: TraceConfig) -> float:
    base_runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    return max(base_runtime_s, cfg.duration_ms / 1e3)


def _major_op_duration_s(artifact: RunArtifacts, cfg: TraceConfig) -> List[float]:
    cycle_to_s = 1.0 / (cfg.npu_freq_mhz * 1e6)
    durations = [
        float(getattr(op, "mm_dur", 0)) * cycle_to_s
        for op in artifact.op_logs
        if getattr(op, "mm_dur", 0) > 0 and getattr(op, "energy_sa", 0.0) > 0
    ]
    return durations


def thermal_bin_count(artifact: RunArtifacts, cfg: TraceConfig) -> int:
    max_bins = max(1, int(cfg.max_bins))
    samples = max(0, int(cfg.major_op_samples))
    if samples <= 0:
        return max_bins
    durations = _major_op_duration_s(artifact, cfg)
    if not durations:
        return max_bins
    percentile = min(100.0, max(0.0, float(cfg.major_op_percentile)))
    target_op_s = float(np.percentile(np.asarray(durations), percentile))
    if target_op_s <= 0:
        return max_bins
    target_dt_s = target_op_s / samples
    requested_bins = int(math.ceil(_target_duration_s(artifact, cfg) / max(target_dt_s, 1e-15)))
    return max(1, min(max_bins, requested_bins))


def temporal_sampling_summary(artifact: RunArtifacts, cfg: TraceConfig, trace: PowerTrace) -> Dict[str, Any]:
    durations = _major_op_duration_s(artifact, cfg)
    if durations:
        percentile = min(100.0, max(0.0, float(cfg.major_op_percentile)))
        target_op_s = float(np.percentile(np.asarray(durations), percentile))
        median_op_s = float(np.percentile(np.asarray(durations), 50.0))
        p75_op_s = float(np.percentile(np.asarray(durations), 75.0))
        ops_with_target_samples = sum(1 for value in durations if value / trace.dt_s >= max(1, cfg.major_op_samples))
    else:
        target_op_s = median_op_s = p75_op_s = 0.0
        ops_with_target_samples = 0
    return {
        "dt_s": trace.dt_s,
        "bins": int(trace.component_power_w.shape[0]),
        "duration_s": trace.duration_s,
        "max_bins": int(cfg.max_bins),
        "major_op_samples_target": int(cfg.major_op_samples),
        "major_op_percentile": float(cfg.major_op_percentile),
        "major_op_count": len(durations),
        "target_major_op_duration_s": target_op_s,
        "median_major_op_duration_s": median_op_s,
        "p75_major_op_duration_s": p75_op_s,
        "major_ops_meeting_target_samples": ops_with_target_samples,
        "major_ops_meeting_target_fraction": (
            ops_with_target_samples / len(durations) if durations else 0.0
        ),
    }


def add_power_event(
    series: np.ndarray,
    component: str,
    start_cycle: int,
    duration_cycle: int,
    energy_pj: float,
    cycle_to_s: float,
    dt_s: float,
) -> None:
    if duration_cycle <= 0 or energy_pj <= 0:
        return
    start_s = start_cycle * cycle_to_s
    end_s = (start_cycle + duration_cycle) * cycle_to_s
    start_bin, end_bin = interval_to_bins(start_s, end_s, dt_s, series.shape[0])
    if end_bin <= start_bin:
        return
    total_duration_s = max(end_s - start_s, 1e-15)
    energy_j = energy_pj / 1e12
    comp_idx = COMPONENTS.index(component)
    for bin_idx in range(start_bin, end_bin):
        bin_start_s = bin_idx * dt_s
        bin_end_s = bin_start_s + dt_s
        overlap_s = max(0.0, min(end_s, bin_end_s) - max(start_s, bin_start_s))
        if overlap_s > 0:
            series[bin_idx, comp_idx] += energy_j * (overlap_s / total_duration_s) / dt_s


def add_block_power_event(
    matrix: np.ndarray,
    layer_offset: int,
    weights: np.ndarray,
    start_cycle: int,
    duration_cycle: int,
    energy_pj: float,
    cycle_to_s: float,
    dt_s: float,
) -> None:
    if duration_cycle <= 0 or energy_pj <= 0:
        return
    start_s = start_cycle * cycle_to_s
    end_s = (start_cycle + duration_cycle) * cycle_to_s
    start_bin, end_bin = interval_to_bins(start_s, end_s, dt_s, matrix.shape[0])
    if end_bin <= start_bin:
        return
    total_duration_s = max(end_s - start_s, 1e-15)
    energy_j = energy_pj / 1e12
    for bin_idx in range(start_bin, end_bin):
        bin_start_s = bin_idx * dt_s
        bin_end_s = bin_start_s + dt_s
        overlap_s = max(0.0, min(end_s, bin_end_s) - max(start_s, bin_start_s))
        if overlap_s > 0:
            matrix[bin_idx, layer_offset:layer_offset + weights.shape[0]] += (
                weights * energy_j * (overlap_s / total_duration_s) / dt_s
            )


SPATIAL_FIELD_CANDIDATES = {
    "core": (
        "active_core_ids",
        "active_cores",
        "core_ids",
        "cores",
        "spatial_core_ids",
        "thermal_core_ids",
    ),
    "link": (
        "active_link_ids",
        "active_noc_link_ids",
        "noc_link_ids",
        "noc_links",
        "link_ids",
        "links",
        "thermal_link_ids",
    ),
    "vault": (
        "active_vault_ids",
        "active_dram_vault_ids",
        "dram_vault_ids",
        "vault_ids",
        "vaults",
        "thermal_vault_ids",
    ),
    "tsv": (
        "active_tsv_group_ids",
        "tsv_group_ids",
        "tsv_groups",
        "active_tsv_ids",
        "tsv_ids",
        "thermal_tsv_group_ids",
    ),
}
SPATIAL_CONTAINER_FIELDS = (
    "spatial_meta",
    "spatial_attribution",
    "spatial_metadata",
    "thermal_spatial_attribution",
    "thermal_spatial_metadata",
)


class AttributionTracker:
    def __init__(self) -> None:
        self.events = {name: 0 for name in COMPONENTS}
        self.attributed_events = {name: 0 for name in COMPONENTS}
        self.fallback_events = {name: 0 for name in COMPONENTS}
        self.attributed_energy_pj = {name: 0.0 for name in COMPONENTS}
        self.fallback_energy_pj = {name: 0.0 for name in COMPONENTS}
        self.fields_seen: Dict[str, set[str]] = {kind: set() for kind in SPATIAL_FIELD_CANDIDATES}
        self.ops_with_spatial_metadata = 0

    def record(self, component: str, energy_pj: float, used: bool) -> None:
        self.events[component] += 1
        if used:
            self.attributed_events[component] += 1
            self.attributed_energy_pj[component] += float(energy_pj)
        else:
            self.fallback_events[component] += 1
            self.fallback_energy_pj[component] += float(energy_pj)

    def as_dict(self, domains: Dict[str, int], grid: int) -> Dict[str, Any]:
        components = [name for name, count in self.attributed_events.items() if count > 0]
        attributed_event_count = sum(self.attributed_events.values())
        return {
            "used": attributed_event_count > 0,
            "components": components,
            "events_by_component": self.events,
            "attributed_events_by_component": self.attributed_events,
            "fallback_events_by_component": self.fallback_events,
            "attributed_energy_pj_by_component": self.attributed_energy_pj,
            "fallback_energy_pj_by_component": self.fallback_energy_pj,
            "attributed_event_count": attributed_event_count,
            "fallback_event_count": sum(self.fallback_events.values()),
            "ops_with_spatial_metadata": self.ops_with_spatial_metadata,
            "fields_seen": {kind: sorted(fields) for kind, fields in self.fields_seen.items() if fields},
            "id_domains": domains,
            "mapping": {
                "grid": grid,
                "core": "active core IDs -> logic tiles",
                "link": "NoC link IDs/endpoints -> logic tiles",
                "vault": "DRAM vault IDs -> DRAM tiles",
                "tsv": "TSV group IDs -> logic tiles",
            },
        }


def _metadata_container(op: object) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for field in SPATIAL_CONTAINER_FIELDS:
        value = getattr(op, field, None)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _spatial_value(op: object, kind: str) -> Tuple[Any, str | None]:
    candidates = SPATIAL_FIELD_CANDIDATES[kind]
    for field in candidates:
        if hasattr(op, field):
            return getattr(op, field), field
    container = _metadata_container(op)
    for field in candidates:
        if field in container:
            return container[field], field
    if kind in container:
        return container[kind], kind
    legacy_value = _legacy_spatial_meta_value(container, kind)
    if legacy_value is not None:
        return legacy_value, "spatial_meta"
    stage_value = _stage_spatial_value(container, kind)
    if stage_value is not None:
        return stage_value, "spatial_attribution.stages"
    return None, None


def _expand_inclusive_ranges(ranges: Any) -> List[int]:
    ids: List[int] = []
    for item in ranges or []:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            start, end = int(item[0]), int(item[1])
            ids.extend(range(start, end + 1))
    return ids


def _legacy_spatial_meta_value(container: Dict[str, Any], kind: str) -> Any:
    if kind == "core" and "active_core_ranges" in container:
        return _expand_inclusive_ranges(container.get("active_core_ranges"))
    if kind == "vault" and isinstance(container.get("dram_vaults"), dict):
        return [
            vault
            for values in container["dram_vaults"].values()
            for vault in values
        ]
    if kind == "tsv" and isinstance(container.get("tsv_groups"), dict):
        return [
            group
            for values in container["tsv_groups"].values()
            for group in values
        ]
    if kind == "link" and isinstance(container.get("noc_links"), dict):
        return [
            link
            for values in container["noc_links"].values()
            for link in values
        ]
    return None


def _stage_spatial_value(container: Dict[str, Any], kind: str) -> Any:
    stages = container.get("stages")
    if not isinstance(stages, dict):
        return None
    if kind == "core":
        for stage_name in ("comp_sa", "comp", "comp_vu", "comp_sram_r", "comp_sram_w"):
            stage = stages.get(stage_name)
            if isinstance(stage, dict) and stage.get("active_core_ids"):
                return stage["active_core_ids"]
    if kind == "link":
        values = []
        for stage_name in ("noc_bcast_sh", "noc_reduce", "noc_shift"):
            stage = stages.get(stage_name)
            if isinstance(stage, dict) and stage.get("noc_logical_link_ids"):
                values.extend(_weighted_ids(stage["noc_logical_link_ids"]))
        return [item for item, _weight in values] if values else None
    if kind == "vault":
        values = []
        for stage_name in ("dram_r", "dram_w"):
            stage = stages.get(stage_name)
            if isinstance(stage, dict) and stage.get("dram_vault_ids"):
                values.extend(_weighted_ids(stage["dram_vault_ids"]))
        return [item for item, _weight in values] if values else None
    if kind == "tsv":
        values = []
        for stage_name in ("dram_r", "dram_w"):
            stage = stages.get(stage_name)
            if isinstance(stage, dict) and stage.get("tsv_group_ids"):
                values.extend(_weighted_ids(stage["tsv_group_ids"]))
        return [item for item, _weight in values] if values else None
    return None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _weighted_ids(value: Any) -> List[Tuple[Any, float]]:
    if value is None:
        return []
    if isinstance(value, dict):
        if value.get("encoding") == "ranges":
            ids: List[Tuple[Any, float]] = []
            for start, end in value.get("ranges", ()):
                ids.extend((idx, 1.0) for idx in range(int(start), int(end)))
            return ids
        if value.get("encoding") == "complete_directed_no_self":
            ids = []
            num_cores = int(value.get("num_cores") or 0)
            for start, end in value.get("core_ranges", ()):
                cores = range(int(start), int(end))
                for src in cores:
                    for dst in cores:
                        if src != dst:
                            ids.append(((src, dst), 1.0))
            if not ids and num_cores > 0:
                for src in range(num_cores):
                    for dst in range(num_cores):
                        if src != dst:
                            ids.append(((src, dst), 1.0))
            return ids
        if "ids" in value:
            return _weighted_ids(value["ids"])
        if "active_ids" in value:
            return _weighted_ids(value["active_ids"])
        weighted: List[Tuple[Any, float]] = []
        for key, weight in value.items():
            if _is_number(weight) and float(weight) > 0:
                weighted.append((key, float(weight)))
        return weighted
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [(part, 1.0) for part in parts] if parts else []
    if _is_number(value):
        return [(value, 1.0)]
    if isinstance(value, np.ndarray):
        return _weighted_ids(value.tolist())
    if isinstance(value, Iterable):
        return [(item, 1.0) for item in value]
    return []


def _numeric_id(value: Any) -> int | None:
    if _is_number(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def _numeric_ids_from_value(value: Any) -> List[int]:
    ids: List[int] = []
    for item, _weight in _weighted_ids(value):
        if isinstance(item, (tuple, list)):
            for endpoint in item:
                numeric = _numeric_id(endpoint)
                if numeric is not None:
                    ids.append(numeric)
        else:
            numeric = _numeric_id(item)
            if numeric is not None:
                ids.append(numeric)
    return ids


def _collect_spatial_domains(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, int]:
    n_tiles = cfg.grid * cfg.grid
    maxima = {kind: -1 for kind in SPATIAL_FIELD_CANDIDATES}
    for op in artifact.op_logs:
        for kind in SPATIAL_FIELD_CANDIDATES:
            value, _field = _spatial_value(op, kind)
            numeric_ids = _numeric_ids_from_value(value)
            if numeric_ids:
                maxima[kind] = max(maxima[kind], max(numeric_ids))
    return {
        "core": max(1, int(artifact.run_id.num_cores), maxima["core"] + 1),
        "link": max(1, maxima["link"] + 1, n_tiles),
        "vault": max(1, maxima["vault"] + 1, n_tiles),
        "tsv": max(1, maxima["tsv"] + 1, n_tiles),
    }


def _id_to_tile(idx: int, domain_size: int, n_tiles: int) -> int | None:
    if idx < 0 or domain_size <= 0:
        return None
    if domain_size <= n_tiles:
        return idx % n_tiles
    return min(n_tiles - 1, int(idx * n_tiles / domain_size))


def _tiles_for_item(kind: str, item: Any, domains: Dict[str, int], n_tiles: int) -> List[int]:
    if kind == "link" and isinstance(item, (tuple, list)) and len(item) >= 2:
        tiles = []
        for endpoint in item[:2]:
            numeric = _numeric_id(endpoint)
            tile = _id_to_tile(numeric, domains["core"], n_tiles) if numeric is not None else None
            if tile is not None:
                tiles.append(tile)
        return tiles
    if kind == "link" and isinstance(item, str) and "-" in item:
        left, right = item.split("-", 1)
        return _tiles_for_item(kind, (left, right), domains, n_tiles)
    numeric = _numeric_id(item)
    if numeric is None:
        return []
    return [_id_to_tile(numeric, domains[kind], n_tiles)] if _id_to_tile(numeric, domains[kind], n_tiles) is not None else []


def _weights_from_ids(
    weighted_ids: List[Tuple[Any, float]],
    kind: str,
    domains: Dict[str, int],
    n_tiles: int,
) -> np.ndarray | None:
    weights = np.zeros(n_tiles, dtype=float)
    for item, item_weight in weighted_ids:
        tiles = _tiles_for_item(kind, item, domains, n_tiles)
        if not tiles:
            continue
        share = float(item_weight) / len(tiles)
        for tile in tiles:
            weights[tile] += share
    total = float(weights.sum())
    if total <= 0:
        return None
    return weights / total


def _component_weights(
    op: object,
    component: str,
    cfg: TraceConfig,
    domains: Dict[str, int],
    tracker: AttributionTracker,
) -> Tuple[np.ndarray, bool]:
    n_tiles = cfg.grid * cfg.grid
    kind_by_component = {
        "sa": "core",
        "vu": "core",
        "sram": "core",
        "noc": "link",
        "dram": "vault",
        "tsv": "tsv",
    }
    fallback = np.full(n_tiles, 1.0 / n_tiles) if component == "dram" else tile_weights(cfg.grid, cfg.spatial_policy).reshape(n_tiles)
    kind = kind_by_component[component]
    value, field = _spatial_value(op, kind)
    if field is not None:
        tracker.fields_seen[kind].add(field)
    weights = _weights_from_ids(_weighted_ids(value), kind, domains, n_tiles)
    if weights is None:
        return fallback, False
    return weights, True


def _record_op_metadata_presence(op: object, tracker: AttributionTracker) -> None:
    for kind in SPATIAL_FIELD_CANDIDATES:
        value, _field = _spatial_value(op, kind)
        if _weighted_ids(value):
            tracker.ops_with_spatial_metadata += 1
            return


def _add_spatial_event(
    matrix: np.ndarray,
    op: object,
    component: str,
    start_cycle: int,
    duration_cycle: int,
    energy_pj: float,
    cfg: TraceConfig,
    domains: Dict[str, int],
    tracker: AttributionTracker,
    cycle_to_s: float,
    dt_s: float,
) -> None:
    if duration_cycle <= 0 or energy_pj <= 0:
        return
    n_tiles = cfg.grid * cfg.grid
    weights, used = _component_weights(op, component, cfg, domains, tracker)
    tracker.record(component, energy_pj, used)
    layer_offset = n_tiles if component == "dram" else 0
    add_block_power_event(matrix, layer_offset, weights, start_cycle, duration_cycle, energy_pj, cycle_to_s, dt_s)


def add_block_event(
    matrix: np.ndarray,
    block_indices: List[int],
    start_cycle: int,
    duration_cycle: int,
    energy_pj: float,
    cycle_to_s: float,
    dt_s: float,
) -> None:
    if duration_cycle <= 0 or energy_pj <= 0 or not block_indices:
        return
    start_s = start_cycle * cycle_to_s
    end_s = (start_cycle + duration_cycle) * cycle_to_s
    start_bin, end_bin = interval_to_bins(start_s, end_s, dt_s, matrix.shape[0])
    if end_bin <= start_bin:
        return
    total_duration_s = max(end_s - start_s, 1e-15)
    energy_j = energy_pj / 1e12 / len(block_indices)
    for bin_idx in range(start_bin, end_bin):
        bin_start_s = bin_idx * dt_s
        bin_end_s = bin_start_s + dt_s
        overlap_s = max(0.0, min(end_s, bin_end_s) - max(start_s, bin_start_s))
        if overlap_s > 0:
            power_w = energy_j * (overlap_s / total_duration_s) / dt_s
            matrix[bin_idx, block_indices] += power_w


def sram_power_window(op: object) -> Tuple[int, int]:
    """Return the interval used to spread SRAM dynamic energy for thermal traces."""
    start = int(getattr(op, "t_comp_shift_start", 0))
    duration = int(getattr(op, "comp_sh_dur", 0))
    if duration <= 0:
        duration = max(
            int(getattr(op, "comp_dur", 0)),
            int(getattr(op, "mm_dur", 0)),
            int(getattr(op, "shift_dur", 0)),
            int(getattr(op, "sram_r_dur", 0)),
            int(getattr(op, "sram_w_dur", 0)),
        )
    return start, duration


def add_weighted_block_event(
    matrix: np.ndarray,
    block_weights: Dict[int, float],
    start_cycle: int,
    duration_cycle: int,
    energy_pj: float,
    cycle_to_s: float,
    dt_s: float,
) -> None:
    if duration_cycle <= 0 or energy_pj <= 0 or not block_weights:
        return
    positive_weights = {
        int(block): float(weight)
        for block, weight in block_weights.items()
        if float(weight) > 0.0
    }
    total_weight = sum(positive_weights.values())
    if total_weight <= 0:
        return
    start_s = start_cycle * cycle_to_s
    end_s = (start_cycle + duration_cycle) * cycle_to_s
    start_bin, end_bin = interval_to_bins(start_s, end_s, dt_s, matrix.shape[0])
    if end_bin <= start_bin:
        return
    total_duration_s = max(end_s - start_s, 1e-15)
    energy_j = energy_pj / 1e12
    block_indices = list(positive_weights)
    normalized = np.asarray([positive_weights[block] / total_weight for block in block_indices], dtype=float)
    for bin_idx in range(start_bin, end_bin):
        bin_start_s = bin_idx * dt_s
        bin_end_s = bin_start_s + dt_s
        overlap_s = max(0.0, min(end_s, bin_end_s) - max(start_s, bin_start_s))
        if overlap_s > 0:
            power_w = energy_j * (overlap_s / total_duration_s) / dt_s
            matrix[bin_idx, block_indices] += power_w * normalized


def _core_grid_dims(num_cores: int) -> Tuple[int, int]:
    n = max(1, int(num_cores))
    rows = 1
    cols = n
    for candidate_rows in range(1, int(math.sqrt(n)) + 1):
        if n % candidate_rows == 0:
            rows = candidate_rows
            cols = n // candidate_rows
    return rows, cols


def _logic_subblock_names(num_cores: int) -> List[str]:
    return [
        f"core_{core:03d}_{unit}"
        for core in range(max(1, num_cores))
        for unit in ("sram", "sa", "vu", "router", "tsv")
    ]


def tsv_region_area_mm2(artifact: RunArtifacts, cfg: TraceConfig) -> float:
    return max(
        0.0,
        float(cfg.tsv_region_area_mm2_at_12tbps)
        * max(0.0, float(artifact.run_id.dram_bw))
        / TSV_AREA_REFERENCE_GBPS,
    )


def tsv_region_names(package_count: int) -> List[str]:
    return [f"tsv_pkg{idx:02d}" for idx in range(max(1, int(package_count)))]


def _tsv_region_layout(
    artifact: RunArtifacts,
    cfg: TraceConfig,
    die_size_mm: float,
    outline_width_mm: float,
    outline_height_mm: float,
    package_count: int,
) -> Dict[str, Any]:
    area_mm2 = tsv_region_area_mm2(artifact, cfg)
    if area_mm2 <= 0:
        return {"area_mm2": 0.0, "blocks": [], "placement": "none"}

    names = tsv_region_names(package_count)
    right_w = max(0.0, outline_width_mm - die_size_mm)
    top_h = max(0.0, outline_height_mm - die_size_mm)
    eps = 1e-9
    if right_w > eps and right_w * die_size_mm >= area_mm2:
        region_w = right_w
        region_h = area_mm2 / region_w
        x0 = die_size_mm
        y0 = 0.0
        placement = "logic_pad_right"
    elif top_h > eps and outline_width_mm * top_h >= area_mm2:
        region_w = area_mm2 / top_h
        region_h = top_h
        x0 = 0.0
        y0 = die_size_mm
        placement = "logic_pad_top"
    else:
        # Fall back to a right-edge stripe inside the logic outline. This keeps
        # the TSV area explicit even for layouts without a non-overlapping pad.
        region_h = min(outline_height_mm, max(math.sqrt(area_mm2), eps))
        region_w = min(outline_width_mm, area_mm2 / region_h)
        x0 = max(0.0, outline_width_mm - region_w)
        y0 = 0.0
        placement = "logic_right_edge"

    block_h = region_h / max(1, len(names))
    blocks = []
    for idx, name in enumerate(names):
        blocks.append({
            "name": name,
            "width_mm": region_w,
            "height_mm": block_h,
            "x_mm": x0,
            "y_mm": y0 + idx * block_h,
        })
    return {
        "area_mm2": area_mm2,
        "region_width_mm": region_w,
        "region_height_mm": region_h,
        "x_mm": x0,
        "y_mm": y0,
        "placement": placement,
        "blocks": blocks,
    }


def _logic_aux_names_with_tsv(
    artifact: RunArtifacts,
    cfg: TraceConfig,
    die_size_mm: float,
    outline_width_mm: float,
    outline_height_mm: float,
    package_count: int,
) -> List[str]:
    return floorplan_pad_names("logic", die_size_mm, die_size_mm, outline_width_mm, outline_height_mm)


def _per_core_tsv_geometry(artifact: RunArtifacts, cfg: TraceConfig, die_size_mm: float) -> Dict[str, float]:
    num_cores = max(1, int(artifact.run_id.num_cores))
    rows, cols = _core_grid_dims(num_cores)
    core_width_mm = die_size_mm / cols
    core_height_mm = die_size_mm / rows
    total_area_mm2 = tsv_region_area_mm2(artifact, cfg)
    per_core_area_mm2 = intracore_unit_areas_mm2(artifact, cfg)["tsv"]
    raw_height_mm = per_core_area_mm2 / core_width_mm if core_width_mm > 0 else 0.0
    if raw_height_mm >= core_height_mm:
        raise ValueError(
            "per-core TSV area does not fit inside the core floorplan: "
            f"{per_core_area_mm2:.6f} mm^2/core over {core_width_mm:.6f} mm x {core_height_mm:.6f} mm"
        )
    return {
        "total_area_mm2": total_area_mm2,
        "per_core_area_mm2": per_core_area_mm2,
        "core_width_mm": core_width_mm,
        "core_height_mm": core_height_mm,
        "tsv_width_mm": core_width_mm,
        "tsv_height_mm": raw_height_mm,
        "rows": float(rows),
        "cols": float(cols),
    }


def intracore_unit_areas_mm2(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, float]:
    """Return absolute per-core logic-unit areas from the TSIM area model.

    These values are used directly in the intra-core thermal floorplan. TSVs
    add area to the core tile; they must not renormalize or shrink preexisting
    SRAM/SA/VU/router blocks.
    """
    sa = max(1, int(artifact.run_id.sa))
    vu = max(1, int(artifact.run_id.vu))
    sram_kb = max(1, int(artifact.run_id.sram_kb))
    noc_bw = max(1, int(artifact.run_id.noc_bw))
    num_cores = max(1, int(artifact.run_id.num_cores))

    sa_area = (65.17333 / 4.0) * (sa / 128.0) ** 2
    vu_area = 22.3942 * (vu / 2048.0)
    if sram_kb <= 640:
        sram_area = 5.74e-4 * sram_kb + 0.133
    else:
        sram_area = 6.7e-4 * sram_kb + 0.322
    router_area = (22545.0 / 8.041) * ((noc_bw * 8.0) / 10.0) / 1_000_000.0
    tsv_area = tsv_region_area_mm2(artifact, cfg) / num_cores
    return {
        "sram": sram_area,
        "sa": sa_area,
        "vu": vu_area,
        "router": router_area,
        "tsv": tsv_area,
    }


def intracore_total_logic_area_mm2(artifact: RunArtifacts, cfg: TraceConfig) -> float:
    num_cores = max(1, int(artifact.run_id.num_cores))
    return num_cores * sum(intracore_unit_areas_mm2(artifact, cfg).values())


def _logic_subblock_indices(num_cores: int) -> Dict[str, Dict[int, int]]:
    indices = {unit: {} for unit in ("sram", "sa", "vu", "router", "tsv")}
    for core in range(max(1, num_cores)):
        base = core * 5
        indices["sram"][core] = base
        indices["sa"][core] = base + 1
        indices["vu"][core] = base + 2
        indices["router"][core] = base + 3
        indices["tsv"][core] = base + 4
    return indices


def expand_ranges(ranges: List[List[int]]) -> List[int]:
    ids: List[int] = []
    for item in ranges or []:
        if len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        ids.extend(range(start, end + 1))
    return ids


def expand_compact_ids(encoded: dict) -> List[int]:
    if not encoded:
        return []
    if encoded.get("encoding") == "ranges":
        ids: List[int] = []
        for start, end in encoded.get("ranges", ()):
            # Worker-side compact ranges are half-open [start, end).
            ids.extend(range(int(start), int(end)))
        return ids
    if encoded.get("encoding") == "complete_directed_no_self":
        ids = []
        num_cores = int(encoded.get("num_cores") or 0)
        for start, end in encoded.get("core_ranges", ()):
            cores = list(range(int(start), int(end)))
            for src in cores:
                for dst in cores:
                    if src != dst:
                        ids.append(src * num_cores + dst)
        return ids
    return []


def meta_from_op(op) -> dict:
    direct = getattr(op, "spatial_meta", {}) or {}
    if direct:
        return direct
    attribution = getattr(op, "spatial_attribution", {}) or {}
    if not attribution:
        return {}

    stages = attribution.get("stages", {})
    comp_stage = stages.get("comp_sa") or stages.get("comp") or {}
    dram_r_stage = stages.get("dram_r") or {}
    dram_w_stage = stages.get("dram_w") or {}
    noc_bcast_stage = stages.get("noc_bcast_sh") or {}
    noc_reduce_stage = stages.get("noc_reduce") or {}
    active_cores = expand_compact_ids(comp_stage.get("active_core_ids", {}))
    num_cores = max(1, int(attribution.get("num_active_cores") or len(active_cores) or 1))
    num_dram_vaults = max(1, int(attribution.get("num_dram_vaults") or 1))

    def link_strings(stage: dict) -> List[str]:
        link_ids = expand_compact_ids(stage.get("noc_logical_link_ids", {}))
        link_num_cores = int(stage.get("noc_logical_link_ids", {}).get("num_cores") or num_cores)
        links = []
        for link_id in link_ids:
            src = int(link_id) // max(1, link_num_cores)
            dst = int(link_id) % max(1, link_num_cores)
            if src != dst:
                a, b = sorted((src, dst))
                links.append(f"{a}-{b}")
        return sorted(set(links))

    return {
        "version": 1,
        "source": "tsim_spatial_attribution",
        "num_cores": int(getattr(op, "spatial_attribution", {}).get("num_active_cores") or num_cores),
        "active_core_count": len(active_cores),
        "active_core_ranges": [[min(active_cores), max(active_cores)]] if active_cores else [],
        "dram_vault_count": num_dram_vaults,
        "dram_vaults": {
            "read": expand_compact_ids(dram_r_stage.get("dram_vault_ids", {})),
            "write": expand_compact_ids(dram_w_stage.get("dram_vault_ids", {})),
        },
        "tsv_groups": {
            "read": expand_compact_ids(dram_r_stage.get("tsv_group_ids", {})),
            "write": expand_compact_ids(dram_w_stage.get("tsv_group_ids", {})),
        },
        "noc_links": {
            "bcast": link_strings(noc_bcast_stage),
            "shift": [],
            "reduce": link_strings(noc_reduce_stage),
        },
    }


def id_to_tile(idx: int, count: int, grid: int) -> int:
    count = max(1, count)
    src_cols = max(1, int(math.ceil(math.sqrt(count))))
    src_rows = max(1, int(math.ceil(count / src_cols)))
    row = min(src_rows - 1, idx // src_cols)
    col = min(src_cols - 1, idx % src_cols)
    tile_y = min(grid - 1, int(row * grid / src_rows))
    tile_x = min(grid - 1, int(col * grid / src_cols))
    return tile_y * grid + tile_x


def core_tiles(meta: dict, grid: int) -> List[int]:
    cores = expand_ranges(meta.get("active_core_ranges", []))
    count = int(meta.get("num_cores") or max(1, len(cores)))
    return sorted({id_to_tile(core, count, grid) for core in cores})


def vault_tiles(meta: dict, kind: str, grid: int) -> List[int]:
    vaults = meta.get("dram_vaults", {}).get(kind, [])
    count = int(meta.get("dram_vault_count") or max(1, len(vaults)))
    return sorted({id_to_tile(int(vault), count, grid) for vault in vaults})


def hbm_package_count(cfg: TraceConfig) -> int:
    return get_hbm_package_count(cfg.dram_capacity_mb, cfg.hbm_package_capacity_mb)


def hbm_package_names(prefix: str, package_count: int) -> List[str]:
    return [f"{prefix}_pkg{idx:02d}" for idx in range(max(1, int(package_count)))]


def hbm_bank_names(prefix: str, package_count: int, banks_per_package: int) -> List[str]:
    return [
        f"{prefix}_pkg{pkg:02d}_bank{bank:02d}"
        for pkg in range(max(1, int(package_count)))
        for bank in range(max(1, int(banks_per_package)))
    ]


def dram_packages(meta: dict, kind: str, package_count: int) -> List[int]:
    vaults = meta.get("dram_vaults", {}).get(kind, [])
    vault_count = int(meta.get("dram_vault_count") or max(1, len(vaults)))
    package_count = max(1, int(package_count))
    packages = set()
    for vault in vaults:
        vault_id = int(vault)
        if vault_id < 0:
            continue
        packages.add(min(package_count - 1, int(vault_id * package_count / max(1, vault_count))))
    return sorted(packages)


def resolve_dram_bank_mapping(cfg: TraceConfig, artifact: RunArtifacts | None = None) -> str:
    mapping = str(cfg.dram_bank_mapping or "from_impl").lower()
    aliases = {
        "address": "address_trace",
        "address-trace": "address_trace",
        "access_trace": "address_trace",
        "access-trace": "address_trace",
        "memory_address": "address_trace",
        "memory-address": "address_trace",
        "best": "software_aware",
        "software-aware": "software_aware",
        "software": "software_aware",
        "program_aware": "software_aware",
        "program-aware": "software_aware",
        "uniform_dram": "uniform",
        "uniform_placement": "uniform",
        "interleave": "interleave_size",
        "interleave_dram": "interleave_size",
        "size_interleave": "interleave_size",
        "fine_interleave": "hbm_interleave",
        "fine-interleave": "hbm_interleave",
        "bank_interleave": "hbm_interleave",
        "bank-interleave": "hbm_interleave",
        "hbm-interleave": "hbm_interleave",
    }
    mapping = aliases.get(mapping, mapping)
    if mapping == "from_impl":
        impl = str(getattr(getattr(artifact, "run_id", None), "impl", "")).lower()
        return aliases.get(impl, "software_aware")
    if mapping not in {"address_trace", "hbm_interleave", "uniform", "interleave_size", "software_aware"}:
        return "hbm_interleave"
    return mapping


def _rotating_contiguous_indices(total: int, start: int, span: int) -> List[int]:
    total = max(1, int(total))
    span = max(1, min(total, int(span)))
    start = int(start) % total
    return [(start + offset) % total for offset in range(span)]


def dram_address_trace_params(artifact: RunArtifacts, cfg: TraceConfig) -> Dict[str, Any]:
    """Reconstruct the coarse DRAM request geometry used for address mapping.

    Existing TSIM pickles preserve per-operator byte counts and timing, but not
    tensor allocation addresses.  This reconstructs a deterministic request
    stream with the same bandwidth and row-size assumptions used by the DRAM
    timing model, then decodes those requests into thermal HBM bank blocks.
    """
    num_cores = max(1, int(getattr(artifact.run_id, "num_cores", 1)))
    dram_bw = max(1, int(getattr(artifact.run_id, "dram_bw", 1)))
    npu_freq_mhz = max(1, int(round(float(cfg.npu_freq_mhz))))
    bytes_per_cycle_per_core = max(1, int(get_per_cycle_bytes_per_core_from_DRAM_config(
        num_cores=num_cores,
        total_bandwidth_GBps=dram_bw,
        npu_freq_MHz=npu_freq_mhz,
    )))
    burst_bytes = max(1, bytes_per_cycle_per_core * num_cores)
    row_accesses = max(1, int(getattr(artifact.run_id, "row", 1)))
    row_bytes = max(burst_bytes, burst_bytes * row_accesses)
    stripe_bytes = max(1, int(cfg.hbm_interleave_stripe_bytes))
    return {
        "source": "synthetic_address_trace_from_tsim_tensor_access_records",
        "bytes_per_cycle_per_core": bytes_per_cycle_per_core,
        "chip_burst_bytes": burst_bytes,
        "row_accesses": row_accesses,
        "row_bytes": row_bytes,
        "stripe_bytes": stripe_bytes,
        "address_decode": "logical_bank = (physical_address // stripe_bytes) % (package_count * banks_per_package)",
        "package_decode": "package = logical_bank // banks_per_package; bank = logical_bank % banks_per_package",
        "tensor_base_addresses": "deterministic synthetic aligned regions keyed by op/subop/tensor/stage",
        "fallback": "older pickles without dram_access_records use aggregate op bytes",
    }


def _stable_u64(*items: object) -> int:
    value = 1469598103934665603
    for item in items:
        data = str(item).encode("utf-8")
        for byte in data:
            value ^= byte
            value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
        value ^= 0xFF
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _align_up(value: int, alignment: int) -> int:
    alignment = max(1, int(alignment))
    return int(math.ceil(max(0, int(value)) / alignment) * alignment)


def _address_stripe_bank_weights(base_address: int, num_bytes: int,
                                 stripe_bytes: int, total_banks: int) -> Dict[int, float]:
    """Map an aligned synthetic physical address range to logical HBM banks."""
    total_banks = max(1, int(total_banks))
    stripe_bytes = max(1, int(stripe_bytes))
    num_bytes = max(0, int(num_bytes))
    if num_bytes <= 0:
        return {}

    start_bank = (int(base_address) // stripe_bytes) % total_banks
    full_stripes, tail_bytes = divmod(num_bytes, stripe_bytes)
    full_rounds, extra_stripes = divmod(full_stripes, total_banks)

    weights: Dict[int, float] = {}
    if full_rounds:
        bytes_per_bank = float(full_rounds * stripe_bytes)
        weights = {bank: bytes_per_bank for bank in range(total_banks)}
    for offset in range(extra_stripes):
        bank = (start_bank + offset) % total_banks
        weights[bank] = weights.get(bank, 0.0) + float(stripe_bytes)
    if tail_bytes:
        bank = (start_bank + extra_stripes) % total_banks
        weights[bank] = weights.get(bank, 0.0) + float(tail_bytes)
    return weights


def _synthetic_tensor_base_address(record: dict, op: object, op_position: int,
                                   kind: str, stripe_bytes: int,
                                   total_banks: int) -> int:
    op_id = int(getattr(op, "op_id", op_position))
    subop_index = int(record.get("subop_index", 0) or 0)
    tensor_index = int(record.get("tensor_index", 0) or 0)
    tensor_role = str(record.get("tensor_role", "tensor"))
    stage = str(record.get("stage", kind))
    key = (op_id, subop_index, tensor_index, tensor_role, stage)
    bank_color = _stable_u64("bank", *key) % max(1, int(total_banks))
    allocation_id = _stable_u64("allocation", *key)
    bytes_total = max(1, int(record.get("total_bytes") or record.get("bytes_per_core") or 1))
    interleave_span = max(1, int(total_banks)) * max(1, int(stripe_bytes))
    region_span = _align_up(bytes_total + interleave_span, interleave_span)
    # Separate load and store pools by a large aligned offset.  This prevents a
    # synthetic output write from always aliasing the input read bank color.
    pool_offset = 0 if kind == "read" else (1 << 48)
    return int(pool_offset + (allocation_id % (1 << 20)) * region_span + bank_color * stripe_bytes)


def _dram_access_records_for_stage(meta: dict, kind: str) -> List[dict]:
    records = []
    for record in (meta or {}).get("dram_access_records", []) or []:
        if not isinstance(record, dict):
            continue
        if str(record.get("stage", "")).lower() != kind:
            continue
        if int(record.get("total_bytes") or record.get("bytes_per_core") or 0) <= 0:
            continue
        records.append(record)
    return records


def _dram_unit_weights_from_bank_weights(bank_weights: Dict[int, float],
                                         cfg: TraceConfig) -> Dict[int, float]:
    if cfg.dram_floorplan_granularity == "bank":
        return dict(bank_weights)
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    package_weights: Dict[int, float] = {}
    for bank, weight in bank_weights.items():
        package = int(bank) // banks_per_package
        package_weights[package] = package_weights.get(package, 0.0) + float(weight)
    return package_weights


def dram_address_trace_record_events(
    meta: dict,
    kind: str,
    cfg: TraceConfig,
    artifact: RunArtifacts,
    op: object,
    op_position: int,
    package_count: int,
) -> List[Dict[str, Any]]:
    records = _dram_access_records_for_stage(meta, kind)
    if not records:
        return []

    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    total_banks = max(1, int(package_count) * banks_per_package)
    params = dram_address_trace_params(artifact, cfg)
    stripe_bytes = max(1, int(params["stripe_bytes"]))
    stage_start = int(getattr(op, "t_dram_ld_start" if kind == "read" else "t_dram_st_start", 0))
    stage_duration = int(getattr(op, "dram_ld_dur" if kind == "read" else "dram_st_dur", 0))
    events: List[Dict[str, Any]] = []

    for record in records:
        total_bytes = max(0, int(record.get("total_bytes") or record.get("bytes_per_core") or 0))
        if total_bytes <= 0:
            continue
        start_cycle = int(record.get("access_start_cycle", stage_start))
        duration_cycle = int(record.get(
            "access_duration_cycle",
            record.get("scheduled_cycles", record.get("cycles_per_core", stage_duration)),
        ))
        base_address = int(record.get(
            "synthetic_base_address",
            _synthetic_tensor_base_address(record, op, op_position, kind, stripe_bytes, total_banks),
        ))
        bank_weights = _address_stripe_bank_weights(base_address, total_bytes, stripe_bytes, total_banks)
        if not bank_weights:
            continue
        event = dict(record)
        event.update({
            "start_cycle": start_cycle,
            "duration_cycle": max(0, duration_cycle),
            "base_address": base_address,
            "stripe_bytes": stripe_bytes,
            "bank_weights": bank_weights,
            "unit_weights": _dram_unit_weights_from_bank_weights(bank_weights, cfg),
        })
        events.append(event)
    return events


def _all_dram_access_records(meta: dict) -> List[dict]:
    records = []
    for record in (meta or {}).get("dram_access_records", []) or []:
        if not isinstance(record, dict):
            continue
        if int(record.get("total_bytes") or record.get("bytes_per_core") or 0) <= 0:
            continue
        records.append(record)
    return records


def _record_signature(record: dict) -> Tuple[int, int, str, str]:
    return (
        int(record.get("subop_index", 0) or 0),
        int(record.get("tensor_index", 0) or 0),
        str(record.get("tensor_role", "tensor")),
        str(record.get("stage", "")),
    )


def _span_count_from_weights(items: List[Tuple[int, float]], total_banks: int) -> Dict[int, int]:
    total_banks = max(1, int(total_banks))
    positive = [(idx, max(0.0, float(weight))) for idx, weight in items]
    total_weight = sum(weight for _idx, weight in positive)
    if total_weight <= 0:
        return {idx: 1 for idx, _weight in positive}
    raw = [(idx, total_banks * weight / total_weight) for idx, weight in positive]
    spans = {idx: max(1, int(math.floor(value))) for idx, value in raw}
    current = sum(spans.values())
    if current > total_banks:
        for idx, _value in sorted(raw, key=lambda item: item[1]):
            if current <= total_banks:
                break
            if spans[idx] > 1:
                spans[idx] -= 1
                current -= 1
    elif current < total_banks:
        for idx, _value in sorted(raw, key=lambda item: item[1] - math.floor(item[1]), reverse=True):
            if current >= total_banks:
                break
            spans[idx] += 1
            current += 1
    return spans


def _contiguous_bank_weights(start_bank: int, span: int, total_banks: int, total_bytes: int) -> Dict[int, float]:
    total_banks = max(1, int(total_banks))
    span = max(1, min(total_banks, int(span)))
    total_bytes = max(0, int(total_bytes))
    if total_bytes <= 0:
        return {}
    bytes_per_bank = float(total_bytes) / span
    return {
        (int(start_bank) + offset) % total_banks: bytes_per_bank
        for offset in range(span)
    }


def _software_aware_record_assignments(meta: dict, total_banks: int, op_seed: int) -> Dict[Tuple[int, int, str, str], Tuple[int, int]]:
    records = _all_dram_access_records(meta)
    if not records:
        return {}
    ordered = sorted(records, key=_record_signature)
    items = [
        (idx, float(record.get("total_bytes") or record.get("bytes_per_core") or 1))
        for idx, record in enumerate(ordered)
    ]
    spans = _span_count_from_weights(items, total_banks)
    start = int(op_seed) % max(1, total_banks)
    assignments: Dict[Tuple[int, int, str, str], Tuple[int, int]] = {}
    cursor = start
    for idx, record in enumerate(ordered):
        span = spans.get(idx, 1)
        assignments[_record_signature(record)] = (cursor, span)
        cursor += span
    return assignments


def dram_mapping_record_events(
    meta: dict,
    kind: str,
    cfg: TraceConfig,
    artifact: RunArtifacts,
    op: object,
    op_position: int,
    package_count: int,
) -> List[Dict[str, Any]]:
    """Return per-tensor DRAM events decoded according to the configured mapping."""
    mapping = resolve_dram_bank_mapping(cfg, artifact)
    if mapping == "address_trace":
        return dram_address_trace_record_events(meta, kind, cfg, artifact, op, op_position, package_count)

    records = _dram_access_records_for_stage(meta, kind)
    if not records:
        return []

    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    total_banks = max(1, int(package_count) * banks_per_package)
    stripe_bytes = max(1, int(cfg.hbm_interleave_stripe_bytes))
    op_id = int(getattr(op, "op_id", op_position))
    stage_start = int(getattr(op, "t_dram_ld_start" if kind == "read" else "t_dram_st_start", 0))
    stage_duration = int(getattr(op, "dram_ld_dur" if kind == "read" else "dram_st_dur", 0))
    max_record_bytes = max(1, max(int(record.get("total_bytes") or record.get("bytes_per_core") or 1) for record in _all_dram_access_records(meta) or records))
    software_assignments = (
        _software_aware_record_assignments(meta, total_banks, op_id * 131 + op_position * 17)
        if mapping == "software_aware"
        else {}
    )

    events: List[Dict[str, Any]] = []
    for record in records:
        total_bytes = max(0, int(record.get("total_bytes") or record.get("bytes_per_core") or 0))
        if total_bytes <= 0:
            continue
        start_cycle = int(record.get("access_start_cycle", stage_start))
        duration_cycle = int(record.get(
            "access_duration_cycle",
            record.get("scheduled_cycles", record.get("cycles_per_core", stage_duration)),
        ))

        if mapping == "uniform":
            bank_weights = {bank: float(total_bytes) / total_banks for bank in range(total_banks)}
        elif mapping == "hbm_interleave":
            base_address = _synthetic_tensor_base_address(record, op, op_position, kind, stripe_bytes, total_banks)
            bank_weights = _address_stripe_bank_weights(base_address, total_bytes, stripe_bytes, total_banks)
        elif mapping == "interleave_size":
            span = max(1, int(math.ceil(total_banks * total_bytes / max_record_bytes)))
            span = min(total_banks, span)
            seed = _stable_u64("interleave_size", op_id, op_position, *_record_signature(record))
            start_bank = int(seed % total_banks)
            bank_weights = _contiguous_bank_weights(start_bank, span, total_banks, total_bytes)
        elif mapping == "software_aware":
            start_bank, span = software_assignments.get(_record_signature(record), (op_id % total_banks, total_banks))
            bank_weights = _contiguous_bank_weights(start_bank, span, total_banks, total_bytes)
        else:
            continue

        if not bank_weights:
            continue
        event = dict(record)
        event.update({
            "start_cycle": start_cycle,
            "duration_cycle": max(0, duration_cycle),
            "stripe_bytes": stripe_bytes,
            "bank_weights": bank_weights,
            "unit_weights": _dram_unit_weights_from_bank_weights(bank_weights, cfg),
            "bank_mapping": mapping,
        })
        events.append(event)
    return events


def dram_address_trace_bank_weights(
    kind: str,
    cfg: TraceConfig,
    artifact: RunArtifacts,
    op: object,
    op_position: int,
    package_count: int,
) -> Dict[int, float]:
    meta = meta_from_op(op)
    record_events = dram_address_trace_record_events(meta, kind, cfg, artifact, op, op_position, package_count)
    if record_events:
        weights: Dict[int, float] = {}
        for event in record_events:
            for bank, weight in event["bank_weights"].items():
                weights[int(bank)] = weights.get(int(bank), 0.0) + float(weight)
        return weights

    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    total_banks = max(1, int(package_count) * banks_per_package)
    stage_bytes = max(0, int(getattr(op, "dram_r_bytes" if kind == "read" else "dram_w_bytes", 0) or 0))
    if stage_bytes <= 0:
        return {}

    params = dram_address_trace_params(artifact, cfg)
    stripe_bytes = max(1, int(params["stripe_bytes"]))
    op_id = int(getattr(op, "op_id", op_position))
    fallback_record = {
        "subop_index": 0,
        "tensor_index": 0,
        "tensor_role": "aggregate",
        "stage": kind,
        "total_bytes": stage_bytes,
    }
    base_address = _synthetic_tensor_base_address(fallback_record, op, op_id, kind, stripe_bytes, total_banks)
    return _address_stripe_bank_weights(base_address, stage_bytes, stripe_bytes, total_banks)


def dram_hbm_interleave_params(cfg: TraceConfig, package_count: int) -> Dict[str, int | str]:
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    total_banks = max(1, int(package_count) * banks_per_package)
    stripe_bytes = max(1, int(cfg.hbm_interleave_stripe_bytes))
    return {
        "source": "synthetic_fine_grain_hbm_bank_interleave_from_tsim_bytes",
        "stripe_bytes": stripe_bytes,
        "package_count": max(1, int(package_count)),
        "banks_per_package": banks_per_package,
        "total_logical_hbm_bank_blocks": total_banks,
        "read_write_offset_banks": max(1, total_banks // 2),
        "decode": "byte_stripes_round_robin_over_logical_hbm_bank_blocks",
    }


def dram_hbm_interleave_bank_weights(
    kind: str,
    cfg: TraceConfig,
    op: object,
    op_position: int,
    package_count: int,
) -> Dict[int, float]:
    params = dram_hbm_interleave_params(cfg, package_count)
    total_banks = int(params["total_logical_hbm_bank_blocks"])
    stripe_bytes = int(params["stripe_bytes"])
    stage_bytes = max(0, int(getattr(op, "dram_r_bytes" if kind == "read" else "dram_w_bytes", 0) or 0))
    if stage_bytes <= 0:
        return {}

    op_id = int(getattr(op, "op_id", op_position))
    base_bank = (op_id * 131 + op_position * 17) % total_banks
    if kind == "write":
        base_bank += int(params["read_write_offset_banks"])

    weights: Dict[int, float] = {}
    remaining = stage_bytes
    stripe_idx = 0
    while remaining > 0:
        bytes_this = min(stripe_bytes, remaining)
        bank = (base_bank + stripe_idx) % total_banks
        weights[bank] = weights.get(bank, 0.0) + float(bytes_this)
        remaining -= bytes_this
        stripe_idx += 1
    return weights


def dram_unit_weights(
    meta: dict,
    kind: str,
    cfg: TraceConfig,
    artifact: RunArtifacts,
    op: object,
    op_position: int,
    package_count: int,
) -> Dict[int, float]:
    mapping = resolve_dram_bank_mapping(cfg, artifact)
    if mapping in {"address_trace", "hbm_interleave"}:
        if mapping == "address_trace":
            bank_weights = dram_address_trace_bank_weights(kind, cfg, artifact, op, op_position, package_count)
        else:
            bank_weights = dram_hbm_interleave_bank_weights(kind, cfg, op, op_position, package_count)
        return _dram_unit_weights_from_bank_weights(bank_weights, cfg)

    if cfg.dram_floorplan_granularity == "bank":
        indices = dram_bank_indices(meta, kind, cfg, artifact, op, op_position, package_count)
    else:
        indices = dram_packages(meta, kind, package_count)
    return {idx: 1.0 for idx in indices}


def dram_bank_indices(
    meta: dict,
    kind: str,
    cfg: TraceConfig,
    artifact: RunArtifacts,
    op: object,
    op_position: int,
    package_count: int,
) -> List[int]:
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    total_banks = max(1, int(package_count) * banks_per_package)
    mapping = resolve_dram_bank_mapping(cfg, artifact)
    read_bytes = max(0, int(getattr(op, "dram_r_bytes", 0) or 0))
    write_bytes = max(0, int(getattr(op, "dram_w_bytes", 0) or 0))
    stage_bytes = read_bytes if kind == "read" else write_bytes
    total_stage_bytes = max(1, read_bytes + write_bytes)

    if mapping == "uniform":
        return list(range(total_banks))

    if mapping == "interleave_size":
        # Approximates the paper's size-based interleaving heuristic: each
        # operator stage receives a contiguous bank stripe proportional to its
        # read/write traffic, and consecutive operators rotate through banks.
        span = max(1, int(math.ceil(total_banks * stage_bytes / total_stage_bytes)))
        start = int(getattr(op, "op_id", op_position)) * max(1, span)
        if kind == "write":
            start += span
        return _rotating_contiguous_indices(total_banks, start, span)

    # Software-aware placement approximates the paper's concurrent-access
    # separation: read and write tensors for the same fused operator are placed
    # on disjoint bank stripes when both are present, while single-sided DRAM
    # traffic can use all banks for balance.
    if read_bytes > 0 and write_bytes > 0:
        half = max(1, total_banks // 2)
        op_offset = int(getattr(op, "op_id", op_position)) % total_banks
        if kind == "read":
            return _rotating_contiguous_indices(total_banks, op_offset, half)
        return _rotating_contiguous_indices(total_banks, op_offset + half, total_banks - half)

    vault_packages = dram_packages(meta, kind, package_count)
    if not vault_packages:
        return list(range(total_banks))
    return [
        package * banks_per_package + bank
        for package in vault_packages
        for bank in range(banks_per_package)
    ]


def tsv_tiles(meta: dict, kind: str, grid: int) -> List[int]:
    groups = meta.get("tsv_groups", {}).get(kind, [])
    count = int(meta.get("dram_vault_count") or max(1, len(groups)))
    return sorted({id_to_tile(int(group), count, grid) for group in groups})


def tsv_cores(meta: dict, kind: str) -> List[int]:
    groups = meta.get("tsv_groups", {}).get(kind, [])
    num_cores = int(meta.get("num_cores") or 1)
    core_group = max(1, int(meta.get("core_group_size") or 1))
    cores = set()
    for group in groups:
        start = int(group) * core_group
        end = min(num_cores, start + core_group)
        cores.update(range(start, end))
    return sorted(cores)


def noc_tiles(meta: dict, stage: str, grid: int) -> List[int]:
    links = meta.get("noc_links", {}).get(stage, [])
    count = int(meta.get("num_cores") or 1)
    tiles = set()
    for link in links:
        try:
            left, right = str(link).split("-", 1)
            u, v = int(left), int(right)
        except ValueError:
            continue
        # Attribute link heat to both endpoint tiles; this is conservative
        # for a coarse floorplan and avoids synthetic wire sub-blocks.
        tiles.add(id_to_tile(u, count, grid))
        tiles.add(id_to_tile(v, count, grid))
    return sorted(tiles)


def noc_endpoint_cores(meta: dict, stage: str) -> List[int]:
    links = meta.get("noc_links", {}).get(stage, [])
    cores = set()
    for link in links:
        try:
            left, right = str(link).split("-", 1)
            cores.add(int(left))
            cores.add(int(right))
        except ValueError:
            continue
    return sorted(cores)


def infer_local_meta(artifact: RunArtifacts, op) -> dict:
    """Infer deterministic local IDs for legacy pickles without metadata."""
    num_cores = int(artifact.run_id.num_cores)
    core_group = max(1, int(artifact.run_id.core_group))
    active_cores = list(range(num_cores))
    vault_count = max(1, int(math.ceil(num_cores / core_group)))
    vaults = list(range(vault_count))

    cols = max(1, int(math.ceil(math.sqrt(num_cores))))
    rows = max(1, int(math.ceil(num_cores / cols)))
    links = set()
    for core in active_cores:
        y, x = divmod(core, cols)
        right = core + 1 if x + 1 < cols and core + 1 < num_cores else None
        down = core + cols if y + 1 < rows and core + cols < num_cores else None
        for other in (right, down):
            if other is not None:
                a, b = sorted((core, other))
                links.add(f"{a}-{b}")

    return {
        "version": 1,
        "source": "thermal_inferred_all_cores",
        "num_cores": num_cores,
        "active_core_count": num_cores,
        "core_group_size": core_group,
        "active_core_ranges": [[0, num_cores - 1]],
        "dram_vault_count": vault_count,
        "dram_vaults": {
            "read": vaults if getattr(op, "dram_r_bytes", 0) > 0 else [],
            "write": vaults if getattr(op, "dram_w_bytes", 0) > 0 else [],
        },
        "tsv_groups": {
            "read": vaults if getattr(op, "dram_r_bytes", 0) > 0 else [],
            "write": vaults if getattr(op, "dram_w_bytes", 0) > 0 else [],
        },
        "noc_links": {
            "bcast": sorted(links),
            "shift": sorted(links),
            "reduce": sorted(links),
        },
    }


def _legacy_build_component_power_trace(artifact: RunArtifacts, cfg: TraceConfig) -> PowerTrace:
    base_runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    target_s = _target_duration_s(artifact, cfg)
    n_bins = thermal_bin_count(artifact, cfg)
    dt_s = target_s / n_bins
    cycle_to_s = 1.0 / (cfg.npu_freq_mhz * 1e6)

    full = np.zeros((n_bins, len(COMPONENTS)), dtype=float)
    dram_layers = max(1, int(cfg.dram_layers))
    names = block_names(cfg.grid, dram_layers)
    n_tiles = cfg.grid * cfg.grid
    block_full = np.zeros((n_bins, len(names)), dtype=float)
    uniform_logic = list(range(n_tiles))
    uniform_dram = list(range(n_tiles, n_tiles * (1 + dram_layers)))
    for component, static_w in artifact.summary.static_component_w.items():
        if component in COMPONENTS:
            full[:, COMPONENTS.index(component)] += static_w
    logic_static_w = sum(
        artifact.summary.static_component_w.get(name, 0.0)
        for name in LOGIC_COMPONENTS
    )
    dram_static_w = artifact.summary.static_component_w.get("dram", 0.0)
    block_full[:, uniform_logic] += logic_static_w / max(1, n_tiles)
    block_full[:, uniform_dram] += dram_static_w / max(1, n_tiles * dram_layers)

    single_bins = max(1, min(n_bins, int(math.ceil(base_runtime_s / dt_s))))
    single = np.zeros((single_bins, len(COMPONENTS)), dtype=float)
    block_single = np.zeros((single_bins, len(names)), dtype=float)
    spatial_events = 0
    inferred_events = 0
    fallback_events = 0
    energy_scales = dynamic_energy_scales(artifact, cfg)
    for op in artifact.op_logs:
        meta = meta_from_op(op)
        if not meta:
            meta = infer_local_meta(artifact, op)
            inferred_events += 1
        else:
            spatial_events += 1
        core_block_ids = core_tiles(meta, cfg.grid) if meta else []
        core_logic_blocks = core_block_ids
        core_dram_blocks = [
            n_tiles + layer * n_tiles + tile
            for layer in range(dram_layers)
            for tile in core_block_ids
        ]
        sa_energy = getattr(op, "energy_sa", 0.0) * energy_scales["sa"]
        vu_energy = getattr(op, "energy_vu", 0.0) * energy_scales["vu"]
        add_power_event(single, "sa", op.t_comp_shift_start, op.mm_dur,
                        sa_energy, cycle_to_s, dt_s)
        add_block_event(block_single, core_logic_blocks or uniform_logic, op.t_comp_shift_start, op.mm_dur,
                        sa_energy, cycle_to_s, dt_s)
        add_power_event(single, "vu", op.t_comp_shift_start, op.ew_dur,
                        vu_energy, cycle_to_s, dt_s)
        add_block_event(block_single, core_logic_blocks or uniform_logic, op.t_comp_shift_start, op.ew_dur,
                        vu_energy, cycle_to_s, dt_s)
        sram_energy = getattr(op, "energy_sram", 0.0) * energy_scales["sram"]
        sram_start, sram_duration = sram_power_window(op)
        add_power_event(single, "sram", sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        add_block_event(block_single, core_logic_blocks or uniform_logic, sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        add_power_event(single, "sram", sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        add_block_event(block_single, core_logic_blocks or uniform_logic, sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)

        noc_energy = modeled_noc_energy_pj(getattr(op, "energy_noc", 0.0), cfg) * energy_scales["noc"]
        noc_total = op.bcast_dur + op.shift_dur + op.reduce_dur
        if noc_total > 0:
            bcast_blocks = noc_tiles(meta, "bcast", cfg.grid) if meta else []
            shift_blocks = noc_tiles(meta, "shift", cfg.grid) if meta else []
            reduce_blocks = noc_tiles(meta, "reduce", cfg.grid) if meta else []
            add_power_event(single, "noc", op.t_bcast_start, op.bcast_dur + op.shift_dur,
                            noc_energy * (op.bcast_dur + op.shift_dur) / noc_total,
                            cycle_to_s, dt_s)
            add_block_event(block_single, bcast_blocks or core_logic_blocks or uniform_logic,
                            op.t_bcast_start, op.bcast_dur,
                            noc_energy * op.bcast_dur / noc_total,
                            cycle_to_s, dt_s)
            add_block_event(block_single, shift_blocks or core_logic_blocks or uniform_logic,
                            op.t_comp_shift_start, op.shift_dur,
                            noc_energy * op.shift_dur / noc_total,
                            cycle_to_s, dt_s)
            add_power_event(single, "noc", op.t_reduce_start, op.reduce_dur,
                            noc_energy * op.reduce_dur / noc_total, cycle_to_s, dt_s)
            add_block_event(block_single, reduce_blocks or core_logic_blocks or uniform_logic,
                            op.t_reduce_start, op.reduce_dur,
                            noc_energy * op.reduce_dur / noc_total, cycle_to_s, dt_s)

        dram_energy = getattr(op, "energy_dram", 0.0) * energy_scales["dram"]
        tsv_energy = getattr(op, "energy_tsv", 0.0) * energy_scales["tsv"]
        dram_total = op.dram_ld_dur + op.dram_st_dur
        if dram_total > 0:
            read_dram_tiles = vault_tiles(meta, "read", cfg.grid) if meta else []
            write_dram_tiles = vault_tiles(meta, "write", cfg.grid) if meta else []
            read_dram_blocks = [
                n_tiles + layer * n_tiles + tile
                for layer in range(dram_layers)
                for tile in read_dram_tiles
            ]
            write_dram_blocks = [
                n_tiles + layer * n_tiles + tile
                for layer in range(dram_layers)
                for tile in write_dram_tiles
            ]
            read_tsv_blocks = tsv_tiles(meta, "read", cfg.grid) if meta else []
            write_tsv_blocks = tsv_tiles(meta, "write", cfg.grid) if meta else []
            add_power_event(single, "dram", op.t_dram_ld_start, op.dram_ld_dur,
                            dram_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            add_block_event(block_single, read_dram_blocks or uniform_dram,
                            op.t_dram_ld_start, op.dram_ld_dur,
                            dram_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            add_power_event(single, "dram", op.t_dram_st_start, op.dram_st_dur,
                            dram_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)
            add_block_event(block_single, write_dram_blocks or uniform_dram,
                            op.t_dram_st_start, op.dram_st_dur,
                            dram_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)
            add_power_event(single, "tsv", op.t_dram_ld_start, op.dram_ld_dur,
                            tsv_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            add_block_event(block_single, read_tsv_blocks or core_logic_blocks or uniform_logic,
                            op.t_dram_ld_start, op.dram_ld_dur,
                            tsv_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            add_power_event(single, "tsv", op.t_dram_st_start, op.dram_st_dur,
                            tsv_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)
            add_block_event(block_single, write_tsv_blocks or core_logic_blocks or uniform_logic,
                            op.t_dram_st_start, op.dram_st_dur,
                            tsv_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)

    normalize_dynamic_trace_to_summary(single, block_single, artifact, cfg, dt_s)
    spread_repeated_template_dynamic(single, artifact, cfg, dt_s)
    spread_repeated_template_dynamic(block_single, artifact, cfg, dt_s)
    normalize_dynamic_trace_to_summary(single, block_single, artifact, cfg, dt_s)
    for start in range(0, n_bins, single.shape[0]):
        end = min(n_bins, start + single.shape[0])
        full[start:end] += single[:end - start]
        block_full[start:end] += block_single[:end - start]
    return PowerTrace(full, dt_s, block_power_w=block_full,
                      block_names=tuple(names), spatial_events=spatial_events,
                      inferred_events=inferred_events,
                      fallback_events=fallback_events)


def build_component_power_trace(artifact: RunArtifacts, cfg: TraceConfig) -> PowerTrace:
    base_runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    target_s = _target_duration_s(artifact, cfg)
    n_bins = thermal_bin_count(artifact, cfg)
    dt_s = target_s / n_bins
    cycle_to_s = 1.0 / (cfg.npu_freq_mhz * 1e6)
    n_tiles = cfg.grid * cfg.grid
    domains = _collect_spatial_domains(artifact, cfg)
    tracker = AttributionTracker()

    full = np.zeros((n_bins, len(COMPONENTS)), dtype=float)
    for component, static_w in artifact.summary.static_component_w.items():
        if component in COMPONENTS:
            full[:, COMPONENTS.index(component)] += static_w

    block_full = np.zeros((n_bins, n_tiles * 2), dtype=float)
    logic_static_w = sum(artifact.summary.static_component_w.get(name, 0.0) for name in LOGIC_COMPONENTS)
    dram_static_w = artifact.summary.static_component_w.get("dram", 0.0)
    block_full[:, :n_tiles] += logic_static_w * tile_weights(cfg.grid, cfg.spatial_policy).reshape(n_tiles)[None, :]
    block_full[:, n_tiles:] += dram_static_w / n_tiles

    single_bins = max(1, min(n_bins, int(math.ceil(base_runtime_s / dt_s))))
    single = np.zeros((single_bins, len(COMPONENTS)), dtype=float)
    block_single = np.zeros((single_bins, n_tiles * 2), dtype=float)
    energy_scales = dynamic_energy_scales(artifact, cfg)
    for op in artifact.op_logs:
        _record_op_metadata_presence(op, tracker)
        sa_energy = getattr(op, "energy_sa", 0.0) * energy_scales["sa"]
        add_power_event(single, "sa", op.t_comp_shift_start, op.mm_dur,
                        sa_energy, cycle_to_s, dt_s)
        _add_spatial_event(block_single, op, "sa", op.t_comp_shift_start, op.mm_dur,
                           sa_energy, cfg, domains, tracker, cycle_to_s, dt_s)
        vu_energy = getattr(op, "energy_vu", 0.0) * energy_scales["vu"]
        add_power_event(single, "vu", op.t_comp_shift_start, op.ew_dur,
                        vu_energy, cycle_to_s, dt_s)
        _add_spatial_event(block_single, op, "vu", op.t_comp_shift_start, op.ew_dur,
                           vu_energy, cfg, domains, tracker, cycle_to_s, dt_s)

        sram_energy = getattr(op, "energy_sram", 0.0) * energy_scales["sram"]
        sram_start, sram_duration = sram_power_window(op)
        add_power_event(single, "sram", sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        _add_spatial_event(block_single, op, "sram", sram_start, sram_duration,
                           sram_energy * 0.5, cfg, domains, tracker, cycle_to_s, dt_s)
        add_power_event(single, "sram", sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        _add_spatial_event(block_single, op, "sram", sram_start, sram_duration,
                           sram_energy * 0.5, cfg, domains, tracker, cycle_to_s, dt_s)

        noc_energy = modeled_noc_energy_pj(getattr(op, "energy_noc", 0.0), cfg) * energy_scales["noc"]
        noc_total = op.bcast_dur + op.shift_dur + op.reduce_dur
        if noc_total > 0:
            bcast_shift_energy = noc_energy * (op.bcast_dur + op.shift_dur) / noc_total
            reduce_energy = noc_energy * op.reduce_dur / noc_total
            add_power_event(single, "noc", op.t_bcast_start, op.bcast_dur + op.shift_dur,
                            bcast_shift_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "noc", op.t_bcast_start, op.bcast_dur + op.shift_dur,
                               bcast_shift_energy, cfg, domains, tracker, cycle_to_s, dt_s)
            add_power_event(single, "noc", op.t_reduce_start, op.reduce_dur,
                            reduce_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "noc", op.t_reduce_start, op.reduce_dur,
                               reduce_energy, cfg, domains, tracker, cycle_to_s, dt_s)

        dram_energy = getattr(op, "energy_dram", 0.0) * energy_scales["dram"]
        tsv_energy = getattr(op, "energy_tsv", 0.0) * energy_scales["tsv"]
        dram_total = op.dram_ld_dur + op.dram_st_dur
        if dram_total > 0:
            dram_ld_energy = dram_energy * op.dram_ld_dur / dram_total
            dram_st_energy = dram_energy * op.dram_st_dur / dram_total
            tsv_ld_energy = tsv_energy * op.dram_ld_dur / dram_total
            tsv_st_energy = tsv_energy * op.dram_st_dur / dram_total
            add_power_event(single, "dram", op.t_dram_ld_start, op.dram_ld_dur,
                            dram_ld_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "dram", op.t_dram_ld_start, op.dram_ld_dur,
                               dram_ld_energy, cfg, domains, tracker, cycle_to_s, dt_s)
            add_power_event(single, "dram", op.t_dram_st_start, op.dram_st_dur,
                            dram_st_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "dram", op.t_dram_st_start, op.dram_st_dur,
                               dram_st_energy, cfg, domains, tracker, cycle_to_s, dt_s)
            add_power_event(single, "tsv", op.t_dram_ld_start, op.dram_ld_dur,
                            tsv_ld_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "tsv", op.t_dram_ld_start, op.dram_ld_dur,
                               tsv_ld_energy, cfg, domains, tracker, cycle_to_s, dt_s)
            add_power_event(single, "tsv", op.t_dram_st_start, op.dram_st_dur,
                            tsv_st_energy, cycle_to_s, dt_s)
            _add_spatial_event(block_single, op, "tsv", op.t_dram_st_start, op.dram_st_dur,
                               tsv_st_energy, cfg, domains, tracker, cycle_to_s, dt_s)

    normalize_dynamic_trace_to_summary(single, block_single, artifact, cfg, dt_s)
    spread_repeated_template_dynamic(single, artifact, cfg, dt_s)
    spread_repeated_template_dynamic(block_single, artifact, cfg, dt_s)
    normalize_dynamic_trace_to_summary(single, block_single, artifact, cfg, dt_s)
    for start in range(0, n_bins, single.shape[0]):
        end = min(n_bins, start + single.shape[0])
        full[start:end] += single[:end - start]
        block_full[start:end] += block_single[:end - start]
    return PowerTrace(full, dt_s, block_power_w=block_full, spatial_attribution=tracker.as_dict(domains, cfg.grid))


_build_component_power_trace_without_inference = build_component_power_trace


def build_component_power_trace(artifact: RunArtifacts, cfg: TraceConfig) -> PowerTrace:
    return _legacy_build_component_power_trace(artifact, cfg)


def tile_weights(grid: int, policy: str) -> np.ndarray:
    if policy == "uniform":
        weights = np.ones((grid, grid), dtype=float)
    elif policy == "edge_hotspot":
        xs = np.linspace(0.0, 1.0, grid)
        yy, xx = np.meshgrid(xs, xs)
        weights = 1.0 + 2.0 * np.exp(-((xx - 0.1) ** 2 + (yy - 0.1) ** 2) / 0.12)
    else:
        xs = np.linspace(-1.0, 1.0, grid)
        yy, xx = np.meshgrid(xs, xs)
        weights = 1.0 + 2.0 * np.exp(-(xx ** 2 + yy ** 2) / 0.35)
    weights /= weights.sum()
    return weights


def block_names(grid: int, dram_layers: int = 1) -> List[str]:
    names = [f"logic_{y}_{x}" for y in range(grid) for x in range(grid)]
    for layer in range(max(1, int(dram_layers))):
        prefix = "dram" if dram_layers == 1 else f"dram{layer}"
        names.extend(f"{prefix}_{y}_{x}" for y in range(grid) for x in range(grid))
    return names


def block_power_matrix(trace: PowerTrace, cfg: TraceConfig) -> np.ndarray:
    if trace.block_power_w is not None:
        return trace.block_power_w
    grid = cfg.grid
    n_tiles = grid * grid
    dram_layers = max(1, int(cfg.dram_layers))
    weights = tile_weights(grid, cfg.spatial_policy).reshape(n_tiles)
    uniform = np.full(n_tiles, 1.0 / n_tiles)
    matrix = np.zeros((trace.component_power_w.shape[0], n_tiles * (1 + dram_layers)), dtype=float)
    component_index = {name: idx for idx, name in enumerate(trace.component_names)}
    logic_power = sum(trace.component_power_w[:, component_index[name]] for name in LOGIC_COMPONENTS)
    dram_power = trace.component_power_w[:, component_index["dram"]]
    matrix[:, :n_tiles] = logic_power[:, None] * weights[None, :]
    per_layer = dram_power[:, None] * (uniform / dram_layers)[None, :]
    for layer in range(dram_layers):
        start = n_tiles * (1 + layer)
        matrix[:, start:start + n_tiles] = per_layer
    return matrix


def export_block_names(artifact: RunArtifacts, cfg: TraceConfig) -> List[str]:
    dram_layers = max(1, int(cfg.dram_layers))
    die_size_mm = effective_die_size_mm(artifact, cfg)
    package_layout = hbm_package_layout(cfg, die_size_mm)
    package_count = int(package_layout["package_count"])
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    outline_width_mm, outline_height_mm = stack_outline_mm(die_size_mm, package_layout)
    if cfg.logic_floorplan == "intra_core":
        names = _logic_subblock_names(int(artifact.run_id.num_cores))
    else:
        names = [f"logic_{y}_{x}" for y in range(cfg.grid) for x in range(cfg.grid)]
    if cfg.logic_floorplan == "intra_core":
        names.extend(_logic_aux_names_with_tsv(
            artifact,
            cfg,
            die_size_mm,
            outline_width_mm,
            outline_height_mm,
            package_count,
        ))
    else:
        names.extend(floorplan_pad_names("logic", die_size_mm, die_size_mm, outline_width_mm, outline_height_mm))
    for layer in range(dram_layers):
        prefix = "dram" if dram_layers == 1 else f"dram{layer}"
        if cfg.dram_floorplan_granularity == "bank":
            names.extend(hbm_bank_names(prefix, package_count, banks_per_package))
        else:
            names.extend(hbm_package_names(prefix, package_count))
        names.extend(floorplan_pad_names(
            prefix,
            package_layout["footprint_width_mm"],
            package_layout["footprint_height_mm"],
            outline_width_mm,
            outline_height_mm,
        ))
    return names


def export_coarse_block_power_matrix(artifact: RunArtifacts, trace: PowerTrace, cfg: TraceConfig) -> np.ndarray:
    grid = cfg.grid
    die_size_mm = effective_die_size_mm(artifact, cfg)
    package_layout = hbm_package_layout(cfg, die_size_mm)
    outline_width_mm, outline_height_mm = stack_outline_mm(die_size_mm, package_layout)
    n_logic_real_blocks = grid * grid
    n_logic_pad_blocks = len(floorplan_pad_names("logic", die_size_mm, die_size_mm, outline_width_mm, outline_height_mm))
    n_logic_blocks = n_logic_real_blocks + n_logic_pad_blocks
    dram_layers = max(1, int(cfg.dram_layers))
    package_count = int(package_layout["package_count"])
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    n_dram_real_blocks_per_layer = package_count * banks_per_package if cfg.dram_floorplan_granularity == "bank" else package_count
    n_dram_pad_blocks_per_layer = len(floorplan_pad_names(
        "dram",
        package_layout["footprint_width_mm"],
        package_layout["footprint_height_mm"],
        outline_width_mm,
        outline_height_mm,
    ))
    n_dram_blocks_per_layer = n_dram_real_blocks_per_layer + n_dram_pad_blocks_per_layer
    n_dram_blocks = dram_layers * n_dram_blocks_per_layer
    matrix = np.zeros((trace.component_power_w.shape[0], n_logic_blocks + n_dram_blocks), dtype=float)
    component_index = {name: idx for idx, name in enumerate(trace.component_names)}
    logic_power = sum(trace.component_power_w[:, component_index[name]] for name in LOGIC_COMPONENTS)
    dram_power = trace.component_power_w[:, component_index["dram"]]
    logic_weights = tile_weights(grid, cfg.spatial_policy).reshape(n_logic_real_blocks)
    matrix[:, :n_logic_real_blocks] = logic_power[:, None] * logic_weights[None, :]
    for layer in range(dram_layers):
        start = n_logic_blocks + layer * n_dram_blocks_per_layer
        matrix[:, start:start + n_dram_real_blocks_per_layer] = (
            dram_power[:, None] / max(1, n_dram_real_blocks_per_layer * dram_layers)
        )
    return matrix


def export_block_power_matrix(artifact: RunArtifacts, trace: PowerTrace, cfg: TraceConfig) -> np.ndarray:
    if cfg.logic_floorplan != "intra_core":
        return export_coarse_block_power_matrix(artifact, trace, cfg)

    base_runtime_s = artifact.summary.exec_cycles / (cfg.npu_freq_mhz * 1e6)
    n_bins = trace.component_power_w.shape[0]
    dt_s = trace.dt_s
    cycle_to_s = 1.0 / (cfg.npu_freq_mhz * 1e6)
    num_cores = max(1, int(artifact.run_id.num_cores))
    n_core_logic_blocks = num_cores * 5
    die_size_mm = effective_die_size_mm(artifact, cfg)
    package_layout = hbm_package_layout(cfg, die_size_mm)
    outline_width_mm, outline_height_mm = stack_outline_mm(die_size_mm, package_layout)
    dram_layers = max(1, int(cfg.dram_layers))
    package_count = int(package_layout["package_count"])
    n_logic_aux_blocks = len(_logic_aux_names_with_tsv(
        artifact,
        cfg,
        die_size_mm,
        outline_width_mm,
        outline_height_mm,
        package_count,
    ))
    n_logic_blocks = n_core_logic_blocks + n_logic_aux_blocks
    banks_per_package = max(1, int(cfg.hbm_banks_per_package))
    n_dram_real_blocks_per_layer = package_count * banks_per_package if cfg.dram_floorplan_granularity == "bank" else package_count
    n_dram_pad_blocks_per_layer = len(floorplan_pad_names(
        "dram",
        package_layout["footprint_width_mm"],
        package_layout["footprint_height_mm"],
        outline_width_mm,
        outline_height_mm,
    ))
    n_dram_blocks_per_layer = n_dram_real_blocks_per_layer + n_dram_pad_blocks_per_layer
    n_dram_blocks = n_dram_blocks_per_layer * dram_layers
    matrix = np.zeros((n_bins, n_logic_blocks + n_dram_blocks), dtype=float)
    subblocks = _logic_subblock_indices(num_cores)
    tsv_block_ids = list(subblocks["tsv"].values())

    static = artifact.summary.static_component_w
    for unit, component in (("sa", "sa"), ("vu", "vu"), ("sram", "sram")):
        block_ids = list(subblocks[unit].values())
        matrix[:, block_ids] += static.get(component, 0.0) / max(1, len(block_ids))
    router_ids = list(subblocks["router"].values())
    matrix[:, router_ids] += static.get("noc", 0.0) / max(1, len(router_ids))
    matrix[:, tsv_block_ids] += static.get("tsv", 0.0) / max(1, len(tsv_block_ids))
    dram_static = static.get("dram", 0.0)
    dram_start = n_logic_blocks
    for layer in range(dram_layers):
        layer_start = dram_start + layer * n_dram_blocks_per_layer
        matrix[:, layer_start:layer_start + n_dram_real_blocks_per_layer] += (
            dram_static / max(1, n_dram_real_blocks_per_layer * dram_layers)
        )

    single_bins = max(1, min(n_bins, int(math.ceil(base_runtime_s / dt_s))))
    single = np.zeros((single_bins, matrix.shape[1]), dtype=float)
    energy_scales = dynamic_energy_scales(artifact, cfg)
    for op_position, op in enumerate(artifact.op_logs):
        meta = meta_from_op(op) or infer_local_meta(artifact, op)
        active_cores = expand_ranges(meta.get("active_core_ranges", [])) or list(range(num_cores))
        active_cores = [core for core in active_cores if 0 <= core < num_cores]

        sa_energy = getattr(op, "energy_sa", 0.0) * energy_scales["sa"]
        add_block_event(single, [subblocks["sa"][core] for core in active_cores],
                        op.t_comp_shift_start, op.mm_dur,
                        sa_energy, cycle_to_s, dt_s)
        vu_energy = getattr(op, "energy_vu", 0.0) * energy_scales["vu"]
        add_block_event(single, [subblocks["vu"][core] for core in active_cores],
                        op.t_comp_shift_start, op.ew_dur,
                        vu_energy, cycle_to_s, dt_s)

        sram_energy = getattr(op, "energy_sram", 0.0) * energy_scales["sram"]
        sram_start, sram_duration = sram_power_window(op)
        sram_blocks = [subblocks["sram"][core] for core in active_cores]
        add_block_event(single, sram_blocks, sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)
        add_block_event(single, sram_blocks, sram_start, sram_duration,
                        sram_energy * 0.5, cycle_to_s, dt_s)

        noc_energy = modeled_noc_energy_pj(getattr(op, "energy_noc", 0.0), cfg) * energy_scales["noc"]
        noc_total = op.bcast_dur + op.shift_dur + op.reduce_dur
        if noc_total > 0:
            bcast_cores = noc_endpoint_cores(meta, "bcast") or active_cores
            shift_cores = noc_endpoint_cores(meta, "shift") or active_cores
            reduce_cores = noc_endpoint_cores(meta, "reduce") or active_cores
            add_block_event(single, [subblocks["router"][core] for core in bcast_cores if core in subblocks["router"]],
                            op.t_bcast_start, op.bcast_dur,
                            noc_energy * op.bcast_dur / noc_total, cycle_to_s, dt_s)
            add_block_event(single, [subblocks["router"][core] for core in shift_cores if core in subblocks["router"]],
                            op.t_comp_shift_start, op.shift_dur,
                            noc_energy * op.shift_dur / noc_total, cycle_to_s, dt_s)
            add_block_event(single, [subblocks["router"][core] for core in reduce_cores if core in subblocks["router"]],
                            op.t_reduce_start, op.reduce_dur,
                            noc_energy * op.reduce_dur / noc_total, cycle_to_s, dt_s)

        dram_energy = getattr(op, "energy_dram", 0.0) * energy_scales["dram"]
        tsv_energy = getattr(op, "energy_tsv", 0.0) * energy_scales["tsv"]
        dram_total = op.dram_ld_dur + op.dram_st_dur
        if dram_total > 0:
            mapping = resolve_dram_bank_mapping(cfg, artifact)
            read_dram_unit_weights = dram_unit_weights(meta, "read", cfg, artifact, op, op_position, package_count)
            write_dram_unit_weights = dram_unit_weights(meta, "write", cfg, artifact, op, op_position, package_count)
            read_dram_blocks = {
                dram_start + layer * n_dram_blocks_per_layer + unit: weight
                for layer in range(dram_layers)
                for unit, weight in read_dram_unit_weights.items()
            }
            write_dram_blocks = {
                dram_start + layer * n_dram_blocks_per_layer + unit: weight
                for layer in range(dram_layers)
                for unit, weight in write_dram_unit_weights.items()
            }
            all_dram_blocks = [
                dram_start + layer * n_dram_blocks_per_layer + unit
                for layer in range(dram_layers)
                for unit in range(n_dram_real_blocks_per_layer)
            ]
            read_record_events = dram_mapping_record_events(meta, "read", cfg, artifact, op, op_position, package_count)
            write_record_events = dram_mapping_record_events(meta, "write", cfg, artifact, op, op_position, package_count)

            def _event_blocks(event: Dict[str, Any]) -> Dict[int, float]:
                return {
                    dram_start + layer * n_dram_blocks_per_layer + int(unit): float(weight)
                    for layer in range(dram_layers)
                    for unit, weight in event.get("unit_weights", {}).items()
                }

            if read_record_events:
                read_energy = dram_energy * op.dram_ld_dur / dram_total
                total_event_bytes = sum(float(event.get("total_bytes", 0.0)) for event in read_record_events)
                for event in read_record_events:
                    event_bytes = float(event.get("total_bytes", 0.0))
                    event_energy = read_energy * (event_bytes / total_event_bytes) if total_event_bytes > 0 else 0.0
                    add_weighted_block_event(single, _event_blocks(event),
                                             int(event.get("start_cycle", op.t_dram_ld_start)),
                                             int(event.get("duration_cycle", op.dram_ld_dur)),
                                             event_energy, cycle_to_s, dt_s)
            elif read_dram_blocks:
                add_weighted_block_event(single, read_dram_blocks,
                                         op.t_dram_ld_start, op.dram_ld_dur,
                                         dram_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            else:
                add_block_event(single, all_dram_blocks,
                                op.t_dram_ld_start, op.dram_ld_dur,
                                dram_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            if write_record_events:
                write_energy = dram_energy * op.dram_st_dur / dram_total
                total_event_bytes = sum(float(event.get("total_bytes", 0.0)) for event in write_record_events)
                for event in write_record_events:
                    event_bytes = float(event.get("total_bytes", 0.0))
                    event_energy = write_energy * (event_bytes / total_event_bytes) if total_event_bytes > 0 else 0.0
                    add_weighted_block_event(single, _event_blocks(event),
                                             int(event.get("start_cycle", op.t_dram_st_start)),
                                             int(event.get("duration_cycle", op.dram_st_dur)),
                                             event_energy, cycle_to_s, dt_s)
            elif write_dram_blocks:
                add_weighted_block_event(single, write_dram_blocks,
                                         op.t_dram_st_start, op.dram_st_dur,
                                         dram_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)
            else:
                add_block_event(single, all_dram_blocks,
                                op.t_dram_st_start, op.dram_st_dur,
                                dram_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)

            read_tsv_cores = tsv_cores(meta, "read") or active_cores
            write_tsv_cores = tsv_cores(meta, "write") or active_cores
            add_block_event(single, [subblocks["tsv"][core] for core in read_tsv_cores if core in subblocks["tsv"]],
                            op.t_dram_ld_start, op.dram_ld_dur,
                            tsv_energy * op.dram_ld_dur / dram_total, cycle_to_s, dt_s)
            add_block_event(single, [subblocks["tsv"][core] for core in write_tsv_cores if core in subblocks["tsv"]],
                            op.t_dram_st_start, op.dram_st_dur,
                            tsv_energy * op.dram_st_dur / dram_total, cycle_to_s, dt_s)

    normalize_block_dynamic_to_summary(single, artifact, cfg, dt_s)
    spread_repeated_template_dynamic(single, artifact, cfg, dt_s)
    normalize_block_dynamic_to_summary(single, artifact, cfg, dt_s)
    for start in range(0, n_bins, single.shape[0]):
        end = min(n_bins, start + single.shape[0])
        matrix[start:end] += single[:end - start]
    return matrix


def export_trace_package(
    artifact: RunArtifacts,
    trace: PowerTrace,
    cfg: TraceConfig,
    out_dir: Path,
    write_visualizations: bool = True,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dram_layers = max(1, int(cfg.dram_layers))
    die_size_mm = effective_die_size_mm(artifact, cfg)
    names = export_block_names(artifact, cfg)
    matrix = export_block_power_matrix(artifact, trace, cfg)

    ptrace = out_dir / "power.ptrace"
    with open(ptrace, "w", encoding="utf-8") as outfile:
        outfile.write(" ".join(names) + "\n")
        for row in matrix:
            outfile.write(" ".join(f"{value:.8f}" for value in row) + "\n")

    csv_path = out_dir / "component_power.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["time_s", *trace.component_names, "total"])
        for idx, row in enumerate(trace.component_power_w):
            writer.writerow([idx * trace.dt_s, *row.tolist(), float(row.sum())])

    spatial_csv_path = out_dir / "spatial_power.csv"
    with open(spatial_csv_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["time_s", *names, "total"])
        for idx, row in enumerate(matrix):
            writer.writerow([idx * trace.dt_s, *row.tolist(), float(row.sum())])

    logic_flp = out_dir / "logic.flp"
    package_layout = hbm_package_layout(cfg, die_size_mm)
    outline_width_mm, outline_height_mm = stack_outline_mm(die_size_mm, package_layout)
    if cfg.logic_floorplan == "intra_core":
        _write_intracore_flp(logic_flp, artifact, cfg, die_size_mm, outline_width_mm, outline_height_mm)
    else:
        _write_grid_flp(logic_flp, "logic", cfg.grid, die_size_mm, outline_width_mm, outline_height_mm)
    dram_flps = []
    for layer in range(dram_layers):
        prefix = "dram" if dram_layers == 1 else f"dram{layer}"
        dram_flp = out_dir / f"{prefix}.flp"
        _write_hbm_package_flp(
            dram_flp,
            prefix,
            package_layout,
            outline_width_mm,
            outline_height_mm,
            cfg.dram_floorplan_granularity,
            cfg.hbm_banks_per_package,
        )
        dram_flps.append(dram_flp)

    bond_flp = out_dir / "bond.flp"
    _write_hbm_package_flp(bond_flp, "bond", package_layout, outline_width_mm, outline_height_mm)

    lcf = out_dir / "stack.lcf"
    bond_thickness_m = max(float(cfg.bond_thickness_um), 1e-6) * 1e-6
    lcf_lines = [
        "# HotSpot layer configuration file.",
        "# Format per layer:",
        "# <Layer Number>",
        "# <Lateral heat flow Y/N?>",
        "# <Power Dissipation Y/N?>",
        "# <Specific heat capacity in J/(m^3K)>",
        "# <Resistivity in (m-K)/W>",
        "# <Thickness in m>",
        "# <floorplan file>",
        "",
        "# Layer 0: logic silicon",
        "0",
        "Y",
        "Y",
        "1.75e6",
        "0.01",
        "50e-6",
        logic_flp.name,
        "",
    ]
    layer_idx = 1
    for dram_idx, dram_flp in enumerate(dram_flps):
        lcf_lines.extend([
            f"# Layer {layer_idx}: polymer/solder microbump bonding layer below HBM die {dram_idx}",
            str(layer_idx),
            "Y",
            "N",
            "4.0e6",
            "0.5",
            f"{bond_thickness_m:.9g}",
            bond_flp.name,
            "",
        ])
        layer_idx += 1
        lcf_lines.extend([
            f"# Layer {layer_idx}: HBM DRAM silicon die {dram_idx}",
            str(layer_idx),
            "Y",
            "Y",
            "1.75e6",
            "0.01",
            "30e-6",
            dram_flp.name,
            "",
        ])
        layer_idx += 1
    lcf.write_text(
        "\n".join(lcf_lines),
        encoding="utf-8",
    )

    hotspot_config = out_dir / "hotspot.config"
    ambient_k = cfg.ambient_c + 273.15
    logic_area_mm2 = die_size_mm * die_size_mm
    dram_footprint_area_mm2 = package_layout["footprint_width_mm"] * package_layout["footprint_height_mm"]
    effective_sink_area_mm2 = max(logic_area_mm2, dram_footprint_area_mm2)
    heat_transfer_coefficient_w_per_um2_k = 1.0 / (
        max(float(cfg.r_convec_k_per_w), 1e-12) * max(effective_sink_area_mm2, 1e-12) * 1.0e6
    )
    outline_max_m = max(outline_width_mm, outline_height_mm) / 1000.0
    spreader_side_m = max(0.03, outline_max_m * 1.10)
    sink_side_m = math.sqrt(effective_sink_area_mm2) / 1000.0
    secondary_side_m = max(0.021, outline_max_m * 1.05)
    hotspot_config.write_text(
        "\n".join([
            "-model_type grid",
            f"-grid_rows {cfg.hotspot_grid}",
            f"-grid_cols {cfg.hotspot_grid}",
            "-grid_map_mode center",
            f"-sampling_intvl {trace.dt_s}",
            f"-grid_layer_file {lcf.name}",
            "-detailed_3D on",
            f"-ambient {ambient_k}",
            "-init_file (null)",
            f"-init_temp {ambient_k}",
            "-model_secondary 1",
            f"-r_convec {cfg.r_convec_k_per_w:.9g}",
            f"-c_convec {cfg.c_convec_j_per_k:.9g}",
            f"-s_sink {sink_side_m:.9g}",
            "-t_sink 0.0069",
            "-k_sink 400.0",
            "-p_sink 3.55e6",
            f"-s_spreader {spreader_side_m:.9g}",
            "-t_spreader 0.001",
            "-k_spreader 400.0",
            "-p_spreader 3.55e6",
            "-t_interface 20e-6",
            "-k_interface 4.0",
            "-p_interface 4.0e6",
            "-r_convec_sec 50.0",
            "-c_convec_sec 40.0",
            "-n_metal 8",
            "-t_metal 100e-6",
            "-t_c4 100e-6",
            "-s_c4 20e-6",
            "-n_c4 400",
            f"-s_sub {secondary_side_m:.9g}",
            "-t_sub 0.001",
            f"-s_solder {secondary_side_m:.9g}",
            "-t_solder 0.00094",
            "-s_pcb 0.1",
            "-t_pcb 0.002",
            "",
        ]),
        encoding="utf-8",
    )
    init_file = out_dir / "init.init"
    init_file.write_text("\n".join(f"{name} {ambient_k}" for name in names) + "\n", encoding="utf-8")

    metadata = out_dir / "metadata.json"
    core_rows, core_cols = _core_grid_dims(max(1, int(artifact.run_id.num_cores)))
    core_width_mm = die_size_mm / core_cols
    core_height_mm = die_size_mm / core_rows
    tsv_geometry = _per_core_tsv_geometry(artifact, cfg, die_size_mm)
    bank_cols = max(1, int(math.ceil(math.sqrt(max(1, int(cfg.hbm_banks_per_package))))))
    bank_rows = max(1, int(math.ceil(max(1, int(cfg.hbm_banks_per_package)) / bank_cols)))
    bank_width_mm = package_layout["package_width_mm"] / bank_cols
    bank_height_mm = package_layout["package_height_mm"] / bank_rows
    hotspot_cell_width_mm = outline_width_mm / cfg.hotspot_grid
    hotspot_cell_height_mm = outline_height_mm / cfg.hotspot_grid
    metadata.write_text(
        json.dumps(
            {
                "run": artifact.run_id.__dict__,
                "log_path": str(artifact.log_path),
                "pickle_path": str(artifact.pickle_path),
                "dt_s": trace.dt_s,
                "duration_s": trace.duration_s,
                "grid": cfg.grid,
                "spatial_policy": cfg.spatial_policy,
                "die_size_mm": die_size_mm,
                "die_size_source": (
                    "explicit"
                    if cfg.die_size_mm is not None
                    else "tsim_intracore_area_model"
                    if cfg.logic_floorplan == "intra_core"
                    else "logic_static_power_estimate"
                ),
                "logic_area_estimate_mm2": logic_area_mm2,
                "logic_static_power_density_w_per_mm2": LOGIC_STATIC_POWER_DENSITY_W_PER_MM2,
                "tsim_intracore_area_model_mm2_per_core": (
                    intracore_unit_areas_mm2(artifact, cfg)
                    if cfg.logic_floorplan == "intra_core"
                    else {}
                ),
                "tsv_region": {
                    "area_mm2": tsv_geometry["total_area_mm2"],
                    "area_mm2_at_12tbps": cfg.tsv_region_area_mm2_at_12tbps,
                    "reference_bandwidth_gbps": TSV_AREA_REFERENCE_GBPS,
                    "run_dram_bw_gbps": artifact.run_id.dram_bw,
                    "scaling": "linear_with_dram_bw",
                    "placement": "per_core",
                    "per_core_area_mm2": tsv_geometry["per_core_area_mm2"],
                    "per_core_width_mm": tsv_geometry["tsv_width_mm"],
                    "per_core_height_mm": tsv_geometry["tsv_height_mm"],
                    "core_width_mm": tsv_geometry["core_width_mm"],
                    "core_height_mm": tsv_geometry["core_height_mm"],
                    "block_count": int(artifact.run_id.num_cores),
                },
                "ambient_c": cfg.ambient_c,
                "dram_layers": dram_layers,
                "dram_capacity_gib": cfg.dram_capacity_mb / 1024,
                "temporal_sampling": temporal_sampling_summary(artifact, cfg, trace),
                "dynamic_energy_reconciliation": dynamic_reconciliation_summary(artifact, cfg, trace),
                "logic_floorplan": cfg.logic_floorplan,
                "hotspot_grid": cfg.hotspot_grid,
                "noc_power_model": noc_power_metadata(cfg),
                "cooling_model": {
                    "profile": cfg.cooling_profile,
                    "r_convec_k_per_w": cfg.r_convec_k_per_w,
                    "c_convec_j_per_k": cfg.c_convec_j_per_k,
                    "logic_direct_to_heatsink": cfg.logic_direct_to_heatsink,
                    "effective_sink_area_mm2": effective_sink_area_mm2,
                    "heat_transfer_coefficient_w_per_um2_k": heat_transfer_coefficient_w_per_um2_k,
                    "heat_transfer_coefficient_w_per_m2_k": heat_transfer_coefficient_w_per_um2_k * 1.0e12,
                },
                "spatial_resolution": {
                    "hotspot_cell_width_mm": hotspot_cell_width_mm,
                    "hotspot_cell_height_mm": hotspot_cell_height_mm,
                    "logic_core_rows": core_rows,
                    "logic_core_cols": core_cols,
                    "logic_core_width_mm": core_width_mm,
                    "logic_core_height_mm": core_height_mm,
                    "hotspot_cells_per_core_width": core_width_mm / hotspot_cell_width_mm,
                    "hotspot_cells_per_core_height": core_height_mm / hotspot_cell_height_mm,
                    "hbm_bank_cols_per_package": bank_cols,
                    "hbm_bank_rows_per_package": bank_rows,
                    "hbm_bank_width_mm": bank_width_mm,
                    "hbm_bank_height_mm": bank_height_mm,
                    "hotspot_cells_per_bank_width": bank_width_mm / hotspot_cell_width_mm,
                    "hotspot_cells_per_bank_height": bank_height_mm / hotspot_cell_height_mm,
                },
                "hbm_package_layout": {
                    "package_count": package_layout["package_count"],
                    "package_capacity_gib": cfg.hbm_package_capacity_mb / 1024,
                    "package_area_mm2": cfg.hbm_package_area_mm2,
                    "package_width_mm": package_layout["package_width_mm"],
                    "package_height_mm": package_layout["package_height_mm"],
                    "package_aspect_ratio": cfg.hbm_package_aspect_ratio,
                    "floorplan_granularity": cfg.dram_floorplan_granularity,
                    "banks_per_package": cfg.hbm_banks_per_package,
                    "bank_mapping": resolve_dram_bank_mapping(cfg, artifact),
                    "grid_rows": package_layout["rows"],
                    "grid_cols": package_layout["cols"],
                    "footprint_width_mm": package_layout["footprint_width_mm"],
                    "footprint_height_mm": package_layout["footprint_height_mm"],
                    "footprint_area_mm2": dram_footprint_area_mm2,
                    "total_hbm_package_area_mm2": package_layout["package_count"] * cfg.hbm_package_area_mm2,
                    "effective_sink_area_mm2": effective_sink_area_mm2,
                    "effective_sink_area_source": (
                        "hbm_footprint" if dram_footprint_area_mm2 >= logic_area_mm2 else "logic_die"
                    ),
                    "hotspot_outline_width_mm": outline_width_mm,
                    "hotspot_outline_height_mm": outline_height_mm,
                    "hotspot_spreader_side_mm": spreader_side_m * 1000.0,
                    "hotspot_sink_side_mm": sink_side_m * 1000.0,
                    "hotspot_secondary_side_mm": secondary_side_m * 1000.0,
                },
                "dram_address_trace": (
                    dram_address_trace_params(artifact, cfg)
                    if resolve_dram_bank_mapping(cfg, artifact) == "address_trace"
                    else {}
                ),
                "dram_hbm_interleave": (
                    dram_hbm_interleave_params(cfg, package_layout["package_count"])
                    if resolve_dram_bank_mapping(cfg, artifact) == "hbm_interleave"
                    else {}
                ),
                "hotspot_material_assumptions": {
                    "logic_silicon": {
                        "volumetric_heat_capacity_j_per_m3k": 1.75e6,
                        "thermal_resistivity_mk_per_w": 0.01,
                        "thermal_conductivity_w_per_mk": 100.0,
                        "thickness_m": 50e-6,
                    },
                    "hbm_dram_silicon": {
                        "volumetric_heat_capacity_j_per_m3k": 1.75e6,
                        "thermal_resistivity_mk_per_w": 0.01,
                        "thermal_conductivity_w_per_mk": 100.0,
                        "thickness_m": 30e-6,
                    },
                    "hbm_polymer_solder_bond": {
                        "volumetric_heat_capacity_j_per_m3k": 4.0e6,
                        "thermal_resistivity_mk_per_w": 0.5,
                        "thermal_conductivity_w_per_mk": 2.0,
                        "thickness_m": bond_thickness_m,
                        "thickness_um": cfg.bond_thickness_um,
                        "power_dissipation": "N",
                    },
                },
                "spatial_attribution": trace.spatial_attribution or {},
                "spatial_trace": trace.block_power_w is not None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    package_paths = {
        "ptrace": ptrace,
        "component_power_csv": csv_path,
        "spatial_power_csv": spatial_csv_path,
        "logic_flp": logic_flp,
        "dram_flp": dram_flps[0],
        "dram_flps": dram_flps,
        "lcf": lcf,
        "hotspot_config": hotspot_config,
        "init": init_file,
        "metadata": metadata,
    }
    if write_visualizations:
        from .visualization import write_layout_visualizations

        package_paths.update(write_layout_visualizations(out_dir))
    return package_paths


def stack_outline_mm(die_size_mm: float, package_layout: Dict[str, Any]) -> Tuple[float, float]:
    return (
        max(float(die_size_mm), float(package_layout["footprint_width_mm"])),
        max(float(die_size_mm), float(package_layout["footprint_height_mm"])),
    )


def floorplan_pad_names(
    prefix: str,
    base_width_mm: float,
    base_height_mm: float,
    outline_width_mm: float,
    outline_height_mm: float,
) -> List[str]:
    names: List[str] = []
    eps = 1e-9
    if outline_width_mm > base_width_mm + eps:
        names.append(f"{prefix}_pad_right")
    if outline_height_mm > base_height_mm + eps:
        names.append(f"{prefix}_pad_top")
    return names


def _floorplan_pad_lines(
    prefix: str,
    base_width_mm: float,
    base_height_mm: float,
    outline_width_mm: float | None,
    outline_height_mm: float | None,
) -> List[str]:
    if outline_width_mm is None or outline_height_mm is None:
        return []
    base_w_m = base_width_mm / 1000.0
    base_h_m = base_height_mm / 1000.0
    outline_w_m = outline_width_mm / 1000.0
    outline_h_m = outline_height_mm / 1000.0
    lines: List[str] = []
    eps_m = 1e-12
    if outline_w_m > base_w_m + eps_m:
        lines.append(
            f"{prefix}_pad_right {outline_w_m - base_w_m:.12f} {base_h_m:.12f} "
            f"{base_w_m:.12f} 0.000000000000"
        )
    if outline_h_m > base_h_m + eps_m:
        lines.append(
            f"{prefix}_pad_top {outline_w_m:.12f} {outline_h_m - base_h_m:.12f} "
            f"0.000000000000 {base_h_m:.12f}"
        )
    return lines


def _write_grid_flp(
    path: Path,
    prefix: str,
    grid: int,
    die_size_mm: float,
    outline_width_mm: float | None = None,
    outline_height_mm: float | None = None,
) -> None:
    die_m = die_size_mm / 1000.0
    tile = die_m / grid
    lines = []
    for y in range(grid):
        for x in range(grid):
            x0 = die_m * x / grid
            x1 = die_m * (x + 1) / grid
            y0 = die_m * y / grid
            y1 = die_m * (y + 1) / grid
            lines.append(f"{prefix}_{y}_{x} {x1 - x0:.12f} {y1 - y0:.12f} {x0:.12f} {y0:.12f}")
    lines.extend(_floorplan_pad_lines(prefix, die_size_mm, die_size_mm, outline_width_mm, outline_height_mm))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_full_die_flp(path: Path, name: str, die_size_mm: float) -> None:
    die_m = die_size_mm / 1000.0
    path.write_text(f"{name} {die_m:.12f} {die_m:.12f} 0.000000000000 0.000000000000\n", encoding="utf-8")


def hbm_package_layout(cfg: TraceConfig, target_die_size_mm: float) -> Dict[str, Any]:
    package_count = hbm_package_count(cfg)
    package_width_mm, package_height_mm = get_hbm_package_footprint_mm(
        cfg.hbm_package_area_mm2,
        cfg.hbm_package_aspect_ratio,
    )
    target_mm = float(target_die_size_mm)
    best: Tuple[float, float, int, int, float, float] | None = None
    for cols in range(1, package_count + 1):
        rows = int(math.ceil(package_count / cols))
        width_mm = cols * package_width_mm
        height_mm = rows * package_height_mm
        score = abs(width_mm - target_mm) + abs(height_mm - target_mm)
        max_overhang = max(width_mm / max(1e-9, target_mm), height_mm / max(1e-9, target_mm))
        candidate = (score, max_overhang, rows * cols - package_count, cols, rows, width_mm, height_mm)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    _score, _overhang, _unused, cols, rows, footprint_width_mm, footprint_height_mm = best
    positions = []
    for idx in range(package_count):
        row, col = divmod(idx, cols)
        positions.append({
            "idx": idx,
            "x_mm": col * package_width_mm,
            "y_mm": row * package_height_mm,
            "width_mm": package_width_mm,
            "height_mm": package_height_mm,
        })
    return {
        "package_count": package_count,
        "package_width_mm": package_width_mm,
        "package_height_mm": package_height_mm,
        "rows": rows,
        "cols": cols,
        "footprint_width_mm": footprint_width_mm,
        "footprint_height_mm": footprint_height_mm,
        "positions": positions,
    }


def _write_hbm_package_flp(
    path: Path,
    prefix: str,
    layout: Dict[str, Any],
    outline_width_mm: float | None = None,
    outline_height_mm: float | None = None,
    granularity: str = "package",
    banks_per_package: int = 16,
) -> None:
    lines = []
    bank_cols = max(1, int(math.ceil(math.sqrt(max(1, int(banks_per_package))))))
    bank_rows = max(1, int(math.ceil(max(1, int(banks_per_package)) / bank_cols)))
    for pos in layout["positions"]:
        if granularity == "bank":
            bank_w_mm = pos["width_mm"] / bank_cols
            bank_h_mm = pos["height_mm"] / bank_rows
            for bank in range(max(1, int(banks_per_package))):
                bank_row, bank_col = divmod(bank, bank_cols)
                lines.append(
                    f"{prefix}_pkg{int(pos['idx']):02d}_bank{bank:02d} "
                    f"{bank_w_mm / 1000.0:.12f} {bank_h_mm / 1000.0:.12f} "
                    f"{(pos['x_mm'] + bank_col * bank_w_mm) / 1000.0:.12f} "
                    f"{(pos['y_mm'] + bank_row * bank_h_mm) / 1000.0:.12f}"
                )
        else:
            lines.append(
                f"{prefix}_pkg{int(pos['idx']):02d} "
                f"{pos['width_mm'] / 1000.0:.12f} {pos['height_mm'] / 1000.0:.12f} "
                f"{pos['x_mm'] / 1000.0:.12f} {pos['y_mm'] / 1000.0:.12f}"
            )
    lines.extend(_floorplan_pad_lines(
        prefix,
        layout["footprint_width_mm"],
        layout["footprint_height_mm"],
        outline_width_mm,
        outline_height_mm,
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_intracore_flp(
    path: Path,
    artifact: RunArtifacts,
    cfg: TraceConfig,
    die_size_mm: float,
    outline_width_mm: float | None = None,
    outline_height_mm: float | None = None,
) -> None:
    num_cores = max(1, int(artifact.run_id.num_cores))
    rows, cols = _core_grid_dims(num_cores)
    die_m = die_size_mm / 1000.0
    core_w = die_m / cols
    core_h = die_m / rows
    unit_areas_mm2 = intracore_unit_areas_mm2(artifact, cfg)
    required_area_mm2 = sum(unit_areas_mm2.values())
    available_area_mm2 = (core_w * 1000.0) * (core_h * 1000.0)
    if available_area_mm2 + 1e-6 < required_area_mm2:
        raise ValueError(
            "logic die size is too small for TSIM intra-core areas: "
            f"{available_area_mm2:.6f} mm^2/core available, {required_area_mm2:.6f} mm^2/core required"
        )

    core_w_mm = core_w * 1000.0
    tsv_h = (unit_areas_mm2["tsv"] / core_w_mm) / 1000.0 if core_w_mm > 0 else 0.0
    sram_h = (unit_areas_mm2["sram"] / core_w_mm) / 1000.0 if core_w_mm > 0 else 0.0
    compute_area_mm2 = unit_areas_mm2["sa"] + unit_areas_mm2["vu"] + unit_areas_mm2["router"]
    compute_h = (compute_area_mm2 / core_w_mm) / 1000.0 if core_w_mm > 0 else 0.0
    compute_h_mm = compute_h * 1000.0
    if compute_h_mm <= 0:
        raise ValueError("TSIM compute-unit area is non-positive; cannot write intra-core floorplan")
    sa_w = (unit_areas_mm2["sa"] / compute_h_mm) / 1000.0
    vu_w = (unit_areas_mm2["vu"] / compute_h_mm) / 1000.0
    router_w = (unit_areas_mm2["router"] / compute_h_mm) / 1000.0

    lines = []
    for core in range(num_cores):
        row, col = divmod(core, cols)
        x0 = col * core_w
        y0 = row * core_h
        lines.append(
            f"core_{core:03d}_sram {core_w:.12f} {sram_h:.12f} {x0:.12f} {y0 + tsv_h + compute_h:.12f}"
        )
        lines.append(
            f"core_{core:03d}_sa {sa_w:.12f} {compute_h:.12f} {x0:.12f} {y0 + tsv_h:.12f}"
        )
        lines.append(
            f"core_{core:03d}_vu {vu_w:.12f} {compute_h:.12f} {x0 + sa_w:.12f} {y0 + tsv_h:.12f}"
        )
        lines.append(
            f"core_{core:03d}_router {router_w:.12f} {compute_h:.12f} {x0 + sa_w + vu_w:.12f} {y0 + tsv_h:.12f}"
        )
        lines.append(
            f"core_{core:03d}_tsv {core_w:.12f} {tsv_h:.12f} {x0:.12f} {y0:.12f}"
        )
    if outline_width_mm is not None and outline_height_mm is not None:
        eps = 1e-9
        if outline_width_mm > die_size_mm + eps:
            right_w_mm = outline_width_mm - die_size_mm
            lines.append(
                f"logic_pad_right {right_w_mm / 1000.0:.12f} {die_size_mm / 1000.0:.12f} "
                f"{die_size_mm / 1000.0:.12f} {0.0:.12f}"
            )
        if outline_height_mm is not None and outline_height_mm > die_size_mm + 1e-9:
            lines.append(
                f"logic_pad_top {outline_width_mm / 1000.0:.12f} {(outline_height_mm - die_size_mm) / 1000.0:.12f} "
                f"{0.0:.12f} {die_size_mm / 1000.0:.12f}"
            )
    else:
        lines.extend(_floorplan_pad_lines("logic", die_size_mm, die_size_mm, outline_width_mm, outline_height_mm))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
