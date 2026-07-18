"""Pluggable thermal backend interface for TSIM traces.

Backends consume the same normalized thermal context: a TSIM artifact, a
time-binned power trace, trace/thermal configs, and an optional exported
package directory.  New simulators should implement ``ThermalBackend`` and
register through ``BACKEND_REGISTRY`` or a small factory wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Protocol

from .adaptive_grid import run_adaptive_grid
from .artifacts import RunArtifacts
from .hotspot import run_hotspot
from .models import ThermalConfig, ThermalResult, analyze
from .operator_analysis import write_operator_hotspot_analysis
from .threedice import run_threedice
from .trace import PowerTrace, TraceConfig


@dataclass(frozen=True)
class ThermalBackendOptions:
    hotspot_bin: str = "hotspot"
    hotspot_timeout_s: float = 300.0
    threedice_bin: str = "3D-ICE"
    threedice_timeout_s: float = 300.0
    threedice_heatmap_count: int = 4
    threedice_heatmap_layers: str = "logic,dram0,dram3,dram7"
    adaptive_grid_heatmap_count: int = 6
    operator_hotspot_analysis: bool = True
    operator_heatmap_count: int = 12


@dataclass
class ThermalBackendContext:
    artifact: RunArtifacts
    trace_cfg: TraceConfig
    thermal_cfg: ThermalConfig
    trace: PowerTrace
    package_dir: Path
    options: ThermalBackendOptions
    prior_results: Dict[str, "ThermalBackendResult"] = field(default_factory=dict)


@dataclass
class ThermalBackendResult:
    name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    raw: Any = None


class ThermalBackend(Protocol):
    """Interface implemented by thermal simulation backends."""

    name: str
    requires_package: bool

    def run(self, context: ThermalBackendContext) -> ThermalBackendResult:
        """Run the backend and return status, metrics, and generated artifacts."""
        ...


class SimpleBackend:
    """Built-in TSIM simple/stack RC backend."""

    name = "simple"
    requires_package = False

    def run(self, context: ThermalBackendContext) -> ThermalBackendResult:
        result = analyze(context.trace, context.trace_cfg, context.thermal_cfg)
        return ThermalBackendResult(
            name=self.name,
            status="ok",
            metrics={
                "simple_peak_c": result.simple_peak_c,
                "stack_logic_peak_c": result.stack_logic_peak_c,
                "stack_dram_peak_c": result.stack_dram_peak_c,
                "stack_peak_c": result.stack_peak_c,
                "stack_p95_peak_c": result.stack_p95_peak_c,
                "simple_slowdown": result.simple_slowdown,
                "stack_slowdown": result.stack_slowdown,
                "slowdown_delta_pct": result.slowdown_delta_pct,
            },
            raw=result,
        )


class HotSpotBackend:
    """HotSpot-compatible backend using the exported package directory."""

    name = "hotspot"
    requires_package = True

    def run(self, context: ThermalBackendContext) -> ThermalBackendResult:
        hs = run_hotspot(
            context.package_dir,
            context.options.hotspot_bin,
            context.options.hotspot_timeout_s,
        )
        status = "missing" if not hs.available else ("ok" if hs.returncode == 0 else f"failed:{hs.returncode}")
        run_json = context.package_dir / "hotspot_run.json"
        artifacts = {"hotspot_run": str(run_json)}
        if status == "ok" and context.options.operator_hotspot_analysis:
            op_artifacts = write_operator_hotspot_analysis(
                context.artifact,
                context.trace,
                context.trace_cfg,
                context.thermal_cfg,
                context.package_dir,
                context.options.operator_heatmap_count,
            )
            artifacts.update({name: str(path) for name, path in op_artifacts.items()})
        run_json.write_text(
            json.dumps({
                "status": status,
                "peak_c": hs.peak_c,
                "stdout": hs.stdout,
                "stderr": hs.stderr,
            }, indent=2),
            encoding="utf-8",
        )
        return ThermalBackendResult(
            name=self.name,
            status=status,
            metrics={"hotspot_peak_c": hs.peak_c},
            artifacts=artifacts,
            raw=hs,
        )


class ThreeDICEBackend:
    """3D-ICE-compatible backend using a TSIM-exported package."""

    name = "threedice"
    requires_package = True

    def run(self, context: ThermalBackendContext) -> ThermalBackendResult:
        result = run_threedice(
            context.package_dir,
            context.options.threedice_bin,
            context.options.threedice_timeout_s,
            context.options.threedice_heatmap_count,
            context.options.threedice_heatmap_layers,
        )
        status = "missing" if not result.available else ("ok" if result.returncode == 0 else f"failed:{result.returncode}")
        run_json = context.package_dir / "threedice_run.json"
        artifacts = {"threedice_run": str(run_json)}
        if result.model_path is not None:
            artifacts["threedice_model"] = str(result.model_path)
        if result.output_path is not None:
            artifacts["threedice_output"] = str(result.output_path)
        if result.temperature_trace is not None:
            artifacts["threedice_temperature_trace"] = str(result.temperature_trace)
        if result.heatmap_dir is not None:
            artifacts["threedice_heatmap_dir"] = str(result.heatmap_dir)
        run_json.write_text(
            json.dumps({
                "status": status,
                "peak_c": result.peak_c,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "model_path": str(result.model_path) if result.model_path else None,
                "output_path": str(result.output_path) if result.output_path else None,
                "temperature_trace": str(result.temperature_trace) if result.temperature_trace else None,
                "heatmap_dir": str(result.heatmap_dir) if result.heatmap_dir else None,
                "runtime_s": result.runtime_s,
                "num_cores": result.num_cores,
            }, indent=2),
            encoding="utf-8",
        )
        return ThermalBackendResult(
            name=self.name,
            status=status,
            metrics={"threedice_peak_c": result.peak_c},
            artifacts=artifacts,
            raw=result,
        )


class AdaptiveGridBackend:
    """TSIM-native variable-size block-grid thermal proxy."""

    name = "adaptive_grid"
    requires_package = True

    def run(self, context: ThermalBackendContext) -> ThermalBackendResult:
        result = run_adaptive_grid(
            context.package_dir,
            context.trace_cfg.ambient_c,
            context.trace.dt_s,
            context.options.adaptive_grid_heatmap_count,
        )
        status = result.status
        artifacts: Dict[str, str] = {}
        if result.temperature_trace is not None:
            artifacts["adaptive_temperature_trace"] = str(result.temperature_trace)
        if result.summary_path is not None:
            artifacts["adaptive_grid_summary"] = str(result.summary_path)
        if result.heatmap_dir is not None:
            artifacts["adaptive_grid_heatmap_dir"] = str(result.heatmap_dir)
        return ThermalBackendResult(
            name=self.name,
            status=status,
            metrics={"adaptive_grid_peak_c": result.peak_c},
            artifacts=artifacts,
            raw=result,
        )


BACKEND_REGISTRY: Dict[str, ThermalBackend] = {
    AdaptiveGridBackend.name: AdaptiveGridBackend(),
    SimpleBackend.name: SimpleBackend(),
    HotSpotBackend.name: HotSpotBackend(),
    ThreeDICEBackend.name: ThreeDICEBackend(),
}


def parse_backend_names(value: str, run_hotspot: bool = False) -> list[str]:
    names = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not names:
        names = ["simple"]
    if run_hotspot and "hotspot" not in names:
        names.append("hotspot")
    return names


def resolve_backends(names: Iterable[str]) -> list[ThermalBackend]:
    backends: list[ThermalBackend] = []
    for name in names:
        try:
            backends.append(BACKEND_REGISTRY[name])
        except KeyError as exc:
            known = ", ".join(sorted(BACKEND_REGISTRY))
            raise ValueError(f"unknown thermal backend '{name}' (known: {known})") from exc
    return backends
