"""Pluggable NoC power backends for thermal power-trace export.

The temporal simulator historically reports NoC energy as a single
``energy_noc`` value per fused operator.  These backends characterize a router
and link with either TSIM's existing constant, DSENT, or ORION, then rescale
that operator energy before it is spatially assigned to router blocks.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping


TSIM_SIMPLE_DATAMOVE_PJ_PER_BYTE = 12.0


class NoCPowerBackendError(RuntimeError):
    """Raised when a requested external NoC power backend cannot be used."""


@dataclass(frozen=True)
class NoCPowerConfig:
    backend: str = "tsim_simple"
    frequency_hz: float = 1.5e9
    flit_bits: int = 64
    router_ports: int = 5
    injection_rate: float = 0.3
    link_length_mm: float = 1.0
    dsent_tech: str = "TG11LVT"
    orion_version: int = 2
    simple_datamove_pj_per_byte: float = TSIM_SIMPLE_DATAMOVE_PJ_PER_BYTE


@dataclass(frozen=True)
class NoCPowerCharacterization:
    backend: str
    tech: str
    router_dynamic_energy_j_per_flit: float
    link_dynamic_energy_j_per_flit: float
    router_leakage_w: float = 0.0
    link_leakage_w: float = 0.0
    router_area_mm2: float = 0.0
    link_area_mm2: float = 0.0
    simple_energy_j_per_flit: float = 0.0
    scale_vs_tsim_simple: float = 1.0
    command: str = ""
    notes: str = ""

    @property
    def total_dynamic_energy_j_per_flit(self) -> float:
        return self.router_dynamic_energy_j_per_flit + self.link_dynamic_energy_j_per_flit

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["total_dynamic_energy_j_per_flit"] = self.total_dynamic_energy_j_per_flit
        return data


def repo_root_from_src() -> Path:
    return Path(__file__).resolve().parents[2]


def simple_energy_j_per_flit(cfg: NoCPowerConfig) -> float:
    return cfg.simple_datamove_pj_per_byte * max(1, cfg.flit_bits) / 8.0 * 1e-12


def supported_backends() -> tuple[str, ...]:
    return ("tsim_simple", "simple", "dsent", "orion")


def normalize_backend_name(name: str) -> str:
    normalized = (name or "tsim_simple").strip().lower().replace("-", "_")
    if normalized == "simple":
        return "tsim_simple"
    if normalized not in supported_backends():
        raise ValueError(f"unknown NoC power backend '{name}'. Known: tsim_simple, dsent, orion")
    return normalized


def _parse_key_value_lines(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        match = re.search(r"^\s*([A-Za-z0-9_>:\-.]+)\s*=\s*([-+0-9.eE]+)", line)
        if match:
            values[match.group(1)] = float(match.group(2))
            continue
        match = re.search(r"^\s*([A-Za-z][A-Za-z0-9_ ]*):\s*([-+0-9.eE]+)", line)
        if match:
            values[match.group(1).strip()] = float(match.group(2))
    return values


def _require_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise NoCPowerBackendError(f"{label} is not positive: {value!r}")
    return value


def _run(cmd: list[str], cwd: Path) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise NoCPowerBackendError(f"failed to execute {' '.join(cmd)}: {exc}") from exc
    output = completed.stdout + ("\n" + completed.stderr if completed.stderr else "")
    if completed.returncode != 0:
        raise NoCPowerBackendError(
            f"{' '.join(cmd)} failed with exit code {completed.returncode}:\n{output.strip()}"
        )
    return output


class NoCPowerBackend:
    name = "base"

    def __init__(self, cfg: NoCPowerConfig, repo_root: Path | None = None) -> None:
        self.cfg = cfg
        self.repo_root = repo_root or repo_root_from_src()

    def characterize(self) -> NoCPowerCharacterization:
        raise NotImplementedError

    def scale_energy_pj(self, base_energy_pj: float) -> float:
        if base_energy_pj <= 0.0:
            return 0.0
        return base_energy_pj * self.characterize().scale_vs_tsim_simple


class TsimSimpleNoCPowerBackend(NoCPowerBackend):
    name = "tsim_simple"

    def characterize(self) -> NoCPowerCharacterization:
        simple_j = simple_energy_j_per_flit(self.cfg)
        return NoCPowerCharacterization(
            backend=self.name,
            tech="tsim_constant",
            router_dynamic_energy_j_per_flit=simple_j,
            link_dynamic_energy_j_per_flit=0.0,
            simple_energy_j_per_flit=simple_j,
            scale_vs_tsim_simple=1.0,
            notes=(
                "Existing TSIM NoC model: operator energy_noc is produced by "
                f"{self.cfg.simple_datamove_pj_per_byte:g} pJ/byte-style data movement."
            ),
        )


class DSENTNoCPowerBackend(NoCPowerBackend):
    name = "dsent"

    def _dsent_dir(self) -> Path:
        return self.repo_root / "external" / "dsent0.91" / "OENOC" / "dsent0.91"

    def characterize(self) -> NoCPowerCharacterization:
        root = self._dsent_dir()
        exe = root / "dsent"
        if not exe.exists():
            raise NoCPowerBackendError(
                f"DSENT executable not found at {exe}. Clone/build zzcnb1/dsent0.91 under external/."
            )
        tech_file = f"tech/tech_models/{self.cfg.dsent_tech}.model"
        common = [
            f"ElectricalTechModelFilename={tech_file}",
            f"Frequency={self.cfg.frequency_hz:.12g}",
            f"InjectionRate={self.cfg.injection_rate:.12g}",
        ]
        router_overwrite = ";".join(
            common
            + [
                f"NumberBitsPerFlit={self.cfg.flit_bits}",
                f"NumberInputPorts={self.cfg.router_ports}",
                f"NumberOutputPorts={self.cfg.router_ports}",
            ]
        )
        link_length_m = max(1.0e-9, self.cfg.link_length_mm * 1.0e-3)
        link_overwrite = ";".join(
            common
            + [
                f"NumberBits={self.cfg.flit_bits}",
                f"WireLength={link_length_m:.12g}",
                f"Delay={1.0 / self.cfg.frequency_hz:.12g}",
            ]
        )
        router_cmd = ["./dsent", "-cfg", "configs/router.cfg", "-overwrite", router_overwrite]
        link_cmd = ["./dsent", "-cfg", "configs/electrical-link.cfg", "-overwrite", link_overwrite]
        router_out = _run(router_cmd, root)
        link_out = _run(link_cmd, root)
        router = _parse_key_value_lines(router_out)
        link = _parse_key_value_lines(link_out)

        router_event_keys = (
            "Energy>>Router:WriteBuffer",
            "Energy>>Router:ReadBuffer",
            "Energy>>Router:TraverseCrossbar->Multicast1",
            "Energy>>Router:ArbitrateSwitch->ArbitrateStage1",
            "Energy>>Router:ArbitrateSwitch->ArbitrateStage2",
            "Energy>>Router:DistributeClock",
        )
        router_energy = sum(router.get(key, 0.0) for key in router_event_keys)
        link_energy = link.get("Energy>>RepeatedLink:Send", 0.0)
        simple_j = simple_energy_j_per_flit(self.cfg)
        total_j = _require_positive(router_energy + link_energy, "DSENT NoC energy per flit")
        area_keys = (
            "Area>>Router:Active->InputPort:Active",
            "Area>>Router:Active->SwitchAllocator:Active",
            "Area>>Router:Active->Crossbar:Active",
            "Area>>Router:Active->Crossbar_Sel_DFF:Active",
            "Area>>Router:Active->ClockTree:Active",
            "Area>>Router:Active->PipelineReg0:Active",
            "Area>>Router:Active->PipelineReg1:Active",
            "Area>>Router:Active->PipelineReg2_0:Active",
        )
        return NoCPowerCharacterization(
            backend=self.name,
            tech=self.cfg.dsent_tech,
            router_dynamic_energy_j_per_flit=router_energy,
            link_dynamic_energy_j_per_flit=link_energy,
            router_leakage_w=router.get("NddPower>>Router:Leakage", 0.0),
            link_leakage_w=link.get("NddPower>>RepeatedLink:Leakage", 0.0),
            router_area_mm2=sum(router.get(key, 0.0) for key in area_keys) * 1.0e6,
            link_area_mm2=link.get("Area>>RepeatedLink:Active", 0.0) * 1.0e6,
            simple_energy_j_per_flit=simple_j,
            scale_vs_tsim_simple=total_j / simple_j,
            command=json.dumps({"router": router_cmd, "link": link_cmd}),
            notes=(
                "DSENT 0.91 characterization. DSENT does not include a TSMC N7 model; "
                "the default uses its TG11LVT technology as the closest shipped node."
            ),
        )


class OrionNoCPowerBackend(NoCPowerBackend):
    name = "orion"

    def _orion_dir(self) -> Path:
        return self.repo_root / "external" / "vnoc20" / "orion3"

    def _probe_path(self) -> Path:
        return self.repo_root / "src" / "tools" / "tsim_orion_probe"

    def characterize(self) -> NoCPowerCharacterization:
        root = self._orion_dir()
        exe = root / "orion_router"
        if not exe.exists():
            raise NoCPowerBackendError(
                f"ORION executable not found at {exe}. Clone/build eigenpi/vnoc20 under external/."
            )
        if not exe.stat().st_mode & 0o111:
            raise NoCPowerBackendError(f"ORION executable is not executable: {exe}")
        load = min(1.0, max(1.0e-9, self.cfg.injection_rate))
        # This checkout's parser expects argv[2] == 2 to select the ORION 2 path.
        router_cmd = ["./orion_router", "tsim", "2", "-p", "-d", "0", "-l", f"{load:.12g}", "router"]
        out = _run(router_cmd, root)
        total_mw = None
        area_um2 = None
        for line in out.splitlines():
            total_match = re.search(r"\bTotal:([-+0-9.eE]+)", line)
            if total_match:
                total_mw = float(total_match.group(1))
            area_match = re.search(r"\bAtotal:([-+0-9.eE]+)", line)
            if area_match:
                area_um2 = float(area_match.group(1))
        if total_mw is None:
            raise NoCPowerBackendError(f"could not parse ORION router output:\n{out.strip()}")
        router_power_w = total_mw * 1.0e-3
        router_energy = router_power_w / (
            self.cfg.frequency_hz * load * max(1, self.cfg.router_ports)
        )

        link_energy = 0.0
        link_leakage = 0.0
        link_area_mm2 = 0.0
        probe = self._probe_path()
        probe_cmd: Iterable[str] | None = None
        if probe.exists() and probe.stat().st_mode & 0o111:
            link_length_m = max(1.0e-9, self.cfg.link_length_mm * 1.0e-3)
            probe_cmd = [str(probe), f"{link_length_m:.12g}", str(self.cfg.flit_bits)]
            probe_out = _run(list(probe_cmd), self.repo_root)
            probe_values = _parse_key_value_lines(probe_out)
            link_energy = probe_values.get("LinkDynamicEnergyPerFlitJ", 0.0)
            link_leakage = probe_values.get("LinkLeakageW", 0.0)
            link_area_mm2 = probe_values.get("LinkAreaUm2", 0.0) * 1.0e-6
        simple_j = simple_energy_j_per_flit(self.cfg)
        total_j = _require_positive(router_energy + link_energy, "ORION NoC energy per flit")
        return NoCPowerCharacterization(
            backend=self.name,
            tech=f"ORION{self.cfg.orion_version}",
            router_dynamic_energy_j_per_flit=router_energy,
            link_dynamic_energy_j_per_flit=link_energy,
            router_area_mm2=(area_um2 or 0.0) * 1.0e-6,
            link_leakage_w=link_leakage,
            link_area_mm2=link_area_mm2,
            simple_energy_j_per_flit=simple_j,
            scale_vs_tsim_simple=total_j / simple_j,
            command=json.dumps({"router": router_cmd, "link_probe": list(probe_cmd or [])}),
            notes=(
                "ORION characterization from the external vnoc20/orion3 checkout. "
                "Router parameters are compile-time SIM_port.h settings; the optional "
                "tsim_orion_probe binary calls ORION's link API for link energy."
            ),
        )


@lru_cache(maxsize=32)
def _characterization_cached(
    backend: str,
    frequency_hz: float,
    flit_bits: int,
    router_ports: int,
    injection_rate: float,
    link_length_mm: float,
    dsent_tech: str,
    orion_version: int,
    simple_datamove_pj_per_byte: float,
    repo_root: str,
) -> NoCPowerCharacterization:
    cfg = NoCPowerConfig(
        backend=backend,
        frequency_hz=frequency_hz,
        flit_bits=flit_bits,
        router_ports=router_ports,
        injection_rate=injection_rate,
        link_length_mm=link_length_mm,
        dsent_tech=dsent_tech,
        orion_version=orion_version,
        simple_datamove_pj_per_byte=simple_datamove_pj_per_byte,
    )
    return make_backend(cfg, Path(repo_root)).characterize()


def make_backend(cfg: NoCPowerConfig, repo_root: Path | None = None) -> NoCPowerBackend:
    name = normalize_backend_name(cfg.backend)
    normalized_cfg = NoCPowerConfig(**{**asdict(cfg), "backend": name})
    if name == "tsim_simple":
        return TsimSimpleNoCPowerBackend(normalized_cfg, repo_root)
    if name == "dsent":
        return DSENTNoCPowerBackend(normalized_cfg, repo_root)
    if name == "orion":
        return OrionNoCPowerBackend(normalized_cfg, repo_root)
    raise ValueError(f"unknown NoC power backend '{cfg.backend}'")


def characterize(cfg: NoCPowerConfig, repo_root: Path | None = None) -> NoCPowerCharacterization:
    root = str((repo_root or repo_root_from_src()).resolve())
    return _characterization_cached(
        normalize_backend_name(cfg.backend),
        float(cfg.frequency_hz),
        int(cfg.flit_bits),
        int(cfg.router_ports),
        float(cfg.injection_rate),
        float(cfg.link_length_mm),
        str(cfg.dsent_tech),
        int(cfg.orion_version),
        float(cfg.simple_datamove_pj_per_byte),
        root,
    )


def scale_energy_pj(base_energy_pj: float, cfg: NoCPowerConfig, repo_root: Path | None = None) -> float:
    if base_energy_pj <= 0.0:
        return 0.0
    return base_energy_pj * characterize(cfg, repo_root).scale_vs_tsim_simple


def describe(cfg: NoCPowerConfig, repo_root: Path | None = None) -> Mapping[str, object]:
    return characterize(cfg, repo_root).to_dict()
