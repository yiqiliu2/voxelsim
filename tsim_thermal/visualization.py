"""Layout visualization helpers for exported HotSpot packages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FlpBlock:
    name: str
    width_mm: float
    height_mm: float
    x_mm: float
    y_mm: float

    @property
    def right_mm(self) -> float:
        return self.x_mm + self.width_mm

    @property
    def top_mm(self) -> float:
        return self.y_mm + self.height_mm


@dataclass(frozen=True)
class StackLayer:
    index: int
    power: bool
    heat_capacity: float
    resistivity: float
    thickness_um: float
    floorplan: str


def read_flp(path: Path) -> List[FlpBlock]:
    blocks: List[FlpBlock] = []
    with open(path, encoding="utf-8") as infile:
        for line in infile:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 5:
                continue
            name, width_m, height_m, x_m, y_m = parts[:5]
            blocks.append(
                FlpBlock(
                    name=name,
                    width_mm=float(width_m) * 1000.0,
                    height_mm=float(height_m) * 1000.0,
                    x_mm=float(x_m) * 1000.0,
                    y_mm=float(y_m) * 1000.0,
                )
            )
    return blocks


def read_lcf(path: Path) -> List[StackLayer]:
    values = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            values.append(line)
    layers: List[StackLayer] = []
    idx = 0
    while idx + 6 < len(values):
        layers.append(
            StackLayer(
                index=int(values[idx]),
                power=values[idx + 2].upper() == "Y",
                heat_capacity=float(values[idx + 3]),
                resistivity=float(values[idx + 4]),
                thickness_um=float(values[idx + 5]) * 1e6,
                floorplan=values[idx + 6],
            )
        )
        idx += 7
    return layers


def floorplan_bounds(blocks: Iterable[FlpBlock]) -> tuple[float, float]:
    width_mm = 0.0
    height_mm = 0.0
    for block in blocks:
        width_mm = max(width_mm, block.right_mm)
        height_mm = max(height_mm, block.top_mm)
    return width_mm, height_mm


def write_layout_visualizations(package_dir: Path) -> Dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    metadata = json.loads((package_dir / "metadata.json").read_text(encoding="utf-8"))
    logic_blocks = read_flp(package_dir / "logic.flp")
    dram_name = "dram0.flp" if (package_dir / "dram0.flp").exists() else "dram.flp"
    dram_blocks = read_flp(package_dir / dram_name)
    layers = read_lcf(package_dir / "stack.lcf")

    artifacts = {
        "logic_floorplan": package_dir / "logic_floorplan.png",
        "dram_floorplan": package_dir / "dram_floorplan.png",
        "stack_cross_section": package_dir / "stack_cross_section.png",
        "layout_overview": package_dir / "layout_overview.png",
    }

    _draw_single_floorplan(
        logic_blocks,
        artifacts["logic_floorplan"],
        title="Logic Layer Floorplan",
        palette=_logic_palette(),
        legend=[
            Patch(facecolor="#4c78a8", edgecolor="#26384f", label="SRAM"),
            Patch(facecolor="#f58518", edgecolor="#6f3b05", label="Systolic array"),
            Patch(facecolor="#54a24b", edgecolor="#244a20", label="Vector unit"),
            Patch(facecolor="#b279a2", edgecolor="#57334b", label="Router/NoC"),
            Patch(facecolor="#d4b896", edgecolor="#8b6914", label="TSV region"),
            Patch(facecolor="#e8e8e8", edgecolor="#a0a0a0", label="Zero-power pad"),
        ],
    )
    _draw_single_floorplan(
        dram_blocks,
        artifacts["dram_floorplan"],
        title="HBM DRAM Package Floorplan",
        palette=_dram_palette(),
        legend=[
            Patch(facecolor="#72b7b2", edgecolor="#2e5c59", label="HBM package"),
            Patch(facecolor="#e8e8e8", edgecolor="#a0a0a0", label="Zero-power pad"),
        ],
        label_packages=True,
    )
    _draw_cross_section(
        layers,
        metadata,
        artifacts["stack_cross_section"],
        legend=[
            Patch(facecolor="#4c78a8", edgecolor="#26384f", label="Logic silicon"),
            Patch(facecolor="#72b7b2", edgecolor="#2e5c59", label="HBM DRAM silicon"),
            Patch(facecolor="#bab0ac", edgecolor="#5c5755", label="Bond/underfill"),
        ],
    )
    _draw_overview(
        logic_blocks,
        dram_blocks,
        layers,
        metadata,
        artifacts["layout_overview"],
        Rectangle,
        Patch,
        plt,
    )
    plt.close("all")
    return artifacts


def _logic_palette():
    def color(block: FlpBlock) -> tuple[str, str]:
        if "_pad_" in block.name:
            return "#e8e8e8", "#a0a0a0"
        if block.name.startswith("tsv_") or block.name.endswith("_tsv"):
            return "#d4b896", "#8b6914"
        if block.name.endswith("_sram"):
            return "#4c78a8", "#26384f"
        if block.name.endswith("_sa"):
            return "#f58518", "#6f3b05"
        if block.name.endswith("_vu"):
            return "#54a24b", "#244a20"
        if block.name.endswith("_router"):
            return "#b279a2", "#57334b"
        return "#9ecae9", "#4a6f8a"

    return color


def _dram_palette():
    def color(block: FlpBlock) -> tuple[str, str]:
        if "_pad_" in block.name:
            return "#e8e8e8", "#a0a0a0"
        return "#72b7b2", "#2e5c59"

    return color


def _draw_single_floorplan(
    blocks: List[FlpBlock],
    path: Path,
    title: str,
    palette,
    legend,
    label_packages: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    _draw_floorplan_axis(ax, blocks, title, palette, label_packages)
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_floorplan_axis(ax, blocks: List[FlpBlock], title: str, palette, label_packages: bool = False) -> None:
    from matplotlib.patches import Rectangle

    for block in blocks:
        face, edge = palette(block)
        ax.add_patch(
            Rectangle(
                (block.x_mm, block.y_mm),
                block.width_mm,
                block.height_mm,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.35 if "_pad_" not in block.name else 0.7,
                hatch="//" if "_pad_" in block.name else None,
            )
        )
        if label_packages and "_pkg" in block.name and "_bank" not in block.name:
            ax.text(
                block.x_mm + block.width_mm / 2.0,
                block.y_mm + block.height_mm / 2.0,
                block.name.rsplit("_", 1)[-1],
                ha="center",
                va="center",
                fontsize=7,
                color="#123a38",
            )
    width_mm, height_mm = floorplan_bounds(blocks)
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, color="#dddddd", linewidth=0.35)
    ax.text(
        0.01,
        0.99,
        f"{width_mm:.3f} mm x {height_mm:.3f} mm",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 2},
    )


def _draw_cross_section(layers: List[StackLayer], metadata: dict, path: Path, legend) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    _draw_cross_section_axis(ax, layers, metadata)
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_cross_section_axis(ax, layers: List[StackLayer], metadata: dict) -> None:
    from matplotlib.patches import Rectangle

    layout = metadata.get("hbm_package_layout", {})
    width_mm = float(layout.get("hotspot_outline_width_mm") or metadata.get("die_size_mm") or 1.0)
    z_um = 0.0
    for layer in layers:
        if layer.floorplan.startswith("logic"):
            face, edge, label = "#4c78a8", "#26384f", "logic"
        elif layer.floorplan.startswith("bond"):
            face, edge, label = "#bab0ac", "#5c5755", "bond"
        else:
            face, edge, label = "#72b7b2", "#2e5c59", layer.floorplan.replace(".flp", "")
        ax.add_patch(
            Rectangle(
                (0.0, z_um),
                width_mm,
                layer.thickness_um,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
                alpha=0.95,
            )
        )
        ax.text(
            width_mm / 2.0,
            z_um + layer.thickness_um / 2.0,
            f"L{layer.index}: {label}, {layer.thickness_um:.0f} um",
            ha="center",
            va="center",
            fontsize=7,
            color="#111111",
        )
        z_um += layer.thickness_um
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, z_um)
    ax.set_title("3D Stack Cross Section")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z thickness (um)")
    ax.grid(True, axis="x", color="#dddddd", linewidth=0.35)
    ax.text(
        0.01,
        0.99,
        f"{len(layers)} HotSpot layers, total active/passive thickness {z_um:.0f} um",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 2},
    )


def _draw_overview(logic_blocks, dram_blocks, layers, metadata, path, _rectangle_cls, patch_cls, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.6))
    _draw_floorplan_axis(axes[0], logic_blocks, "Logic Floorplan", _logic_palette(), False)
    _draw_floorplan_axis(axes[1], dram_blocks, "HBM DRAM Floorplan", _dram_palette(), True)
    _draw_cross_section_axis(axes[2], layers, metadata)
    axes[0].legend(
        handles=[
            patch_cls(facecolor="#4c78a8", edgecolor="#26384f", label="SRAM"),
            patch_cls(facecolor="#f58518", edgecolor="#6f3b05", label="SA"),
            patch_cls(facecolor="#54a24b", edgecolor="#244a20", label="VU"),
            patch_cls(facecolor="#b279a2", edgecolor="#57334b", label="Router"),
            patch_cls(facecolor="#d4b896", edgecolor="#8b6914", label="TSV"),
            patch_cls(facecolor="#e8e8e8", edgecolor="#a0a0a0", label="Pad"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )
    axes[1].legend(
        handles=[
            patch_cls(facecolor="#72b7b2", edgecolor="#2e5c59", label="HBM package"),
            patch_cls(facecolor="#e8e8e8", edgecolor="#a0a0a0", label="Pad"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )
    axes[2].legend(
        handles=[
            patch_cls(facecolor="#4c78a8", edgecolor="#26384f", label="Logic"),
            patch_cls(facecolor="#72b7b2", edgecolor="#2e5c59", label="HBM DRAM"),
            patch_cls(facecolor="#bab0ac", edgecolor="#5c5755", label="Bond"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
