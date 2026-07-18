#!/usr/bin/env python3
"""Write per-smoothing component-density tables from aggregate CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COMPONENTS = ["sa", "sram", "tsv", "router", "vu"]
ORDER = [
    ("llama2-13", "prefill"),
    ("llama3-70", "prefill"),
    ("opt-30", "prefill"),
    ("gemma2", "prefill"),
    ("llama2-13", "decode"),
    ("llama3-70", "decode"),
    ("opt-30", "decode"),
    ("gemma2", "decode"),
    ("dit-xl", "decode"),
]
WINDOW_LABELS = {
    0: "Raw",
    500: "500us",
    1000: "1ms",
    2000: "2ms",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def display_model(model: str) -> str:
    return model.replace("llama2-", "llama").replace("opt-30", "opt")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as infile:
        return list(csv.DictReader(infile))


def lookup(rows: list[dict[str, str]], model: str, mode: str, window_us: int, component: str) -> float:
    row = next(
        item
        for item in rows
        if item["model"] == model
        and item["mode"] == mode
        and int(item["window_us"]) == window_us
        and item["component"] == component
    )
    return float(row["stacked_group_w_per_mm2"])


def write_tables(rows: list[dict[str, str]], out_path: Path) -> None:
    chunks = [
        "# Component Power Density Tables",
        "",
        "Values are grouped stacked component density in W/mm2:",
        "",
        "`logic component dynamic + logic component static + overlapping DRAM dynamic + overlapping DRAM static`,",
        "normalized by logic component area.",
        "",
    ]
    for window_us, label in WINDOW_LABELS.items():
        chunks.append(f"## {label}")
        chunks.append("")
        chunks.append("| config | SA | SRAM | TSV | router | VU | max |")
        chunks.append("|---|---:|---:|---:|---:|---:|---|")
        for model, mode in ORDER:
            vals = {component: lookup(rows, model, mode, window_us, component) for component in COMPONENTS}
            max_component = max(COMPONENTS, key=lambda component: vals[component])
            chunks.append(
                f"| {display_model(model)} {mode} | "
                f"{vals['sa']:.3f} | {vals['sram']:.3f} | {vals['tsv']:.3f} | "
                f"{vals['router']:.3f} | {vals['vu']:.3f} | "
                f"{vals[max_component]:.3f} {max_component.upper()} |"
            )
        chunks.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(chunks), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_tables(load_rows(args.csv), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
