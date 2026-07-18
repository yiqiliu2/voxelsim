#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/thermal/settings.env"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

python3 benchmark_scripts/compute_thermal_power_density.py \
  --root "$THERMAL_ROOT" \
  --windows-us 0,500,1000,2000

python3 "$ROOT_DIR/thermal/summarize_component_density.py" \
  --csv "$THERMAL_ROOT/aggregate_power_density_smoothed.csv" \
  --out "$THERMAL_ROOT/component_density_tables.md"

echo "Power density CSVs and Markdown table written under ${THERMAL_ROOT}"
