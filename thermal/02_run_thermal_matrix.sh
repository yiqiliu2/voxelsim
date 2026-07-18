#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/thermal/settings.env"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

COMMON_ARGS=(
  --results-dir "$TRACE_ROOT"
  --cooling-profiles "$COOLING_PROFILE"
  --ambient-c "$AMBIENT_C"
  --bond-thickness-um "$BOND_THICKNESS_UM"
  --hotspot-grid "$HOTSPOT_GRID"
  --dram-layers "$DRAM_LAYERS"
  --jobs "$THERMAL_JOBS"
  --cores-per-job "$THERMAL_CORES_PER_JOB"
  --skip-gif
  --noc-power-backend "$NOC_POWER_BACKEND"
  --noc-power-flit-bits "$NOC_POWER_FLIT_BITS"
  --noc-power-dsent-tech "$NOC_POWER_DSENT_TECH"
  --timeout-s 7200
)

echo "[thermal] prefill bs=${PREFILL_BATCH} seq=${PREFILL_SEQ}"
python3 benchmark_scripts/run_thermal_cooler_matrix.py \
  "${COMMON_ARGS[@]}" \
  --out-dir "$THERMAL_ROOT/prefill" \
  --models llama2-13,llama3-70,opt-30,gemma2 \
  --modes prefill \
  --batch-size "$PREFILL_BATCH" \
  --prefill-bins "$PREFILL_BINS"

echo "[thermal] decode bs=${DECODE_BATCH} isl=${DECODE_ISL} iterations=${DECODE_ITERATIONS}"
python3 benchmark_scripts/run_thermal_cooler_matrix.py \
  "${COMMON_ARGS[@]}" \
  --out-dir "$THERMAL_ROOT/decode" \
  --models llama2-13,llama3-70,opt-30,gemma2,dit-xl \
  --modes decode \
  --batch-size "$DECODE_BATCH" \
  --decode-iterations "$DECODE_ITERATIONS" \
  --decode-samples-per-iteration "$DECODE_SAMPLES_PER_ITERATION"

echo "Thermal packages written under ${THERMAL_ROOT}"
