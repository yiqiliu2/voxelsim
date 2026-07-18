#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source "$ROOT_DIR/thermal/settings.env"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

run_prefill() {
  local model="$1"
  local layers="$2"
  local split_factor="$3"
  echo "[prefill] ${model} layers=${layers} bs=${PREFILL_BATCH} seq=${PREFILL_SEQ}"
  python3 icbm_launch.py "$model" "$CORE_COUNT" \
    --core_mem_kb "$CORE_MEM_KB" \
    --output_dir "$PREFILL_PICKLE_ROOT" \
    --layers "$layers" \
    --batch_size "$PREFILL_BATCH" \
    --sequence_length "$PREFILL_SEQ" \
    --use_pickle \
    --split_factor "$split_factor" \
    --hw_json "$HW_JSON" \
    --core_group "$CORE_GROUP" \
    --dram_name "$DRAM_NAME" \
    --dram_bw "$DRAM_BW" \
    --sim_layers 1 \
    --trace_out_dir_base "$TRACE_ROOT" \
    --prefill
}

run_decode() {
  local model="$1"
  local layers="$2"
  local split_factor="$3"
  echo "[decode] ${model} layers=${layers} bs=${DECODE_BATCH} isl=${DECODE_ISL}"
  python3 icbm_launch.py "$model" "$CORE_COUNT" \
    --core_mem_kb "$CORE_MEM_KB" \
    --output_dir "$DECODE_PICKLE_ROOT" \
    --layers "$layers" \
    --batch_size "$DECODE_BATCH" \
    --sequence_length "$DECODE_ISL" \
    --use_pickle \
    --split_factor "$split_factor" \
    --hw_json "$HW_JSON" \
    --core_group "$CORE_GROUP" \
    --dram_name "$DRAM_NAME" \
    --dram_bw "$DRAM_BW" \
    --sim_layers 0 \
    --trace_out_dir_base "$TRACE_ROOT"
}

PREFILL_RUNS=(
  "llama2-13 40 1.1"
  "llama3-70 80 1.1"
  "opt-30 48 1.1"
  "gemma2 46 1.1"
)

DECODE_RUNS=(
  "llama2-13 40 3"
  "llama3-70 80 3"
  "opt-30 48 4"
  "gemma2 46 3"
  "dit-xl 32 3"
)

for row in "${PREFILL_RUNS[@]}"; do
  # shellcheck disable=SC2086
  run_prefill $row
done

active=0
for row in "${DECODE_RUNS[@]}"; do
  # shellcheck disable=SC2086
  run_decode $row &
  active=$((active + 1))
  if (( active >= TRACE_JOBS )); then
    wait -n
    active=$((active - 1))
  fi
done
wait

echo "TSIM traces written under ${TRACE_ROOT}"
