#!/usr/bin/env bash
# ============================================================================
# run_figure_data.sh — parallel runner for figures 8-20 data
#
# Every generated data point is consumed by at least one figure.
# No unnecessary models, phases, params, or modes are run.
#
# Parallel groups:
#   G1. mode 2 prefill fwd (dit-xl needed for Fig 15 prefill row)
#   G2. mode 2 prefill rev
#   G3. modes 2,5,7,9 decode + 5,7,9 prefill fwd (no dit-xl)
#   G4. modes 2,5,7,9 decode + 5,7,9 prefill rev
#   G5. mode 3 decode (dit-xl needed for Fig 17)
#   G6. modes 8,11 decode (llama2-13 only)
#   G7. op-breakdown sweep (Fig 20)
#
# Figure → Mode mapping:
#   Fig 10  sw_compiler              → 5,7,9 (prefill+decode)
#   Fig 11  eval_noc_topo            → 2     (prefill+decode; noc_topo sweep)
#   Fig 12  eval_rowchange_overhead  → 8,11  (decode only, llama2-13)
#   Fig 13  sw_dram_map              → 5,9   (prefill+decode)
#   Fig 15  eval_all_combined        → 2     (prefill+decode; dit-xl prefill only)
#   Fig 17  eval_smile_curve         → 3     (decode only; dit-xl included)
#   Fig 18  eval_energy_power        → 2     (prefill+decode)
#   Fig 19  energy_breakdown         → 2     (prefill+decode)
#   Fig 20  eval_op_breakdown_sweep  → run_seq_batch_sweep.py
#
# Usage:
#   ./run_figure_data.sh
#   ./run_figure_data.sh --dry-run
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    source "$ROOT_DIR/venv/bin/activate"
elif ! python3 -c "import numpy, matplotlib, sklearn, ujson, scipy" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies and no venv found."
    echo "       Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

DRY_RUN=false
case "${1:-}" in
    --dry-run|--stats)
        DRY_RUN=true
        shift
        ;;
    "")
        ;;
    -h|--help)
        echo "Usage: $0 [--dry-run|--stats]"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--dry-run|--stats]"
        exit 1
        ;;
esac

# Mode 2 prefill: dit-xl needed (Fig 15 prefill row)
PREFILL_MODE2="2"
# Modes with no dit-xl need (both phases)
NO_VIT_MODES="5,7,9"
# Decode (no dit-xl: modes 2,5,7,9)
DECODE_NO_VIT="2,5,7,9"
# Mode 3 decode: dit-xl needed (Fig 17)
DECODE_MODE3="3"
# llama2-13 only
LLAMA13_MODES="8,11"
DECODE_PARALLEL_LIMIT="${DECODE_PARALLEL_LIMIT:-5}"
SWEEP_PARAMS_OVERRIDE="noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo"
CG_LIST_OVERRIDE="1,2,4,8"
LOG_DIR="$ROOT_DIR/test_logs"
mkdir -p "$LOG_DIR"

RUNNER_EXTRA_ARGS=()
if [[ "$DRY_RUN" == true ]]; then
    RUNNER_EXTRA_ARGS+=(--dry-run)
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ "$DRY_RUN" == true ]]; then
    M2_PFWD_LOG=""; M2_PREV_LOG=""
    NOVIT_PFWD_LOG=""; NOVIT_PREV_LOG=""
    MODE3_LOG=""; L13_LOG=""; SWEEP_LOG=""
else
    M2_PFWD_LOG="$LOG_DIR/m2_prefill_fwd_${TIMESTAMP}.log"
    M2_PREV_LOG="$LOG_DIR/m2_prefill_rev_${TIMESTAMP}.log"
    NOVIT_PFWD_LOG="$LOG_DIR/no_vit_fwd_${TIMESTAMP}.log"
    NOVIT_PREV_LOG="$LOG_DIR/no_vit_rev_${TIMESTAMP}.log"
    MODE3_LOG="$LOG_DIR/mode3_${TIMESTAMP}.log"
    L13_LOG="$LOG_DIR/llama13_${TIMESTAMP}.log"
    SWEEP_LOG="$LOG_DIR/sweep_${TIMESTAMP}.log"
fi

echo "============================================"
echo " Figure Data Runner (figs 8-20)"
echo " Started at: $(date)"
if [[ "$DRY_RUN" == true ]]; then
    echo " Mode: statistics only (--dry-run)"
else
    echo " Mode: full execution"
fi
echo " Prefill (dit-xl needed): $PREFILL_MODE2"
echo " Prefill+decode (no dit-xl): $NO_VIT_MODES"
echo " Decode (no dit-xl): $DECODE_NO_VIT"
echo " Decode (dit-xl needed): $DECODE_MODE3"
echo " llama2-13 only: $LLAMA13_MODES"
echo " Decode parallel limit: $DECODE_PARALLEL_LIMIT"
echo "============================================"
echo ""

# ── Dry-run ──
if [[ "$DRY_RUN" == true ]]; then
    _bar() {  # $1=done $2=total
        local w=20 d=${1:-0} t=${2:-0} i=0 s=""
        if [ "$t" -le 0 ]; then printf "%*s" "$w" ""; return; fi
        local f=$(( w * d / t ))
        while [ $i -lt $f ]; do s="${s}#"; i=$((i+1)); done
        while [ $i -lt $w ]; do s="${s} "; i=$((i+1)); done
        printf "[%s]" "$s"
    }
    _run_dry() { python3 run_all_modes.py --dry-run-raw "$@" 2>/dev/null; }
    _run_dry_sweep() {
        local d_count=0 p_count=0
        for seq_batch in "1024 32" "2048 16" "2048 32" "2048 64" "4096 32"; do
            set -- $seq_batch
            local f="results/logs_seq${1}/llama3-70/bs_${2}/core_256/decode/sa_32-vu_32/sram_2048-drambw_12288_PLACEHOLDER/topo_1-nocbw16/best/output_cg_8_row_8192.log"
            [ -f "$f" ] && d_count=$((d_count + 1))
        done
        for seq_batch in "1024 1" "2048 1" "2048 2" "2048 4" "4096 1"; do
            set -- $seq_batch
            local f="results/logs_seq${1}/llama3-70/bs_${2}/core_256/prefill/sa_32-vu_32/sram_2048-drambw_12288_PLACEHOLDER/topo_1-nocbw16/best/output_cg_8_row_8192.log"
            [ -f "$f" ] && p_count=$((p_count + 1))
        done
        echo "${d_count} ${p_count}"
    }

    echo "[dry-run] G1: mode 2 prefill..."
    RAW1=$(_run_dry --modes "$PREFILL_MODE2" --prefill --sweep-params "$SWEEP_PARAMS_OVERRIDE")
    echo "[dry-run] G3: modes 2,5,7,9 decode + 5,7,9 prefill (no dit-xl)..."
    RAW3=$(_run_dry --run-both --modes "2,5,7,9" --prefill-modes "$NO_VIT_MODES" --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" --sweep-params "$SWEEP_PARAMS_OVERRIDE" --exclude-models dit-xl)
    echo "[dry-run] G5: mode 3 decode..."
    RAW5=$(_run_dry --modes "$DECODE_MODE3" --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" --cg-list "$CG_LIST_OVERRIDE")
    echo "[dry-run] G6: modes 8,11 llama2-13..."
    RAW6=$(_run_dry --modes "$LLAMA13_MODES" --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" --model llama2-13)
    echo "[dry-run] G7: op-breakdown sweep..."
    RAW7=$(_run_dry_sweep)

    _extract() { echo "$1" | grep "^ALL P" | awk '{print $3, $4}'; }
    _extract_d() { echo "$1" | grep "^ALL D" | awk '{print $3, $4}'; }
    read p1_d p1_t <<< "$(_extract "$RAW1")"     # G1: prefill
    read p3a_d p3a_t <<< "$(_extract "$RAW3")"   # G3: prefill portion
    read d3_d d3_t <<< "$(_extract_d "$RAW3")"   # G3: decode portion
    read d5_d d5_t <<< "$(_extract_d "$RAW5")"   # G5: mode 3 decode
    read d6_d d6_t <<< "$(_extract_d "$RAW6")"   # G6: modes 8,11 decode
    read sd_done sp_done <<< "$RAW7"

    p_done=$(( ${p1_d:-0} + ${p3a_d:-0} ))
    p_tot=$(( ${p1_t:-0} + ${p3a_t:-0} ))
    d_done=$(( ${d3_d:-0} + ${d5_d:-0} + ${d6_d:-0} ))
    d_tot=$(( ${d3_t:-0} + ${d5_t:-0} + ${d6_t:-0} ))
    # sweep
    sd_done=${sd_done:-0}; sp_done=${sp_done:-0}
    p_done=$(( p_done + sp_done )); p_tot=$(( p_tot + 5 ))
    d_done=$(( d_done + sd_done )); d_tot=$(( d_tot + 5 ))

    echo ""
    echo "============================================================"
    echo " Dry-Run Combined Totals"
    echo "============================================================"
    printf "ALL P %s %4d/%-4d\n" "$(_bar $p_done $p_tot)" "$p_done" "$p_tot"
    printf "    D %s %4d/%-4d\n" "$(_bar $d_done $d_tot)" "$d_done" "$d_tot"
    echo ""
    echo "Dry-run complete."
    exit 0
fi

# ── Full execution ──

echo "[G1 mode 2 prefill fwd] Launching..."
python3 run_all_modes.py --modes "$PREFILL_MODE2" --prefill --sweep-params "$SWEEP_PARAMS_OVERRIDE" "${RUNNER_EXTRA_ARGS[@]}" > "$M2_PFWD_LOG" 2>&1 &
PID_G1=$!

echo "[G2 mode 2 prefill rev] Launching..."
python3 run_all_modes.py --modes "$PREFILL_MODE2" --prefill --reverse --sweep-params "$SWEEP_PARAMS_OVERRIDE" "${RUNNER_EXTRA_ARGS[@]}" > "$M2_PREV_LOG" 2>&1 &
PID_G2=$!

echo "[G3 no-vit fwd] Launching..."
python3 run_all_modes.py --run-both --modes "2,5,7,9" --prefill-modes "$NO_VIT_MODES" \
    --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" --sweep-params "$SWEEP_PARAMS_OVERRIDE" \
    --exclude-models dit-xl "${RUNNER_EXTRA_ARGS[@]}" > "$NOVIT_PFWD_LOG" 2>&1 &
PID_G3=$!

echo "[G4 no-vit rev] Launching..."
python3 run_all_modes.py --run-both --modes "2,5,7,9" --prefill-modes "$NO_VIT_MODES" --reverse \
    --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" --sweep-params "$SWEEP_PARAMS_OVERRIDE" \
    --exclude-models dit-xl "${RUNNER_EXTRA_ARGS[@]}" > "$NOVIT_PREV_LOG" 2>&1 &
PID_G4=$!

echo "[G5 mode 3 decode] Launching..."
python3 run_all_modes.py --modes "$DECODE_MODE3" --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" \
    --cg-list "$CG_LIST_OVERRIDE" "${RUNNER_EXTRA_ARGS[@]}" > "$MODE3_LOG" 2>&1 &
PID_G5=$!

echo "[G6 llama2-13] Launching..."
python3 run_all_modes.py --modes "$LLAMA13_MODES" --decode-parallel-limit "$DECODE_PARALLEL_LIMIT" \
    --model llama2-13 "${RUNNER_EXTRA_ARGS[@]}" > "$L13_LOG" 2>&1 &
PID_G6=$!

echo "[G7 op-breakdown sweep] Launching..."
python3 run_seq_batch_sweep.py > "$SWEEP_LOG" 2>&1 &
PID_G7=$!

echo ""
echo "All 7 groups launched: G1=$PID_G1 G2=$PID_G2 G3=$PID_G3 G4=$PID_G4 G5=$PID_G5 G6=$PID_G6 G7=$PID_G7"
echo "Logs: $LOG_DIR/"
echo "Waiting for all to finish..."
echo ""

FAILED=0
for pid in $PID_G1 $PID_G2 $PID_G3 $PID_G4 $PID_G5 $PID_G6 $PID_G7; do
    wait "$pid" || { echo "ERROR: process $pid failed with exit code $?"; FAILED=1; }
done

echo ""
echo "============================================"
echo " All processes finished at: $(date)"
if [ "$FAILED" -eq 0 ]; then
    echo " Status: ALL SUCCESS"
else
    echo " Status: SOME FAILED (check logs in $LOG_DIR/)"
fi
echo "============================================"
