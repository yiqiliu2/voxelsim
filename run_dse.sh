#!/usr/bin/env bash
# ============================================================================
# run_dse.sh — run DSE Pareto frontier analysis (Fig 8)
#
# Usage:
#   bash run_dse.sh           # decode + prefill + plot
#   bash run_dse.sh --decode  # decode only
#   bash run_dse.sh --prefill # prefill only
#   bash run_dse.sh --plot    # plot only (from existing data)
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_DECODE=0
RUN_PREFILL=0
RUN_PLOT=0

case "${1:-}" in
    --decode)
        RUN_DECODE=1; RUN_PLOT=1 ;;
    --prefill)
        RUN_PREFILL=1; RUN_PLOT=1 ;;
    --plot)
        RUN_PLOT=1 ;;
    "")
        RUN_DECODE=1; RUN_PREFILL=1; RUN_PLOT=1 ;;
    -h|--help)
        echo "Usage: $0 [--decode|--prefill|--plot]"
        echo "  (default)    Run decode + prefill DSE, then plot"
        echo "  --decode     Decode DSE only + plot"
        echo "  --prefill    Prefill DSE only + plot"
        echo "  --plot       Plot only from existing data"
        exit 0 ;;
    *)
        echo "Unknown option: $1"
        echo "Usage: $0 [--decode|--prefill|--plot]"
        exit 1 ;;
esac

if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    source "$ROOT_DIR/venv/bin/activate"
elif ! python3 -c "import numpy, matplotlib, sklearn, ujson, scipy" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies and no venv found."
    echo "       Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

mkdir -p figures

if [ "$RUN_DECODE" -eq 1 ]; then
    echo "=== Decode DSE ==="
    python3 -u dse_pareto.py --mode decode
fi

if [ "$RUN_PREFILL" -eq 1 ]; then
    echo "=== Prefill DSE ==="
    python3 -u dse_pareto.py --mode prefill
fi

if [ "$RUN_PLOT" -eq 1 ]; then
    echo "=== Plot ==="
    python3 dse_pareto.py --plot-only
    echo "Pareto figure: figures/pareto_front.png"
fi

echo "ALL DONE"
