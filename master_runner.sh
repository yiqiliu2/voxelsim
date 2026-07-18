#!/usr/bin/env bash
# ============================================================================
# master_runner.sh — end-to-end figure generation for 3D-stack AE
#
# Usage:
#   bash master_runner.sh           # Full run (simulation + figures + DSE)
#   bash master_runner.sh --figures # Only generate figures from existing data
#   bash master_runner.sh --dse     # Only run DSE Pareto
#   bash master_runner.sh --dry-run # Check data completeness (no simulation)
#   bash master_runner.sh --help    # Show this message
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_SIM=1
RUN_FIGURES=0
RUN_DSE=0
DRY_RUN=0

print_help() {
    cat <<EOF
Usage: $0 [OPTION]

Options:
  (no args)      Full run: simulation data + figures + DSE Pareto
  --figures      Only generate figures from existing simulation data
  --dse          Only run DSE Pareto frontier analysis
  --dry-run      Check data completeness without running simulations
  --stats        Alias for --dry-run
  -h, --help     Show this help message

Examples:
  bash master_runner.sh                # Everything end-to-end
  bash master_runner.sh --figures      # Draw figures only
  bash master_runner.sh --dse          # DSE only
  bash master_runner.sh --dry-run      # Check what data exists
EOF
}

case "${1:-}" in
    --figures)
        RUN_SIM=0; RUN_FIGURES=1 ;;
    --dse)
        RUN_SIM=0; RUN_DSE=1 ;;
    --dry-run|--stats)
        RUN_SIM=0; RUN_FIGURES=0; RUN_DSE=0; DRY_RUN=1 ;;
    -h|--help)
        print_help; exit 0 ;;
    "")
        RUN_FIGURES=1; RUN_DSE=1 ;;
    -*)
        echo "Unknown option: $1"
        echo "Usage: $0 [--figures|--dse|--dry-run|--stats|--help]"
        exit 1 ;;
esac

if [ "$DRY_RUN" -eq 1 ]; then
    if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
        source "$ROOT_DIR/venv/bin/activate"
    elif ! python3 -c "import numpy, matplotlib, sklearn, ujson, scipy" 2>/dev/null; then
        echo "ERROR: Missing Python dependencies and no venv found."
        echo "       Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        exit 1
    fi
    echo "============================================"
    echo " Dry-Run: Checking data completeness"
    echo "============================================"
    bash "$ROOT_DIR/run_figure_data.sh" --dry-run
    exit 0
fi

# Activate venv if it exists, else fall back to system Python
if [ -f "$ROOT_DIR/venv/bin/activate" ]; then
    source "$ROOT_DIR/venv/bin/activate"
elif ! python3 -c "import numpy, matplotlib, sklearn, ujson, scipy" 2>/dev/null; then
    echo "ERROR: Missing Python dependencies and no venv found."
    echo "       Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

mkdir -p figures results models/TExpr models/parsed hw_config noc_distance_tables

# ── Step 1: Simulation data ──
if [ "$RUN_SIM" -eq 1 ]; then
    echo "============================================"
    echo " Step 1: Generating simulation data"
    echo " Started at: $(date)"
    echo "============================================"

    # 4-group decomposition.

    # Group A1: Mode 2 prefill (dit-xl needed for Fig 15 prefill row).
    #   Mode 2 → Fig 11,15,18,19 (prefill rows).
    python3 -u run_all_modes.py --modes 2 --prefill \
        --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo'
    echo "Group A1 (mode 2 prefill) done at: $(date)"

    # Group A2: Modes 2,5,7,9 decode + modes 5,7,9 prefill (no dit-xl).
    #   Mode 2 → Fig 11,15,18,19 (decode rows; 4 LLM only)
    #   Mode 5 → Fig 10,13
    #   Mode 7 → Fig 10
    #   Mode 9 → Fig 10,13
    python3 -u run_all_modes.py --run-both --modes 2,5,7,9 \
        --prefill-modes 5,7,9 \
        --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo' \
        --decode-parallel-limit 3 \
        --exclude-models dit-xl
    echo "Group A2 (modes 2,5,7,9 decode + 5,7,9 prefill) done at: $(date)"

    # Group B: Mode 3 decode only (dit-xl needed for Fig 17 smile curve).
    #   Mode 3 → Fig 17
    python3 -u run_all_modes.py --modes 3 \
        --cg-list '1,2,4,8' \
        --decode-parallel-limit 3
    echo "Group B (mode 3 decode) done at: $(date)"

    # Group C: Modes 8,11 decode only (llama2-13 only).
    #   Mode 8 → Fig 12 left panel
    #   Mode 11 → Fig 12 right panel
    python3 -u run_all_modes.py --modes 8,11 --decode-parallel-limit 3 --model llama2-13
    echo "Group C (modes 8,11 decode, llama2-13) done at: $(date)"

    # Run op-breakdown sweep (Fig 20)
    echo "Running op-breakdown sweep..."
    python3 -u run_seq_batch_sweep.py
    echo "Sweep completed at: $(date)"
fi

# ── Step 2: Drawing scripts ──
if [ "$RUN_FIGURES" -eq 1 ]; then
    echo "============================================"
    echo " Step 2: Generating figures"
    echo "============================================"

    echo "--- Fig 10: Compiler strategy comparison ---"
    python3 benchmark_scripts/draw_sw_diff.py

    echo "--- Fig 11: NoC topology comparison ---"
    python3 benchmark_scripts/draw_topo.py

    echo "--- Fig 12: DRAM row-conflict overhead ---"
    python3 benchmark_scripts/draw_merged_rowchange_overhead.py

    echo "--- Fig 13: DRAM mapping comparison ---"
    python3 benchmark_scripts/draw_sw_diff_dram.py

    echo "--- Fig 15: Full design-space sweep ---"
    python3 benchmark_scripts/draw_all_lines_1_col.py

    echo "--- Fig 17: Core group smile curve ---"
    python3 benchmark_scripts/draw_curve.py

    echo "--- Fig 18: Energy vs bandwidth ---"
    python3 benchmark_scripts/draw_energy_power.py

    echo "--- Fig 19: Component energy breakdown ---"
    python3 benchmark_scripts/draw_component_energy_breakdown.py

    echo "--- Fig 20: Op breakdown sweep ---"
    python3 benchmark_scripts/draw_op_breakdown_sweep.py

    echo "Figures written to figures/"
fi

# ── Step 3: DSE Pareto ──
if [ "$RUN_DSE" -eq 1 ]; then
    echo "============================================"
    echo " Step 3: DSE Pareto frontier (Fig 8)"
    echo "============================================"

    echo "--- Pareto decode ---"
    python3 dse_pareto.py --mode decode

    echo "--- Pareto prefill ---"
    python3 dse_pareto.py --mode prefill

    echo "Pareto figure: figures/pareto_front.png"
fi

echo "============================================"
echo " ALL DONE at: $(date)"
echo "============================================"
