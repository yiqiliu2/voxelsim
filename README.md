# VoxelSim — Artifact Evaluation Guide

Reproduces key experiments in the [paper](https://arxiv.org/abs/2604.26821) (Figures 8, 10–13, 15, 17–20).  Covers environment
setup, simulation data generation, figure plotting, and the Pareto-frontier
design-space exploration (DSE).

> **Interactive UI (experimental):** an experimental web workbench is also
> available in `web_profiler/` for browsing simulation results, reproducing
> paper figures, and launching runs — see
> [web_profiler/README.md](web_profiler/README.md).  It is offered as an
> optional convenience and is still evolving; for artifact evaluation we
> recommend the standard steps below (Section 2) to reproduce the paper's
> results.
> 
> ![VoxelSim Workbench — Pareto explorer](web_profiler/webui.png)
---

## 1. Environment Setup

### 1.1 System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| Python | 3.10+ | 3.10.12 |
| RAM | 128 GB | 256 GB |
| Disk | 30 GB | 50 GB |
| CPU cores | 64 | 128 |

> Prefill (batch-1 full-sequence) simulations are the memory bottleneck.
> The pipeline auto-limits concurrency.  At 96 GB RAM use
> `--decode-parallel-limit 2`.

### 1.2 Python Virtual Environment

```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y python3.10-venv python3-pip

python3 -m venv venv
source venv/bin/activate
```

### 1.3 Python Dependencies

```bash
pip install -r requirements.txt
```

Packages: `numpy==1.26.4`, `ujson`, `scikit-learn`, `scipy`, `linear-tree`, `matplotlib`

### 1.4 External Backends (Optional)

```bash
bash scripts/setup_external_backends.sh
```

Builds HotSpot, 3D-ICE, DSENT, and ORION for thermal/NoC power analysis.
Not mandatory for AE — the simulator can fall back to built-in analytical models and the provided cached data.

---

## 2. Quick Start — Recommended

**This is the recommended way for artifact reviewers.**  One command runs
everything sequentially (no need to juggle parallel jobs or manually chain
invocations):

```bash
bash master_runner.sh
```

Runs everything end-to-end:

| Stage | Contents | ~Time |
|-------|----------|-------|
| 1. Simulation data | Modes 2,3,5,7,8,9,11 + op-breakdown sweep | 3–6 h |
| 2. Drawing scripts | Figures 10–13, 15, 17–20 (9 PDFs) | < 1 min |
| 3. DSE Pareto | Decode + prefill, merged (Fig 8) | 2–14 h |

Partial runs:

```bash
bash master_runner.sh --figures    # draw only (existing data)
bash master_runner.sh --dse        # Pareto DSE only
bash master_runner.sh --dry-run    # check data completeness
```

> **Note:** `master_runner.sh` auto-activates the `venv/` virtual environment
> if it exists — no need to manually `source venv/bin/activate` beforehand.
>
> All commands are sequential — run one command at a time and wait for it to
> finish.  The simulation groups (A1, A2, B, C) run one after another
> internally; do NOT launch them as parallel background processes.

---

## 3. Step-by-Step Guide

> For your convenience, we recommend using `bash master_runner.sh` (Section 2).
> This section explains each step and is useful for partial / incremental runs.
>
> For faster simulation on high-RAM machines, `bash run_figure_data.sh` runs
> 7 parallel groups (forward *and* reverse sweeps) and internally uses the
> same `run_all_modes.py` commands shown below; see its header comment for
> the 1:1 correspondence with the groups listed here.
>
> Usage: `bash run_figure_data.sh` (full) or `bash run_figure_data.sh --dry-run` (check progress only).

### 3.1 Generate Simulation Data

> Ensure your virtual environment is activated (`source venv/bin/activate`)
> before running the commands below.

**Important:** Run each group sequentially — do **not** launch them as parallel
background jobs.  Each invocation reads and modifies `run_all_tests.py` before
executing it; concurrent runs from the same terminal are safe (in-memory
modification via `python3 -c`), but parallel shell background jobs may interfere.

```bash
# Group A1 — Mode 2 prefill only (dit-xl needed for Fig 15 prefill row)
python3 run_all_modes.py --modes 2 --prefill \
    --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo'

# Group A2 — Modes 2,5,7,9 decode + modes 5,7,9 prefill (no dit-xl)
#   Mode 2 → Fig 11,15,18,19 (decode rows)
#   Mode 5 → Fig 10,13
#   Mode 7 → Fig 10
#   Mode 9 → Fig 10,13
python3 run_all_modes.py --run-both --modes 2,5,7,9 \
    --prefill-modes 5,7,9 \
    --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo' \
    --decode-parallel-limit 3 \
    --exclude-models dit-xl

# Group B — Mode 3 decode only (dit-xl needed for Fig 17 smile curve)
python3 run_all_modes.py --modes 3 \
    --cg-list '1,2,4,8' \
    --decode-parallel-limit 3

# Group C — Modes 8,11 decode, llama2-13 only
python3 run_all_modes.py --modes 8,11 --decode-parallel-limit 3 --model llama2-13
```

Flags:

| Flag | What it skips | Why |
|------|---------------|-----|
| `--exclude-models dit-xl` | dit-xl from modes that don't need it | Only mode 2 prefill (Fig 15) and mode 3 decode (Fig 17) use dit-xl |
| `--sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo'` | `core_group` sweep from mode 2 | No figure reads core_group data from mode 2 |
| `--cg-list '1,2,4,8'` | `cg=16` from mode 3 | Fig 17 smile curve uses {1,2,4,8} only |
| `--prefill-modes 5,7,9` | prefill for mode 2 in Group A2 | mode 2 prefill already covered by Group A1 (dit-xl needed) |

Completed work is cached — re-running skips existing outputs.

| Mode | Description | Figures |
|------|------------|---------|
| 2 | Individual parameter sweeps (SA, SRAM, DRAM BW, cores, NoC) | Fig 11, 15, 18, 19 |
| 3 | Paired 3-way sweep: SA × SRAM × cores across CG groups | Fig 17 |
| 5 | Compiler: SPMD, uniform-DRAM mapping | Fig 10, 13 |
| 7 | Dataflow paradigm (per-model pipeline depth) | Fig 10 |
| 8 | DRAM mapping vs DRAM bandwidth (llama2-13 only) | Fig 12 |
| 9 | Default configuration baseline | Fig 10, 13 |
| 11 | DRAM tRP row-conflict overhead sweep (llama2-13 only) | Fig 12 |

> Mode 1 (NoC topology × bandwidth dense sweep) is not required for any figure
> — Fig 11 draws its noc_topo data from Mode 2's individual parameter sweeps
> (which already provide both "best" and "seq_noc" for noc_topo={1,2,3} at the
> default noc_bw).
>
> The `--exclude-models` flag injects a `SKIP_MODELS` filter into the
> underlying script, removing specified models from `all_list` before any
> simulation runs.

### 3.2 Operator Breakdown Sweep (Fig 20)

```bash
python3 run_seq_batch_sweep.py
```

Generates operator-level timing for Llama2-70B across 5 decode (seq,batch) and
5 prefill combos (10 jobs, ~15–25 min).

### 3.3 Check Simulation Progress

```bash
# Count completed output log files (~405 when all modes finish)
find results/logs/ -name "output_*.log" | wc -l

# Dry-run: count expected vs existing outputs
bash run_figure_data.sh --dry-run
```

### 3.4 Generate Figures (Figs 10–20)

Once simulation data is ready (verify with the dry-run in Section 3.3), each
script below produces one PDF in `figures/`.

```bash
# Fig 10 — Compiler strategy comparison
python3 benchmark_scripts/draw_sw_diff.py

# Fig 11 — NoC topology comparison
python3 benchmark_scripts/draw_topo.py

# Fig 12 — DRAM row-conflict overhead
python3 benchmark_scripts/draw_merged_rowchange_overhead.py

# Fig 13 — DRAM mapping: uniform vs software-aware
python3 benchmark_scripts/draw_sw_diff_dram.py

# Fig 15 — Full design-space sweep curves
python3 benchmark_scripts/draw_all_lines_1_col.py

# Fig 17 — Core-group smile curve
python3 benchmark_scripts/draw_curve.py

# Fig 18 — Total energy vs DRAM BW and core count
python3 benchmark_scripts/draw_energy_power.py

# Fig 19 — Component energy breakdown (Llama2-70B)
python3 benchmark_scripts/draw_component_energy_breakdown.py

# Fig 20 — Operator-level time breakdown sweep
python3 benchmark_scripts/draw_op_breakdown_sweep.py
```

Expected output files in `figures/`:

| Script | Output |
|--------|--------|
| `draw_sw_diff.py` | `eval_sw_compiler.pdf` |
| `draw_topo.py` | `eval_noc_topo_combined.pdf` |
| `draw_merged_rowchange_overhead.py` | `eval_merged_rowchange_overhead_llama2-13.pdf` |
| `draw_sw_diff_dram.py` | `eval_sw_dram_map.pdf` |
| `draw_all_lines_1_col.py` | `eval_lines_all_combined.pdf` |
| `draw_curve.py` | `eval_core_group_decode_smile_curve.pdf` |
| `draw_energy_power.py` | `eval_total_energy_mj_total_energy_mj_vs_dram_bw_num_cores.pdf` |
| `draw_component_energy_breakdown.py` | `llama3-70_energy_breakdown_absolute.pdf` |
| `draw_op_breakdown_sweep.py` | `eval_op_breakdown_sweep.pdf` |

### 3.5 Pareto Frontier DSE (Fig 8)

Multi-level area-constrained coordinate descent traces the optimal area-vs-latency
trade-off.  Defaults to 15 area levels with 5 coordinate-descent cycles each in **large→small (reverse)** order,
which provides better Pareto coverage by warm-starting from the globally
unconstrained optimum (figure in paper used higher level and cycle counts, but would take too long).  Each run merges results with prior runs on disk.
The coordinate descent evaluates many configs per level; **decode ~2 h**,
**prefill ~12 h** (batch-1 full-sequence is slower).  Interrupted runs
resume from cached evaluations.

Override with `--num-sweeps N` and `--max-cycles C`.

```bash
# Convenience wrapper (auto-activates venv):
bash run_dse.sh              # decode + prefill + plot
bash run_dse.sh --decode     # decode only
bash run_dse.sh --plot       # plot only (from existing data)

# Direct invocation:
python3 dse_pareto.py --mode decode   # Decode Pareto (4 LLM models, no ViT)
python3 dse_pareto.py --mode prefill  # Prefill Pareto (all 5 models)

# Both auto-merged into figures/pareto_front.png

# Re-plot without re-running simulations
python3 dse_pareto.py --plot-only

# Optional: forward sweep for additional small-area coverage
python3 dse_pareto.py --mode decode --forward
```

Search space (6300 configs):

> SA ∈ {16, 32, 64, 128, 256} ×
> SRAM/KB ∈ {256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576} ×
> Cores ∈ {32, 48, 64, 96, 128, 192, 256, 384, 512} ×
> DRAM BW ∈ {1000, 1500, 2000, 3000, 4000, 6000, 8000, 12000, 16000, 24000}

---

## 4. Troubleshooting

### OOM during prefill
```bash
# On RAM-constrained machines, split into the 4 groups from Section 3.1:
python3 run_all_modes.py --modes 2 --prefill \
    --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo'
python3 run_all_modes.py --run-both --modes 2,5,7,9 --prefill-modes 5,7,9 \
    --sweep-params 'noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo' \
    --decode-parallel-limit 2 --exclude-models dit-xl
python3 run_all_modes.py --modes 3 --cg-list '1,2,4,8' --decode-parallel-limit 2
python3 run_all_modes.py --modes 8,11 --model llama2-13 --decode-parallel-limit 2
```

### DSE killed / orphaned subprocesses
```bash
killall -9 python3       # CAUTION: kills all Python
python3 dse_pareto.py --mode prefill   # re-run; disk-cached results are reused
```

### Verify data completeness
```bash
source venv/bin/activate           # required for the python calls inside
bash run_figure_data.sh --dry-run
```

### Incremental runs
The pipeline skips output files containing `Overall Util:` — simply re-run the
same command.  DSE uses disk-backed cache; forward+reverse runs merge results.

---

## 5. Useful Commands

```bash
# Efficient: modes that need both prefill and decode
python3 run_all_modes.py --run-both --modes 2,5,7,9

# Prefill or decode only
python3 run_all_modes.py --modes 2,5,7,9 --prefill
python3 run_all_modes.py --modes 2,3,5,7,9

# llama2-13 only (modes 8,11 needed only for this model)
python3 run_all_modes.py --modes 8,11 --model llama2-13

# Single model for debugging
python3 run_all_modes.py --modes 9 --model llama2-13

# Force re-run (ignore cache)
python3 run_all_modes.py --modes 2 --run-all

# Limit mode 2 to only sweep params needed by figures
python3 run_all_modes.py --modes 2 --sweep-params 'dram_bw,num_cores'

# Limit mode 3 cg sweep (cg=16 unused by figures)
python3 run_all_modes.py --modes 3 --cg-list '1,2,4,8'

# Completion count
find results/logs/ -name "output_*.log" | wc -l
```

---

## 6. Key Files

```
├── benchmark_scripts/      Drawing scripts + run_all_tests.py
│   ├── draw_sw_diff.py            Fig 10   Compiler comparison
│   ├── draw_topo.py               Fig 11   NoC topology
│   ├── draw_merged_rowchange_overhead.py  Fig 12   DRAM row conflict
│   ├── draw_sw_diff_dram.py       Fig 13   DRAM mapping
│   ├── draw_all_lines_1_col.py    Fig 15   Parameter sweeps
│   ├── draw_curve.py              Fig 17   Smile curve
│   ├── draw_energy_power.py       Fig 18   Energy vs BW
│   ├── draw_component_energy_breakdown.py  Fig 19   Energy breakdown
│   └── draw_op_breakdown_sweep.py Fig 20   Op breakdown
├── figures/                All output PDFs and PNGs
├── hw_config/              Generated hardware configuration JSONs
├── models/                 Model specs (JSON) + parser + TExpr compiler
├── results/                Simulation logs (logs/) and pickles (pickles/)
├── results_pareto_*/       DSE results and checkpoints
├── tsim_simple.py           Core simulation engine (compute, DRAM, NoC pipeline)
├── tsim_components/         Compute, DRAM, NoC, power simulation core
│
├── icbm_launch.py          Single simulation entry point
├── run_all_modes.py        Multi-mode batch scheduler (called by both
│                           master_runner.sh and run_figure_data.sh; internally
│                           drives benchmark_scripts/run_all_tests.py)
├── run_figure_data.sh      Parallel simulation runner (7 groups, faster than
│                           sequential master_runner.sh; alternative entry point)
├── run_seq_batch_sweep.py  Fig 20 input data generator
├── dse_pareto.py           Fig 8 Pareto frontier DSE
├── run_dse.sh              Fig 8 convenience wrapper (auto-activates venv)
├── master_runner.sh        One-command end-to-end (sequential; recommended)
└── requirements.txt        Python dependencies

============================
 Call chain
============================
  master_runner.sh              1-command, sequential (recommended)
   ├─ run_all_modes.py          4 groups: A1, A2, B, C
   │   └─ benchmark_scripts/run_all_tests.py   actual simulation engine
   ├─ run_seq_batch_sweep.py    Fig 20 data
   ├─ draw_*.py                 Figs 10–20
   └─ dse_pareto.py             Fig 8

  run_figure_data.sh            7-group parallel alternative
   └─ run_all_modes.py          splits groups into fwd/rev pairs
```
