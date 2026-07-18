# Thermal Reproduction

This folder contains the scripts needed to reproduce the thermal simulations and
component power-density tables used in the current thermal analysis.

The scripts call the simulator at the repository root and write outputs under
`results/`.  They assume the current checked-in simulator code, including:

- Jouppi TPUv4i arithmetic-energy constants in `t10_TensorExpression.py`
  (`ADD_PJ=0.11`, `MUL_PJ=0.21`).
- TSIM-derived absolute logic areas for SRAM, SA, VU, router, and per-core TSV
  blocks in the intra-core floorplan.
- 3D-ICE stack export with 8 HBM DRAM layers, bank-level DRAM floorplans, address
  trace mapping, and DSENT TG11LVT NoC power.

## Workloads

The reproduction matrix is:

- Prefill: `bs=1`, `seq=2048`, full model layer count.
- Decode: `bs=32`, `ISL=2048`, 100 thermal iterations.
- Models: `llama2-13`, `llama3-70`, `opt-30`, `gemma2`.
- Vision model: `dit-xl` decode only.

Layer counts used by these scripts:

| Model | Layers | Prefill | Decode |
|---|---:|---:|---:|
| llama2-13 | 40 | yes | yes |
| llama3-70 | 80 | yes | yes |
| opt-30 | 48 | yes | yes |
| gemma2 | 46 | yes | yes |
| dit-xl | 32 | no | yes |

## Thermal Settings

The thermal run settings are:

- Engine: 3D-ICE emulator via `benchmark_scripts/run_thermal_cooler_matrix.py`.
- Cooling profile: `sxm_air`.
- Ambient: 35 C.
- Bond/underfill thickness: 10 um.
- DRAM layers: 8.
- HotSpot/thermal grid: 128.
- Logic floorplan: intra-core, one SRAM/SA/VU/router/TSV block per core.
- NoC power: DSENT, `TG11LVT`, 256-bit flits.
- Prefill bins: 1024.
- Decode bins: `100 * 64 = 6400`.

## Running

From the repository root:

```bash
./thermal/01_generate_tsim_traces.sh
./thermal/02_run_thermal_matrix.sh
./thermal/03_compute_power_density.sh
```

The first step is the long one. It generates the TSIM logs/pickles consumed by
the thermal runner. The second step runs the 3D-ICE matrix. The third step
computes grouped component power densities and writes Markdown tables split by
smoothing level.

## Outputs

Default output roots:

- TSIM logs: `results/logs_matrix_bs1_seq2048_paper_sa`
- TSIM pickles:
  - Prefill: `results/pickles/outputs_icbm_2048_training_matrix_bs1_paper_sa`
  - Decode: `results/pickles/outputs_icbm_2048_matrix_decode_bs32_paper_sa`
- Thermal packages: `results/thermal_matrix_prefill_bs1_seq2048_decode_bs32_iter100_tsim_area`
- Grouped power density CSV:
  `results/thermal_matrix_prefill_bs1_seq2048_decode_bs32_iter100_tsim_area/aggregate_power_density_smoothed.csv`
- Per-block power density CSV:
  `results/thermal_matrix_prefill_bs1_seq2048_decode_bs32_iter100_tsim_area/aggregate_power_density_blocks.csv`
- Markdown summary:
  `results/thermal_matrix_prefill_bs1_seq2048_decode_bs32_iter100_tsim_area/component_density_tables.md`

The density tables report stacked component density:

```text
logic component static + logic component dynamic
+ spatially overlapping DRAM static + spatially overlapping DRAM dynamic
```

normalized by the logic component area.

## Parallelism Knobs

The scripts default to the settings used for the current run:

- `TRACE_JOBS=5` for decode trace generation.
- `THERMAL_JOBS=4`
- `THERMAL_CORES_PER_JOB=32`

Override them as environment variables if the machine has a different core or
memory budget.
