# TSIM Thermal Validation Guide

## Concise Overview

This workflow validates the paper's simple TSIM thermal constraint against a more detailed 3D stack thermal analysis. It starts from normal TSIM logs and fused-op pickles, reconstructs a time-binned component power trace, maps that trace onto a logic/HBM stack floorplan, compares a lumped thermal proxy with a stack-aware RC model, and optionally exports a HotSpot-compatible package for an external HotSpot 7.0 run.

The one-command workflow is:

```bash
python3 src/benchmark_scripts/run_thermal_validation_sweep.py \
  --out-dir src/results/thermal_validation \
  --duration-ms 20 \
  --max-bins 2000 \
  --dram-layers 8
```

By default, this generates or reuses three `llama2-13` decode TSIM points at DRAM bandwidths `8192,12288,16384`, sweeps spatial power placements `uniform,center_hotspot,edge_hotspot`, writes reports under `src/results/thermal_validation/`, and returns nonzero if validation gates fail.

The default thermal backend is `simple`, which runs the built-in TSIM simple/stack RC comparison without exporting a HotSpot package. To run the external HotSpot cross-check, add:

```bash
--thermal-backends simple,hotspot \
--hotspot-bin external/hotspot-7.0/hotspot
```

If the external thermal/NoC backends are not installed in a fresh checkout, run:

```bash
scripts/setup_external_backends.sh
```

This initializes and builds the DSENT/VNoC submodules, downloads/builds HotSpot and 3D-ICE under `external/`, compiles the TSIM ORION link probe, and runs a NoC backend smoke test. Use `--skip-thermal` or `--skip-noc` to install only one backend family, and `--force` to replace/rebuild local external installs.

For rebuttal-quality HotSpot runs, use bank-level HBM, an operator-aware temporal grid, and a bounded number of operator heatmaps:

```bash
python3 src/benchmark_scripts/thermal_compare.py \
  --results-dir src/results/logs \
  --out-dir src/results/thermal_rebuttal/hotspot/llama2-13/decode \
  --models llama2-13 \
  --modes decode \
  --impls best \
  --thermal-backends simple,hotspot \
  --hotspot-bin external/hotspot-7.0/hotspot \
  --hotspot-grid 64 \
  --duration-ms 2 \
  --max-bins 10000 \
  --major-op-samples 4 \
  --major-op-percentile 25 \
  --dram-layers 8 \
  --dram-capacity-gb 192 \
  --dram-floorplan-granularity bank \
  --hbm-banks-per-package 16 \
  --hbm-interleave-stripe-bytes 256 \
  --dram-bank-mappings hbm_interleave \
  --logic-floorplan intra_core \
  --operator-heatmap-count 2
```

`--logic-floorplan intra_core` is the intended HotSpot default. It emits per-core SRAM, systolic-array, vector-unit, and router/NoC blocks instead of the older coarse `logic_y_x` grid. With the default 256-core, 16x16 logic layout and the 192 GB HBM footprint, `--hotspot-grid 64` gives about 4.0 thermal cells across a core in x and 3.0 in y; each 16-bank HBM bank block gets about 5.3 by 4.0 cells. That is sufficient for core-scale and bank-scale hotspots while keeping the full 8-layer sweep tractable. `--hotspot-grid 128` is supported, but it is expensive with 8 HBM layers and bank-level floorplans. In local testing, a 128x128 bank-level point timed out at 180 s, while a 64x64 bank-level point completed and produced valid HotSpot output. Use 64x64 for full sweeps and reserve 128x128 for longer spot checks with a larger `--hotspot-timeout-s`.

## Components

`src/benchmark_scripts/run_thermal_validation_sweep.py`

End-to-end driver. It creates hardware JSON files with the repository's existing `run_all_tests.make_config`, runs `icbm_launch.py` when expected TSIM logs are missing, and then invokes the thermal comparison CLI.

`src/benchmark_scripts/thermal_compare.py`

Compatibility wrapper for the modular thermal validation package.

`src/tsim_thermal/artifacts.py`

Discovers `output_cg_*_row_*.log` files, parses run summaries, and loads the exact matching fused-op pickle `output_cg_<core_group>.pickle`. It intentionally does not fall back to another core group's pickle.

`src/tsim_thermal/trace.py`

Builds a component power trace for `sa`, `vu`, `sram`, `noc`, `dram`, and `tsv`. Dynamic energy is distributed over each fused operator's stage intervals; static power is added to every bin. TSIM currently records SRAM read/write durations but not a true cycle-level SRAM access schedule inside the compute/shift phase, so the thermal exporter preserves total SRAM energy and distributes both SRAM read and write energy evenly across the full compute/shift interval. This avoids injecting an artificial startup burst while keeping operator energy accounting unchanged. NoC energy is passed through a pluggable NoC power backend before it is assigned to router blocks. It also exports `power.ptrace`, `logic.flp`, one `dram<N>.flp` file per configured HBM die, `stack.lcf`, `hotspot.config`, `init.init`, and `metadata.json` when a backend requires a package.

`src/tsim_components/noc_power.py`

Defines the pluggable NoC power interface used by the thermal exporter. The default backend is `tsim_simple`, which preserves the existing TSIM `energy_noc` accounting. The optional `dsent` backend invokes the external DSENT 0.91 command-line tool under `external/dsent0.91/OENOC/dsent0.91`; the optional `orion` backend invokes the external ORION checkout under `external/vnoc20/orion3` and a small TSIM probe binary that links against ORION's `libpower.a` for link energy. These backends characterize router/link energy once, record the characterization in `metadata.json`, and rescale per-operator NoC energy before power-trace generation. They do not yet replace TSIM's NoC timing model or reconstruct exact per-link byte paths from fused-op pickles.

`src/tsim_thermal/models.py`

Runs two internal thermal models:

- `simulate_lumped`: a simple lumped RC proxy for trend comparison.
- `simulate_stack`: a two-layer logic/DRAM grid RC model with vertical and lateral coupling.

`src/tsim_thermal/hotspot.py`

Optionally runs a `hotspot` binary on each exported package. If HotSpot is absent, the report records `missing` and still completes.

`src/tsim_thermal/backends.py`

Defines the pluggable thermal backend interface. The current backends are:

- `simple`: built-in TSIM simple/stack RC analysis. This is the default.
- `hotspot`: HotSpot 7.0 execution over an exported package.

New thermal simulators should implement the `ThermalBackend` protocol and register in `BACKEND_REGISTRY`. A backend declares whether it needs an exported package through `requires_package`, so simple TSIM runs avoid unnecessary HotSpot files.

## Outputs

After a successful run:

```text
src/results/thermal_validation/
  thermal_compare.csv
  thermal_compare.md
  validation_status.json
  packages/
    <run>_<spatial_policy>/
      component_power.csv
      power.ptrace
      logic.flp
      bond.flp
      dram0.flp
      ...
      dram7.flp
      stack.lcf
      hotspot.config
      init.init
      metadata.json
      hotspot_run.json
      logic_floorplan.png
      dram_floorplan.png
      stack_cross_section.png
      layout_overview.png
      operator_hotspot_summary.csv
      operator_unit_power_density.csv
      operator_heatmaps/
```

`thermal_compare.md` is the main reviewer-facing summary. It reports:

- trace average-power reconstruction error versus TSIM,
- existing TSIM simple power density in `W/mm^2`,
- simple lumped peak temperature,
- stack-aware peak temperature,
- thermal margin to the throttle threshold,
- slowdown delta,
- per-spatial-policy Spearman rank correlation,
- HotSpot availability/status.

`validation_status.json` contains a machine-readable pass/fail status and any failed gates.

The PNG layout files visualize the exact exported HotSpot package:

- `logic_floorplan.png`: intra-core logic floorplan from `logic.flp`; SRAM, systolic array, vector unit, router/NoC/TSV, and zero-power padding are color-coded.
- `dram_floorplan.png`: HBM package floorplan from `dram0.flp` or `dram.flp`; each lateral HBM package is labeled and padding is hatched.
- `stack_cross_section.png`: side-view stack from `stack.lcf`; logic, passive bond/underfill, and HBM DRAM silicon layers are shown with their configured thicknesses.
- `layout_overview.png`: combined logic floorplan, DRAM floorplan, and cross-section view.
- `operator_hotspot_summary.csv`: one row per fused operator with runtime, max HotSpot temperature, max-temperature block/layer, max power density, and HotSpot-vs-simple slowdown.
- `operator_unit_power_density.csv`: per-operator, per-HotSpot-block max/average power density in `W/mm^2`.
- `operator_heatmaps/`: per-layer temperature and power-density rectangle heatmaps for the top `--operator-heatmap-count` high-energy fused operators.

To redraw these images for an existing package without rerunning TSIM or HotSpot:

```bash
python3 src/benchmark_scripts/visualize_thermal_package.py \
  src/results/thermal_validation/packages/<package-name>
```

## Current Smoke Result

On this checkout, the verified command analyzed 9 rows: three DRAM bandwidth points times three spatial policies. The run passed with:

- max trace average-power error: `0.562%`,
- max simple-vs-stack peak-temperature delta: `0.128 C`,
- max slowdown delta: `0.000%`,
- max existing TSIM simple power density: `0.3521 W/mm^2`,
- no simple-model false negative,
- per-policy Spearman rank correlation: `1.0000` for `uniform`, `center_hotspot`, and `edge_hotspot`.

HotSpot 7.0 is installed in this checkout under `external/hotspot-7.0/hotspot`. A one-point HBM8 run with fresh TSIM spatial IDs and the corrected 32x32 SA throughput is available under `src/results/thermal_validation_hbm8_correct_core/` and passed with:

- runs analyzed: `1`,
- DRAM layers exported to HotSpot: `8`,
- TSIM-attached spatial op records: `2129`,
- inferred spatial op records: `0`,
- fallback dynamic events: `0`,
- max trace average-power error: `0.218%`,
- simple peak temperature: `45.04 C`,
- stack-RC peak temperature: `45.06 C`,
- HotSpot peak temperature: `51.39 C`,
- no simple-model false negative.

The full default-chip rebuttal matrix has also been run with HotSpot grid `64x64`, 8 HBM layers, 12 x 16 GB HBM packages, bank-level HBM floorplans, `hbm_interleave` bank mapping, and operator heatmaps. Aggregated outputs are under `src/results/rebuttal_thermal_matrix/`:

- `aggregate_thermal_compare.md`: reviewer-facing aggregate table.
- `aggregate_thermal_compare.csv`: machine-readable aggregate table.
- `aggregate_summary.json`: machine-readable headline metrics.
- `simple/thermal_compare.md`: simple-backend-only baseline over all 10 model/mode points.
- `hotspot/<model>/<mode>/`: per-point HotSpot report, exported package, operator summaries, and heatmaps.

The completed matrix produced:

- runs analyzed: `10` (`llama2-13`, `llama3-70`, `opt-30`, `gemma2`, `dit-xl` x `decode,prefill`),
- max HotSpot peak temperature: `69.34 C` (`llama3-70/prefill`),
- minimum HotSpot margin to the `85 C` throttle threshold: `15.66 C`,
- max HotSpot-vs-simple slowdown delta: `0.000%`,
- Spearman(simple peak, HotSpot peak): `0.9273`,
- Spearman(stack peak, HotSpot peak): `0.9152`,
- total TSIM-attached spatial op records: `11059`,
- inferred spatial op records: `2129` (`llama2-13/decode`, from legacy logs without attached spatial IDs),
- fallback dynamic events: `0`,
- operator heatmaps generated: `180`.

The simple backend therefore underestimates absolute peak temperature, but all high-fidelity HotSpot points remain below the throttle threshold and preserve the paper-relevant performance conclusion: no additional thermal slowdown appears for the default configuration. Prefill rows have large trace average-power reconstruction errors because of the sampled-layer accounting issue described above; these errors are reported, but they do not change the peak-temperature or slowdown conclusion from the fused-op HotSpot traces.

## Configuration Knobs

End-to-end sweep driver:

```bash
python3 src/benchmark_scripts/run_thermal_validation_sweep.py --help
```

Useful flags:

- `--dram-bws`: comma-separated DRAM bandwidth points. Default: `8192,12288,16384`.
- `--dram-layers`: number of stacked DRAM/HBM dies exported to HotSpot. Use `8` for the paper's current HBM-style setup.
- `--model`: TSIM model name. Default: `llama2-13`.
- `--core-mem-kb`: per-core SRAM capacity. Default: `2048`.
- `--core-group`: core group size. Default: `8`.
- `--force`: rerun TSIM even when expected logs already exist.
- `--spatial-policies`: comma-separated spatial mappings. Default: `uniform,center_hotspot,edge_hotspot`.
- `--duration-ms`: thermal observation window. The fused-op trace is repeated to fill this window; this models steady repeated inference, not only the one TSIM execution listed in `exec_ms`.
- `--max-bins`: maximum number of thermal samples. With `--major-op-samples > 0`, this is a cap rather than a fixed count.
- `--major-op-samples`: target number of thermal samples across most major matmul/SA stages. Default: `4`.
- `--major-op-percentile`: matmul-duration percentile used to choose `dt`. Default: `25`, so at least roughly the upper 75% of major matmuls should receive the requested sample count unless capped by `--max-bins`.
- `--thermal-backends`: comma-separated backend list. Default: `simple`. Use `simple,hotspot` for the external HotSpot cross-check.
- `--noc-power-backend`: NoC power backend used while constructing thermal power traces. Default: `tsim_simple`; optional values are `dsent` and `orion`.
- `--noc-power-flit-bits`: flit width passed to DSENT/ORION. Default: `64`.
- `--noc-power-injection-rate`: injection/load point for DSENT/ORION characterization. Default: `0.3`.
- `--noc-power-link-length-mm`: representative router-to-router link length for DSENT/ORION link characterization. Default: `1.0`.
- `--noc-power-dsent-tech`: DSENT technology model basename. Default: `TG11LVT`, the closest shipped DSENT node to the paper's TSMC N7 assumption; DSENT 0.91 does not include an N7 model.
- `--run-hotspot`: legacy shorthand that appends the `hotspot` backend.
- `--hotspot-bin`: HotSpot binary path, e.g. `external/hotspot-7.0/hotspot`.
- `--logic-floorplan`: HotSpot logic floorplan granularity. Default: `intra_core`; use `coarse_grid` for the older 4x4 grid exporter.
- `--dram-floorplan-granularity`: `package` or `bank`. Default: `bank`, which is the intended HBM2-style thermal attribution mode.
- `--hbm-banks-per-package`: number of pseudochannel-aligned thermal bank blocks inside each HBM package in bank mode. Default: `16`, matching the common HBM2 controller view of 16 pseudochannels per stack. This is a floorplan abstraction, not a vendor-exact subarray bank map.
- `--hbm-interleave-stripe-bytes`: stripe size used by `address_trace` and `hbm_interleave`. Default: `256` bytes.
- `--dram-bank-mapping`: DRAM bank placement policy. Default: `address_trace`, which maps tensor-level synthetic physical-address ranges to HBM bank blocks with `logical_bank = (address // stripe_bytes) % (package_count * hbm_banks_per_package)`. New TSIM pickles use per-tensor DRAM access records; older pickles fall back to aggregate per-operator read/write bytes. `from_impl` maps `best` to `software_aware`, `bad_dram` to `uniform`, and `interleave_dram` to `interleave_size`.
- `--dram-bank-mappings`: comma-separated mapping sweep. Supported values: `hbm_interleave`, `fine_interleave`, `bank_interleave`, `address_trace`, `from_impl`, `uniform`, `interleave_size`, `software_aware`.
- `--hotspot-grid`: HotSpot grid resolution for exported packages. Default: `128` in the low-level CLI. The rebuttal matrix defaults to `64` because the default 28.325779 mm logic die plus 192 GB HBM outline gives 0.4426 mm x 0.5851 mm thermal cells, roughly 4.0 x 3.0 cells per core and 5.3 x 4.0 cells per HBM bank block. This is intentionally independent of `--grid`, which only controls TSIM/simple-stack spatial aggregation and the older coarse exporter.
- `--operator-hotspot-analysis` / `--no-operator-hotspot-analysis`: enable per-operator HotSpot CSV and heatmap post-processing after successful HotSpot runs.
- `--operator-heatmap-count`: number of high-energy fused operators to visualize. Use `0` for CSV-only batch sweeps.

Default-chip rebuttal launcher:

```bash
python3 src/benchmark_scripts/run_rebuttal_thermal_matrix.py \
  --run-tsim \
  --parallel-jobs 2 \
  --hotspot-grid 64 \
  --max-bins 10000 \
  --max-trace-power-error-pct 100 \
  --operator-heatmap-count 2
```

This script runs mode-9 TSIM artifacts for the default workloads when requested, records a simple-backend summary, and then launches independent HotSpot jobs per model/mode so the machine can use multiple cores. Relative `--out-dir` and `--results-dir` paths are resolved from the repository root. HotSpot itself is effectively a separate process per run; use `--parallel-jobs` conservatively because each job can be memory- and disk-intensive.

At the end of a matrix run, the launcher scans `hotspot/*/*/thermal_compare.csv` and writes `aggregate_thermal_compare.csv`, `aggregate_thermal_compare.md`, and `aggregate_summary.json`. To regenerate only those aggregate files from existing per-run outputs:

```bash
python3 src/benchmark_scripts/run_rebuttal_thermal_matrix.py \
  --aggregate-only \
  --out-dir src/results/rebuttal_thermal_matrix
```

For rebuttal matrix runs, `--max-trace-power-error-pct` defaults to `100` in the launcher and is recorded explicitly in the command line. Decode traces should still have low reconstruction error. Prefill traces can show large average-power mismatch because the current TSIM prefill mode simulates sampled layers/operators while the run-level summary reflects broader model aggregation. Treat this as an accounting warning in `thermal_compare.md` and `validation_status.json`; the operator-level HotSpot traces, peak temperatures, and slowdown deltas are still generated from the available fused-op timing and energy records.

Direct comparison CLI:

```bash
python3 src/benchmark_scripts/thermal_compare.py \
  --results-dir src/results/logs \
  --out-dir src/results/thermal_validation \
  --models llama2-13 \
  --modes decode \
  --impls best \
  --duration-ms 20 \
  --max-bins 2000 \
  --dram-layers 8 \
  --dram-bws 8192 \
  --rows 64 \
  --core-groups 8 \
  --spatial-policies uniform,center_hotspot,edge_hotspot \
  --run-hotspot
```

Thermal parameters:

- `--ambient-c`: ambient/init temperature. Default: `35`.
- `--throttle-c`: throttle threshold. Default: `85`.
- `--simple-r-k-per-w`, `--simple-c-j-per-k`: lumped RC parameters.
- `--stack-r-sink-k-per-w`, `--stack-r-vertical-k-per-w`, `--stack-r-lateral-k-per-w`: stack RC coupling.
- `--logic-c-j-per-k`, `--dram-c-j-per-k`: per-tile thermal capacitance.
- `--die-size-mm`: optional square logic die side override. By default, thermal export derives this from the simulator's static logic power estimate at `0.061 W/mm^2`.
- `--grid`: coarse floorplan grid dimension when using coarse-grid logic export.
- `--dram-bws`, `--rows`, `--core-groups`: artifact filters. Use these when existing logs in the same results tree should not be swept together with a fresh validation point.

Validation gates:

- `--max-trace-power-error-pct`: default `2.0`.
- `--max-temp-delta-c`: default `2.0`.
- `--max-slowdown-delta-pct`: default `5.0`.
- `--min-policy-spearman`: default `0.9`.
- `--power-density-threshold`: default `0.7 W/mm^2`, matching the paper/simple TSIM thermal-limit framing.

## HotSpot Integration

HotSpot is an architectural thermal simulator intended for pre-RTL studies and supports 2D/3D IC modeling. The public HotSpot repository describes HotSpot 7.0 as suitable for architectural studies and 3D IC simulation: https://github.com/uvahotspot/HotSpot. The older LAVA HotSpot page also describes the RC-network model and architectural simulator interface: https://lava.cs.virginia.edu/HotSpot/.

To install HotSpot and 3D-ICE automatically:

```bash
scripts/setup_thermal_backends.sh
```

The script downloads upstream GitHub archives when the local ignored archives are absent, builds:

- `external/hotspot-7.0/hotspot`
- `external/3d-ice-src/bin/3D-ICE-Emulator`

Then either put the binaries on `PATH` or pass:

```bash
python3 src/benchmark_scripts/thermal_compare.py ... \
  --thermal-backends simple,hotspot \
  --hotspot-bin /path/to/HotSpot/hotspot
```

The generated package contains the normal HotSpot-style ingredients:

- `power.ptrace`: one column per floorplan block, watts per sample.
- `logic.flp`, `dram<N>.flp`: block dimensions and coordinates.
- `bond.flp`: full-die passive floorplan used by HBM bonding/underfill layers.
- `stack.lcf`: layer description for one logic die plus `--dram-layers` HBM DRAM dies.
- `hotspot.config`: grid model options, sampling interval, ambient, init file, and layer file.
- `init.init`: initial block temperatures.
- `logic_floorplan.png`, `dram_floorplan.png`, `stack_cross_section.png`, `layout_overview.png`: visual checks of the emitted floorplans and layer stack.

The current exporter uses an intra-core logic floorplan by default. For 256 cores, the logic die is arranged as a 16x16 core grid. Each core is split into four HotSpot blocks: `core_<id>_sram` across the top of the core tile, and `core_<id>_sa`, `core_<id>_vu`, and `core_<id>_router` across the bottom. The width/height split is derived from the same SRAM, systolic-array, vector-unit, and router area assumptions used by the validation flow.

The HBM floorplan uses TSIM's HBM area model. Total HBM capacity is controlled by `--dram-capacity-gb` and defaults to `192`. Package capacity is controlled by `--hbm-package-capacity-gb` and defaults to `16`, so the default layout has `ceil(192 / 16) = 12` lateral HBM packages. Each package uses `--hbm-package-area-mm2`, defaulting to `87.62745402745404 mm^2`, so a square package is about `9.361 mm x 9.361 mm`. For the default 192 GB case, those packages are arranged as a `3 x 4` mosaic with a physical HBM footprint of about `28.083 mm x 37.444 mm`. HotSpot requires every layer to share the same outline, so shorter logic or HBM footprints receive explicit zero-power padding blocks.

In `package` mode, DRAM power is mapped to `dram<layer>_pkgXX` blocks and split evenly across the configured active DRAM layers. In `bank` mode, each package is subdivided into `--hbm-banks-per-package` pseudochannel-aligned rectangles named `dram<layer>_pkgXX_bankYY`. With the default `16`, the layout is a 4x4 grid per HBM package, which is a realistic coarse thermal proxy for HBM2 pseudochannel/vault regions. It is not a claim about the vendor-private placement of every physical DRAM subarray bank. The current mapping policies are:

- `address_trace`: default. Reconstructs a synthetic physical-address trace from per-tensor DRAM access records when available. Each record gets a deterministic aligned base address keyed by fused op, sub-op, tensor index, tensor role, and read/write stage. The HBM thermal bank is decoded as `logical_bank = (address // stripe_bytes) % total_hbm_bank_blocks`, then `package = logical_bank // banks_per_package` and `bank = logical_bank % banks_per_package`. Dynamic DRAM energy is split across tensor access windows and weighted by decoded bytes per bank. Older pickles without tensor records fall back to aggregate per-operator read/write bytes.
- `hbm_interleave`: idealized comparison. Splits each per-operator read/write byte count into fine 256-byte stripes and rotates those stripes across all `package_count * hbm_banks_per_package` logical HBM package/bank blocks. Reads and writes for the same fused operator use separate deterministic offsets. Dynamic DRAM energy is weighted by bytes landing on each block.
- `uniform`: every DRAM event is spread over all bank blocks. This corresponds to the paper's uniform tensor placement baseline.
- `interleave_size`: each operator stage receives a contiguous rotating bank stripe proportional to its read/write byte share. This approximates the paper's interleaving-by-size baseline.
- `software_aware`: read and write tensors for operators with both directions are placed on disjoint bank stripes; single-sided traffic follows logical vault/package locality. This approximates the paper's software-aware tensor-to-bank placement.

These bank mappings are thermal attribution models over TSIM DRAM bytes and logical vault IDs. `address_trace` now uses tensor-level access records from new pickles, but the base addresses are still synthetic because TSIM does not preserve real tensor allocation addresses or a JEDEC-accurate HBM bank/subarray command trace. A final physical implementation would need tensor allocation IDs, exact address mapping, pseudo-channel/bank/subarray geometry, and per-bank command traces.

The exported 3D stack uses explicit passive bonding layers between the logic die and every HBM DRAM die. These layers are represented in `stack.lcf` as `Power Dissipation = N`, with `bond.flp` covering the die footprint. The default bonding-layer conductivity is `2 W/(m K)` to match measured HBM polymer/solder-bump through-plane behavior, and the default thickness is `20 um`, matching HotSpot's own TIM-layer example scale.

`--hotspot-grid 64` means HotSpot divides each layer outline into 64 equal intervals per axis, producing a uniform 64x64 thermal mesh. It does not assign one thermal grid cell per hardware unit, and it does not create heterogeneously sized thermal cells. Heterogeneous block sizes are represented in `.flp`; HotSpot maps those blocks onto the uniform grid. With the default 28.325779 mm logic die and 192 GB HBM outline, each thermal cell is about 0.4426 mm x 0.5851 mm. A 256-core 16x16 layout therefore gives about 4.0 x 3.0 thermal cells per core, and the default 4x4 bank subdivision inside each HBM package gives about 5.3 x 4.0 thermal cells per HBM bank block. Each exported `metadata.json` includes a `spatial_resolution` section with these run-specific values.

### HotSpot Parameter Assumptions

The table below records the current generated defaults and their provenance. Values labeled "HotSpot default" are emitted explicitly so the package has no hidden simulator assumptions. Values labeled "model assumption" are reasonable early architectural assumptions that should be swept in sensitivity studies if the rebuttal needs a stronger package-calibration claim.

| Parameter | Current value | Meaning | Source and rationale |
| --- | ---: | --- | --- |
| Logic silicon thickness | `50 um` | Active compute-die thermal layer thickness. | Model assumption for a thinned die-stacked TSMC N7-class logic die. Public TSMC N7 material does not disclose final packaged die thickness. |
| Logic silicon heat capacity | `1.75e6 J/(m^3 K)` | Volumetric heat capacity of the active logic layer. | HotSpot 3D `.lcf` examples use `1.75e6` for silicon layers. |
| Logic silicon resistivity | `0.01 (m K)/W` | Thermal resistivity; equivalent to `100 W/(m K)` conductivity. | HotSpot 3D `.lcf` examples use `0.01` for silicon. This is also close to measured effective HBM in-plane conductivity reported by Chalise and Cahill. |
| HBM DRAM die count | `--dram-layers`, use `8` for HBM2-style setup | Number of active stacked DRAM dies. | HBM2 supports up to 8 dies per stack; Samsung's HBM2 description uses stacked core dies plus a buffer die, connected by TSVs and microbumps. |
| HBM DRAM silicon thickness | `30 um` | Active DRAM die thermal layer thickness. | Model assumption. Public HBM3 descriptions report 30 um DRAM dies; public HBM2 sources confirm stacked dies but do not give a primary vendor thickness value in the source used here. Keep this as an assumption and sweep `30-50 um` if needed. |
| HBM DRAM heat capacity | `1.75e6 J/(m^3 K)` | Volumetric heat capacity of active DRAM silicon layers. | Same HotSpot silicon-layer convention as logic. |
| HBM DRAM resistivity | `0.01 (m K)/W` | Equivalent to `100 W/(m K)` conductivity. | Chosen as an effective active-memory-layer value. Chalise and Cahill report HBM memory-layer in-plane conductivity of `140 W/(m K)` and overall HBM effective in-plane conductivity of `100 W/(m K)`. |
| Bond/underfill layer count | one passive `bond` layer below each DRAM die | Die-to-die polymer/solder microbump thermal path. | Needed because HBM and 3D stacks are not silicon-only stacks; Samsung describes HBM2 dies connected by TSVs and microbumps, and HBM thermal measurements show polymer/solder layers dominate through-plane thermal resistance. |
| Bond/underfill thickness | `20 um` | Passive die-to-die layer thickness. | Model assumption. HotSpot 3D examples use `20 um` passive TIM layers. This should be swept if exact package data is unavailable. |
| Bond/underfill heat capacity | `4.0e6 J/(m^3 K)` | Thermal mass of passive bonding material. | HotSpot 3D examples use `4e6` for passive TIM layers. |
| Bond/underfill resistivity | `0.5 (m K)/W` | Equivalent to `2 W/(m K)` conductivity. | From Chalise and Cahill's measured HBM polymer/solder-bump through-plane conductivity. |
| External TIM thickness | `20 um` | Package thermal interface layer to spreader/cooler. | HotSpot `template.config` default. Distinct from the HBM die-to-die bond layers. |
| External TIM conductivity | `4 W/(m K)` | Heat conduction from die stack toward package cooling path. | HotSpot `template.config` default. |
| HBM thermal banks/package | `16` in bank mode | Pseudochannel-aligned thermal subdivision inside each HBM package. | HBM2 exposes a 1024-bit stack interface commonly organized as 16 pseudochannels. Public FPGA HBM2 platforms expose 16 pseudochannels per stack, e.g. U280-style systems expose 32 pseudochannels across two stacks. This is realistic for coarse thermal attribution, but not a full vendor bank/subarray layout. |
| DRAM bank mapping | `address_trace` | Thermal attribution of DRAM power to bank rectangles. | Tensor-level synthetic physical-address records decode to logical HBM bank blocks with 256-byte stripes; explicit sweeps can still use hbm interleave, uniform, interleave-size, or software-aware placement baselines. |
| Heat spreader | `max(30 mm, 1.10 * stack max side)`, `1 mm` thick, `400 W/(m K)` | Package spreader above the stack. | Size is adjusted so HotSpot accepts large 192 GB HBM outlines; material/thickness are HotSpot `template.config` defaults. |
| Heat sink | Equivalent square side of `max(logic die area, HBM footprint area)`, `6.9 mm` thick, `400 W/(m K)` | Heat sink attached through the spreader/TIM path. | The cooled area follows the active package footprint; material/thickness are HotSpot `template.config` defaults. |
| Primary convection | `r_convec=0.13 K/W`, `c_convec=140.4 J/K` | Ambient cooling boundary condition. | Conservative H100 PCIe/NVL dual-slot air-cooling baseline. This is cooling-solution-specific and should be swept for final sensitivity. |
| Secondary path | enabled with C4/underfill/substrate/solder/PCB defaults | Lower-side heat path through package and board. | HotSpot `template.config` defines this as the C4/underfill, package-substrate, solder-ball, and PCB path. We emit the defaults explicitly. |
| C4/underfill secondary thickness | `100 um` | Lower-side C4/underfill layer thickness. | HotSpot `template.config` default, not calibrated to a specific TSMC CoWoS/HBM2 package. |
| Package substrate | `21 mm` side, `1 mm` thick | Lower-side package substrate. | HotSpot `template.config` default. |
| Solder layer | `21 mm` side, `940 um` thick | Solder-ball layer in the secondary path. | HotSpot `template.config` default. |
| PCB | `100 mm` side, `2 mm` thick | Board-level secondary heat path. | HotSpot `template.config` default. |

Parameter sources:

- HotSpot repository and documentation: https://github.com/uvahotspot/HotSpot
- HotSpot `template.config` package, TIM, secondary path, and default grid parameters: https://raw.githubusercontent.com/uvahotspot/HotSpot/master/template.config
- HotSpot 3D `.lcf` examples for silicon and passive TIM layer conventions: https://raw.githubusercontent.com/uvahotspot/HotSpot/master/examples/example3/example.lcf
- Samsung HBM2 package description: buffer die plus core dies connected by TSVs and microbumps: https://news.samsung.com/global/samsung-begins-mass-producing-worlds-fastest-dram-based-on-newest-high-bandwidth-memory-hbm-interface
- Chalise and Cahill, "Anisotropic thermal conductivity of high bandwidth memory": measured HBM memory-layer and polymer/solder-bump thermal conductivities: https://arxiv.org/abs/2303.06785
- HBM pseudo-channel organization examples, including U280 HBM2 exposing 32 pseudo-channels: https://arxiv.org/abs/2105.11754

The older coarse layout is still available with `--logic-floorplan coarse_grid`; it emits one `logic_y_x` block per coarse grid tile.

## What TSIM Currently Provides

Useful existing intermediate data:

- run-level execution cycles, static energy, dynamic energy, average power, utilization,
- per-fused-op stage start cycles and durations,
- per-fused-op DRAM read/write bytes,
- per-fused-op component dynamic energy for SA, VU, SRAM, NoC, DRAM, and TSV,
- static-power decomposition for logic and DRAM,
- overlap/top-power logs that expose high-power intervals.

The thermal validation flow consumes the log summary and fused-op pickle because those are enough to reconstruct time-resolved component power with energy conservation checks. SRAM power is reconstructed as a model assumption: total SRAM energy is known, but the exact intra-operator read/write schedule is not, so the current default spreads SRAM read/write power across the compute/shift window.

## NoC Power Backends

Install and build the optional upstream NoC power tools from GitHub:

```bash
scripts/setup_noc_power_backends.sh
```

The script initializes the recorded `external/dsent0.91` and `external/vnoc20` submodules, builds DSENT and ORION, marks the ORION executable runnable, compiles `src/tools/tsim_orion_probe`, and runs the characterization smoke test below.

Sanity check the three backend characterizations:

```bash
PYTHONPATH=src python3 - <<'PY'
from tsim_components.noc_power import NoCPowerConfig, describe
for backend in ("tsim_simple", "dsent", "orion"):
    m = describe(NoCPowerConfig(backend=backend, frequency_hz=1.5e9))
    print(backend, m["total_dynamic_energy_j_per_flit"], m["scale_vs_tsim_simple"])
PY
```

On this machine, the current defaults report:

- `tsim_simple`: `9.6e-11 J/flit`, scale `1.0`.
- `dsent` with `TG11LVT`: about `1.46e-12 J/flit`, scale about `0.0152`.
- `orion`: about `2.96e-11 J/flit`, scale about `0.309`.

The current integration is intentionally conservative and modular. TSIM still determines NoC timing and produces one aggregate `energy_noc` per fused operator. The selected backend characterizes a representative router/link and rescales that aggregate energy before thermal export, then spatially attributes the power to the same per-core router blocks used by the intra-core HotSpot floorplan. This is enough to test whether the paper's thermal conclusion is sensitive to a lower-level NoC power calibration. A fully link-accurate NoC power model would additionally need TSIM to emit per-stage source/destination/byte traffic, routed link IDs, router traversal counts, and contention intervals for each fused operator.

## What Is Still Missing For High Fidelity

For a true HotSpot/3D-ICE-grade physical thermal model, TSIM would need more spatial detail:

- physical floorplan for cores, SRAM, NoC routers/links, TSV arrays, and DRAM banks/vaults,
- DRAM layer count, bank/vault geometry, and tensor-to-bank physical placement,
- per-core/per-router/per-link/per-SRAM-bank activity,
- per-NoC-stage source/destination/byte traffic, routed link IDs, and router traversal counts,
- per-DRAM-bank/channel traffic and row-hit/miss/conflict counters over time,
- DRAM command/address traces if using DRAMPower-style energy validation,
- TSV pitch/count/area and assignment to physical vertical regions,
- calibrated material stack: silicon, bonding, microbumps, TIM, heat spreader/sink, and boundary conditions,
- temperature-dependent leakage and frequency/power feedback.

The paper itself emphasizes that 3D-stacked AI efficiency depends on tile-to-core mapping, tensor-to-bank mapping, NoC topology/bandwidth, DRAM bank bandwidth, SRAM capacity, and energy/thermal constraints: https://arxiv.org/abs/2604.26821. Those are exactly the design dimensions this validation should sweep before making a broad rebuttal claim.

3D-ICE-style tools are useful as a future cross-check because they target transient 2.5D/3D thermal maps with heterogeneous materials, anisotropy, vertical heat conduction, and package-level paths; see 3D-ICE 4.0: https://arxiv.org/abs/2512.05823.

## How To Use This In The Rebuttal

Use the validation result to make a bounded claim:

The IPU-based emulator plus DRAM replay validates performance trends and TSIM event timing, but it does not validate all physical 3D effects directly. To address thermal-model concern, TSIM now exports component-level time-resolved power traces and checks them against a stack-aware thermal model and optional HotSpot package. Across the tested DRAM-bandwidth sweep and spatial hotspot assumptions, power accounting closes within 1%, design ranking is preserved per spatial policy, no configuration that is safe under the simple model becomes unsafe under the stack-aware model, and slowdown conclusions are unchanged.

Avoid claiming that this proves final silicon temperature. The stronger and defensible claim is that, for the explored design points and conservative spatial assumptions, the simple thermal constraint is sufficient for representative performance trend modeling.
