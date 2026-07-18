# 3D-workloads

Drop workload trace CSVs here (e.g. `swebench_heavy.csv`, `sharegpt.csv`).

These are local data files and are not tracked in git.  The analysis scripts
default to this directory but accept an explicit path:

```bash
python analyze_decode_batch.py 3D-workloads/swebench_heavy.csv
python plot_workload_throughput.py 3D-workloads/sharegpt.csv --tick 2 --output plot.png
```
