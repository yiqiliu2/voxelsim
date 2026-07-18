# VoxelSim Workbench

An all-in-one web workbench for VoxelSim (3D-stacked AI-chip LLM-inference
simulator): result browsing, deep-dive analysis, paper-figure reproduction,
launching simulations / sweeps / DSE / thermal analyses from the browser,
job management, hardware/model/NoC catalogs and system monitoring.

![VoxelSim Workbench — Pareto explorer](webui.png)

## Quick Start

```bash
# from the project root (venv needs flask / pyyaml; tests need pytest)
venv/bin/python web_profiler/app.py
# or
web_profiler/launch.sh
```

Open **http://127.0.0.1:5000**. Environment variables:
`FLASK_HOST` / `FLASK_PORT` / `FLASK_DEBUG=1` /
`WEB_PROFILER_MAX_WORKERS` (parallel sim slots, default 2).

## Feature Map

| Page | Route | What it does |
|---|---|---|
| Dashboard | `#/dashboard` | Fleet-wide result distribution (donuts/bars), latest results |
| Results | `#/results` | 12-dimension filters + text search + paging + sorting + CSV export (3080+ rows) |
| Result detail | `#/result?id=` | Metric cards, static/dynamic energy donuts, windowed operator Gantt, Fig.20-style op-type breakdown, top-power intervals, overlap/power curve, reproduce command, trace.json/CSV download |
| Compare | `#/compare` | 2–12 results side by side: small-multiple metric charts + best/worst highlighted table |
| Sweep | `#/sweep` | Any parameter × any metric line chart (mean/min/max error band, auto log axis), drill-down to results |
| Paper figures | `#/paper` | One-click reproduction of Fig.10–20 (generic panel renderer, mode/model params) |
| Pareto | `#/pareto` | DSE front scatter/line, parallel coordinates, front table |
| Run | `#/run` | **Multi-stop tier sliders** (model/impl chips + 10 parameter tiers) → cartesian expansion into 1–64 sims; plus DSE and thermal tabs |
| Jobs | `#/jobs` | Queue/progress/cancel/delete, live log streaming (incremental offset polling) |
| HW configs | `#/hwconfigs` | 71 configs searchable, JSON view, create new (strict validation) |
| Models | `#/models` | 24 model cards + operator browser (search/type filter/paging) |
| NoC | `#/noc` | 12 distance-table heatmaps (auto downsampled above 256) + avg/max hops |
| LLM Serving | `#/serving` | llmservingsim profile curves and metadata |
| Logs | `#/logs` | test_logs/root/dse/thermal categories, byte-window paging, tail polling, keyword highlight |
| System | `#/system` | CPU/RAM/disk, RAM timeline (10s auto-refresh) |

## Architecture

```
web_profiler/
├── app.py                 # entry point (thin launcher → server.create_app)
├── launch.sh              # launcher script (auto-uses venv)
├── requirements.txt
├── server/                # Flask backend (blueprints)
│   ├── __init__.py        #   create_app() application factory
│   ├── config.py          #   paths/constants
│   ├── parsers.py         #   log parsers (pure functions)
│   ├── index.py           #   results index + summary cache
│   ├── classify.py        #   operator classification (FFN/Attn/Other,
│   │                      #   ported from draw_op_breakdown.py)
│   ├── commands.py        #   sim/DSE/thermal command builders + hw_config
│   │                      #   generation (never overwrites)
│   ├── jobs.py            #   JobManager (queue/concurrency/cancel/persist)
│   ├── api_results.py     #   /api/results /api/result/*
│   ├── api_analysis.py    #   /api/sweep /api/compare /api/pareto /api/paper/*
│   ├── api_catalog.py     #   /api/hwconfigs /api/models /api/noc /api/serving
│   ├── api_system.py      #   /api/system /api/logs
│   └── api_jobs.py        #   /api/jobs*
├── templates/index.html   # SPA shell (sidebar + router)
├── static/
│   ├── css/app.css        # design system (dark theme)
│   ├── js/core.js         # framework: hash router + API client + UI kit
│   ├── js/pages/*.js      # 15 page modules (self-register via App.route)
│   └── vendor/plotly-2.35.2.min.js
├── runtime/               # runtime state (summary cache, jobs/*.json|*.log)
└── tests/                 # pytest suite
```

Conventions:

- A result id is its log path relative to the project root; web-launched runs
  land in `results/logs_web/`.
- The backend only ever *adds* files — existing project files are never
  modified or overwritten (a name clash with different content gets a
  `web_`-prefixed hw_config).
- On server restart, unfinished jobs are marked `interrupted` and are never
  auto-rerun.

## Tests

```bash
venv/bin/python -m pytest web_profiler/tests/ -q    # 159 tests
```

## API Overview

`GET /api/overview` · `GET /api/filters` · `GET /api/metrics_meta` ·
`GET /api/results?page=&page_size=&with_metrics=1&sort=&q=` ·
`GET /api/result/detail|operators|op_energy|op_breakdown|top_power|overlap|reproduce|export.csv|trace.json?id=` ·
`GET /api/sweep?x=&metric=&group=` · `GET /api/compare?ids=` ·
`GET /api/pareto?mode=` · `GET /api/paper/fig10..fig20` ·
`POST /api/index/rebuild` ·
`GET|POST /api/jobs` · `GET /api/jobs/<id>/log?offset=` · `POST /api/jobs/<id>/cancel|delete` ·
`GET|POST /api/hwconfigs` · `GET /api/models/<name>/ops` · `GET /api/noc/tables/<topo>/<n>` ·
`GET /api/serving/profile?path=` · `GET /api/logs/view?path=` ·
`GET /api/system/host` · `GET /api/system/ram_timeline`
