# TransformerEngine microbenchmark dashboard

A small static dashboard that tracks TE microbenchmark performance over time and
flags regressions using a **run-to-run noise band** (a drop counts only if it
exceeds `max(3% gate, 2σ)` of the kernel's own run-to-run variation). It renders
entirely client-side — no server or build step — from per-family CSV "shards".

Adapted from the [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) CI dashboard
(Apache-2.0); Chart.js is vendored under `vendor/` (MIT).

## Layout

```
dashboard/
  index.html            landing (Health), All Benchmarks, PR Check tabs
  app.js                all logic (vanilla JS; no framework)
  styles.css
  vendor/chart.umd.min.js   Chart.js (trend charts)
  data/                 generated CSV shards (git-ignored)
    index.csv           catalog: file,family,ref,pr
    perf-<family>-<ref>.csv   long-format rows, appended per run
../dashboard_ingest.py    wide benchmark CSV -> per-family shards (stdlib only)
../build_bundle.py        emit a single self-contained dashboard.html
```

Each shard row is `ts,commit,run_id,model,runner,op,shape,dtype,metric,value,time_ms,pr`.
Shards are **append-only**; every ingest call is one run (unique `run_id`).

## Quickstart

Run the benchmarks and ingest one run (GPU + TE + torch required). Each
`benchmark_<family>.py` writes a wide `benchmark_<family>.csv`; `dashboard_ingest.py`
appends them to the shards as one run (a unique `run_id`):

```bash
cd benchmarks/microbenchmarks
families="benchmark_gemm.py benchmark_casting.py benchmark_grouped_gemm.py benchmark_normalization.py"

for f in $families; do python "$f" --csv --kernel-profile; done   # omit --kernel-profile for wall-clock
python dashboard_ingest.py benchmark_*.csv --ref dev --out-dir dashboard/data
```

With `--kernel-profile` (via `torch.profiler`) the shards hold **GPU kernel
(device) time** and its throughput, excluding host launch/timing overhead. Drop
the flag to record **host wall-clock** time instead. Pick one timing mode per
shard and stick with it — kernel and wall-clock values aren't comparable, so
mixing them in one shard makes the trend meaningless (start a fresh `--ref`/`--pr`
shard when switching).

A noise band needs **≥4 prior runs**; build a baseline by repeating run+ingest
(each ingest is one `run_id`):

```bash
for run in 1 2 3 4 5; do
  for f in $families; do python "$f" --csv --kernel-profile; done
  python dashboard_ingest.py benchmark_*.csv --ref dev --out-dir dashboard/data
done
```

To also track **compute-kernel-only** numbers (the op's own GPU kernels, with
host/torch scaffolding like `randn`/copies excluded), run with `--compute-kernel`
and ingest under a distinct `--ref` with an op-suffix, so they form their own
series (the front-end keys a series on `op`/`shape`/`dtype`, so the suffix keeps
them separate):

```bash
for f in $families; do python "$f" --csv --compute-kernel; done
python dashboard_ingest.py benchmark_*.csv --ref dev-kernel --op-suffix ' [kernel]' --out-dir dashboard/data
```

The GPU model label (e.g. `MI355X`) is auto-detected via `rocminfo`.

View it locally (no server dependency other than a static file server, because
the front-end `fetch()`es the shards):

```bash
cd dashboard && python3 -m http.server 8000     # http://localhost:8000
```

## Share as a single file

Bundle everything (front-end + Chart.js + the CSV data) into one offline HTML you
can email/Teams:

```bash
python3 build_bundle.py --data-dir dashboard/data     # -> dashboard/dist/dashboard.html
```

Open by double-click — no server, no network. (Some orgs quarantine `.html`
attachments; zip it if needed.)

## Publish to GitHub Pages (optional)

The weekly CI (below) publishes automatically. For a one-off / self-hosted deploy,
ingest into a checkout of a Pages repo you own, build the single-file bundle, and
push:

```bash
git clone <your-pages-repo-url> /tmp/te-dash
# run the benchmarks (see Quickstart), then ingest into the checkout and bundle:
python dashboard_ingest.py benchmark_*.csv --ref dev --out-dir /tmp/te-dash/data
python3 build_bundle.py --data-dir /tmp/te-dash/data --out /tmp/te-dash/dashboard.html
git -C /tmp/te-dash add -A && git -C /tmp/te-dash commit -m "dashboard update" && git -C /tmp/te-dash push
```

> GitHub Pages sites are public even from a private repo (private Pages needs
> Enterprise Cloud). For internal-only use, prefer the single-file bundle or a
> local/internal static server instead.

## CI (weekly snapshot)

`.github/workflows/perf-dashboard-weekly.yml` runs the suite on a self-hosted GPU
runner every **Sunday night (Central Time)**, ingests the result as a new weekly
point on `dev`, rebuilds the single-file `dashboard.html`, and publishes it to the
external GitHub Pages repo (`AMD-ROCm-Internal/TE-dashboard`). You can also run it
on demand via `workflow_dispatch` (with an optional GPU-model override). Each run
adds one point to the baseline shown in **Health** / **Trends**.

## Adding a GPU model

Models are fully data-driven: the dashboard discovers them from the ingested
rows (sorted alphabetically) and colors each by a palette slot (`--series-N` in
`styles.css`), so no per-model list to maintain. The GPU model label (e.g.
`MI355X`) is auto-detected at ingest (rocminfo/torch) and carried in each shard
row's `model` column, which the dashboard uses as the series key. Any GPU is
accepted — ingest auto-detects it (or pass `--model`), so there is nothing to add
for a new GPU.
