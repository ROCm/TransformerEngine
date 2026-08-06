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
../run_all_benchmarks.sh  run the suite (+ optional ingest / bundle)
../dashboard_redeploy.sh  publish the front-end (+ optional data / bundle) to gh-pages
../build_bundle.py        emit a single self-contained dashboard.html
```

Each shard row is `ts,commit,run_id,arch,model,runner,op,shape,dtype,metric,value,time_ms,pr`.
Shards are **append-only**; every ingest call is one run (unique `run_id`).

## Quickstart

Run benchmarks and ingest one run (GPU + TE + torch required):

```bash
cd benchmarks/microbenchmarks
./run_all_benchmarks.sh --ingest --out-dir dashboard/data       # one run
./run_all_benchmarks.sh --ingest --out-dir dashboard/data --runs 5   # a fresh baseline (needs >=4)
```

By default the shards hold **GPU kernel (device) time** and its throughput (the
benchmarks run with `--kernel-profile`, via `torch.profiler`, excluding host
launch/timing overhead). Pass `--python-time` to instead record **host
wall-clock** time:

```bash
./run_all_benchmarks.sh --ingest --python-time --out-dir dashboard/data
```

Pick one timing mode per shard and stick with it — kernel and wall-clock values
aren't comparable, so appending both into one shard makes the trend meaningless
(start a fresh `--ref`/`--pr` shard when switching).

To also track **compute-kernel-only** numbers (the op's own GPU kernels, with
host/torch scaffolding like `randn`/copies excluded), add `--compute-kernel`.
Those rows are ingested with an op-suffix ` [kernel]`, so they show up as their
own trend series alongside the e2e ops (the front-end keys a series on
`op`/`shape`/`dtype`, so the suffix is what keeps them separate). Give them their
own `--ref` to keep a clean shard/history:

```bash
./run_all_benchmarks.sh --ingest --runs 5 --out-dir dashboard/data                       # e2e (default)
./run_all_benchmarks.sh --ingest --runs 5 --compute-kernel --ref dev-kernel --out-dir dashboard/data
```

The arch (`gfx942` / `gfx950` / `gfx1250`) is auto-detected via `rocminfo`.

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
# or in one shot:
./run_all_benchmarks.sh --ingest --out-dir dashboard/data --bundle
```

Open by double-click — no server, no network. (Some orgs quarantine `.html`
attachments; zip it if needed.)

## Publish to GitHub Pages (optional)

`dashboard_redeploy.sh` copies the front-end into a **gh-pages checkout** you own
and pushes it. Point it at your own Pages repo:

```bash
git clone -b gh-pages <your-gh-pages-repo-url> /tmp/te-dash
export TE_DASH_DST=/tmp/te-dash
# ingest directly into the checkout, then publish front-end + data (+ a bundle):
./run_all_benchmarks.sh --ingest --out-dir "$TE_DASH_DST/data"
./dashboard_redeploy.sh --bundle
```

> GitHub Pages sites are public even from a private repo (private Pages needs
> Enterprise Cloud). For internal-only use, prefer the single-file bundle or a
> local/internal static server instead.

## CI (opt-in)

`.github/workflows/perf-dashboard.yml` runs the suite on a self-hosted GPU runner
and publishes, but only **on demand** — it skips normal pushes/PRs. Trigger it by:

- adding the **`ci-perf-test`** label to a PR, or
- a push whose head commit title contains **`[ci-perf-test]`**, or
- a manual `workflow_dispatch`.

PR runs ingest as `--pr <N>` (isolated from the `dev` baseline) and surface in the
**PR Check** tab; dev runs build the baseline shown in **Health** / **Trends**.

## Adding an arch

Arches are fully data-driven: the dashboard discovers them from the ingested
rows (sorted alphabetically) and colors each by a palette slot (`--series-N` in
`styles.css`), so no per-arch list to maintain. The GPU model label (e.g.
`MI355X`) is auto-detected at ingest (rocminfo/torch) and carried in each shard
row's `model` column; the dashboard shows it in place of the `gfx…` arch. Any
arch is accepted — ingest auto-detects it (or pass `--arch`/`--model`), so there
is nothing to add for a new GPU.
