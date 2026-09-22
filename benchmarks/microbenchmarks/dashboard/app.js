/* TransformerEngine CI Performance Dashboard — regression-first.
 *
 * Adapted from the ROCm/FlyDSL CI dashboard (Apache-2.0): the run-to-run
 * noise-band regression logic and Chart.js views are reused, retargeted to TE's
 * microbenchmark data. Landing = "is dev regressing?"; PR Check = "does this PR
 * regress vs dev?"; Trends = per-kernel series with a run-to-run noise band.
 */
"use strict";

const CFG = {
  repo: "ROCm/TransformerEngine",
  data: "./data/",               // per-family CSV shards + index.csv live here
  dataBranch: "./data/",         // local-only: no external data branch
  bundled: "./data/",
  api: "",                        // empty -> no live GitHub Actions board
  regressionPct: -5.0,   // fixed gate
  warnPct: -1.0,         // surfaced as "watch"
  noiseK: 2.0,           // a drop must exceed K * (run-to-run relative std) to count as real
  minSamples: 3,         // prior main runs needed to size a noise band (else: low confidence)
  sparkFloorPct: 6,      // sparkline min half-window (% of baseline) so trivial noise stays flat
  gapPct: 10,            // a backend is "slower" when it trails its family's reference by > this %
};

const S = {
  records: [], series: new Map(), runs: [], updated: null, runMeta: new Map(),
  models: [], modelColor: new Map(),   // GPU models present in the data, sorted (discovered at load)
  backends: [], backendColor: new Map(),   // backends present in the data (parsed from op), sorted
  view: "health", noiseAware: true, boardFilter: "all",
  pr: { sel: null },
  trend: { key: null, model: "all", metric: null, q: "", range: "all", xmode: "commits", by: "arch", railMode: "kernels" },
  byType: { q: "", facets: { family: "all", mode: "all", dtype: "all", model: "all" },
    trend: { key: null, model: "all", metric: null, range: "all", xmode: "commits", by: "arch" } },
  theme: "dark",
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const kkey = r => `${r.op} ${r.shape} ${r.dtype}`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const VIEWS = ["health", "bytype", "gaps", "prcheck", "board"];

// theme-aware colors: each model gets a distinct hue by golden-angle spacing, so
// any number of models renders with well-separated, legible colors -- no palette
// size limit. Hues are re-tuned per theme (see setModelColors / toggleTheme).
const cssVal = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || n;
const hslHex = (h, s, l) => {                     // HSL(0-360, 0-100, 0-100) -> #rrggbb
  s /= 100; l /= 100;
  const k = n => (n + h / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, 9 - k(n), 1));
  const to = n => Math.round(255 * f(n)).toString(16).padStart(2, "0");
  return `#${to(0)}${to(8)}${to(4)}`;
};
const modelColorFor = i => hslHex((i * 137.508) % 360, S.theme === "light" ? 62 : 70, S.theme === "light" ? 44 : 62);
const setModelColors = () => { S.modelColor = new Map(S.models.map((m, i) => [m, modelColorFor(i)])); };
const modelVar = m => S.modelColor.get(m) || cssVal("--ink-2");   // hex; used inline and on <canvas>
const modelCol = modelVar;                                       // alias kept for <canvas> call sites
const setBackendColors = () => { S.backendColor = new Map(S.backends.map((b, i) => [b, modelColorFor(i)])); };
const backendCol = b => S.backendColor.get(b) || cssVal("--ink-2");   // hex; same golden-angle palette as archs
const commitUrl = sha => sha ? `https://github.com/${CFG.repo}/commit/${sha}` : "#";

function relTime(iso) {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso)) / 1000;
  if (s < 60) return "just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 60) return `${m | 0}m ago`;
  if (h < 24) return `${h | 0}h ago`;
  if (d < 30) return `${d | 0}d ago`;
  return new Date(iso).toISOString().slice(0, 10);
}
function fmtVal(v, metric) {
  if (v == null) return "—";
  if (metric === "speedup") return v.toFixed(2) + "×";
  if (metric === "TB/s") return v.toFixed(3);
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}
function fmtPct(d) { return (d > 0 ? "+" : "") + d.toFixed(1) + "%"; }
function toast(msg) {
  let t = $(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show"); clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2400);
}
// Persistent, high-visibility banner for load failures (missing/undeployed data files).
function showBanner(msg) {
  const b = $("#banner"); if (!b) return;
  const m = $("#bannerMsg"); if (m) m.textContent = msg;
  b.hidden = false;
}
function hideBanner() { const b = $("#banner"); if (b) b.hidden = true; }

/* ----------------------------------------------------------------- loading -- */
async function getJSON(url, timeout = 8000) {
  const c = new AbortController(); const id = setTimeout(() => c.abort(), timeout);
  try { const r = await fetch(url, { signal: c.signal, cache: "no-store" }); if (!r.ok) throw 0; return await r.json(); }
  catch { return null; } finally { clearTimeout(id); }
}
async function getText(url, timeout = 8000) {
  // Single-file bundle mode: each data shard is embedded in the page as a
  // `<script type="text/csv" data-file="<name>">` block (see build_bundle.py) and
  // read here instead of fetched (fetch is blocked on file://). Falls back to
  // fetch for the normal server-hosted mode, so the same code works both ways.
  const key = String(url).split("/").pop();
  const embedded = document.querySelector(`script[type="text/csv"][data-file="${key}"]`);
  if (embedded) return embedded.textContent.replace(/^\n/, "");
  const c = new AbortController(); const id = setTimeout(() => c.abort(), timeout);
  try { const r = await fetch(url, { signal: c.signal, cache: "no-store" }); if (!r.ok) throw 0; return await r.text(); }
  catch { return null; } finally { clearTimeout(id); }
}
// Minimal RFC-4180-ish CSV parser: handles quoted fields with embedded commas,
// escaped quotes ("") and CRLF. Returns an array of field arrays.
function parseCSV(text) {
  const rows = []; let row = [], field = "", i = 0, q = false;
  while (i < text.length) {
    const c = text[i];
    if (q) {
      if (c === '"') { if (text[i + 1] === '"') { field += '"'; i += 2; } else { q = false; i++; } }
      else { field += c; i++; }
    } else if (c === '"') { q = true; i++; }
    else if (c === ',') { row.push(field); field = ""; i++; }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ""; i++; }
    else if (c === '\r') { i++; }
    else { field += c; i++; }
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  return rows;
}
function csvToObjects(text) {
  const rows = parseCSV(text).filter(r => !(r.length === 1 && r[0] === ""));
  if (rows.length < 2) return [];
  const hdr = rows[0];
  return rows.slice(1).map(r => Object.fromEntries(hdr.map((h, i) => [h, r[i] ?? ""])));
}
// family from a shard file name: perf-<family>-<ref>.csv -> <family>
function _familyFromFile(file) {
  const stem = String(file).replace(/^perf-/, "").replace(/\.csv$/, "");
  const i = stem.lastIndexOf("-");
  return i > 0 ? stem.slice(0, i) : stem;
}
// one long-CSV row -> the record shape the rest of the app consumes
function toRecord(o) {
  const num = v => (v === "" || v == null) ? null : +v;
  // Backend is a first-class column (new schema); legacy shards encode it as a
  // "<base> · <backend>" op suffix, so fall back to parsing that. The composite op
  // is kept for identity/display (current UI unchanged); base + backend are exposed
  // as clean fields so a future de-duplicated/faceted layout is a UI-only change.
  let base = o.op, backend = (o.backend || "").trim();
  if (!backend) { const sep = o.op.lastIndexOf(" · "); if (sep >= 0) { base = o.op.slice(0, sep); backend = o.op.slice(sep + 3); } }
  const op = backend ? `${base} · ${backend}` : base;
  return {
    op, base, backend,
    shape: o.shape, dtype: o.dtype, metric: o.metric,
    value: num(o.value), ts: o.ts, commit: o.commit, run_id: num(o.run_id),
    model: o.model, runner: o.runner, pr: num(o.pr), source: "ci",
    mode: /\[kernel\]/.test(o.op) ? "kernel" : "wall-clock",   // compute-kernel series carry a " [kernel]" op suffix
    status: "ok", regression: false, vs_main: null, vs_tag: null,
    extra: o.time_ms ? { median_ms: +o.time_ms } : {},
  };
}
async function loadAll() {
  // Discover shards from the catalog (no hardcoded benchmark list), then load
  // each CSV shard. dev shards feed Health/Trends; pr shards feed PR Check.
  const idx = await getText(CFG.data + "index.csv");
  const failed = [];
  if (idx == null) failed.push("index.csv");
  const shards = idx ? csvToObjects(idx) : [];
  const texts = await Promise.all(shards.map(s => getText(CFG.data + s.file)));
  const records = []; let latest = "";
  texts.forEach((t, i) => {
    if (t == null) { failed.push(shards[i].file); return; }
    const family = shards[i].family || _familyFromFile(shards[i].file);
    for (const o of csvToObjects(t)) {
      const r = toRecord(o);
      if (r.value == null) continue;
      r.family = family;
      records.push(r);
      if ((r.ts || "") > latest) latest = r.ts;
    }
  });
  S.records = records.concat(buildAggregateRecords(records));   // append synthetic Σwork/Σtime rollup series
  S.series = buildSeriesIndex(S.records);
  _healthRows = null; _aggRows = null;   // invalidate the memos for the fresh dataset
  // Discover GPU models (sorted) from the rows; the model is the key for
  // grouping/colors/baselines and is shown directly as the series label.
  S.models = [...new Set(records.map(r => r.model).filter(Boolean))].sort();
  setModelColors();
  S.backends = [...new Set(records.map(r => r.backend).filter(Boolean))].sort();
  setBackendColors();
  computePRDeltas();   // derive vs_main (PR value vs dev baseline) so PR Check works
  S.runs = [];
  S.runMeta = new Map();
  S.updated = latest || null;
  const up = $("#updated"); up.classList.remove("syncing");
  up.textContent = S.updated ? `snapshot ${relTime(S.updated)}` : "no data";
  if (failed.length) {
    const noun = failed.length > 1 ? "data files" : "data file";
    showBanner(`Couldn't load ${failed.length} ${noun} (${failed.join(", ")}). Charts may be incomplete \u2014 verify ${CFG.data} is deployed alongside index.csv.`);
    console.warn("[dashboard] failed to fetch:", failed);
  } else {
    hideBanner();
  }
  renderAll();
}
async function enhanceLiveBoard() {
  if (!CFG.api) return;   // live CI board disabled for the local TE dashboard
  const live = await getJSON(`${CFG.api}/actions/runs?per_page=25`);
  if (!live || !live.workflow_runs) return;
  const wf = live.workflow_runs.filter(r => /fly\s*dsl\s*test/i.test(r.name || ""));
  const byId = new Map(S.runs.map(r => [r.run_id, r]));
  // The live API often returns pull_requests:[] for fork PRs; reuse the PR the ingest
  // snapshot already resolved for this head branch so new runs aren't shown as branch cards.
  const branchToPr = new Map();
  for (const r of S.runs) if (r.pr && r.branch) branchToPr.set(r.branch, r.pr);
  for (const r of wf) {
    const cur = byId.get(r.id);
    const pr = r.pull_requests?.[0]?.number ?? cur?.pr ?? branchToPr.get(r.head_branch) ?? null;
    byId.set(r.id, {
      run_id: r.id, pr, commit: r.head_sha,
      branch: r.head_branch, event: r.event, title: r.display_title, status: r.status, conclusion: r.conclusion,
      url: r.html_url, created_at: r.created_at, updated_at: r.updated_at, actor: r.actor?.login, jobs: cur?.jobs || [],
    });
  }
  S.runs = [...byId.values()].sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
  S.runMeta = new Map(S.runs.map(r => [r.run_id, r]));
  const active = S.runs.filter(r => r.status !== "completed" || !r.jobs?.length).slice(0, 5);
  await Promise.all(active.map(async r => {
    const j = await getJSON(`${CFG.api}/actions/runs/${r.run_id}/jobs?per_page=100`);
    if (!j || !j.jobs) return;
    r.jobs = j.jobs.filter(x => /linux-flydsl-(mi355|mi325|navi)/.test(x.name)).map(x => ({
      runner: (x.name.match(/\((linux-flydsl-[^)]+)\)/) || [])[1], model: modelOf(x.name),
      status: x.status, conclusion: x.conclusion, url: x.html_url,
    }));
  }));
  if (S.view === "board") renderBoard();
}
function modelOf(n) { return /mi355/.test(n) ? "MI355X" : /mi325/.test(n) ? "MI325X" : "?"; }

/* ----------------------------------------------------------- noise model --- */
function isMainRec(r) { const m = S.runMeta.get(r.run_id); return m ? m.branch === "dev" : r.pr == null; }

// Work-weighted harmonic mean of throughput == Σwork/Σtime == Σ(value·time_ms)/Σ(time_ms).
// Emitted as synthetic "kernel" records (shape = AGG_SHAPE) so a per-(op,dtype,model,metric)
// rollup trends through the chart like any config -- tagged aggregate:true with a per-run
// config count (n) and shape signature (sig) to flag when the set of shapes changed.
const AGG_SHAPE = "\u03a3 aggregate";
const AGG_METRICS = new Set(["TFLOPS", "GB/s"]);   // value·time is meaningful "work" only for throughput
const AGG_WARN = "#e0a94f";                         // amber: point where the config set changed
function buildAggregateRecords(records) {
  const g = new Map();   // op|dtype|model|metric|run_id -> {wt, t, shapes, any}
  for (const r of records) {
    const t = r.extra && r.extra.median_ms;
    if (r.source !== "ci" || !AGG_METRICS.has(r.metric) || !(t > 0) || !(r.value > 0)) continue;
    const k = `${r.op}|${r.dtype}|${r.model}|${r.metric}|${r.run_id}`;
    let e = g.get(k); if (!e) g.set(k, e = { wt: 0, t: 0, shapes: new Set(), any: r });
    e.wt += r.value * t; e.t += t; e.shapes.add(r.shape);
  }
  const out = [];
  for (const e of g.values()) {
    if (!(e.t > 0)) continue;
    const r = e.any;
    out.push({
      op: r.op, base: r.base, backend: r.backend, shape: AGG_SHAPE, dtype: r.dtype, metric: r.metric,
      value: e.wt / e.t, ts: r.ts, commit: r.commit, run_id: r.run_id, model: r.model, runner: r.runner,
      pr: r.pr, source: "ci", mode: r.mode, status: "ok", regression: false, vs_main: null, vs_tag: null,
      family: r.family, extra: { median_ms: e.t }, aggregate: true, n: e.shapes.size, sig: [...e.shapes].sort().join("|"),
    });
  }
  return out;
}

// One-pass index (built in loadAll): series key -> chronological records, so the
// per-series lookups below are O(1) instead of re-scanning every record.
const seriesKey = r => `${r.op}|${r.shape}|${r.dtype}|${r.model}|${r.metric}`;
function buildSeriesIndex(records) {
  const m = new Map();
  for (const r of records) {
    if (r.source !== "ci" || r.value == null) continue;
    let arr = m.get(seriesKey(r)); if (!arr) m.set(seriesKey(r), arr = []);
    arr.push(r);
  }
  for (const arr of m.values()) arr.sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  return m;
}
// chronological main-branch series for one kernel/model/metric
function mainSeries(op, shape, dtype, model, metric) {
  const arr = S.series.get(`${op}|${shape}|${dtype}|${model}|${metric}`);
  return arr ? arr.filter(isMainRec) : [];
}
function noiseOf(values) {
  const n = values.length;
  if (n < 2) return { n, mean: values[0] ?? null, std: 0, relStd: null, lo: null, hi: null };
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
  const relStd = mean ? (std / mean) * 100 : null;
  return { n, mean, std, relStd, lo: mean - CFG.noiseK * std, hi: mean + CFG.noiseK * std };
}
// "real" regression: drop beyond max(|gate|, K*relStd). With too few samples, just the gate.
function realRegression(deltaPct, noise) {
  const thr = (S.noiseAware && noise.n >= CFG.minSamples && noise.relStd != null)
    ? Math.max(Math.abs(CFG.regressionPct), CFG.noiseK * noise.relStd) : Math.abs(CFG.regressionPct);
  return deltaPct <= -thr;
}
function sev(deltaPct, real) { return real ? "bad" : deltaPct <= CFG.warnPct ? "warn" : deltaPct > 0 ? "ok" : "flat"; }

// Baseline noise for "did main regress?": the main history EXCLUDING the latest main run.
// On push-to-main, flydsl.yaml rebuilds origin/main in the same job, so a main run's own
// vs_main is current-vs-itself (re-run variance) — useless as a historical signal. We instead
// compare the latest main value against prior main runs.
function mainBaseline(op, shape, dtype, model, metric) {
  const series = mainSeries(op, shape, dtype, model, metric);
  if (!series.length) return noiseOf([]);
  const latestRun = series[series.length - 1].run_id;
  return noiseOf(series.filter(s => s.run_id !== latestRun).map(s => s.value));
}
// Δ% and whether it is a real regression, given a prior-main baseline.
//  - main run:  value vs the prior-main mean (a true historical comparison)
//  - PR run:    vs_main is a real PR-commit-vs-main-commit diff, so use it directly
function regOf(r, base) {
  if (!r || r.value == null || r.metric === "speedup") return { d: null, real: false };
  if (isMainRec(r)) {
    if (base.mean == null) return { d: null, real: false, lowConf: true };
    const d = (r.value - base.mean) / base.mean * 100;
    const haveBand = base.n >= CFG.minSamples && base.relStd != null;
    // confident only when enough prior history sizes the band; otherwise it's a
    // raw-gate guess that can't be told apart from run-to-run noise.
    const real = S.noiseAware
      ? (haveBand && d <= -Math.max(Math.abs(CFG.regressionPct), CFG.noiseK * base.relStd))
      : d <= CFG.regressionPct;
    return { d, real, lowConf: !haveBand };
  }
  // PR run: vs_main is a real PR-commit-vs-main-commit diff in the same job — valid as-is.
  if (!r.vs_main) return { d: null, real: false };
  return { d: r.vs_main.delta_pct, real: realRegression(r.vs_main.delta_pct, base) };
}

// Derive vs_main for PR records client-side. Our CSV ingest doesn't precompute
// the PR-vs-dev diff (unlike upstream FlyDSL's CI job), so compare each PR
// kernel's value against the dev baseline (prior-dev mean) here. Populates the
// PR Check tab; a safe no-op until a `--pr` shard is ingested.
function computePRDeltas() {
  for (const r of S.records) {
    if (r.pr == null || r.value == null || r.metric === "speedup") continue;
    const base = mainBaseline(r.op, r.shape, r.dtype, r.model, r.metric);
    if (base.mean == null) continue;    // no dev baseline for this kernel yet
    r.vs_main = {
      delta_pct: (r.value - base.mean) / base.mean * 100,
      baseline: base.mean,
      label: "dev",
    };
  }
}

let _sparkN = 0;
function sparkline(values, noise, lastReal) {
  const W = 128, H = 34, pad = 4, GUT = 28;   // GUT: right gutter for the % axis
  if (values.length < 2) return `<svg class="spark" width="${W}" height="${H}"></svg>`;
  const XR = W - GUT;                          // plot right edge (x)
  // Consistent vertical scale: center on the baseline mean and use a *symmetric*
  // window with a floor (sparkFloorPct). Without the floor the range collapses to
  // the values' own min/max, so trivial run-to-run noise gets stretched to fill
  // the box ("jitter"). The window still expands to include the noise band and any
  // real excursion, so genuine drops stay visible — matching the big trend chart.
  const center = (noise && noise.mean != null)
    ? noise.mean : values.reduce((a, b) => a + b, 0) / values.length;
  const floorHalf = Math.abs(center) * (CFG.sparkFloorPct / 100);
  const bandHalf = (noise && noise.lo != null && noise.hi != null)
    ? Math.max(noise.hi - center, center - noise.lo) : 0;
  const dataHalf = Math.max(0, ...values.map(v => Math.abs(v - center)));
  const half = Math.max(floorHalf, bandHalf, dataHalf) || 1;
  const lo = center - half, hi = center + half;
  const span = hi - lo || 1;
  const x = i => pad + (i / (values.length - 1)) * (XR - pad);
  const y = v => H - pad - ((v - lo) / span) * (H - 2 * pad);
  const pts = values.map((v, i) => [x(i), y(v)]);
  const col = lastReal ? cssVal("--bad") : cssVal("--ink-2");
  const gid = "sg" + (_sparkN++);
  const line = "M" + pts.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L");
  const area = `M${pts[0][0].toFixed(1)},${(H - pad).toFixed(1)} ` +
    pts.map(p => `L${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ") +
    ` L${pts[pts.length - 1][0].toFixed(1)},${(H - pad).toFixed(1)} Z`;
  let band = "";
  if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
    const yh = y(noise.hi), yl = y(noise.lo);
    band = `<rect x="0" y="${yh.toFixed(1)}" width="${XR}" height="${Math.max(1, yl - yh).toFixed(1)}" fill="${cssVal("--band")}"/>`;
  }
  // y-axis: 0 = baseline mean; the top/bottom labels state the window extent in %
  // of that mean (auto-scaled, so labels never crowd and always say what the box
  // height represents). A point at "-H%" means H% below the prior-dev mean.
  const axis = cssVal("--ink-3");
  const midY = y(center);
  const pct = Math.abs(center) ? half / Math.abs(center) * 100 : 0;
  const pctLab = pct >= 10 ? String(Math.round(pct)) : pct.toFixed(1);
  const zeroLine = `<line x1="${pad}" y1="${midY.toFixed(1)}" x2="${XR}" y2="${midY.toFixed(1)}" stroke="${axis}" stroke-width="0.6" stroke-dasharray="2 2" opacity="0.55"/>`;
  const lx0 = XR + 4;
  const axisTxt =
    `<text x="${lx0}" y="${pad}" font-size="8" fill="${axis}" dominant-baseline="hanging">+${pctLab}%</text>` +
    `<text x="${lx0}" y="${(H - pad).toFixed(1)}" font-size="8" fill="${axis}">-${pctLab}%</text>`;
  const [lx, ly] = pts[pts.length - 1];
  return `<svg class="spark" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<defs><linearGradient id="${gid}" x1="0" y1="0" x2="0" y2="1">` +
    `<stop offset="0" stop-color="${col}" stop-opacity="0.20"/><stop offset="1" stop-color="${col}" stop-opacity="0"/></linearGradient></defs>` +
    `${band}${zeroLine}<path d="${area}" fill="url(#${gid})"/><path d="${line}" fill="none" stroke="${col}" stroke-width="1.5" stroke-linejoin="round"/>` +
    `<circle cx="${lx.toFixed(1)}" cy="${ly.toFixed(1)}" r="2.8" fill="${col}"/>${axisTxt}</svg>`;
}

/* latest main-run record per (kernel,model) */
function latestMainByKernelModel() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.aggregate || r.metric === "speedup" || r.value == null || !isMainRec(r)) continue;
    const k = `${r.model}|${kkey(r)}`; const ex = m.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) m.set(k, r);
  }
  return m;
}

/* one row per latest-main kernel/model: latest value vs PRIOR main history.
   Memoized (reset on data reload / noise-gate toggle) -- Health, By-type and the
   kernel rail each recompute this on every render. */
let _healthRows = null;
function healthRows() {
  if (_healthRows) return _healthRows;
  _healthRows = [...latestMainByKernelModel().values()].map(r => {
    const noise = mainBaseline(r.op, r.shape, r.dtype, r.model, r.metric);
    const vals = mainSeries(r.op, r.shape, r.dtype, r.model, r.metric).map(s => s.value);
    const { d, real } = regOf(r, noise);
    return { r, vals, noise, d, real, sev: d == null ? "flat" : sev(d, real) };
  });
  return _healthRows;
}

// same, but for the synthetic aggregate rollups (excluded from the per-shape rows above)
function latestMainAggByModel() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || !r.aggregate || r.value == null || !isMainRec(r)) continue;
    const k = `${r.model}|${kkey(r)}`; const ex = m.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) m.set(k, r);
  }
  return m;
}
let _aggRows = null;
function aggRows() {
  if (_aggRows) return _aggRows;
  _aggRows = [...latestMainAggByModel().values()].map(r => {
    const noise = mainBaseline(r.op, r.shape, r.dtype, r.model, r.metric);
    const vals = mainSeries(r.op, r.shape, r.dtype, r.model, r.metric).map(s => s.value);
    const { d, real } = regOf(r, noise);
    return { r, vals, noise, d, real, sev: d == null ? "flat" : sev(d, real) };
  });
  return _aggRows;
}

/* ----------------------------------------------------------- 1 · HEALTH --- */
function renderHealth() {
  const rows = healthRows();
  const reals = rows.filter(x => x.real);
  // "watch" = dropped past the gate but not confident (thin main history can't size the noise band)
  const watch = rows.filter(x => !x.real && x.d != null && x.d <= CFG.regressionPct);
  const list = [...reals, ...watch].sort((a, b) => a.d - b.d);

  // hero
  const card = $("#heroCard");
  const n = reals.length;
  card.className = "hero " + (n ? "alert" : "clear");
  $("#heroNum").textContent = n;
  const glyph = n
    ? `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 2l6 11H2z"/><path d="M8 6.4v3.2M8 11.5v.01"/></svg>`
    : `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>`;
  $("#heroLabel").innerHTML = glyph + (n ? `confirmed regression${n > 1 ? "s" : ""} on dev` : watch.length ? "no confirmed regressions" : "dev is clean");
  const lastRun = rows.reduce((a, x) => (x.r.ts || "") > a ? x.r.ts : a, "");
  $("#heroNote").innerHTML =
    `latest dev vs <b>prior dev history</b> · confirmed = below the ${CFG.noiseK}σ noise band ` +
    `(needs ≥${CFG.minSamples} prior dev runs)`;
  $("#heroStats").innerHTML =
    `<div class="stat"><span class="v">${rows.length}</span><span class="l">kernel × arch</span></div>` +
    `<div class="stat warn"><span class="v">${watch.length}</span><span class="l">to check</span></div>` +
    `<div class="stat"><span class="v">${new Set(rows.map(x => x.r.model)).size}</span><span class="l">arches</span></div>` +
    `<div class="stat"><span class="v" style="font-size:15px">${relTime(lastRun)}</span><span class="l">last run</span></div>`;
  const badge = $("#healthBadge"); badge.hidden = false; badge.textContent = n; badge.classList.toggle("zero", n === 0);
  $("#regHeadTitle").textContent = list.length ? `${reals.length} confirmed · ${watch.length} to check` : "Regressions on dev";

  // list
  if (!list.length) {
    $("#regList").innerHTML = `<div class="reg-list-empty"><div class="big">` +
      `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>` +
      `all kernels within budget</div>no kernel on dev is slower than the gate or its noise band.</div>`;
    return;
  }
  const maxAbs = Math.max(6, ...list.map(x => Math.abs(x.d)));
  $("#regList").innerHTML = list.map(({ r, vals, noise, real, d, sev }, i) => {
    const run = S.runMeta.get(r.run_id);
    const sha = (r.commit || "").slice(0, 7);
    const href = commitUrl(r.commit);
    const w = Math.max(4, Math.min(46, Math.abs(d) / maxAbs * 46));
    const bc = sev === "bad" ? "var(--bad)" : sev === "warn" ? "var(--warn)" : "var(--good)";
    return `<div class="reg-row s-${sev}" style="--i:${i}" data-k="${esc(kkey(r))}" data-model="${r.model}">
      <span class="op">${esc(r.op)} <span class="metric-tag">${r.metric}</span></span>
      <span class="shape">${esc(r.shape)} · ${esc(r.dtype)}</span>
      <span class="reg-arch" style="color:${modelVar(r.model)}">${esc(r.model)}</span>
      ${sparkline(vals, noise, real)}
      <span class="reg-delta ${sev}"><span class="dbar" style="width:${w}px;background:${bc}"></span>${fmtPct(d)}</span>
      <span class="commit"><a href="${href}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${r.pr ? "#" + r.pr : "dev"}·${sha}</a></span>
    </div>`;
  }).join("");
}

/* ---------------------------------------------------------- 2 · BY TYPE --- */
// Pretty labels for known families; unknown families (e.g. future e2e suites)
// fall back to a title-cased version of the raw family name -- so newly ingested
// benchmark types appear here automatically without code changes.
const FAMILY_LABELS = {
  gemm: "GEMM", grouped_gemm: "Grouped GEMM",
  casting: "Casting", normalization: "Normalization",
};
// Display order: known microbenchmark families first, then anything else
// (e.g. e2e suites) alphabetically.
const FAMILY_ORDER = ["gemm", "grouped_gemm", "casting", "normalization"];
const familyLabel = f => FAMILY_LABELS[f] ||
  (f ? f.replace(/[_-]+/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : "Other");
const familyRank = f => { const i = FAMILY_ORDER.indexOf(f); return i < 0 ? FAMILY_ORDER.length : i; };

// Populate the by-type facet <select>s from the current snapshot, preserving the
// active selection (dropping any value that no longer exists in the data).
function populateFacets() {
  const rows = healthRows();
  const buildOpt = (vals, cur, labelFn) =>
    [`<option value="all">all</option>`,
     ...vals.map(v => `<option value="${esc(v)}"${v === cur ? " selected" : ""}>${esc(labelFn ? labelFn(v) : v)}</option>`)].join("");
  const uniq = acc => [...new Set(rows.map(acc).filter(Boolean))];
  const fams = uniq(x => x.r.family).sort((a, b) => familyRank(a) - familyRank(b) || a.localeCompare(b));
  const set = (id, key, vals, labelFn) => {
    const el = $("#" + id); if (!el) return;
    if (!vals.includes(S.byType.facets[key])) S.byType.facets[key] = "all";   // drop stale selection
    el.innerHTML = buildOpt(vals, S.byType.facets[key], labelFn);
    el.value = S.byType.facets[key];
    // nothing to choose (0/1 distinct value in the data) -> hide the facet entirely
    const wrap = el.closest(".facet");
    if (wrap) wrap.hidden = vals.length <= 1;
  };
  set("facetFamily", "family", fams, familyLabel);
  set("facetMode", "mode", uniq(x => x.r.mode).sort(), null);
  set("facetDtype", "dtype", uniq(x => x.r.dtype).sort(), null);
  // arch isn't a facet: the table already groups by arch and the chart has its own arch selector.
}

// Every latest dev kernel/model, grouped by benchmark family -- the full picture,
// not just regressions. Same latest-value + noise-band trend as Health.
function renderByType() {
  const q = (S.byType.q || "").toLowerCase();
  const f = S.byType.facets;
  const rows = healthRows().filter(x => {
    const r = x.r;
    if (f.family !== "all" && (r.family || "other") !== f.family) return false;
    if (f.mode !== "all" && r.mode !== f.mode) return false;
    if (f.dtype !== "all" && r.dtype !== f.dtype) return false;
    if (f.model !== "all" && r.model !== f.model) return false;
    if (q && !`${r.op} ${r.shape} ${r.dtype} ${r.family || ""} ${r.model || ""}`.toLowerCase().includes(q)) return false;
    return true;
  });
  const anyFilter = q || Object.values(f).some(v => v !== "all");
  const openAttr = anyFilter ? " open" : "";   // folded by default; auto-open while filtering/faceting

  // group model -> family -> rows
  const byModel = new Map();
  for (const x of rows) {
    const a = x.r.model || "?";
    if (!byModel.has(a)) byModel.set(a, new Map());
    const fm = byModel.get(a), f = x.r.family || "other";
    (fm.get(f) || fm.set(f, []).get(f)).push(x);
  }
  const models = [...byModel.keys()].sort((a, b) => a.localeCompare(b));
  const nFam = new Set(rows.map(x => x.r.family || "other")).size;
  $("#typeSummary").innerHTML = rows.length
    ? `${rows.length} kernel × arch · <b>${models.length}</b> arch${models.length === 1 ? "" : "es"} · <b>${nFam}</b> type${nFam === 1 ? "" : "s"} · latest dev value + Δ vs prior-dev history`
    : "no results in snapshot";

  // aggregate (Σwork/Σtime) summary rows, filtered by the same facets, indexed by model|family
  const aggByMF = new Map();
  for (const x of aggRows()) {
    const r = x.r;
    if (f.family !== "all" && (r.family || "other") !== f.family) continue;
    if (f.mode !== "all" && r.mode !== f.mode) continue;
    if (f.dtype !== "all" && r.dtype !== f.dtype) continue;
    if (f.model !== "all" && r.model !== f.model) continue;
    if (q && !`${r.op} ${r.dtype} ${r.family || ""} ${r.model || ""}`.toLowerCase().includes(q)) continue;
    const key = `${r.model}|${r.family || "other"}`;
    (aggByMF.get(key) || aggByMF.set(key, []).get(key)).push(x);
  }

  // one collapsible <details> per benchmark family, for a given model's family-map
  const familyPanels = fm => {
    const fams = [...fm.keys()].sort((a, b) => familyRank(a) - familyRank(b) || a.localeCompare(b));
    return fams.map(fam => {
      const list = fm.get(fam).slice().sort((a, b) =>
        a.r.op.localeCompare(b.r.op) || a.r.shape.localeCompare(b.r.shape));
      const model = list[0].r.model;
      const aggs = (aggByMF.get(`${model}|${fam}`) || []).slice()
        .sort((a, b) => a.r.op.localeCompare(b.r.op) || a.r.dtype.localeCompare(b.r.dtype));
      const units = [...new Set(list.map(x => x.r.metric))].join(", ");
      // Σwork/Σtime rollups pinned atop the table -- clickable through to the aggregate trend
      const aggBody = aggs.map(({ r, vals, noise, d, real, sev }) => {
        const dcell = d == null ? `<td class="num k-dim">—</td>` : `<td class="num delta ${sev}">${fmtPct(d)}</td>`;
        return `<tr class="agg-row" data-k="${esc(kkey(r))}" data-model="${r.model}" title="work-weighted harmonic mean (Σwork ÷ Σtime) over ${r.n} configs">
          <td>${esc(r.op)} <span class="metric-tag agg-tag">Σ</span></td>
          <td class="k-dim">Σ aggregate · n=${r.n}</td>
          <td>${esc(r.dtype)}</td>
          <td style="color:${modelVar(r.model)}">${esc(r.model)}</td>
          <td class="spark-cell">${sparkline(vals, noise, real)}</td>
          <td class="num">${fmtVal(r.value, r.metric)}</td>
          ${dcell}
        </tr>`;
      }).join("");
      const body = list.map(({ r, vals, noise, d, real, sev }) => {
        const dcell = d == null ? `<td class="num k-dim">—</td>` : `<td class="num delta ${sev}">${fmtPct(d)}</td>`;
        return `<tr data-k="${esc(kkey(r))}" data-model="${r.model}">
          <td>${esc(r.op)} <span class="metric-tag">${r.metric}</span></td>
          <td class="k-dim">${esc(r.shape)}</td>
          <td>${esc(r.dtype)}</td>
          <td style="color:${modelVar(r.model)}">${esc(r.model)}</td>
          <td class="spark-cell">${sparkline(vals, noise, real)}</td>
          <td class="num">${fmtVal(r.value, r.metric)}</td>
          ${dcell}
        </tr>`;
      }).join("");
      return `<details class="panel type-panel"${openAttr}>
        <summary class="panel-head"><span class="type-caret" aria-hidden="true">▸</span><span class="t">${esc(familyLabel(fam))}</span>
          <span class="type-count">${list.length} kernel${list.length === 1 ? "" : "s"}</span>
          <span class="spacer"></span><span class="type-unit">${esc(units)}</span></summary>
        <div class="table-wrap"><table class="data"><thead><tr>
          <th>kernel</th><th>shape</th><th>dtype</th><th>arch</th><th>recent trend</th><th class="num">latest</th><th class="num">Δ vs dev</th>
        </tr></thead><tbody>${aggBody}${body}</tbody></table></div>
      </details>`;
    }).join("");
  };

  if (models.length <= 1) {
    // single model: family panels directly (no extra model layer)
    $("#typeSections").innerHTML = models.length
      ? familyPanels(byModel.get(models[0]))
      : `<div class="empty">no results in snapshot</div>`;
  } else {
    // multiple models: model (outer <details>) -> family (inner <details>)
    $("#typeSections").innerHTML = models.map(a => {
      const fm = byModel.get(a);
      const count = [...fm.values()].reduce((n, l) => n + l.length, 0);
      return `<details class="panel arch-panel"${openAttr}>` +
        `<summary class="panel-head"><span class="type-caret" aria-hidden="true">▸</span>` +
        `<span class="t arch-name" style="color:${modelVar(a)}">${esc(a)}</span>` +
        `<span class="type-count">${count} kernel${count === 1 ? "" : "s"}</span></summary>` +
        `<div class="arch-body">${familyPanels(fm)}</div>` +
        `</details>`;
    }).join("");
  }

  // Drive the All-Benchmarks chart from the tables: keep the current selection if it
  // survived the filter, else default to the first visible row (empty if none).
  const domRows = $$("#typeSections tr[data-k]");
  const bt = S.byType.trend;
  const present = new Set(domRows.map(tr => tr.dataset.k));
  if (bt.key && present.has(bt.key)) {
    selectKernel(bt.key, BYTYPE_CTX, false);
  } else if (domRows.length) {
    bt.model = "all";   // overview across arches; clicking a row drills into that arch
    selectKernel(domRows[0].dataset.k, BYTYPE_CTX, false);
  } else {
    selectKernel(null, BYTYPE_CTX, false);
  }
}

/* ---------------------------------------------------------- 3 · PR CHECK --- */
function prsWithData() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || !r.pr || r.metric === "speedup" || !r.vs_main) continue;
    if (!m.has(r.pr)) m.set(r.pr, { pr: r.pr, ts: r.ts, title: S.runMeta.get(r.run_id)?.title || "" });
    const e = m.get(r.pr); if ((r.ts || "") > (e.ts || "")) e.ts = r.ts;
  }
  return [...m.values()].sort((a, b) => b.pr - a.pr);
}
function renderPRSelect() {
  const prs = prsWithData();
  if (S.pr.sel == null && prs.length) S.pr.sel = prs[0].pr;
  $("#prSelect").innerHTML = prs.map(p =>
    `<option value="${p.pr}" ${p.pr === S.pr.sel ? "selected" : ""}>#${p.pr} — ${esc((p.title || "").slice(0, 60))}</option>`).join("")
    || `<option>no PR data</option>`;
}
function renderPRCheck() {
  renderPRSelect();
  const pr = S.pr.sel;
  const pane = $("#prPane");
  if (pr == null) { pane.innerHTML = `<div class="empty">no PR benchmark data in the snapshot yet</div>`; return; }
  // latest run of this PR
  const recs = S.records.filter(r => r.source === "ci" && r.pr === pr && r.vs_main && r.metric !== "speedup" && r.value != null);
  const latestRun = recs.reduce((a, r) => (r.ts || "") > (a.ts || "") ? r : a, { ts: "" }).run_id;
  const cur = recs.filter(r => r.run_id === latestRun);
  const rows = cur.map(r => {
    const { d, real } = regOf(r, mainBaseline(r.op, r.shape, r.dtype, r.model, r.metric));
    return { r, real, d, sev: sev(d, real) };
  }).sort((a, b) => a.d - b.d);
  const nbad = rows.filter(x => x.real).length;
  const nwatch = rows.filter(x => !x.real && x.d <= CFG.warnPct).length;
  $("#prSummary").innerHTML = `${rows.length} kernels · <b style="color:${nbad ? "var(--bad)" : "var(--good)"}">${nbad} real regression${nbad === 1 ? "" : "s"}</b> · ${nwatch} watch`;
  const runUrl = S.runMeta.get(latestRun)?.url || `https://github.com/${CFG.repo}/actions/runs/${latestRun}`;
  pane.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>
    <th>kernel</th><th>shape</th><th>dtype</th><th>arch</th><th class="num">PR</th><th class="num">dev</th><th class="num">Δ vs dev</th><th>baseline</th>
    </tr></thead><tbody>${rows.map(({ r, d, sev }) => `<tr class="${sev === "bad" ? "row-bad" : ""}">
      <td>${esc(r.op)} <span class="metric-tag">${r.metric}</span></td><td class="k-dim">${esc(r.shape)}</td><td>${esc(r.dtype)}</td>
      <td style="color:${modelVar(r.model)}">${esc(r.model)}</td>
      <td class="num">${fmtVal(r.value, r.metric)}</td>
      <td class="num k-dim">${fmtVal(r.vs_main.baseline, r.metric)}</td>
      <td class="num delta ${sev}">${fmtPct(d)}</td>
      <td class="k-dim">${esc(r.vs_main.label || "dev")}</td></tr>`).join("")
    || `<tr><td colspan="8" class="empty">no vs-dev rows for this PR run</td></tr>`}</tbody></table>
    <div class="noise-note" style="padding:10px 12px">latest run of #${pr} · <a href="${esc(runUrl)}" target="_blank" rel="noopener">view on GitHub</a> · “real” = beyond the kernel’s run-to-run noise on dev</div></div>`;
}

/* --------------------------------------------------- 2b · BACKEND GAPS --- */
// Native reference backend per family; every other backend is measured against it.
const GAP_REF_BACKEND = { gemm: "hipblaslt", grouped_gemm: "hipblaslt", casting: "hip", normalization: "hip" };

// Latest-dev cross-backend comparison. For each config (arch, family, base, shape,
// dtype, metric) where the family's reference backend has a value, measure how far
// each other backend's throughput trails it, then roll up per (arch, family, backend).
// Throughput is higher-is-better, so a negative gap % means slower than the reference.
function backendGapFindings() {
  const groups = new Map();                         // config -> { backend: record }
  for (const x of healthRows()) {
    const r = x.r;
    if (!(r.value > 0)) continue;
    const k = [r.model, r.family, r.base, r.shape, r.dtype, r.metric].join("\u0000");
    (groups.get(k) || groups.set(k, {}).get(k))[r.backend] = r;
  }
  const cand = new Map();                            // arch\0family\0backend -> [{gap, r, ref}]
  for (const byBackend of groups.values()) {
    const any = Object.values(byBackend)[0];
    const ref = GAP_REF_BACKEND[any.family];
    const refR = ref && byBackend[ref];
    if (!refR || !(refR.value > 0)) continue;
    for (const [b, r] of Object.entries(byBackend)) {
      if (b === ref) continue;
      const gap = (r.value / refR.value - 1) * 100;
      const ck = [r.model, any.family, b].join("\u0000");
      (cand.get(ck) || cand.set(ck, []).get(ck)).push({ gap, r, refR, ref });
    }
  }
  const findings = [];
  for (const [ck, list] of cand) {
    const [model, family, backend] = ck.split("\u0000");
    const gaps = list.map(e => e.gap).sort((a, b) => a - b);
    const median = gaps[Math.floor((gaps.length - 1) / 2)];
    const worst = list.reduce((a, e) => e.gap < a.gap ? e : a, list[0]);
    const nSlower = list.filter(e => e.gap <= -CFG.gapPct).length;
    // work-weighted aggregate throughput (Σ(thr·ms)/Σms) for backend vs ref, over timed shapes
    let wb = 0, tb = 0, wr = 0, tr = 0;
    for (const e of list) {
      const tB = e.r.extra && e.r.extra.median_ms, tR = e.refR.extra && e.refR.extra.median_ms;
      if (!(tB > 0) || !(tR > 0)) continue;
      wb += e.r.value * tB; tb += tB;
      wr += e.refR.value * tR; tr += tR;
    }
    const aggGap = (tb > 0 && tr > 0) ? ((wb / tb) / (wr / tr) - 1) * 100 : median;
    findings.push({ model, family, backend, ref: list[0].ref, n: list.length, aggGap, median, nSlower, worst });
  }
  // surface a combo when it's slower overall (work-weighted) or on a big share of shapes
  return findings
    .filter(f => f.aggGap <= -CFG.gapPct || f.nSlower >= Math.max(2, Math.ceil(f.n * 0.3)))
    .sort((a, b) => a.aggGap - b.aggGap);
}

function renderBackendGaps() {
  const findings = backendGapFindings();
  const sum = $("#gapsSummary");
  if (sum) sum.innerHTML = findings.length
    ? `${findings.length} backend\u00d7family\u00d7arch ${findings.length === 1 ? "combo" : "combos"} > ${CFG.gapPct}% slower than native`
    : `nothing > ${CFG.gapPct}% slower than native`;
  const pane = $("#gapsList"); if (!pane) return;
  if (!findings.length) {
    pane.innerHTML = `<div class="reg-list-empty"><div class="big">` +
      `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3.2L13 5"/></svg>` +
      `all backends within ${CFG.gapPct}% of native</div>no backend trails its family reference (hip / hipBLASLt) by more than ${CFG.gapPct}%.</div>`;
    return;
  }
  pane.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>` +
    `<th>backend</th><th>family</th><th>arch</th><th class="num" title="work-weighted aggregate throughput (Σwork ÷ Σtime) vs reference">overall vs ref</th><th class="num">median</th><th class="num">shapes ${CFG.gapPct}%+ slower</th><th>worst shape</th>` +
    `</tr></thead><tbody>${findings.map(f => {
      const w = f.worst;
      return `<tr data-k="${esc(kkey(w.r))}" data-model="${f.model}" title="open the backend comparison chart for the worst shape">
        <td>${esc(f.backend)} <span class="metric-tag">vs ${esc(f.ref)}</span></td>
        <td>${esc(familyLabel(f.family))}</td>
        <td style="color:${modelVar(f.model)}">${esc(f.model)}</td>
        <td class="num delta ${f.aggGap <= -CFG.gapPct ? "bad" : "warn"}">${fmtPct(f.aggGap)}</td>
        <td class="num k-dim">${fmtPct(f.median)}</td>
        <td class="num">${f.nSlower}/${f.n}</td>
        <td class="k-dim">${fmtPct(w.gap)} on ${esc(w.r.shape)} · ${esc(w.r.dtype)}</td>
      </tr>`;
    }).join("")}</tbody></table></div>`;
}

/* --------------------------------------------------- 2c · FEATURE GAPS --- */
// Declared backend capabilities (from the harness backend-selection code + kernel
// support). Edit as backends gain/lose support; a backend with no entry falls back to
// its observed coverage. A reference-covered config a backend can't satisfy is a gap,
// attributed to the first violated axis: dtype, then arch, then shape (a supported
// dtype+arch whose specific shapes weren't covered).
const BACKEND_CAPS = {
  gemm:         { triton: { dtypes: ["bf16"], archs: ["MI355X"] } },
  grouped_gemm: {
    ck_tile:    { dtypes: ["bf16", "fp8"] },
    hipkittens: { dtypes: ["mxfp8"], archs: ["MI355X", "MI455X"] },
    triton:     { dtypes: ["bf16"] },
  },
  casting:      { triton: { dtypes: ["fp8-e4m3", "fp8-e5m2", "mxfp4", "mxfp8-e4m3", "mxfp8-e5m2"] } },
};

function featureGapFindings() {
  const cfgOf = r => [r.model, r.base, r.shape, r.dtype].join("\u0000");
  const covered = new Map();                          // family -> backend -> Set(configKey)
  for (const x of healthRows()) {
    const r = x.r; if (!(r.value > 0)) continue;
    const fam = covered.get(r.family) || covered.set(r.family, new Map()).get(r.family);
    (fam.get(r.backend) || fam.set(r.backend, new Set()).get(r.backend)).add(cfgOf(r));
  }
  const findings = [];
  for (const [family, byBackend] of covered) {
    const ref = GAP_REF_BACKEND[family];
    const refSet = byBackend.get(ref);
    if (!refSet) continue;
    const caps = BACKEND_CAPS[family] || {};
    for (const [backend, set] of byBackend) {
      if (backend === ref) continue;
      const cap = caps[backend] || {};
      // declared caps if present, else the backend's own observed dtypes/archs
      const dtypes = cap.dtypes ? new Set(cap.dtypes) : new Set([...set].map(k => k.split("\u0000")[3]));
      const archs = cap.archs ? new Set(cap.archs) : new Set([...set].map(k => k.split("\u0000")[0]));
      const buckets = new Map();                       // reason\0detail -> config count
      for (const key of refSet) {
        if (set.has(key)) continue;
        const [arch, , , dtype] = key.split("\u0000");
        const bk = !dtypes.has(dtype) ? `dtype\u0000${dtype}`
                 : !archs.has(arch) ? `arch\u0000${arch}`
                 : `shape\u0000${dtype}`;
        buckets.set(bk, (buckets.get(bk) || 0) + 1);
      }
      for (const [bk, n] of buckets) {
        const [reason, detail] = bk.split("\u0000");
        findings.push({ family, backend, reason, detail, n });
      }
    }
  }
  const rank = { dtype: 0, arch: 1, shape: 2 };
  return findings.sort((a, b) => (rank[a.reason] - rank[b.reason]) || b.n - a.n);
}

function renderFeatureGaps() {
  const findings = featureGapFindings();
  const sum = $("#featSummary");
  if (sum) sum.innerHTML = findings.length
    ? `${findings.length} gaps · ${findings.reduce((a, f) => a + f.n, 0)} configs not covered`
    : "no feature gaps";
  const pane = $("#featList"); if (!pane) return;
  if (!findings.length) {
    pane.innerHTML = `<div class="reg-list-empty"><div class="big">every backend covers its reference's configs</div></div>`;
    return;
  }
  const detailText = f => f.reason === "shape"
    ? `partial — some ${esc(f.detail)} shapes`
    : `no ${esc(f.detail)}`;
  pane.innerHTML = `<div class="table-wrap"><table class="data"><thead><tr>` +
    `<th>backend</th><th>family</th><th>reason</th><th>missing</th><th class="num">configs</th>` +
    `</tr></thead><tbody>${findings.map(f => `<tr>
      <td>${esc(f.backend)}</td>
      <td>${esc(familyLabel(f.family))}</td>
      <td><span class="metric-tag">${f.reason}</span></td>
      <td class="k-dim">${detailText(f)}</td>
      <td class="num">${f.n}</td>
    </tr>`).join("")}</tbody></table></div>`;
}

/* ------------------------------------------------------------ 3 · TRENDS --- */
function kernelIndex() {
  const regKeys = new Set(healthRows().filter(x => x.real).map(x => kkey(x.r)));
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.value == null) continue;
    const k = kkey(r);
    if (!m.has(k)) m.set(k, { op: r.op, base: r.base, shape: r.shape, dtype: r.dtype, metrics: new Set(), runs: new Set(), reg: false, aggregate: !!r.aggregate });
    const e = m.get(k); e.metrics.add(r.metric); e.runs.add(r.run_id);
  }
  for (const [k, e] of m) { e.reg = regKeys.has(k); e.n = e.runs.size; }
  return m;
}
function renderKernelRail() {
  const idx = kernelIndex();
  const agg = S.trend.railMode === "aggregates";
  $("#railMode").innerHTML = [["kernels", "kernels"], ["aggregates", "\u03a3 aggregates"]].map(([v, t]) =>
    `<button data-rm="${v}" class="${v === S.trend.railMode ? "is-active" : ""}">${t}</button>`).join("");
  let keys = [...idx.keys()].filter(k => !!idx.get(k).aggregate === agg);
  if (S.trend.q) { const q = S.trend.q.toLowerCase(); keys = keys.filter(k => k.toLowerCase().includes(q)); }
  keys.sort();
  // default: the worst confirmed regression on dev; if there is none, leave the chart empty.
  if (!S.trend.key) {
    const worst = healthRows().filter(x => x.real).sort((a, b) => a.d - b.d)[0];
    const worstK = worst ? kkey(worst.r) : null;
    selectKernel(worstK && keys.includes(worstK) ? worstK : null, HEALTH_CTX, false);
  }
  $("#kernelList").innerHTML = keys.map(k => {
    const e = idx.get(k);
    const sub = e.aggregate ? `\u03a3 ${esc(e.dtype)} · work-weighted H-mean` : `${esc(e.shape)} · ${esc(e.dtype)}`;
    return `<button class="kitem ${k === S.trend.key ? "is-active" : ""} ${e.reg ? "has-reg" : ""}" data-k="${esc(k)}">
      ${esc(e.op)}<span class="ks">${sub}</span></button>`;
  }).join("") || `<div class="empty" style="padding:20px">no ${agg ? "aggregates" : "match"}</div>`;
}
// Each tab hosts its own trend chart: Health drives from S.trend (+ the kernel rail),
// All Benchmarks from S.byType.trend (+ the benchmark tables). A ctx bundles the
// per-tab state, DOM element ids, and the live Chart.js instance so selectKernel /
// drawTrend render into either without touching the other.
const HEALTH_CTX = { state: S.trend, hasRail: true, chart: null, hash: true, ids: {
  title: "trendTitle", chart: "trendChart", by: "trendBy", model: "trendModel",
  range: "trendRange", xmode: "trendXMode", note: "noiseNote", headRow: "trendHeadRow", body: "trendBody" } };
const BYTYPE_CTX = { state: S.byType.trend, hasRail: false, chart: null, hash: false, ids: {
  title: "btTitle", chart: "btChart", by: "btBy", model: "btModel",
  range: "btRange", xmode: "btXMode", note: "btNote", headRow: "btHeadRow", body: "btBody" } };
const cq = (ctx, name) => document.getElementById(ctx.ids[name]);

// Empty state: tear the chart down and blank the title/note/table (Health with no
// regression to show, or All Benchmarks with no rows in the current filter).
function clearChart(ctx, msg) {
  if (ctx.chart) { ctx.chart.destroy(); ctx.chart = null; }
  const t = cq(ctx, "title"); if (t) t.innerHTML = esc(msg);
  const n = cq(ctx, "note"); if (n) n.innerHTML = "";
  const hr = cq(ctx, "headRow"); if (hr) hr.innerHTML = "<th>commit</th><th>date</th><th>pr</th>";
  const b = cq(ctx, "body"); if (b) b.innerHTML = "";
}

function selectKernel(k, ctx = HEALTH_CTX, rerail = true) {
  const st = ctx.state;
  const idx = kernelIndex(); const e = k ? idx.get(k) : null;
  if (!e) {
    st.key = null;
    clearChart(ctx, ctx.hasRail ? "No confirmed regressions on dev" : "Select a benchmark");
    if (ctx.hash) writeHash();
    return;
  }
  st.key = k;
  const metrics = [...e.metrics];
  // one metric per kernel: auto-select it (the unit is shown in the trend title)
  if (!metrics.includes(st.metric)) st.metric = metrics.find(m => m !== "speedup") || metrics[0];
  const byBackend = st.by === "backend";
  // backend mode compares backends at one arch, so drop the "all" arch option
  if (byBackend && (st.model === "all" || !S.models.includes(st.model))) st.model = S.models[0] || "all";
  cq(ctx, "by").innerHTML = [["arch", "by arch"], ["backend", "by backend"]].map(([v, t]) =>
    `<button data-b="${v}" class="${v === st.by ? "is-active" : ""}">${t}</button>`).join("");
  cq(ctx, "model").innerHTML = (byBackend ? S.models : ["all", ...S.models]).map(a =>
    `<button data-a="${a}" class="${a === st.model ? "is-active" : ""}">${esc(a)}</button>`).join("");
  cq(ctx, "range").innerHTML = [["7d", "7 days"], ["30d", "30 days"], ["all", "all"]].map(([v, t]) =>
    `<button data-r="${v}" class="${v === st.range ? "is-active" : ""}">${t}</button>`).join("");
  cq(ctx, "xmode").innerHTML = [["commits", "by commit"], ["daily", "by day"]].map(([v, t]) =>
    `<button data-x="${v}" class="${v === st.xmode ? "is-active" : ""}">${t}</button>`).join("");
  cq(ctx, "title").innerHTML = byBackend
    ? `${esc(e.base)} <small>${esc(e.aggregate ? "\u03a3 aggregate" : e.shape)} · ${esc(e.dtype)} · ${esc(st.model)} · ${st.metric} · backends</small>`
    : e.aggregate
      ? `${esc(e.op)} <small>\u03a3 work-weighted H-mean · ${esc(e.dtype)} · ${st.metric}</small>`
      : `${esc(e.op)} <small>${esc(e.shape)} · ${esc(e.dtype)} · ${st.metric}</small>`;
  if (ctx.hasRail && rerail) $$("#kernelList .kitem").forEach(b => b.classList.toggle("is-active", b.dataset.k === k));
  drawTrend(e, ctx);
  if (ctx.hash) writeHash();
}
function drawTrend(e, ctx = HEALTH_CTX) {
  const st = ctx.state;
  const metric = st.metric, shape = e.shape, dtype = e.dtype;
  const byBackend = st.by === "backend";
  const base = e.base, arch = (st.model === "all") ? (S.models[0] || "") : st.model;
  // Overlay dimension: archs of one exact op (default), or backends of one op-family
  // (base) at a single arch. opFor() maps a dim value back to its full op for baselines.
  const dimOf = byBackend ? (r => r.backend) : (r => r.model);
  const dimCol = byBackend ? backendCol : modelCol;
  const opFor = byBackend ? (v => `${base} · ${v}`) : (() => e.op);
  const recs = S.records.filter(r => r.source === "ci" && r.shape === shape && r.dtype === dtype && r.metric === metric &&
    (byBackend ? (r.base === base && r.model === arch) : r.op === e.op));
  // aggregate rollups carry a per-run config count (n) + shape signature (sig); used below to
  // annotate points and flag config-set changes (single-arch only; skipped in backend overlay).
  const aggInfo = (e.aggregate && !byBackend) ? new Map(recs.map(r => [r.run_id, { n: r.n, sig: r.sig }])) : null;
  let runIds = [...new Set(recs.map(r => r.run_id))].map(id => {
    const any = recs.find(r => r.run_id === id);
    return { id, ts: any.ts, commit: any.commit, pr: any.pr, main: isMainRec(any) };
  }).sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  // time-range filter. range is "<N>d" (last N days) or "all" (no filter).
  const days = st.range === "all" ? Infinity : parseInt((st.range.match(/\d+/) || ["30"])[0], 10);
  if (Number.isFinite(days)) {
    const cutoff = Date.now() - days * 86400000;
    runIds = runIds.filter(ri => !ri.ts || new Date(ri.ts).getTime() >= cutoff);
  }
  const val = new Map();                       // run_id|dim -> value
  for (const r of recs) val.set(r.run_id + "|" + dimOf(r), r.value);

  // x-axis points: one per commit, or one per day (daily mean) when xmode=daily
  const daily = st.xmode === "daily";
  let points;
  if (daily) {
    const byDay = new Map();
    for (const ri of runIds) { const d = (ri.ts || "").slice(0, 10); (byDay.get(d) || byDay.set(d, []).get(d)).push(ri); }
    points = [...byDay.keys()].sort().map(d => {
      const rs = byDay.get(d), last = rs[rs.length - 1];
      return { date: d, dateLabel: d.slice(5), sha: "", commit: last.commit, pr: last.pr, main: rs.some(x => x.main), runs: rs };
    });
  } else {
    points = runIds.map(ri => ({ date: (ri.ts || "").slice(0, 10), dateLabel: (ri.ts || "").slice(5, 10), sha: (ri.commit || "").slice(0, 7), commit: ri.commit, pr: ri.pr, main: ri.main, id: ri.id }));
  }
  const labels = points.map((p, i) => p.sha || p.dateLabel || String(i));
  const valueAt = (p, dv) => {
    if (daily) { const vs = p.runs.map(ri => val.get(ri.id + "|" + dv)).filter(v => v != null); return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null; }
    return val.get(p.id + "|" + dv) ?? null;
  };
  // single-arch mode draws a noise band; multi-arch and backend modes overlay for comparison
  const single = !byBackend && st.model !== "all";
  // backend mode lists only backends that actually ran this config+arch (drop absent ones)
  const dims = single ? [st.model]
    : byBackend ? [...new Set(recs.map(r => r.backend))].sort()
    : S.models;
  if (aggInfo) {                              // tag points with config count + set-change flag
    let prev = null;
    for (const p of points) {
      const info = aggInfo.get(p.id) || (p.runs && aggInfo.get(p.runs[p.runs.length - 1].id));
      p.aggN = info ? info.n : null;
      p.aggChanged = single && !!info && prev != null && info.sig !== prev;
      if (info) prev = info.sig;
    }
  }

  const datasets = [];
  let note = "";
  const span = `${points.length} ${daily ? "day" + (points.length === 1 ? "" : "s") : "commits"}` +
    (st.range === "all" ? "" : ` · last ${days}d`);
  if (single) {
    const noise = mainBaseline(e.op, shape, dtype, st.model, metric);
    if (noise.lo != null && noise.relStd != null && noise.n >= CFG.minSamples) {
      datasets.push({ label: "+2σ", data: points.map(() => noise.hi), borderColor: "transparent", pointRadius: 0, fill: "+1", backgroundColor: cssVal("--band"), order: 20 });
      datasets.push({ label: "-2σ", data: points.map(() => noise.lo), borderColor: "transparent", pointRadius: 0, fill: false, order: 20 });
      datasets.push({ label: "dev mean", data: points.map(() => noise.mean), borderColor: cssVal("--ink-4"), borderDash: [4, 4], borderWidth: 1, pointRadius: 0, order: 19 });
      note = `${span} · band = prior-dev mean ${fmtVal(noise.mean, metric)} ± ${CFG.noiseK}σ (σ≈<b>${noise.relStd.toFixed(1)}%</b>, n=${noise.n}).` +
        (daily ? " Daily mean smooths run-to-run jitter." : " A point below the band is a real regression.");
    } else {
      note = `${span} · n=${noise.n} prior dev runs — too few for a noise band; fixed <b>${CFG.regressionPct}%</b> gate.`;
    }
  } else {
    const unit = byBackend ? "backend" : "model";
    note = `${span} · one line per ${unit}.` + (daily ? ` Daily mean per ${unit} — smooths CI jitter to expose real drift.` : " Red = dev below its prior-dev band, or a PR slower than dev.");
  }
  if (aggInfo) {
    const nNow = points.length ? points[points.length - 1].aggN : null;
    const changes = points.filter(p => p.aggChanged).length;
    note = `${span} · work-weighted harmonic mean (Σwork ÷ Σtime)${nNow != null ? `, n=${nNow} configs` : ""}.` +
      (single ? (changes ? ` Amber = config set changed (${changes}×) — values across a change aren't directly comparable.` : " Config set stable across this range.")
              : " Select a single arch to flag config-set changes.");
  }
  for (const dv of dims) {
    const noise = mainBaseline(opFor(dv), shape, dtype, byBackend ? arch : dv, metric);
    const data = points.map(p => valueAt(p, dv));
    if (data.every(v => v == null)) continue;
    const ptColor = points.map(p => {
      if (aggInfo && p.aggChanged) return AGG_WARN;   // config set changed at this point
      if (daily) return dimCol(dv);             // daily means aren't per-run regression calls
      const r = recs.find(x => x.run_id === p.id && dimOf(x) === dv);
      return (r && regOf(r, noise).real) ? cssVal("--bad") : dimCol(dv);
    });
    datasets.push({
      label: dv, data, borderColor: dimCol(dv), backgroundColor: dimCol(dv) + "22",
      pointBackgroundColor: ptColor, pointBorderColor: ptColor,
      pointRadius: daily ? 4 : points.length > 30 ? 1.5 : 3, pointHoverRadius: 6,
      borderWidth: daily ? 2.4 : 2, tension: .25, spanGaps: true, order: 1, fill: single ? "origin" : false,
    });
  }
  cq(ctx, "note").innerHTML =
    `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.3v4M8 5v.01" stroke-linecap="round"/></svg>` + note;

  const tickCol = cssVal("--ink-3"), gridCol = cssVal("--grid");
  const canvasEl = cq(ctx, "chart");
  // Robust teardown: destroy our instance AND any chart still bound to this canvas
  // (Chart.getChart), so a re-entrant/interrupted redraw can't leave the canvas
  // "already in use" -> a blank chart that only a full page reload clears.
  if (ctx.chart) { ctx.chart.destroy(); ctx.chart = null; }
  const bound = (window.Chart && Chart.getChart) ? Chart.getChart(canvasEl) : null;
  if (bound) bound.destroy();
  ctx.chart = new Chart(canvasEl, {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 220 },
      interaction: { mode: "index", intersect: false },
      onHover: (ev, els) => { if (ev.native) ev.native.target.style.cursor = els.length ? "pointer" : "default"; },
      onClick: (ev, els) => { if (!els.length) return; const p = points[els[0].index]; if (p && p.commit) window.open(commitUrl(p.commit), "_blank", "noopener"); },
      plugins: {
        legend: { labels: { color: cssVal("--ink-2"), font: { family: "IBM Plex Mono", size: 11 }, boxWidth: 10, usePointStyle: true, filter: i => !/σ|mean/.test(i.text) } },
        tooltip: {
          backgroundColor: cssVal("--bg-2"), borderColor: cssVal("--border"), borderWidth: 1, titleColor: cssVal("--ink"), bodyColor: cssVal("--ink-2"),
          titleFont: { family: "IBM Plex Mono" }, bodyFont: { family: "IBM Plex Mono" },
          // Drop the band/mean helper lines AND null (gap) points. With sparse
          // multi-model data, hovering a gap x-position otherwise yields an empty
          // items array -> the title callback threw on items[0], which broke
          // Chart.js's draw loop (blank chart until reload).
          filter: i => i && i.dataset && !/σ|mean/.test(i.dataset.label) && i.parsed && i.parsed.y != null,
          callbacks: {
            title: items => {
              const it = items && items[0];
              const p = it && points[it.dataIndex];
              if (!p) return "";
              const head = daily ? `${p.date} · ${p.runs.length} run${p.runs.length === 1 ? "" : "s"}` : `${p.sha} · ${p.main ? "dev" : "#" + p.pr} · click to open`;
              return (aggInfo && p.aggN != null) ? `${head} · n=${p.aggN}${p.aggChanged ? " · config set changed" : ""}` : head;
            },
            label: i => (i && i.dataset && i.parsed && i.parsed.y != null) ? ` ${i.dataset.label}: ${fmtVal(i.parsed.y, metric)} ${metric}` : "",
          },
        },
      },
      scales: {
        x: { grid: { color: gridCol }, ticks: { color: tickCol, font: { family: "IBM Plex Mono", size: 10 }, maxRotation: 0, autoSkipPadding: 12,
          callback: function (v, i) { const p = points[i]; return p ? (p.sha ? [p.dateLabel, p.sha] : [p.dateLabel]) : v; } } },
        y: { grid: { color: gridCol }, ticks: { color: tickCol, font: { family: "IBM Plex Mono", size: 10 } },
          title: { display: true, text: metric, color: tickCol, font: { family: "IBM Plex Mono", size: 10 } } },
      },
    },
  });
  // status-aware table — always per-commit, with a real link to each commit.
  // Columns are the overlay dimension (archs, or backends in backend mode) so the
  // header stays aligned with the data cells below.
  const cols = byBackend ? dims : S.models;
  cq(ctx, "headRow").innerHTML = `<th>commit</th><th>date</th><th>pr</th>` +
    cols.map(c => `<th class="num">${esc(c)}</th>`).join("");
  cq(ctx, "body").innerHTML = runIds.slice().reverse().map(ri => {
    const cell = dv => {
      const r = recs.find(x => x.run_id === ri.id && dimOf(x) === dv)
        || S.records.find(x => x.run_id === ri.id && dimOf(x) === dv && x.shape === shape && x.dtype === dtype && x.metric === metric &&
             (byBackend ? (x.base === base && x.model === arch) : x.op === e.op));
      if (!r) return `<td class="num st-na">—</td>`;
      if (r.value == null) return `<td class="num cell-status ${r.status === "skip" ? "st-skip" : "st-missing"}">${r.status}</td>`;
      const real = regOf(r, mainBaseline(opFor(dv), shape, dtype, byBackend ? arch : dv, metric)).real;
      return `<td class="num" style="color:${real ? "var(--bad)" : dimCol(dv)}">${fmtVal(r.value, metric)}</td>`;
    };
    const sha = (ri.commit || "").slice(0, 7);
    return `<tr><td><a class="commit-link" href="${commitUrl(ri.commit)}" target="_blank" rel="noopener">${sha || "—"}</a></td>` +
      `<td class="k-dim">${(ri.ts || "").slice(0, 10)}</td>` +
      `<td class="k-dim">${ri.main ? "dev" : ri.pr ? `<a href="https://github.com/${CFG.repo}/pull/${ri.pr}" target="_blank" rel="noopener">#${ri.pr}</a>` : "branch"}</td>` +
      `${cols.map(cell).join("")}</tr>`;
  }).join("");
  // If the chart was built while its container was briefly unsized (view switch or
  // first paint), force a resize on the next frame so it isn't left blank.
  const _ch = ctx.chart;
  requestAnimationFrame(() => { if (_ch && _ch === ctx.chart) ctx.chart.resize(); });
}

/* ------------------------------------------------------------- 4 · BOARD --- */
function chipState(j) { if (!j) return "none"; if (j.status && j.status !== "completed") return "running"; return j.conclusion || "none"; }
function worstDeltaForRun(runId) {
  let worst = null, kernel = null;
  for (const r of S.records) {
    if (r.run_id !== runId || !r.vs_main || r.metric === "speedup") continue;
    if (worst == null || r.vs_main.delta_pct < worst) { worst = r.vs_main.delta_pct; kernel = r.op; }
  }
  return worst == null ? null : { worst, kernel };
}
function renderBoard() {
  const grid = $("#boardGrid");
  const groups = new Map();
  for (const r of S.runs) {
    const key = r.pr ? `pr:${r.pr}` : `br:${r.branch}:${r.commit}`;
    const ex = groups.get(key);
    if (!ex || (r.created_at || "") > (ex.created_at || "")) groups.set(key, r);
  }
  let list = [...groups.values()];
  const f = S.boardFilter;
  if (f === "pr") list = list.filter(r => r.pr);
  else if (f === "main") list = list.filter(r => !r.pr && r.branch === "main");
  else if (f === "active") list = list.filter(r => r.status !== "completed" || (r.jobs || []).some(j => j.status !== "completed"));
  list.sort((a, b) => (a.status !== "completed" ? 0 : 1) - (b.status !== "completed" ? 0 : 1) || (b.created_at || "").localeCompare(a.created_at || ""));

  const jobs = list.flatMap(r => r.jobs || []);
  $("#boardCounts").innerHTML =
    `<span class="c-warn"><b>${jobs.filter(j => j.status !== "completed").length}</b>running</span>` +
    `<span class="c-good"><b>${jobs.filter(j => j.conclusion === "success").length}</b>passed</span>` +
    `<span class="c-bad"><b>${jobs.filter(j => j.conclusion === "failure").length}</b>failed</span>` +
    `<span><b>${list.length}</b><i>runs</i></span>`;

  if (!list.length) { grid.innerHTML = `<div class="empty">no recent runs in snapshot</div>`; return; }
  grid.innerHTML = list.map(r => {
    const byRunner = {}; for (const j of (r.jobs || [])) byRunner[j.model] = j;
    const overall = (r.status !== "completed") ? "running" : (r.conclusion || "none");
    const chips = S.models.map(model => {
      const j = byRunner[model]; const st = chipState(j);
      const label = st === "running" ? "run" : st === "success" ? "pass" : st === "failure" ? "fail" : st === "none" ? "—" : st.slice(0, 4);
      const link = j?.url ? `<a href="${esc(j.url)}" target="_blank" rel="noopener" title="${model} · ${st}"></a>` : "";
      return `<div class="chip" data-c="${esc(st)}"><span class="arch">${esc(model)}</span><span class="st">${esc(label)}</span>${link}</div>`;
    }).join("");
    // vs_main is only a real diff for PR runs (a main run rebuilds its own commit as baseline).
    const wd = r.pr ? worstDeltaForRun(r.run_id) : null;
    const wsev = wd == null ? "none" : wd.worst <= CFG.regressionPct ? "bad" : wd.worst <= CFG.warnPct ? "warn" : "ok";
    const perf = `<div class="pr-perf"><span class="lab">${r.pr ? "worst Δ vs dev" : "branch"}</span>` +
      (!r.pr ? `<span class="worst none">${esc(r.branch || "—")} commit</span>`
        : wd == null ? `<span class="worst none">no perf data</span>`
          : `<span class="worst ${wsev}">${fmtPct(wd.worst)}</span><span class="lab">${esc(wd.kernel)}</span>`) + `</div>`;
    const who = r.pr ? `#${r.pr}` : esc(r.branch || "—");
    return `<div class="pr-card ${overall}">
      <div class="pr-top"><span class="pr-num">${who}</span><span class="pr-event">${esc(r.event || "")}</span>
        <span class="spacer"></span><span class="pr-meta" style="margin:0">${relTime(r.created_at)}</span></div>
      <a class="pr-title" href="${esc(r.url || "#")}" target="_blank" rel="noopener" style="color:inherit">${esc(r.title || r.branch || "")}</a>
      <div class="pr-meta"><span>@${esc(r.actor || "?")}</span><a class="commit-link" href="${commitUrl(r.commit)}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${esc((r.commit || "").slice(0, 7))}</a></div>
      ${perf}<div class="chips">${chips}</div></div>`;
  }).join("");
}

/* ------------------------------------------------------------------ shell -- */
function renderAll() { renderModelLegend(); renderHealth(); populateFacets(); renderByType(); renderBackendGaps(); renderFeatureGaps(); renderKernelRail(); renderPRCheck(); renderBoard(); }

// --- shareable URL state: the location hash carries the active view AND its
// selection (trend kernel/metric/model/range/x, or by-type facets/search, or the
// selected PR) so a copied link restores exactly what you're looking at.
function writeHash() {
  const p = new URLSearchParams();
  if (S.view === "health") {
    if (S.trend.key) p.set("k", S.trend.key);
    if (S.trend.metric) p.set("m", S.trend.metric);
    if (S.trend.model && S.trend.model !== "all") p.set("model", S.trend.model);
    if (S.trend.range && S.trend.range !== "all") p.set("r", S.trend.range);
    if (S.trend.xmode && S.trend.xmode !== "commits") p.set("x", S.trend.xmode);
    if (S.trend.by && S.trend.by !== "arch") p.set("by", S.trend.by);
    if (S.trend.railMode && S.trend.railMode !== "kernels") p.set("rail", S.trend.railMode);
  } else if (S.view === "bytype") {
    for (const [k, v] of Object.entries(S.byType.facets)) if (v && v !== "all") p.set(k, v);
    if (S.byType.q) p.set("q", S.byType.q);
  } else if (S.view === "prcheck" && S.pr.sel != null) {
    p.set("pr", String(S.pr.sel));
  }
  const qs = p.toString();
  const h = "#" + S.view + (qs ? "?" + qs : "");
  if (location.hash !== h) history.replaceState(null, "", h);
}
function readHash() {
  const raw = location.hash.slice(1);
  const qi = raw.indexOf("?");
  const view = (qi >= 0 ? raw.slice(0, qi) : raw) || "health";
  const p = new URLSearchParams(qi >= 0 ? raw.slice(qi + 1) : "");
  if (VIEWS.includes(view)) S.view = view;
  if (S.view === "health") {
    if (p.has("k")) S.trend.key = p.get("k");
    if (p.has("m")) S.trend.metric = p.get("m");
    S.trend.model = p.get("model") || "all";
    S.trend.range = p.get("r") || "all";
    S.trend.xmode = p.get("x") || "commits";
    S.trend.by = p.get("by") || "arch";
    S.trend.railMode = p.get("rail") || "kernels";
  } else if (S.view === "bytype") {
    S.byType.facets = {
      family: p.get("family") || "all", mode: p.get("mode") || "all",
      dtype: p.get("dtype") || "all", model: p.get("model") || "all",
    };
    S.byType.q = p.get("q") || "";
  } else if (S.view === "prcheck" && p.has("pr")) {
    S.pr.sel = +p.get("pr");
  }
}
// Re-render the active view after the hash changes out-of-band (a link pasted
// into the same tab while the page is already open).
function applyState() {
  showView(S.view);
  if (S.view === "health") {
    if (S.trend.key && kernelIndex().get(S.trend.key)) selectKernel(S.trend.key, HEALTH_CTX);
    else { S.trend.key = null; renderKernelRail(); }
  } else if (S.view === "bytype") {
    $("#typeSearch").value = S.byType.q;
    populateFacets(); renderByType();
  } else if (S.view === "gaps") {
    renderBackendGaps(); renderFeatureGaps();
  } else if (S.view === "prcheck") {
    renderPRCheck();
  }
}
function showView(v) {
  if (v === "trends") v = "health";
  if (!VIEWS.includes(v)) v = "health";
  S.view = v;
  $$(".tab").forEach(t => { const on = t.dataset.view === v; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on ? "true" : "false"); });
  $$(".view").forEach(s => s.classList.toggle("is-active", s.dataset.view === v));
  writeHash();
  // charts built while their tab was hidden come up unsized -> resize the now-visible one
  const ctx = v === "health" ? HEALTH_CTX : v === "bytype" ? BYTYPE_CTX : null;
  if (ctx && ctx.chart) requestAnimationFrame(() => { if (ctx.chart) ctx.chart.resize(); });
}
function goTrend(k, model) {
  S.trend.model = model || "all";
  const e = kernelIndex().get(k);
  S.trend.railMode = (e && e.aggregate) ? "aggregates" : "kernels";   // match the rail to the target
  S.trend.key = k;
  if (S.view !== "health") showView("health");
  renderKernelRail();          // rebuild the rail for the (possibly switched) mode + highlight k
  selectKernel(k, HEALTH_CTX);
  const el = document.getElementById("trendsSection");
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

// model legend chips, generated from S.models so the set/order matches the
// table, chart and selector (single source of truth for models).
function renderModelLegend() {
  const el = $("#modelLegend"); if (!el) return;
  el.innerHTML = S.models.map(a =>
    `<span class="arch-chip" data-model="${esc(a)}" style="--c:${modelVar(a)}"><i></i>${esc(a)}</span>`).join("");
}

function wire() {
  $("#tabs").addEventListener("click", e => { const t = e.target.closest(".tab"); if (t) showView(t.dataset.view); });
  document.addEventListener("keydown", e => {
    if (e.target.matches("input,select")) return;
    const map = { 1: "health", 2: "bytype", 3: "gaps", 4: "prcheck", 5: "board" };
    if (map[e.key]) showView(map[e.key]);
    if (e.key.toLowerCase() === "r") doRefresh();
  });
  $("#refresh").addEventListener("click", doRefresh);
  $("#bannerClose").addEventListener("click", hideBanner);
  $("#noiseAware").addEventListener("change", e => {
    S.noiseAware = e.target.checked; _healthRows = null; _aggRows = null;
    renderHealth(); renderByType();   // renderByType redraws the All-Benchmarks chart
    if (S.trend.key) drawTrend(kernelIndex().get(S.trend.key), HEALTH_CTX);
  });
  $("#regList").addEventListener("click", e => { const row = e.target.closest(".reg-row"); if (row) goTrend(row.dataset.k, row.dataset.model); });
  $("#typeSearch").addEventListener("input", e => { S.byType.q = e.target.value; renderByType(); writeHash(); });
  $("#typeFacets").addEventListener("change", e => { const s = e.target.closest("select[data-facet]"); if (!s) return; S.byType.facets[s.dataset.facet] = s.value; renderByType(); writeHash(); });
  $("#facetReset").addEventListener("click", () => { S.byType.facets = { family: "all", mode: "all", dtype: "all", model: "all" }; S.byType.q = ""; $("#typeSearch").value = ""; populateFacets(); renderByType(); writeHash(); });
  // A row click selects it in the All-Benchmarks chart (in place, no tab jump).
  $("#typeSections").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (!tr) return; S.byType.trend.model = tr.dataset.model || "all"; selectKernel(tr.dataset.k, BYTYPE_CTX); });
  // a gap row opens the by-backend comparison chart for the worst shape
  $("#gapsList").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (!tr) return; S.trend.by = "backend"; goTrend(tr.dataset.k, tr.dataset.model); });
  $("#prSelect").addEventListener("change", e => { S.pr.sel = +e.target.value; renderPRCheck(); writeHash(); });
  $("#boardFilter").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.boardFilter = b.dataset.f; $$("#boardFilter button").forEach(x => x.classList.toggle("is-active", x === b)); renderBoard(); });
  $("#railMode").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.railMode = b.dataset.rm; S.trend.key = null; renderKernelRail(); });
  $("#kernelSearch").addEventListener("input", e => { S.trend.q = e.target.value; renderKernelRail(); });
  $("#kernelList").addEventListener("click", e => { const b = e.target.closest(".kitem"); if (b) selectKernel(b.dataset.k, HEALTH_CTX); });
  // Health chart controls
  $("#trendBy").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.by = b.dataset.b; selectKernel(S.trend.key, HEALTH_CTX); });
  $("#trendModel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.model = b.dataset.a; selectKernel(S.trend.key, HEALTH_CTX); });
  $("#trendRange").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.range = b.dataset.r; selectKernel(S.trend.key, HEALTH_CTX); });
  $("#trendXMode").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.xmode = b.dataset.x; selectKernel(S.trend.key, HEALTH_CTX); });
  // All-Benchmarks chart controls
  $("#btBy").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.byType.trend.by = b.dataset.b; selectKernel(S.byType.trend.key, BYTYPE_CTX); });
  $("#btModel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.byType.trend.model = b.dataset.a; selectKernel(S.byType.trend.key, BYTYPE_CTX); });
  $("#btRange").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.byType.trend.range = b.dataset.r; selectKernel(S.byType.trend.key, BYTYPE_CTX); });
  $("#btXMode").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.byType.trend.xmode = b.dataset.x; selectKernel(S.byType.trend.key, BYTYPE_CTX); });
  $("#themeBtn").addEventListener("click", toggleTheme);
}

const SUN_SVG = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="10" cy="10" r="3.4"/><path d="M10 2v2M10 16v2M2 10h2M16 10h2M4.5 4.5l1.4 1.4M14.1 14.1l1.4 1.4M15.5 4.5l-1.4 1.4M5.9 14.1l-1.4 1.4"/></svg>`;
const MOON_SVG = `<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M16 11.5A6.5 6.5 0 1 1 8.5 4a5 5 0 0 0 7.5 7.5z"/></svg>`;
function applyTheme(t) {
  S.theme = t === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", S.theme);
  const btn = $("#themeBtn");
  if (btn) { btn.innerHTML = S.theme === "light" ? MOON_SVG : SUN_SVG; btn.title = `Switch to ${S.theme === "light" ? "dark" : "light"} mode`; }
}
function toggleTheme() {
  applyTheme(S.theme === "light" ? "dark" : "light");
  try { localStorage.setItem("flydsl-theme", S.theme); } catch { /* ignore */ }
  setModelColors();                    // re-tune model hues for the new theme
  setBackendColors();                  // ...and backend hues
  renderAll();                         // re-render SVG sparklines with theme colors
  if (S.trend.key) selectKernel(S.trend.key, HEALTH_CTX);   // redraw Health chart with new theme colors
}
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("flydsl-theme"); } catch { /* ignore */ }
  const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
  applyTheme(saved || (prefersLight ? "light" : "dark"));
}
let refreshing = false;
async function doRefresh() {
  if (refreshing) return; refreshing = true; $("#refresh").classList.add("spin");
  await loadAll(); $("#refresh").classList.remove("spin"); refreshing = false; toast("data reloaded");
}

window.addEventListener("hashchange", () => { readHash(); applyState(); });
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  wire();
  readHash();
  showView(S.view);
  await loadAll();
  if (S.view === "health" && S.trend.key) {
    if (kernelIndex().get(S.trend.key)) selectKernel(S.trend.key, HEALTH_CTX);
    else { S.trend.key = null; renderKernelRail(); }   // stale link -> fall back to default
  }
  setInterval(enhanceLiveBoard, 90000);
});
