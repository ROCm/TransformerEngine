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
  regressionPct: -3.0,   // fixed gate
  warnPct: -1.0,         // surfaced as "watch"
  noiseK: 2.0,           // a drop must exceed K * (run-to-run relative std) to count as real
  minSamples: 3,         // prior main runs needed to size a noise band (else: low confidence)
  sparkFloorPct: 6,      // sparkline min half-window (% of baseline) so trivial noise stays flat
  archOrder: ["gfx950"], // TE runs; add arches here (also update index.html trend header)
};

const S = {
  records: [], runs: [], updated: null, runMeta: new Map(),
  view: "health", noiseAware: true, boardFilter: "all",
  pr: { sel: null },
  trend: { key: null, arch: "all", metric: null, q: "", range: "all", xmode: "commits" },
  byType: { q: "" },
  theme: "dark",
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const kkey = r => `${r.op} ${r.shape} ${r.dtype}`;
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const VIEWS = ["health", "bytype", "prcheck", "board"];

// theme-aware colors: read the live CSS variables so canvas/SVG match the active theme
const ARCHVAR = { gfx950: "--gfx950", gfx942: "--gfx942", gfx1201: "--gfx1201" };
const cssVal = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim() || n;
const archVar = a => `var(${ARCHVAR[a] || "--ink-2"})`;     // for inline style="" (CSS var)
const archCol = a => cssVal(ARCHVAR[a] || "--ink-2");        // resolved value for <canvas>
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
  return {
    op: o.op, shape: o.shape, dtype: o.dtype, metric: o.metric,
    value: num(o.value), ts: o.ts, commit: o.commit, run_id: num(o.run_id),
    arch: o.arch, runner: o.runner, pr: num(o.pr), source: "ci",
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
  S.records = records;
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
      runner: (x.name.match(/\((linux-flydsl-[^)]+)\)/) || [])[1], arch: archOf(x.name),
      status: x.status, conclusion: x.conclusion, url: x.html_url,
    }));
  }));
  if (S.view === "board") renderBoard();
}
function archOf(n) { return /mi355/.test(n) ? "gfx950" : /mi325/.test(n) ? "gfx942" : /navi/.test(n) ? "gfx1201" : "?"; }

/* ----------------------------------------------------------- noise model --- */
function isMainRec(r) { const m = S.runMeta.get(r.run_id); return m ? m.branch === "dev" : r.pr == null; }

// chronological main-branch series for one kernel/arch/metric
function mainSeries(op, shape, dtype, arch, metric) {
  return S.records
    .filter(r => r.source === "ci" && r.op === op && r.shape === shape && r.dtype === dtype &&
      r.arch === arch && r.metric === metric && r.value != null && isMainRec(r))
    .sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
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
function mainBaseline(op, shape, dtype, arch, metric) {
  const series = mainSeries(op, shape, dtype, arch, metric);
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

/* latest main-run record per (kernel,arch) */
function latestMainByKernelArch() {
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.metric === "speedup" || r.value == null || !isMainRec(r)) continue;
    const k = `${r.arch}|${kkey(r)}`; const ex = m.get(k);
    if (!ex || (r.ts || "") > (ex.ts || "")) m.set(k, r);
  }
  return m;
}

/* one row per latest-main kernel/arch: latest value vs PRIOR main history */
function healthRows() {
  return [...latestMainByKernelArch().values()].map(r => {
    const noise = mainBaseline(r.op, r.shape, r.dtype, r.arch, r.metric);
    const vals = mainSeries(r.op, r.shape, r.dtype, r.arch, r.metric).map(s => s.value);
    const { d, real } = regOf(r, noise);
    return { r, vals, noise, d, real, sev: d == null ? "flat" : sev(d, real) };
  });
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
    `<div class="stat"><span class="v">${new Set(rows.map(x => x.r.arch)).size}</span><span class="l">arches</span></div>` +
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
    return `<div class="reg-row s-${sev}" style="--i:${i}" data-k="${esc(kkey(r))}" data-arch="${r.arch}">
      <span class="op">${esc(r.op)} <span class="metric-tag">${r.metric}</span></span>
      <span class="shape">${esc(r.shape)} · ${esc(r.dtype)}</span>
      <span class="reg-arch" style="color:${archVar(r.arch)}">${esc(r.arch)}</span>
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
  gemm: "GEMM", gemm_fp8: "FP8 GEMM", grouped_gemm: "Grouped GEMM",
  casting: "Casting", normalization: "Normalization",
};
// Display order: known microbenchmark families first, then anything else
// (e.g. e2e suites) alphabetically.
const FAMILY_ORDER = ["gemm", "gemm_fp8", "grouped_gemm", "casting", "normalization"];
const familyLabel = f => FAMILY_LABELS[f] ||
  (f ? f.replace(/[_-]+/g, " ").replace(/\b\w/g, c => c.toUpperCase()) : "Other");
const familyRank = f => { const i = FAMILY_ORDER.indexOf(f); return i < 0 ? FAMILY_ORDER.length : i; };

// Every latest dev kernel/arch, grouped by benchmark family -- the full picture,
// not just regressions. Same latest-value + noise-band trend as Health.
function renderByType() {
  const q = (S.byType.q || "").toLowerCase();
  const rows = healthRows().filter(x => {
    if (!q) return true;
    const r = x.r;
    return `${r.op} ${r.shape} ${r.dtype} ${r.family || ""}`.toLowerCase().includes(q);
  });
  const groups = new Map();
  for (const x of rows) {
    const f = x.r.family || "other";
    (groups.get(f) || groups.set(f, []).get(f)).push(x);
  }
  const fams = [...groups.keys()].sort((a, b) => familyRank(a) - familyRank(b) || a.localeCompare(b));
  const openAttr = q ? " open" : "";   // folded by default; auto-open sections while filtering
  $("#typeSummary").innerHTML = rows.length
    ? `${rows.length} kernel × arch across <b>${fams.length}</b> type${fams.length === 1 ? "" : "s"} · latest dev value + Δ vs prior-dev history`
    : "no results in snapshot";
  $("#typeSections").innerHTML = fams.map(f => {
    const list = groups.get(f).slice().sort((a, b) =>
      a.r.op.localeCompare(b.r.op) || a.r.shape.localeCompare(b.r.shape) || (a.r.arch || "").localeCompare(b.r.arch || ""));
    const units = [...new Set(list.map(x => x.r.metric))].join(", ");
    const body = list.map(({ r, vals, noise, d, real, sev }) => {
      const dcell = d == null ? `<td class="num k-dim">—</td>` : `<td class="num delta ${sev}">${fmtPct(d)}</td>`;
      return `<tr data-k="${esc(kkey(r))}" data-arch="${r.arch}">
        <td>${esc(r.op)} <span class="metric-tag">${r.metric}</span></td>
        <td class="k-dim">${esc(r.shape)}</td>
        <td>${esc(r.dtype)}</td>
        <td style="color:${archVar(r.arch)}">${esc(r.arch)}</td>
        <td class="spark-cell">${sparkline(vals, noise, real)}</td>
        <td class="num">${fmtVal(r.value, r.metric)}</td>
        ${dcell}
      </tr>`;
    }).join("");
    return `<details class="panel type-panel"${openAttr}>
      <summary class="panel-head"><span class="type-caret" aria-hidden="true">▸</span><span class="t">${esc(familyLabel(f))}</span>
        <span class="type-count">${list.length} kernel${list.length === 1 ? "" : "s"}</span>
        <span class="spacer"></span><span class="type-unit">${esc(units)}</span></summary>
      <div class="table-wrap"><table class="data"><thead><tr>
        <th>kernel</th><th>shape</th><th>dtype</th><th>arch</th><th>recent trend</th><th class="num">latest</th><th class="num">Δ vs dev</th>
      </tr></thead><tbody>${body}</tbody></table></div>
    </details>`;
  }).join("") || `<div class="empty">no results in snapshot</div>`;
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
    const { d, real } = regOf(r, mainBaseline(r.op, r.shape, r.dtype, r.arch, r.metric));
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
      <td style="color:${archVar(r.arch)}">${esc(r.arch)}</td>
      <td class="num">${fmtVal(r.value, r.metric)}</td>
      <td class="num k-dim">${fmtVal(r.vs_main.baseline, r.metric)}</td>
      <td class="num delta ${sev}">${fmtPct(d)}</td>
      <td class="k-dim">${esc(r.vs_main.label || "dev")}</td></tr>`).join("")
    || `<tr><td colspan="8" class="empty">no vs-dev rows for this PR run</td></tr>`}</tbody></table>
    <div class="noise-note" style="padding:10px 12px">latest run of #${pr} · <a href="${esc(runUrl)}" target="_blank" rel="noopener">view on GitHub</a> · “real” = beyond the kernel’s run-to-run noise on dev</div></div>`;
}

/* ------------------------------------------------------------ 3 · TRENDS --- */
function kernelIndex() {
  const regKeys = new Set(healthRows().filter(x => x.real).map(x => kkey(x.r)));
  const m = new Map();
  for (const r of S.records) {
    if (r.source !== "ci" || r.value == null) continue;
    const k = kkey(r);
    if (!m.has(k)) m.set(k, { op: r.op, shape: r.shape, dtype: r.dtype, metrics: new Set(), runs: new Set(), reg: false });
    const e = m.get(k); e.metrics.add(r.metric); e.runs.add(r.run_id);
  }
  for (const [k, e] of m) { e.reg = regKeys.has(k); e.n = e.runs.size; }
  return m;
}
function renderKernelRail() {
  const idx = kernelIndex();
  let keys = [...idx.keys()];
  if (S.trend.q) { const q = S.trend.q.toLowerCase(); keys = keys.filter(k => k.toLowerCase().includes(q)); }
  keys.sort();
  // default to a regressed kernel, else the best-sampled one (so the chart isn't a lone point)
  if (!S.trend.key && keys.length) {
    const reg = keys.find(k => idx.get(k).reg);
    const dense = keys.slice().sort((a, b) => idx.get(b).n - idx.get(a).n)[0];
    selectKernel(reg || dense, false);
  }
  $("#kernelList").innerHTML = keys.map(k => {
    const e = idx.get(k);
    return `<button class="kitem ${k === S.trend.key ? "is-active" : ""} ${e.reg ? "has-reg" : ""}" data-k="${esc(k)}">
      ${esc(e.op)}<span class="ks">${esc(e.shape)} · ${esc(e.dtype)}</span></button>`;
  }).join("") || `<div class="empty" style="padding:20px">no match</div>`;
}
let trendChart = null;
function selectKernel(k, rerail = true) {
  S.trend.key = k;
  const idx = kernelIndex(); const e = idx.get(k); if (!e) return;
  const metrics = [...e.metrics];
  if (!metrics.includes(S.trend.metric)) S.trend.metric = metrics.find(m => m !== "speedup") || metrics[0];
  $("#metricSel").innerHTML = metrics.map(m => `<button data-m="${m}" class="${m === S.trend.metric ? "is-active" : ""}">${m}</button>`).join("");
  $("#trendArch").innerHTML = ["all", ...CFG.archOrder].map(a =>
    `<button data-a="${a}" class="${a === S.trend.arch ? "is-active" : ""}">${a}</button>`).join("");
  $("#trendRange").innerHTML = [["7d", "7 days"], ["30d", "30 days"], ["all", "all"]].map(([v, t]) =>
    `<button data-r="${v}" class="${v === S.trend.range ? "is-active" : ""}">${t}</button>`).join("");
  $("#trendXMode").innerHTML = [["commits", "by commit"], ["daily", "by day"]].map(([v, t]) =>
    `<button data-x="${v}" class="${v === S.trend.xmode ? "is-active" : ""}">${t}</button>`).join("");
  $("#trendTitle").innerHTML = `${esc(e.op)} <small>${esc(e.shape)} · ${esc(e.dtype)} · ${S.trend.metric}</small>`;
  if (rerail) $$("#kernelList .kitem").forEach(b => b.classList.toggle("is-active", b.dataset.k === k));
  drawTrend(e);
}
function drawTrend(e) {
  const metric = S.trend.metric, op = e.op, shape = e.shape, dtype = e.dtype;
  const recs = S.records.filter(r => r.source === "ci" && r.op === op && r.shape === shape && r.dtype === dtype && r.metric === metric);
  let runIds = [...new Set(recs.map(r => r.run_id))].map(id => {
    const any = recs.find(r => r.run_id === id);
    return { id, ts: any.ts, commit: any.commit, pr: any.pr, main: isMainRec(any) };
  }).sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  // time-range filter. range is "<N>d" (last N days) or "all" (no filter).
  const days = S.trend.range === "all" ? Infinity : parseInt((S.trend.range.match(/\d+/) || ["30"])[0], 10);
  if (Number.isFinite(days)) {
    const cutoff = Date.now() - days * 86400000;
    runIds = runIds.filter(ri => !ri.ts || new Date(ri.ts).getTime() >= cutoff);
  }
  const val = new Map();                       // run_id|arch -> value
  for (const r of recs) val.set(r.run_id + "|" + r.arch, r.value);

  // x-axis points: one per commit, or one per day (daily mean) when xmode=daily
  const daily = S.trend.xmode === "daily";
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
  const valueAt = (p, arch) => {
    if (daily) { const vs = p.runs.map(ri => val.get(ri.id + "|" + arch)).filter(v => v != null); return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null; }
    return val.get(p.id + "|" + arch) ?? null;
  };
  const single = S.trend.arch !== "all";
  const archs = single ? [S.trend.arch] : CFG.archOrder;

  const datasets = [];
  let note = "";
  const span = `${points.length} ${daily ? "day" + (points.length === 1 ? "" : "s") : "commits"}` +
    (S.trend.range === "all" ? "" : ` · last ${days}d`);
  if (single) {
    const noise = mainBaseline(op, shape, dtype, S.trend.arch, metric);
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
    note = `${span} · one line per arch.` + (daily ? " Daily mean per arch — smooths CI jitter to expose real drift." : " Red = dev below its prior-dev band, or a PR slower than dev.");
  }
  for (const arch of archs) {
    const noise = mainBaseline(op, shape, dtype, arch, metric);
    const data = points.map(p => valueAt(p, arch));
    if (data.every(v => v == null)) continue;
    const ptColor = points.map((p, i) => {
      if (daily) return archCol(arch);          // daily means aren't per-run regression calls
      const r = recs.find(x => x.run_id === p.id && x.arch === arch);
      return (r && regOf(r, noise).real) ? cssVal("--bad") : archCol(arch);
    });
    datasets.push({
      label: arch, data, borderColor: archCol(arch), backgroundColor: archCol(arch) + "22",
      pointBackgroundColor: ptColor, pointBorderColor: ptColor,
      pointRadius: daily ? 4 : points.length > 30 ? 1.5 : 3, pointHoverRadius: 6,
      borderWidth: daily ? 2.4 : 2, tension: .25, spanGaps: true, order: 1, fill: single ? "origin" : false,
    });
  }
  $("#noiseNote").innerHTML =
    `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.3v4M8 5v.01" stroke-linecap="round"/></svg>` + note;

  const tickCol = cssVal("--ink-3"), gridCol = cssVal("--grid");
  if (trendChart) trendChart.destroy();
  trendChart = new Chart($("#trendChart"), {
    type: "line", data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 220 },
      interaction: { mode: "index", intersect: false },
      onHover: (ev, els) => { if (ev.native) ev.native.target.style.cursor = els.length ? "pointer" : "default"; },
      onClick: (ev, els) => { if (!els.length) return; const p = points[els[0].index]; if (p && p.commit) window.open(commitUrl(p.commit), "_blank", "noopener"); },
      plugins: {
        legend: { labels: { color: cssVal("--ink-2"), font: { family: "IBM Plex Mono", size: 11 }, boxWidth: 10, usePointStyle: true, filter: i => !/σ|mean/.test(i.text) } },
        tooltip: {
          backgroundColor: cssVal("--panel"), borderColor: cssVal("--border"), borderWidth: 1, titleColor: cssVal("--ink"), bodyColor: cssVal("--ink-2"),
          titleFont: { family: "IBM Plex Mono" }, bodyFont: { family: "IBM Plex Mono" },
          filter: i => !/σ|mean/.test(i.dataset.label),
          callbacks: {
            title: items => { const p = points[items[0].dataIndex]; return daily ? `${p.date} · ${p.runs.length} run${p.runs.length === 1 ? "" : "s"}` : `${p.sha} · ${p.main ? "dev" : "#" + p.pr} · click to open`; },
            label: i => ` ${i.dataset.label}: ${fmtVal(i.parsed.y, metric)} ${metric}`,
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
  // status-aware table — always per-commit, with a real link to each commit
  $("#trendBody").innerHTML = runIds.slice().reverse().map(ri => {
    const cell = arch => {
      const r = recs.find(x => x.run_id === ri.id && x.arch === arch)
        || S.records.find(x => x.run_id === ri.id && x.arch === arch && x.op === op && x.shape === shape && x.dtype === dtype);
      if (!r) return `<td class="num st-na">—</td>`;
      if (r.value == null) return `<td class="num cell-status ${r.status === "skip" ? "st-skip" : "st-missing"}">${r.status}</td>`;
      const real = regOf(r, mainBaseline(op, shape, dtype, arch, metric)).real;
      return `<td class="num" style="color:${real ? "var(--bad)" : archVar(arch)}">${fmtVal(r.value, metric)}</td>`;
    };
    const sha = (ri.commit || "").slice(0, 7);
    return `<tr><td><a class="commit-link" href="${commitUrl(ri.commit)}" target="_blank" rel="noopener">${sha || "—"}</a></td>` +
      `<td class="k-dim">${(ri.ts || "").slice(0, 10)}</td>` +
      `<td class="k-dim">${ri.main ? "dev" : ri.pr ? `<a href="https://github.com/${CFG.repo}/pull/${ri.pr}" target="_blank" rel="noopener">#${ri.pr}</a>` : "branch"}</td>` +
      `${CFG.archOrder.map(cell).join("")}</tr>`;
  }).join("");
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
    const byRunner = {}; for (const j of (r.jobs || [])) byRunner[j.arch] = j;
    const overall = (r.status !== "completed") ? "running" : (r.conclusion || "none");
    const chips = CFG.archOrder.map(arch => {
      const j = byRunner[arch]; const st = chipState(j);
      const label = st === "running" ? "run" : st === "success" ? "pass" : st === "failure" ? "fail" : st === "none" ? "—" : st.slice(0, 4);
      const link = j?.url ? `<a href="${esc(j.url)}" target="_blank" rel="noopener" title="${arch} · ${st}"></a>` : "";
      return `<div class="chip" data-c="${esc(st)}"><span class="arch">${esc(arch)}</span><span class="st">${esc(label)}</span>${link}</div>`;
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
function renderAll() { renderHealth(); renderByType(); renderKernelRail(); renderPRCheck(); renderBoard(); }
function showView(v) {
  if (v === "trends") v = "health";
  if (!VIEWS.includes(v)) v = "health";
  S.view = v;
  $$(".tab").forEach(t => { const on = t.dataset.view === v; t.classList.toggle("is-active", on); t.setAttribute("aria-selected", on ? "true" : "false"); });
  $$(".view").forEach(s => s.classList.toggle("is-active", s.dataset.view === v));
  if (location.hash.slice(1) !== v) history.replaceState(null, "", "#" + v);
  if (v === "health" && trendChart) trendChart.resize();
}
function goTrend(k, arch) {
  S.trend.key = null; S.trend.arch = arch || "all";
  if (S.view !== "health") showView("health");
  selectKernel(k);
  const el = document.getElementById("trendsSection");
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function wire() {
  $("#tabs").addEventListener("click", e => { const t = e.target.closest(".tab"); if (t) showView(t.dataset.view); });
  document.addEventListener("keydown", e => {
    if (e.target.matches("input,select")) return;
    const map = { 1: "health", 2: "bytype", 3: "prcheck", 4: "board" };
    if (map[e.key]) showView(map[e.key]);
    if (e.key.toLowerCase() === "r") doRefresh();
  });
  $("#refresh").addEventListener("click", doRefresh);
  $("#bannerClose").addEventListener("click", hideBanner);
  $("#noiseAware").addEventListener("change", e => { S.noiseAware = e.target.checked; renderHealth(); if (S.trend.key) drawTrend(kernelIndex().get(S.trend.key)); });
  $("#regList").addEventListener("click", e => { const row = e.target.closest(".reg-row"); if (row) goTrend(row.dataset.k, row.dataset.arch); });
  $("#typeSearch").addEventListener("input", e => { S.byType.q = e.target.value; renderByType(); });
  $("#typeSections").addEventListener("click", e => { const tr = e.target.closest("tr[data-k]"); if (tr) goTrend(tr.dataset.k, tr.dataset.arch); });
  $("#prSelect").addEventListener("change", e => { S.pr.sel = +e.target.value; renderPRCheck(); });
  $("#boardFilter").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.boardFilter = b.dataset.f; $$("#boardFilter button").forEach(x => x.classList.toggle("is-active", x === b)); renderBoard(); });
  $("#kernelSearch").addEventListener("input", e => { S.trend.q = e.target.value; renderKernelRail(); });
  $("#kernelList").addEventListener("click", e => { const b = e.target.closest(".kitem"); if (b) selectKernel(b.dataset.k); });
  $("#metricSel").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.metric = b.dataset.m; selectKernel(S.trend.key); });
  $("#trendArch").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.arch = b.dataset.a; selectKernel(S.trend.key); });
  $("#trendRange").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.range = b.dataset.r; selectKernel(S.trend.key); });
  $("#trendXMode").addEventListener("click", e => { const b = e.target.closest("button"); if (!b) return; S.trend.xmode = b.dataset.x; selectKernel(S.trend.key); });
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
  renderAll();                         // re-render SVG sparklines with theme colors
  if (S.trend.key) selectKernel(S.trend.key);   // redraw the canvas chart
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

window.addEventListener("hashchange", () => showView(location.hash.slice(1)));
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  wire();
  if (VIEWS.includes(location.hash.slice(1))) showView(location.hash.slice(1));
  loadAll();
  setInterval(enhanceLiveBoard, 90000);
});
