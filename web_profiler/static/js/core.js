/* ==========================================================================
   VoxelSim Workbench — core framework
   Hash router + API client + DOM builder + design-system components.
   Every page module registers itself via App.route(name, {title, render}).
   ========================================================================== */

"use strict";

const App = {
  routes: {},          // name -> {title, render, group}
  _cleanup: null,      // cleanup fn returned by previous page render
  _timers: [],         // per-page timers, cleared on navigation
  cache: {},           // shared caches: filters, metricsMeta, jobsDefs
};

/* ============================ i18n ============================
   English is the default UI language; the English text is the key.
   t("Results") -> "结果浏览" when lang=zh.  {name} placeholders are
   filled from the optional vars object. */
App.lang = localStorage.getItem("vs-lang") || "en";

function t(key, vars) {
  let s = key;
  if (App.lang !== "en") {
    const d = (window.I18N_DICT || {})[App.lang];
    if (d && d[key] !== undefined) s = d[key];
  }
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split("{" + k + "}").join(String(v));
  return s;
}
App.t = t;

App.setLang = function (lang) {
  localStorage.setItem("vs-lang", lang);
  location.reload();
};

/** Translate the static shell (sidebar etc.) + wire the language switch. */
App._applyShellI18n = function () {
  document.documentElement.lang = App.lang === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-title]").forEach(el => { el.title = t(el.dataset.i18nTitle); });
  const sw = document.getElementById("lang-switch");
  if (sw) sw.querySelectorAll(".seg-opt").forEach(o => {
    o.classList.toggle("on", o.dataset.lang === App.lang);
    o.onclick = () => App.setLang(o.dataset.lang);
  });
};

/* ============================ DOM builder ============================ */

/** h('div.card#id', {onclick: fn, style:...}, child1, child2, ...) */
function h(tag, attrs, ...children) {
  const [name, ...classes] = tag.split(".");
  let id = null;
  const cls = classes.map(c => {
    const i = c.indexOf("#");
    if (i >= 0) { id = c.slice(i + 1); return c.slice(0, i); }
    return c;
  }).filter(Boolean);
  const el = document.createElement(name);
  if (cls.length) el.className = cls.join(" ");
  if (id) el.id = id;
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "style" && typeof v === "object") Object.assign(el.style, v);
      else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
      else if (k === "dataset") Object.assign(el.dataset, v);
      else if (k === "html") el.innerHTML = v;          // trusted internal markup only
      else if (k === "class" || k === "className") {    // merge with tag-derived classes
        const extra = String(v).trim();
        if (extra) el.className = el.className ? el.className + " " + extra : extra;
      }
      else el.setAttribute(k, v);
    }
  }
  for (const c of children.flat(9)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(c));
  }
  return el;
}

/* ============================ API client ============================ */

App.api = async function (url, opts = {}) {
  let resp;
  try {
    resp = await fetch(url, opts.json !== undefined ? {
      method: opts.method || "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(opts.json),
    } : { method: opts.method || "GET" });
  } catch (e) {
    throw new Error(t("Network error: {msg}", { msg: e.message }));
  }
  let data = null;
  const text = await resp.text();
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!resp.ok) {
    const msg = (data && data.error) ? data.error : `HTTP ${resp.status}`;
    throw new Error(msg);
  }
  return data;
};

/** Shared lookups, fetched once. */
App.filters = async () => App.cache.filters ??= await App.api("/api/filters");
App.metricsMeta = async () => App.cache.metricsMeta ??= await App.api("/api/metrics_meta");
App.jobsDefs = async () => App.cache.jobsDefs ??= await App.api("/api/jobs/defs");

/* ============================ Router ============================ */

App.route = function (name, def) { App.routes[name] = def; };

App.navigate = function (name, params) {
  const q = params ? "?" + new URLSearchParams(params).toString() : "";
  location.hash = `#/${name}${q}`;
};

App._parseHash = function () {
  const raw = (location.hash || "#/dashboard").replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  return { name: path || "dashboard", params: Object.fromEntries(new URLSearchParams(qs || "")) };
};

App._render = async function () {
  const { name, params } = App._parseHash();
  const route = App.routes[name] || App.routes.dashboard;

  // cleanup previous page
  App._timers.forEach(clearInterval); App._timers = [];
  if (typeof App._cleanup === "function") { try { App._cleanup(); } catch {} }
  App._cleanup = null;

  // sidebar highlight + title
  document.querySelectorAll(".nav-item").forEach(a =>
    a.classList.toggle("active", a.dataset.route === name));
  document.getElementById("page-title").textContent = route.title || name;
  document.title = `${route.title || name} · VoxelSim Workbench`;
  const crumb = document.getElementById("page-crumb");
  crumb.textContent = params && Object.keys(params).length
    ? Object.entries(params).map(([k, v]) => `${k}=${v.length > 46 ? v.slice(0, 46) + "…" : v}`).join("  ")
    : "";
  document.getElementById("topbar-actions").replaceChildren();

  const page = document.getElementById("page");
  page.classList.remove("page-enter");
  void page.offsetWidth;                    // restart animation
  page.classList.add("page-enter");
  page.replaceChildren();
  page.scrollTop = 0;

  try {
    const cleanup = await route.render(page, params || {});
    if (typeof cleanup === "function") App._cleanup = cleanup;
  } catch (e) {
    console.error(e);
    page.replaceChildren(UI.empty("⚠", t("Page failed to load: {msg}", { msg: e.message })));
  }
};

/** Register a per-page interval (auto-cleared on navigation, paused while tab hidden). */
App.every = function (ms, fn) {
  App._timers.push(setInterval(() => { if (!document.hidden) fn(); }, ms));
};

App.boot = function () {
  App._applyShellI18n();
  window.addEventListener("hashchange", App._render);
  App._render();
  UI._jobBadgeLoop();
};

/* ============================ Formatting ============================ */

const fmt = {
  num(v, digits = 2) {
    if (v == null || Number.isNaN(v)) return "—";
    if (typeof v === "string") return v;
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(digits) + "G";
    if (a >= 1e6) return (v / 1e6).toFixed(digits) + "M";
    if (a >= 1e4) return (v / 1e3).toFixed(1) + "k";
    if (a >= 100 || Number.isInteger(v)) return v.toLocaleString("en-US");
    return (+v).toFixed(digits);
  },
  int(v) { return v == null ? "—" : Math.round(v).toLocaleString("en-US"); },
  pct(v, digits = 1) { return v == null ? "—" : (+v).toFixed(digits) + "%"; },
  ms(v) {
    if (v == null) return "—";
    if (v >= 1e6) return (v / 1e6).toFixed(2) + " s";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + " ms";
    return (+v).toFixed(0) + " µs";
  },
  bytes(v) {
    if (v == null) return "—";
    for (const [u, s] of [["GB", 1e9], ["MB", 1e6], ["KB", 1e3]]) {
      if (Math.abs(v) >= s) return (v / s).toFixed(1) + " " + u;
    }
    return v + " B";
  },
  time(ts) {  // unix seconds -> local
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toLocaleString(App.lang === "en" ? "en-US" : "zh-CN",
      { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
  },
  ago(ts) {
    if (!ts) return "—";
    const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return t("{n}s ago", { n: s | 0 });
    if (s < 3600) return t("{n}m ago", { n: s / 60 | 0 });
    if (s < 86400) return t("{n}h ago", { n: s / 3600 | 0 });
    return t("{n}d ago", { n: s / 86400 | 0 });
  },
  dur(sec) {
    if (sec == null) return "—";
    sec = Math.round(sec);
    if (sec < 60) return sec + "s";
    if (sec < 3600) return `${sec / 60 | 0}m ${sec % 60}s`;
    return `${sec / 3600 | 0}h ${(sec % 3600) / 60 | 0}m`;
  },
  metric(key, v) {  // format by metrics_meta unit
    if (v == null) return "—";
    const meta = (App.cache.metricsMeta || {})[key] || {};
    if (key === "total_time" || key === "time_ms") return fmt.ms(key === "time_ms" ? v * 1e3 : v);
    if (meta.unit === "%") return fmt.pct(v);
    if (meta.unit === "mJ") return fmt.num(v) + " mJ";
    if (meta.unit === "W") return fmt.num(v) + " W";
    if (meta.unit === "GFLOPS") return fmt.num(v) + " GF";
    return fmt.num(v);
  },
};

/* ============================ Plotly ============================ */

const Plot = {
  COLORS: ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#f472b6",
           "#22d3ee", "#a3e635", "#fb923c", "#818cf8", "#2dd4bf", "#e879f9"],
  baseLayout(over = {}) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: "#9fb0cc", family: "inherit", size: 12 },
      margin: { l: 58, r: 24, t: 34, b: 46 },
      xaxis: { gridcolor: "#1e2a3f66", zerolinecolor: "#1e2a3f", linecolor: "#1e2a3f" },
      yaxis: { gridcolor: "#1e2a3f66", zerolinecolor: "#1e2a3f", linecolor: "#1e2a3f" },
      legend: { orientation: "h", y: 1.14, font: { size: 11 } },
      colorway: Plot.COLORS,
      hoverlabel: { bgcolor: "#111a2a", bordercolor: "#2a3b5e", font: { color: "#e8eefb" } },
    }, over);
  },
  /** Render into a fresh div inside container. Returns the div. */
  chart(container, traces, layout = {}, config = {}) {
    const div = h("div.plot");
    div.style.height = (layout.height || 340) + "px";
    container.append(div);
    Plotly.newPlot(div, traces, Plot.baseLayout(layout),
      Object.assign({ responsive: true, displaylogo: false,
                      modeBarButtonsToRemove: ["lasso2d", "select2d"] }, config));
    // the container is often still off-DOM (or mid-layout) at newPlot time,
    // leaving the SVG sized wrong and overflowing its card — re-fit next frame
    requestAnimationFrame(() =>
      requestAnimationFrame(() => { if (div.isConnected) Plotly.Plots.resize(div); }));
    return div;
  },
  empty(container, msg = t("No data")) {
    container.append(UI.empty("📉", msg));
  },
};

/* ============================ UI components ============================ */

const UI = {};

UI.loading = (msg = t("Loading…")) =>
  h("div.loading-wrap", {}, h("div.spinner"), h("div", {}, msg));

UI.empty = (icon, text) =>
  h("div.empty", {}, h("div.empty-icon", {}, icon), h("div.empty-text", {}, text));

UI.skeletons = (n = 3) => {
  const wrap = h("div.grid.cols-3");
  for (let i = 0; i < n; i++) wrap.append(h("div.skeleton"));
  return wrap;
};

UI.toast = function (msg, kind = "info", ms = 3400) {
  const t = h("div.toast", { class: `toast ${kind}` }, msg);
  document.getElementById("toast-root").append(t);
  setTimeout(() => { t.classList.add("out"); setTimeout(() => t.remove(), 320); }, ms);
};

UI.modal = function (title, contentEl, actions = []) {
  const root = document.getElementById("modal-root");
  const close = () => root.replaceChildren();
  const box = h("div.modal", {},
    h("h3", {}, title),
    contentEl,
    actions.length ? h("div.modal-actions", {},
      actions.map(a => h("button.btn", { class: `btn ${a.kind || ""}`, onclick: () => { if (!a.onClick || a.onClick() !== false) close(); } }, a.label))) : null);
  const overlay = h("div.modal-overlay", { onclick: e => { if (e.target === overlay) close(); } }, box);
  root.replaceChildren(overlay);
  return close;
};

/** Stat card with count-up animation. */
UI.stat = function (label, value, sub = "", delay = 0) {
  const valEl = h("div.stat-value", {}, "0");
  const card = h("div.card.stat.hoverable", { style: { animationDelay: delay + "ms" } },
    h("div.stat-label", {}, label), valEl,
    sub ? h("div.stat-sub", {}, sub) : null);
  const target = typeof value === "number" ? value : parseFloat(String(value).replace(/[^0-9.\-]/g, ""));
  if (Number.isFinite(target)) {
    const t0 = performance.now(), dur = 750;
    const suffix = String(value).replace(/[0-9.,\-]/g, "").trim();
    const step = t => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      valEl.textContent = fmt.num(target * e) + (suffix ? " " + suffix : "");
      if (k < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  } else valEl.textContent = String(value);
  return card;
};

/** Section title row. */
UI.sectionTitle = (text, hint = "") =>
  h("h3.card-title", {}, text, hint ? h("span.hint", {}, hint) : null);

/** Chip. kind: accent|green|warn|red|violet */
UI.chip = (text, kind = "") => h("span.chip", { class: `chip ${kind}` }, text);

UI.statusPill = status => h("span.status", { class: `status ${status}` }, status);

/** Copy-to-clipboard button. */
UI.copyBtn = function (text, label = t("Copy")) {
  return h("button.btn.sm.ghost", {
    onclick: async e => {
      e.stopPropagation();
      try { await navigator.clipboard.writeText(text); UI.toast(t("Copied to clipboard"), "success", 1600); }
      catch { UI.toast(t("Copy failed"), "error"); }
    }
  }, "⧉ " + label);
};

/* ---------------- Filter bar (driven by /api/filters) ---------------- */
/**
 * defs: [{key, label, multi=false, allowAll=true}]
 * values: initial {key: value}
 * onChange(values) fired on any change.
 * Returns {el, get()}.
 */
UI.filterBar = function (defs, values, onChange) {
  const state = Object.assign({}, values);
  const bar = h("div.card.filterbar");
  const sel = {};
  for (const d of defs) {
    const s = h("select", {
      onchange: () => {
        const v = s.value;
        if (v === "") delete state[d.key];
        else state[d.key] = d.numeric ? +v : v;
        onChange && onChange({ ...state });
      }
    });
    sel[d.key] = s;
    if (d.allowAll !== false) s.append(h("option", { value: "" }, t("All")));
    bar.append(h("label.field", {}, d.label || d.key, s));
  }
  UI.filterBar._fill(sel, values);
  const api = {
    el: bar,
    get: () => ({ ...state }),
    async fill() { await UI.filterBar._fill(sel, values); },
  };
  UI.filterBar._last = api;
  return api;
};
UI.filterBar._fill = async function (sel, values) {
  const filters = await App.filters();
  for (const [key, s] of Object.entries(sel)) {
    const cur = s.value || (values && values[key] != null ? String(values[key]) : "");
    const seen = new Set([...s.options].map(o => o.value));
    for (const v of (filters[key] || [])) {
      const sv = String(v);
      if (!seen.has(sv)) { s.append(h("option", { value: sv }, sv)); seen.add(sv); }
    }
    if (cur) s.value = cur;
  }
};

/* ---------------- Data table ---------------- */
/**
 * cols: [{key, label, num=false, render?: (v,row)=>node|string, sortable=true}]
 * rows: array of objects. opts: {onRow(row), pageSize=0, maxHeight}
 * Returns {el, setRows(rows)}.
 */
UI.table = function (cols, rows, opts = {}) {
  let sortKey = null, sortDir = 1, data = rows || [];
  const tbody = h("tbody");
  const thead = h("thead", {}, h("tr", {}, cols.map(c =>
    h("th", {
      class: c.num ? "num" : "",
      onclick: () => {
        if (c.sortable === false) return;
        if (sortKey === c.key) sortDir *= -1; else { sortKey = c.key; sortDir = 1; }
        paint();
      }
    }, c.label, h("span.arrow", {}, sortKey === c.key ? (sortDir > 0 ? "▲" : "▼") : "")))));

  function paint() {
    thead.replaceChildren(h("tr", {}, cols.map(c =>
      h("th", {
        class: c.num ? "num" : "",
        onclick: () => {
          if (c.sortable === false) return;
          if (sortKey === c.key) sortDir *= -1; else { sortKey = c.key; sortDir = 1; }
          paint();
        }
      }, c.label, h("span.arrow", {}, sortKey === c.key ? (sortDir > 0 ? "▲" : "▼") : "")))));
    let view = [...data];
    if (sortKey) {
      view.sort((a, b) => {
        const x = a[sortKey], y = b[sortKey];
        if (x == null) return 1; if (y == null) return -1;
        return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
      });
    }
    if (opts.pageSize) view = view.slice(0, opts.pageSize);
    tbody.replaceChildren(...view.map(row =>
      h("tr", { onclick: () => opts.onRow && opts.onRow(row) },
        cols.map(c => {
          const v = c.render ? c.render(row[c.key], row) : (row[c.key] ?? "—");
          return h("td", { class: c.num ? "num" : "" }, v);
        }))));
    if (!view.length) {
      tbody.replaceChildren(h("tr", {}, h("td", { colspan: cols.length, style: { textAlign: "center", padding: "26px" } }, t("No matching data"))));
    }
  }
  paint();
  const wrap = h("div.tbl-wrap", { style: opts.maxHeight ? { maxHeight: opts.maxHeight } : {} },
    h("table.tbl", {}, thead, tbody));
  return { el: wrap, setRows(r) { data = r || []; paint(); } };
};

/* ---------------- Tabs ---------------- */
/** tabs: [{key, label}], onChange(key). Returns {el, set(key)} */
UI.tabs = function (tabs, active, onChange) {
  const el = h("div.tabs");
  const paint = act => {
    el.replaceChildren(...tabs.map(t =>
      h("button.tab", { class: t.key === act ? "active" : "", onclick: () => { paint(t.key); onChange(t.key); } }, t.label)));
  };
  paint(active);
  return { el, set: paint };
};

/* ---------------- Tier select (slider-knob multi picker) ---------------- */
/**
 * The "拉调节钮" component: a track of discrete stops; click to toggle.
 * def: {key, label, unit, stops: [...], numeric=true}
 * selected: initial array. onChange(key, selectedArray).
 * Returns {el, get(), set(vals)}.
 */
UI.tierSelect = function (def, selected, onChange) {
  let sel = new Set((selected && selected.length ? selected : [def.stops[Math.floor(def.stops.length / 2)]]).map(String));
  const selInfo = h("span.tier-sel");
  const track = h("div.tier-track");
  const stops = def.stops.map(v => {
    const el = h("div.tier-stop", {
      onclick: () => {
        const sv = String(v);
        if (sel.has(sv)) { if (sel.size > 1) sel.delete(sv); }  // keep >=1 selected
        else sel.add(sv);
        paint();
        onChange && onChange(def.key, api.get());
      }
    }, String(v));
    el.dataset.v = String(v);
    return el;
  });
  track.append(...stops);
  function paint() {
    stops.forEach(el => el.classList.toggle("on", sel.has(el.dataset.v)));
    const n = sel.size;
    selInfo.replaceChildren(t("Selected "), h("b", {}, String(n)), n > 1 ? t(" stops → ×{n} combos", { n }) : t(" stops"));
  }
  paint();
  const api = {
    el: h("div.tier", {},
      h("div.tier-head", {},
        h("span.tier-name", {}, def.label || def.key),
        def.unit ? h("span.tier-unit", {}, def.unit) : null,
        selInfo,
        h("span.tier-actions", {},
          h("button", { onclick: () => { sel = new Set(def.stops.map(String)); paint(); onChange && onChange(def.key, api.get()); } }, t("Select all")),
          h("button", { onclick: () => { sel = new Set([String(def.stops[0])]); paint(); onChange && onChange(def.key, api.get()); } }, t("Reset")))),
      track),
    get: () => def.numeric === false ? [...sel] : [...sel].map(Number).sort((a, b) => a - b),
    set(vals) { sel = new Set(vals.map(String)); paint(); },
  };
  return api;
};

/* ---------------- Job badge (nav) ---------------- */
/* Adaptive poll: 8 s while jobs are active, backs off to 60 s when idle,
   and pauses entirely while the tab is hidden.  Uses a self-rescheduling
   timeout (never overlapping) instead of a fixed interval. */
UI._jobBadgeLoop = async function () {
  const badge = document.getElementById("jobs-badge");
  let timer = null, idleStreak = 0;
  const schedule = ms => { clearTimeout(timer); timer = setTimeout(tick, ms); };
  const tick = async () => {
    if (document.hidden) { schedule(30000); return; }
    try {
      const d = await App.api("/api/jobs?limit=30");
      const n = (d.jobs || []).filter(j => ["running", "queued"].includes(j.status)).length;
      badge.hidden = n === 0;
      badge.textContent = n;
      App.state = App.state || {};
      App.state.activeJobs = n;
      idleStreak = n ? 0 : idleStreak + 1;
    } catch { badge.hidden = true; idleStreak++; }
    schedule(idleStreak === 0 ? 8000 : Math.min(60000, 15000 * idleStreak));
  };
  document.addEventListener("visibilitychange", () => { if (!document.hidden) schedule(0); });
  schedule(0);
};

/* ---------------- config label helpers ---------------- */
UI.implChip = impl => {
  const map = { best: ["best", "accent"], uniform_dram: ["uniform", ""], spmd_compiler: ["spmd", "violet"], seq_noc: ["seq_noc", "warn"], dataflow: ["dataflow", "green"], ipu_tsim: ["ipu", "red"] };
  const [label, kind] = map[impl] || [impl, ""];
  return UI.chip(label, kind);
};
UI.modeChip = mode => UI.chip(mode, mode === "decode" ? "accent" : "violet");

/** Compact config description used across pages. */
UI.cfgLabel = c =>
  `${c.model} · ${c.mode} · bs${c.batch_size} · c${c.num_cores} · sa${c.sa_size} · sram${c.sram_kb}K · bw${c.dram_bw} · topo${c.noc_topo}/noc${c.noc_bw} · cg${c.core_group}`;
