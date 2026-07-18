/* Serving — llmservingsim profile browser: profile cards (files + meta
   summary), per-file line charts from columnar JSON, and meta.yaml
   structures (attention_grid / engine_effective / source) as kv blocks. */

App.route("serving", {
  title: t("LLM Serving"),
  async render(el) {
    el.append(UI.loading(t("Loading serving profiles…")));
    let profiles = [];
    try { profiles = (await App.api("/api/serving/profiles")).profiles || []; }
    catch (e) { el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }
    if (!profiles.length) {
      el.replaceChildren(UI.empty("◍", t("No profiles under llmservingsim/3D-chip")));
      return;
    }

    const detail = h("div");

    /* ============================ profile cards ============================ */
    function kvPairs(pairs) {   // pairs: [[key, value], ...]
      return pairs.map(([k, v]) => [
        h("dt", {}, k),
        h("dd", {}, (v && typeof v === "object") ? JSON.stringify(v) : String(v)),
      ]);
    }

    function profileCard(p, i) {
      const meta = p.meta || {};
      const src = meta.source || {};
      const hasMeta = Object.keys(meta).length > 0;
      return h("div.card.hoverable", {
        style: { cursor: "pointer", animationDelay: (i % 6) * 60 + "ms" },
        onclick: () => openProfile(p),
      },
        h("div", { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
          h("b", { style: { fontSize: "15px" } }, p.model || "?"),
          UI.chip(p.tp, "accent"),
          h("span", { style: { flex: "1" } }),
          (p.files || []).map(f => UI.chip(f.replace(/\.csv$/, ""), "violet"))),
        h("div.small.mono.muted", { style: { marginTop: "6px", wordBreak: "break-all" } }, p.path),
        hasMeta
          ? h("dl.kv", { style: { marginTop: "10px" } }, [
              h("dt", {}, t("GPU config")), h("dd", { style: { wordBreak: "break-all" } }, meta.gpu || "—"),
              h("dt", {}, t("Source model")), h("dd", {}, src.model || "—"),
              h("dt", {}, t("Simulator")), h("dd", {}, src.simulator || "—"),
              h("dt", {}, t("Time unit")), h("dd", {}, src.time_unit || "—"),
              h("dt", {}, "profiled_at"), h("dd", {}, meta.profiled_at || "—"),
            ])
          : h("div.small.muted", { style: { marginTop: "10px" } }, t("No meta.yaml")));
    }

    /* ============================ meta card ============================ */
    function buildMetaCard(meta) {
      if (!meta || !Object.keys(meta).length) {
        return h("div.card", {}, UI.sectionTitle("meta.yaml"), UI.empty("◍", t("No meta info")));
      }
      const scalars = Object.entries(meta).filter(([, v]) => !v || typeof v !== "object");
      const nested = Object.entries(meta).filter(([, v]) => v && typeof v === "object");
      return h("div.card", {},
        UI.sectionTitle("meta.yaml", t("Full contents")),
        scalars.length ? h("dl.kv", {}, kvPairs(scalars)) : null,
        nested.map(([k, v]) => h("div", { style: { marginTop: "14px" } },
          h("div", {
            style: {
              fontSize: "11px", fontWeight: 700, color: "var(--accent)",
              letterSpacing: "1.5px", textTransform: "uppercase", marginBottom: "8px",
            },
          }, k),
          h("dl.kv", {}, kvPairs(Object.entries(v))))));
    }

    /* ============================ profile detail ============================ */
    async function openProfile(p) {
      detail.replaceChildren(UI.loading(t("Loading profile data…")));
      let d;
      try { d = await App.api("/api/serving/profile?path=" + encodeURIComponent(p.path)); }
      catch (e) { detail.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }
      const files = Object.keys(d.files || {});
      if (!files.length) { detail.replaceChildren(UI.empty("◍", t("This profile has no CSV files"))); return; }

      const chartHint = h("span.hint");
      const chartBody = h("div");

      function paintFile(name) {
        const f = (d.files || {})[name];
        if (!f || !(f.rows || []).length) {
          chartBody.replaceChildren(UI.empty("📉", t("No data in this file")));
          chartHint.textContent = name;
          return;
        }
        const cols = f.columns, rows = f.rows;
        const firstNumeric = rows.every(r => typeof r[0] === "number");
        const isNumCol = j => rows.every(r => typeof r[j] === "number");
        let x, xLabel, numIdx;
        if (firstNumeric) {
          x = rows.map((_, i) => i + 1);
          xLabel = t("Sample #");
          numIdx = cols.map((c, j) => j).filter(isNumCol);
        } else {
          x = rows.map(r => String(r[0]));
          xLabel = cols[0];
          numIdx = cols.map((c, j) => j).slice(1).filter(isNumCol);
        }
        chartHint.textContent = t("{name} · {rows} rows · cols: {cols}", { name, rows: rows.length, cols: cols.join(" / ") });
        chartBody.replaceChildren();
        if (!numIdx.length) { chartBody.append(UI.empty("📉", t("No plottable numeric columns"))); return; }
        const traces = numIdx.map(j => ({
          type: "scatter", mode: "lines+markers", name: cols[j],
          x: x, y: rows.map(r => r[j]),
          line: { width: 2 }, marker: { size: 4 },
        }));
        Plot.chart(chartBody, traces, {
          height: 340,
          xaxis: { title: xLabel, type: firstNumeric ? "linear" : "category" },
          yaxis: { title: t("Value") },
        });
      }

      const tabs = UI.tabs(files.map(f => ({ key: f, label: f })), files[0], paintFile);
      const headCard = h("div.card.mb", {},
        h("h3.card-title", {}, (p.model || "?") + " · " + p.tp, h("span.hint", {}, p.path)),
        tabs.el);
      const chartCard = h("div.card.mb", {},
        h("h3.card-title", {}, t("Curves"), chartHint), chartBody);

      detail.replaceChildren(headCard, chartCard, buildMetaCard(p.meta));
      paintFile(files[0]);
      headCard.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    /* ============================ page ============================ */
    const grid = h("div.grid.cols-2.mb");
    profiles.forEach((p, i) => grid.append(profileCard(p, i)));
    el.replaceChildren(
      h("div.grid.cols-3.mb", {},
        UI.stat(t("Profiles"), profiles.length, "llmservingsim/3D-chip", 0),
        UI.stat(t("Models"), profiles.map(p => p.model).filter((v, i, a) => v && a.indexOf(v) === i).length,
          profiles.map(p => p.model).filter(Boolean).join(" / "), 60),
        UI.stat(t("CSV files"), profiles.reduce((n, p) => n + (p.files || []).length, 0),
          "attention / dense / per_sequence", 120)),
      grid, detail);
  },
});
