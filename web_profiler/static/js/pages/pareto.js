/* Pareto — DSE Pareto 探索页。
   数据:GET /api/pareto?mode=decode|prefill
   响应:{mode, models, num_points, num_pareto, num_evaluated, eval_time_sec,
         timestamp, fixed_params:{...}, search_space:{dram_bw,num_cores,sa,sram_kb},
         all_points:[{config:{dram_bw,num_cores,sa,sram_kb}, geo_mean_exe, area_mm2}],
         pareto_front:[...同上]} */

(function () {
  /* 散点可选维度:[key, label, 默认 log 轴] */
  const DIMS = [
    ["geo_mean_exe", t("Geo-mean latency (cyc)"), true],
    ["area_mm2", t("Area (mm²)"), true],
    ["config.num_cores", t("Core count"), false],
    ["config.sa", t("SA size"), false],
    ["config.sram_kb", "SRAM (KB)", true],
    ["config.dram_bw", t("DRAM bandwidth (GB/s)"), true],
  ];
  /* 着色维度(散点 marker.color)。 */
  const COLOR_DIMS = [
    ["", t("No coloring")], ["config.sa", t("SA size")], ["config.dram_bw", t("DRAM bandwidth")],
    ["config.num_cores", t("Core count")], ["config.sram_kb", "SRAM (KB)"],
  ];
  const DIM_LABEL = Object.fromEntries(DIMS.concat(COLOR_DIMS.filter(d => d[0])));
  const DIM_LOG = Object.fromEntries(DIMS.map(([k, , l]) => [k, l]));

  const getV = (p, key) => key.indexOf("config.") === 0 ? p.config[key.slice(7)] : p[key];
  const pKey = p =>
    [p.config.sa, p.config.dram_bw, p.config.num_cores, p.config.sram_kb, p.geo_mean_exe].join("|");
  const cfgText = p =>
    `sa${p.config.sa} · c${p.config.num_cores} · sram${p.config.sram_kb}K · bw${p.config.dram_bw}`;

  App.route("pareto", {
    title: t("Pareto"),
    async render(el) {
      const body = h("div");
      const tabs = UI.tabs([
        { key: "decode", label: t("Decode") },
        { key: "prefill", label: t("Prefill") },
      ], "decode", load);
      el.append(tabs.el, body);
      load("decode");

      async function load(mode) {
        body.replaceChildren(UI.loading(t("Loading Pareto results…")));
        let d;
        try {
          d = await App.api("/api/pareto?mode=" + mode);
        } catch (e) {
          body.replaceChildren(UI.empty("◭",
            t("Pareto results unavailable for mode={mode}: {msg} (launch a DSE sweep from the Run page first)", { mode, msg: e.message })));
          return;
        }
        if (!d.all_points || !d.all_points.length) {
          body.replaceChildren(UI.empty("📉", t("Pareto results are empty")));
          return;
        }
        paint(d);
      }

      /* 点详情 modal:完整 config + 指标 + 是否在前沿 */
      function pointModal(d, p) {
        const onFront = (d.__frontSet || (d.__frontSet = new Set(d.pareto_front.map(pKey)))).has(pKey(p));
        const kv = h("dl.kv", {},
          [[t("SA size"), p.config.sa], [t("DRAM bandwidth"), p.config.dram_bw + " GB/s"],
           [t("Core count"), p.config.num_cores], ["SRAM", p.config.sram_kb + " KB"],
           [t("Geo-mean latency"), fmt.num(p.geo_mean_exe) + " cyc"],
           [t("Area"), p.area_mm2.toFixed(2) + " mm²"],
           [t("Pareto front"), onFront ? t("✓ on the front") : t("no")]].map(([k, v]) =>
            [h("dt", {}, k), h("dd", {}, String(v))]));
        UI.modal(t("Design point details"), h("div", {},
          h("div.row", { style: { marginBottom: "12px" } },
            onFront ? UI.chip(t("Pareto front"), "warn") : UI.chip(t("Dominated point")),
            UI.chip(d.mode, d.mode === "decode" ? "accent" : "violet")),
          kv), [{ label: t("Close") }]);
      }

      function paint(d) {
        body.replaceChildren();

        /* ---------- 统计行 ---------- */
        const stats = h("div.grid.cols-4.mb");
        stats.append(
          UI.stat(t("Evaluated points"), d.num_points, t("Models: {list}", { list: (d.models || []).join(" / ") }), 0),
          UI.stat(t("Pareto front points"), d.num_pareto, t("{pct}% of all", { pct: (100 * d.num_pareto / Math.max(1, d.num_points)).toFixed(1) }), 60),
          UI.stat(t("Search combos"), Object.keys(d.search_space || {}).reduce(
            (n, k) => n * (d.search_space[k] || []).length, 1), t("theoretical full enumeration"), 120),
          UI.stat(t("Evaluation time"), (d.eval_time_sec / 3600).toFixed(1) + "h",
            String(d.timestamp || "").replace("T", " ").slice(0, 16), 180));
        body.append(stats);

        /* ---------- 固定参数 / 搜索空间 ---------- */
        const fixCard = h("div.card.mb", { style: { animationDelay: "80ms" } },
          UI.sectionTitle(t("Fixed params & search space")),
          h("div.row", { style: { marginBottom: "8px" } },
            Object.entries(d.fixed_params || {}).map(([k, v]) => UI.chip(`${k} = ${v}`))),
          h("div.small.muted", {}, t("Search space: ") + Object.entries(d.search_space || {})
            .map(([k, v]) => t("{k} ∈ [{lo} ~ {hi}] ({n} stops)", { k, lo: Math.min(...v), hi: Math.max(...v), n: v.length }))
            .join(" · ")));
        body.append(fixCard);

        /* ---------- 主散点图 ---------- */
        const mkDimSelect = (options, initial, onChange) => {
          const s = h("select", {
            style: { width: "auto", padding: "5px 9px", fontSize: "12px" },
            onchange: () => onChange(s.value),
          }, options.map(([v, l]) =>
            h("option", { value: v, selected: v === initial ? "" : null }, l)));
          return s;
        };
        const plotWrap = h("div");
        const xSel = mkDimSelect(DIMS.map(([k, l]) => [k, l]), "geo_mean_exe", () => drawScatter());
        const ySel = mkDimSelect(DIMS.map(([k, l]) => [k, l]), "area_mm2", () => drawScatter());
        const cSel = mkDimSelect(COLOR_DIMS.map(([k, l]) => [k, l]), "", () => drawScatter());
        const scatterCard = h("div.card.hoverable.mb", { style: { animationDelay: "140ms" } },
          UI.sectionTitle(t("Design-space scatter"), t("gold diamond line = Pareto front · click a point for details")),
          h("div.row", { style: { marginBottom: "10px", gap: "10px" } },
            h("label.field", { style: { flexDirection: "row", alignItems: "center", gap: "6px" } }, t("X axis"), xSel),
            h("label.field", { style: { flexDirection: "row", alignItems: "center", gap: "6px" } }, t("Y axis"), ySel),
            h("label.field", { style: { flexDirection: "row", alignItems: "center", gap: "6px" } }, t("Color"), cSel)),
          plotWrap);
        body.append(scatterCard);

        function drawScatter() {
          plotWrap.replaceChildren();
          const xd = xSel.value, yd = ySel.value, cd = cSel.value;
          const all = d.all_points;
          const vals = (pts, k) => pts.map(p => getV(p, k));
          const logOK = k => all.every(p => getV(p, k) > 0);
          const hover = p =>
            `${cfgText(p)}<br>${t("latency")} ${fmt.num(p.geo_mean_exe)} cyc<br>${t("area")} ${p.area_mm2.toFixed(1)} mm²`;
          const allTrace = {
            type: "scatter", mode: "markers", name: t("All points ({n})", { n: all.length }),
            x: vals(all, xd), y: vals(all, yd), text: all.map(hover),
            hovertemplate: "%{text}<extra></extra>",
            marker: { size: 7, opacity: 0.4, color: "#38bdf8" },
            customdata: all,
          };
          if (cd) {
            allTrace.marker.color = vals(all, cd);
            allTrace.marker.opacity = 0.7;
            allTrace.marker.colorscale = "Viridis";
            allTrace.marker.showscale = true;
            allTrace.marker.colorbar = {
              title: { text: DIM_LABEL[cd] || cd, font: { size: 11 } },
              tickfont: { size: 10 },
            };
          }
          const front = d.pareto_front.slice().sort((a, b) => getV(a, xd) - getV(b, xd));
          const frontTrace = {
            type: "scatter", mode: "lines+markers", name: t("Pareto front ({n})", { n: front.length }),
            x: vals(front, xd), y: vals(front, yd), text: front.map(hover),
            hovertemplate: "%{text}<extra>" + t("front") + "</extra>",
            line: { color: "#fbbf24", width: 2 },
            marker: { size: 10, color: "#fbbf24", symbol: "diamond" },
            customdata: front,
          };
          const div = Plot.chart(plotWrap, [allTrace, frontTrace], {
            height: 430,
            xaxis: { title: DIM_LABEL[xd] || xd, type: (DIM_LOG[xd] && logOK(xd)) ? "log" : "linear" },
            yaxis: { title: DIM_LABEL[yd] || yd, type: (DIM_LOG[yd] && logOK(yd)) ? "log" : "linear" },
          });
          div.on("plotly_click", ev => {
            const pt = ev.points && ev.points[0];
            if (pt && pt.customdata) pointModal(d, pt.customdata);
          });
        }
        drawScatter();

        /* ---------- 平行坐标图 ---------- */
        const pcCard = h("div.card.hoverable.mb", { style: { animationDelay: "200ms" } },
          UI.sectionTitle(t("Multi-dimensional trade-off"), t("parallel coords: hardware params → latency / area (color = log10 latency)")));
        const pts = d.all_points;
        const dim = (label, values) => ({ label, values });
        Plot.chart(pcCard, [{
          type: "parcoords",
          line: {
            color: pts.map(p => Math.log10(Math.max(1e-9, p.geo_mean_exe))),
            colorscale: "Viridis", showscale: true,
            colorbar: { title: { text: t("log10 latency"), font: { size: 11 } }, tickfont: { size: 10 } },
          },
          dimensions: [
            dim("SA", pts.map(p => p.config.sa)),
            dim("DRAM BW", pts.map(p => p.config.dram_bw)),
            dim(t("Cores"), pts.map(p => p.config.num_cores)),
            dim("SRAM KB", pts.map(p => p.config.sram_kb)),
            dim(t("log10 latency"), pts.map(p => +Math.log10(Math.max(1e-9, p.geo_mean_exe)).toFixed(3))),
            dim(t("log10 area"), pts.map(p => +Math.log10(Math.max(1e-9, p.area_mm2)).toFixed(3))),
          ],
        }], { height: 360, margin: { l: 60, r: 70, t: 36, b: 28 } });
        body.append(pcCard);

        /* ---------- 前沿点表格 ---------- */
        const frontRows = d.pareto_front.map(p => ({
          sa: p.config.sa, bw: p.config.dram_bw, cores: p.config.num_cores,
          sram: p.config.sram_kb, exe: p.geo_mean_exe, area: p.area_mm2, __p: p,
        }));
        const tblCard = h("div.card", { style: { animationDelay: "260ms" } },
          UI.sectionTitle(t("Pareto front points"), t("{n} points · click a row for details", { n: frontRows.length })));
        const tbl = UI.table([
          { key: "sa", label: "SA", num: true },
          { key: "bw", label: "DRAM BW", num: true },
          { key: "cores", label: t("Cores"), num: true },
          { key: "sram", label: "SRAM (KB)", num: true },
          { key: "exe", label: t("Geo-mean latency (cyc)"), num: true, render: v => fmt.num(v) },
          { key: "area", label: t("Area (mm²)"), num: true, render: v => v.toFixed(1) },
        ], frontRows, { onRow: r => pointModal(d, r.__p), maxHeight: "420px" });
        tblCard.append(tbl.el);
        body.append(tblCard);
      }
    },
  });
})();
