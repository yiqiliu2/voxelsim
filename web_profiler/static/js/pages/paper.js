/* Paper — 论文图表库:fig10~fig20 的一键图数据。
   数据:GET /api/paper/<fig>?mode=&model=
   响应:{fig, title, metric?, mode?, model?, panels:[...]}
   panel 三种结构(通用渲染器全覆盖):
     ① series 面板 (fig10/11/12/13/15/17/18):
        {title, series:[{name, points:[{x,y,ymin,ymax,n,ids,relaxed?}]}],
         x?, x_log?, x_labels?, mode?, model?, sweep?, x_key?, relaxed?}
     ② components 面板 (fig19): {title, sweep, x:[...], components:[{name, values}]}
     ③ categories 面板 (fig20): {title, mode, seq_length, x_labels, categories:[{key,label,values_ms}]} */

(function () {
  const METRIC_LABEL = {
    total_time: t("Execution time"), time_ms: t("Latency"), total_energy_mj: t("Total energy"),
    dynamic_energy_mj: t("Dynamic energy"), static_energy_mj: t("Static energy"),
    avg_power_w: t("Average power"), dynamic_power_w: t("Dynamic power"), static_power_w: t("Static power"),
    sa_util: t("SA utilization"), vu_util: t("VU utilization"), noc_util: t("NoC utilization"),
    overall_util: t("Overall utilization"), dram_r_util: t("DRAM read utilization"), dram_w_util: t("DRAM write utilization"),
    mm_gflops: t("Matrix throughput"), vu_gflops: t("Vector throughput"),
  };

  /* 图库静态目录(与 api_analysis.py 的 build_figN 一一对应)。 */
  const FIGS = [
    { id: "fig10", en: "Compute Paradigm Comparison",
      desc: t("Comparison of the three implementations best / spmd_compiler / dataflow across models, in decode + prefill dual panels."),
      params: [] },
    { id: "fig11", en: "NoC Topology x Mapping",
      desc: t("Mesh / Torus / All-to-All topologies combined with best / seq_noc mappings, one panel per model."),
      params: ["mode"] },
    { id: "fig12", en: "DRAM Bandwidth & tRP Sensitivity",
      desc: t("DRAM bandwidth sweep and tRP sweep curves for llama2-13 decode."), params: [] },
    { id: "fig13", en: "Tensor-to-Bank Placement",
      desc: t("Comparison of the best and uniform_dram placement strategies across models, in decode + prefill dual panels."),
      params: [] },
    { id: "fig15", en: "Hardware Sweep",
      desc: t("Five-dimension sweep over noc_bw / dram_bw / sa_size / num_cores / sram_kb, with models as series."),
      params: ["mode"] },
    { id: "fig17", en: "Core Group Smile Curve",
      desc: t("Latency smile curves of cg=1/2/4/8 over equal-FLOPS (core count / SA) configuration pairs."), params: [] },
    { id: "fig18", en: "Energy vs Bandwidth / Cores",
      desc: t("Total energy vs DRAM bandwidth and core count, with models as series."), params: ["mode"] },
    { id: "fig19", en: "Component Energy Breakdown",
      desc: t("Stacked bars of SA / VU / SRAM / DRAM / TSV / NoC static + dynamic energy."),
      params: ["model", "mode"] },
    { id: "fig20", en: "Operator Type Breakdown",
      desc: t("Operator-category time breakdown stacked across sequence lengths (1024/2048/4096) and batch sizes."), params: ["model"] },
  ];

  /* series 面板的 x 轴标题兜底(panel 未带 sweep/x_key 时按图猜)。 */
  const X_FALLBACK = {
    fig10: "model", fig11: "noc_topo", fig13: "model", fig17: t("num_cores / sa_size (equal FLOPS)"),
  };

  let META = {};   // metrics_meta 缓存(render 时填充)

  const metricLabel = key => {
    const m = META[key] || {};
    const base = METRIC_LABEL[key] || m.label || key;
    return m.unit ? `${base} (${m.unit})` : base;
  };

  const fmtM = (key, v) => {
    if (v == null || Number.isNaN(v)) return "—";
    if (key === "time_ms") return fmt.ms(v * 1e3);
    const unit = (META[key] || {}).unit;
    if (key === "total_time" || unit === "cycles") return fmt.num(v) + " cyc";
    if (unit === "%") return fmt.pct(v);
    if (unit === "mJ") return fmt.num(v) + " mJ";
    if (unit === "W") return fmt.num(v) + " W";
    if (unit === "GFLOPS") return fmt.num(v) + " GF";
    return fmt.num(v);
  };

  /* 点击数据点 → 来源结果 id 列表 modal(与 sweep 页一致)。 */
  const showIds = (ids, title) => {
    let close = () => {};
    const list = h("div",
      { style: { maxHeight: "380px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px" } },
      ids.map((id, i) => h("a", {
        href: "#/result?id=" + encodeURIComponent(id), title: id,
        onclick: () => close(),
        style: {
          fontFamily: "var(--mono)", fontSize: "11.5px", padding: "6px 10px",
          borderRadius: "7px", background: "#0d1422", border: "1px solid var(--border)",
          display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        },
      }, `${i + 1}. …/${id.split("/").slice(-4).join("/")}`)));
    close = UI.modal(title, h("div", {},
      h("div.small.muted", { style: { marginBottom: "10px" } },
        t("{n} results, click to view the result detail", { n: ids.length })),
      list), [{ label: t("Close") }]);
  };

  /* ---------------- 通用 panel 渲染器 ----------------
     返回 {traces, layout, hasIds} | {empty:true} | null(结构不支持)。 */
  function buildPanelTraces(panel, resp) {
    const yTitle = resp.metric ? metricLabel(resp.metric)
      : panel.components ? t("Energy (mJ)")
      : panel.categories ? t("Time (ms)") : "";

    /* ② fig19:components → 堆叠柱状图 */
    if (panel.components && panel.components.length) {
      const traces = panel.components.map((c, i) => ({
        type: "bar", name: c.name, x: panel.x, y: c.values,
        marker: { color: Plot.COLORS[i % Plot.COLORS.length] },
        hovertemplate: "%{x}<br>" + c.name + ": %{y:.3g} mJ<extra></extra>",
      }));
      return {
        traces, hasIds: false,
        layout: {
          height: 340, barmode: "stack",
          xaxis: {
            title: panel.sweep || "", type: "category",
            categoryorder: "array", categoryarray: panel.x,
          },
          yaxis: { title: yTitle },
        },
      };
    }

    /* ③ fig20:categories → 堆叠柱状图(值为 ms) */
    if (panel.categories && panel.categories.length) {
      const traces = panel.categories.map((c, i) => ({
        type: "bar", name: c.label, x: panel.x_labels, y: c.values_ms,
        marker: { color: Plot.COLORS[i % Plot.COLORS.length] },
        hovertemplate: "%{x}<br>" + c.label + ": %{y:.3g} ms<extra></extra>",
      }));
      return {
        traces, hasIds: false,
        layout: {
          height: 340, barmode: "stack",
          xaxis: { title: "batch size", type: "category" },
          yaxis: { title: yTitle },
        },
      };
    }

    /* ① series 面板:数值 x → 折线(+error_y);字符串 x → 分组柱状;
       panel.x_labels 存在(fig17)时按类别轴画折线。 */
    if (panel.series && panel.series.length) {
      const totalPts = panel.series.reduce((n, s) => n + s.points.length, 0);
      if (!totalPts) return { empty: true };
      const cat = panel.series.some(s => s.points.some(p => typeof p.x === "string"));
      const order = panel.x_labels || panel.x || null;
      const isBar = cat && !panel.x_labels;
      const xTitle = panel.sweep || panel.x_key || X_FALLBACK[resp.fig] || "";
      const single = panel.series.length === 1;
      const traces = [];
      let hasIds = false;
      panel.series.forEach((s, i) => {
        const color = Plot.COLORS[i % Plot.COLORS.length];
        const xs = s.points.map(p => p.x);
        const err = {
          type: "data", symmetric: false, color, thickness: 1.4, width: 4,
          array: s.points.map(p => Math.max(0, p.ymax - p.y)),
          arrayminus: s.points.map(p => Math.max(0, p.y - p.ymin)),
        };
        const text = s.points.map(p =>
          `x = ${p.x}<br>` + t("Mean {v}", { v: fmtM(resp.metric, p.y) }) + `<br>` +
          t("Range {min} ~ {max}", { min: fmtM(resp.metric, p.ymin), max: fmtM(resp.metric, p.ymax) }) + `<br>` +
          `n = ${p.n}` + (p.relaxed ? "<br>" + t("⚠ relaxed (loose match)") : ""));
        if (s.points.some(p => p.ids && p.ids.length)) hasIds = true;
        if (isBar) {
          traces.push({
            type: "bar", name: s.name, x: xs, y: s.points.map(p => p.y),
            error_y: err, marker: { color }, text,
            hovertemplate: "%{text}<extra>" + s.name + "</extra>",
            customdata: s.points,
          });
        } else {
          traces.push({
            type: "scatter", mode: "lines+markers", name: s.name,
            x: xs, y: s.points.map(p => p.y), error_y: err,
            line: { color, width: 2.4 }, marker: { color, size: 8 }, text,
            hovertemplate: "%{text}<extra>" + s.name + "</extra>",
            customdata: s.points,
          });
        }
      });
      let xaxis;
      if (cat) {
        xaxis = { title: xTitle, type: "category", categoryorder: "array" };
        if (order) xaxis.categoryarray = order;
      } else {
        xaxis = { title: xTitle, type: panel.x_log ? "log" : "linear" };
      }
      return {
        traces, hasIds,
        layout: {
          height: 340, barmode: "group", showlegend: !single,
          xaxis, yaxis: { title: yTitle },
        },
      };
    }

    return null;   // 未识别结构
  }

  function renderPanel(panel, resp, idx) {
    const sub = [];
    if (panel.relaxed) sub.push(UI.chip(t("Contains relaxed points"), "warn"));
    if (panel.sweep) sub.push(UI.chip("x: " + panel.sweep));
    if (panel.x_key) sub.push(UI.chip("x: " + panel.x_key));
    if (panel.mode) sub.push(UI.modeChip(panel.mode));
    if (panel.seq_length) sub.push(UI.chip("seq " + panel.seq_length));
    const card = h("div.card.hoverable", { style: { animationDelay: idx * 70 + "ms" } },
      h("h3.card-title", {}, panel.title || t("Panel {n}", { n: idx + 1 }),
        h("span.hint", {}, sub)));
    const built = buildPanelTraces(panel, resp);
    if (!built) {
      card.append(UI.empty("🧩",
        t("This panel structure cannot be rendered yet (fields: {fields})", { fields: Object.keys(panel).join(", ") })));
      return card;
    }
    if (built.empty) {
      card.append(UI.empty("📉", t("No data for this panel under the current parameters")));
      return card;
    }
    const div = Plot.chart(card, built.traces, built.layout);
    if (built.hasIds) {
      div.on("plotly_click", ev => {
        const pt = ev.points && ev.points[0];
        const pd = pt && pt.customdata;
        if (pd && pd.ids && pd.ids.length)
          showIds(pd.ids, (panel.title || "") + " · x = " + pd.x);
      });
    }
    return card;
  }

  /* ---------------- 路由 ---------------- */
  App.route("paper", {
    title: t("Paper Figures"),
    async render(el, params) {
      try { META = await App.metricsMeta(); } catch (e) { META = {}; }
      if (params.fig && FIGS.some(f => f.id === params.fig)) {
        await paintDetail(el, params.fig, params);
      } else {
        paintGallery(el);
      }
    },
  });

  /* ---------------- 图库 ---------------- */
  function paintGallery(el) {
    el.append(h("div.card.mb", {},
      UI.sectionTitle(t("Paper Figure Gallery"), t("fig10 ~ fig20 · click a card to load its data")),
      h("div.small.muted", {},
        t("Data is aggregated in one shot by /api/paper/<fig>, consistent with the paper evaluation; some (model, impl) buckets relax num_cores/batch_size when strict matching finds no data, and panels and points are marked relaxed."))));
    const grid = h("div.grid.cols-3");
    FIGS.forEach((f, i) => {
      const card = h("div.card.hoverable", {
        style: { cursor: "pointer", animationDelay: i * 50 + "ms" },
        onclick: () => App.navigate("paper", { fig: f.id }),
      },
        h("div", { style: { display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" } },
          h("span", {
            style: {
              fontFamily: "var(--mono)", fontWeight: "800", fontSize: "13px",
              color: "var(--accent)", background: "#38bdf81a",
              border: "1px solid #38bdf844", borderRadius: "7px", padding: "3px 9px",
            },
          }, f.id.toUpperCase()),
          h("span", { class: "grow" }),
          f.params.indexOf("mode") >= 0 ? UI.chip(t("mode optional")) : null,
          f.params.indexOf("model") >= 0 ? UI.chip(t("model optional"), "violet") : null),
        h("div", { style: { fontWeight: "650", fontSize: "14.5px", marginBottom: "3px" } }, t(f.en)),
        h("div.small.muted", { style: { marginBottom: "8px" } }, App.lang === "en" ? "" : f.en),
        h("div.small", { style: { color: "var(--text-2)" } }, f.desc));
      grid.append(card);
    });
    el.append(grid);
  }

  /* ---------------- 详情视图 ---------------- */
  async function paintDetail(el, figId, params) {
    const def = FIGS.find(f => f.id === figId);
    const state = {
      mode: params.mode === "prefill" ? "prefill" : "decode",
      model: params.model || "llama3-70",
    };

    const body = h("div");
    el.append(body);

    /* 参数控件(仅该图声明支持的才显示) */
    const useMode = def.params.indexOf("mode") >= 0;
    const useModel = def.params.indexOf("model") >= 0;
    const modeSeg = h("div.seg", {},
      [{ value: "decode", label: "Decode" }, { value: "prefill", label: "Prefill" }].map(o =>
        h("span.seg-opt", {
          class: "seg-opt" + (o.value === state.mode ? " on" : ""),
          dataset: { v: o.value },
          onclick: () => {
            if (state.mode === o.value) return;
            state.mode = o.value; syncSeg(); load();
          },
        }, o.label)));
    const syncSeg = () => modeSeg.querySelectorAll(".seg-opt").forEach(o =>
      o.classList.toggle("on", o.dataset.v === state.mode));

    const modelSel = h("select", {
      onchange: () => { state.model = modelSel.value; load(); },
    }, h("option", { value: state.model }, state.model));
    if (useModel) {
      App.filters().then(filters => {
        (filters.model || []).forEach(m => {
          if (m !== state.model) modelSel.append(h("option", { value: m }, m));
        });
      }).catch(() => {});
    }

    const headCard = h("div.card.mb", {},
      h("div.row", {},
        h("button.btn.ghost.sm", { onclick: () => App.navigate("paper") }, "← " + t("Gallery")),
        h("div", { class: "grow" },
          h("div", { style: { fontWeight: "700", fontSize: "15px" } },
            `${def.id.toUpperCase()} · ${t(def.en)}`),
          h("div.small.muted", {}, App.lang === "en" ? "" : def.en)),
        useMode ? modeSeg : null,
        useModel ? h("label.field", { style: { minWidth: "150px" } }, t("Model"), modelSel) : null));
    el.insertBefore(headCard, body);

    async function load() {
      body.replaceChildren(UI.loading(t("Aggregating figure data…")));
      const qs = new URLSearchParams();
      if (useMode) qs.set("mode", state.mode);
      if (useModel) qs.set("model", state.model);
      let d;
      try {
        d = await App.api(`/api/paper/${figId}?` + qs.toString());
      } catch (e) {
        body.replaceChildren(UI.empty("⚠", `${t(def.en)}: ${e.message}`));
        return;
      }
      paintFig(d);
    }

    function paintFig(d) {
      body.replaceChildren();
      const panels = d.panels || [];
      if (!panels.length) {
        body.replaceChildren(UI.empty("📉", t("No data for this figure under the current parameters")));
        return;
      }
      body.append(h("div.row.mb", {},
        d.metric ? UI.chip(t("Metric: {label}", { label: metricLabel(d.metric) }), "accent") : null,
        d.mode ? UI.modeChip(d.mode) : null,
        d.model ? UI.chip(d.model, "violet") : null,
        h("span.small.muted", {}, t("{n} panels · click a data point to view source results", { n: panels.length }))));
      const grid = h("div", { class: panels.length > 1 ? "grid cols-2" : "grid" });
      panels.forEach((p, i) => grid.append(renderPanel(p, d, i)));
      body.append(grid);
    }

    load();
  }
})();
