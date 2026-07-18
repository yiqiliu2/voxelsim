/* Sweep — 参数扫描页:单维参数扫描的聚合指标曲线。
   数据:GET /api/sweep?x=&metric=&group=&<固定过滤>
   响应:{x_key, metric, group_key, count, x:[...], x_log:bool,
         series:[{name, points:[{x, y, ymin, ymax, n, ids}]}]} */

(function () {
  /* metrics_meta 的 label 为英文,这里提供中文映射(缺省回退英文 label)。 */
  const ZH_METRIC = {
    total_time: t("Execution time"), time_ms: t("Latency"), total_energy_mj: t("Total energy"),
    dynamic_energy_mj: t("Dynamic energy"), static_energy_mj: t("Static energy"),
    avg_power_w: t("Average power"), dynamic_power_w: t("Dynamic power"), static_power_w: t("Static power"),
    sa_util: t("SA utilization"), vu_util: t("VU utilization"), noc_util: t("NoC utilization"),
    overall_util: t("Overall utilization"), dram_r_util: t("DRAM read utilization"), dram_w_util: t("DRAM write utilization"),
    mm_gflops: t("Matrix throughput"), vu_gflops: t("Vector throughput"),
  };

  /* x / group 可选维度(与后端 X_KEYS 对齐:FILTER_KEYS + trp)。 */
  const X_DEFS = [
    ["dram_bw", t("DRAM bandwidth")], ["num_cores", t("Core count")], ["sa_size", t("SA array size")],
    ["sram_kb", t("SRAM capacity")], ["noc_bw", t("NoC bandwidth")], ["noc_topo", t("NoC Topology")],
    ["core_group", "Core Group"], ["batch_size", "Batch Size"],
    ["seq_length", t("Sequence length")], ["row", "DRAM Row"], ["trp", "tRP"],
    ["impl", t("Impl")], ["model", t("Model")], ["mode", t("Mode")], ["root", t("Root")],
  ];
  const X_LABEL = Object.fromEntries(X_DEFS);

  const FIX_FILTERS = [
    ["model", t("Model")], ["mode", t("Mode")], ["impl", t("Impl")],
    ["batch_size", "Batch"], ["num_cores", t("Core count")],
  ];

  const metricLabel = (meta, key) => {
    const m = meta[key] || {};
    const base = ZH_METRIC[key] || m.label || key;
    return m.unit ? `${base} (${m.unit})` : base;
  };

  const fmtM = (meta, key, v) => {
    if (v == null || Number.isNaN(v)) return "—";
    if (key === "time_ms") return fmt.ms(v * 1e3);
    const unit = (meta[key] || {}).unit;
    if (key === "total_time" || unit === "cycles") return fmt.num(v) + " cyc";
    if (unit === "%") return fmt.pct(v);
    if (unit === "mJ") return fmt.num(v) + " mJ";
    if (unit === "W") return fmt.num(v) + " W";
    if (unit === "GFLOPS") return fmt.num(v) + " GF";
    return fmt.num(v);
  };

  const hexA = (hex, a) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };

  /* 点击数据点 → 列出该聚合桶的来源结果 id,可跳转结果详情页。 */
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
        t("{n} results, click to open the result detail", { n: ids.length })),
      list), [{ label: t("Close") }]);
  };

  App.route("sweep", {
    title: t("Sweep"),
    async render(el) {
      el.append(UI.loading(t("Loading metrics and filter definitions…")));
      let meta, filters;
      try {
        [meta, filters] = await Promise.all([App.metricsMeta(), App.filters()]);
      } catch (e) {
        el.replaceChildren(UI.empty("⚠", t("Failed to load definitions: {msg}", { msg: e.message })));
        return;
      }
      el.replaceChildren();

      const state = {
        x: "dram_bw", metric: "total_time", group: "",
        model: "llama2-13", mode: "decode", impl: "", batch_size: "", num_cores: "",
      };
      let seq = 0;   // 防旧响应覆盖新图

      const mkSelect = (label, options, initial, onChange) => {
        const s = h("select", { onchange: () => onChange(s.value) },
          options.map(o => h("option",
            { value: o.value, selected: String(o.value) === String(initial) ? "" : null },
            o.label)));
        return h("label.field", {}, label, s);
      };

      /* 指标选项:常用指标置顶,其余按 metrics_meta 顺序。 */
      const preferred = ["total_time", "time_ms", "total_energy_mj", "avg_power_w"];
      const metricKeys = preferred.filter(k => meta[k])
        .concat(Object.keys(meta).filter(k => preferred.indexOf(k) < 0));
      const metricOpts = metricKeys.map(k => ({ value: k, label: metricLabel(meta, k) }));

      const countChip = h("span.chip.accent", {}, "—");
      const axisChip = h("span.chip", {}, "");
      const bar = h("div.card.filterbar", {},
        mkSelect(t("X-axis parameter"), X_DEFS.map(([v, l]) => ({ value: v, label: l })), state.x,
          v => { state.x = v; reload(); }),
        mkSelect(t("Metric"), metricOpts, state.metric, v => { state.metric = v; reload(); }),
        mkSelect(t("Group by"), [{ value: "", label: t("No grouping") }]
          .concat(X_DEFS.map(([v, l]) => ({ value: v, label: l }))), state.group,
          v => { state.group = v; reload(); }),
        h("span", { style: { width: "1px", alignSelf: "stretch", background: "var(--border)" } }),
        FIX_FILTERS.map(([key, label]) => mkSelect(label,
          [{ value: "", label: t("All") }].concat(
            (filters[key] || []).map(v => ({ value: String(v), label: String(v) }))),
          state[key], v => { state[key] = v; reload(); })),
        h("span", { class: "spacer" }),
        h("span.row", { style: { gap: "6px" } }, countChip, axisChip));

      const chartBody = h("div");
      const chartCard = h("div.card.hoverable", { style: { animationDelay: "80ms" } },
        UI.sectionTitle(t("Sweep curve"), t("Mean line + min/max envelope · hover for n · click a point to see source results")),
        chartBody);
      el.append(bar, chartCard);

      async function reload() {
        const mySeq = ++seq;
        chartBody.replaceChildren(UI.loading(t("Aggregating…")));
        const qs = new URLSearchParams({ x: state.x, metric: state.metric });
        if (state.group) qs.set("group", state.group);
        for (const [k] of FIX_FILTERS) if (state[k] !== "") qs.set(k, state[k]);
        let data;
        try {
          data = await App.api("/api/sweep?" + qs.toString());
        } catch (e) {
          if (mySeq !== seq) return;
          countChip.textContent = t("Query failed");
          axisChip.textContent = "";
          chartBody.replaceChildren(UI.empty("⚠", e.message));
          return;
        }
        if (mySeq !== seq) return;
        draw(data);
      }

      function draw(data) {
        const nPts = data.series.reduce((n, s) => n + s.points.length, 0);
        countChip.textContent = t("{count} matches · {n} aggregated points", { count: data.count, n: nPts });
        if (!nPts) {
          axisChip.textContent = "";
          chartBody.replaceChildren(UI.empty("📉", t("No data under current filters; adjust the filters or the X axis")));
          return;
        }
        const xIsCat = data.x.some(v => typeof v === "string");
        axisChip.textContent = xIsCat ? t("category x-axis") : (data.x_log ? t("log x-axis") : t("linear x-axis"));

        const single = data.series.length === 1;
        const xName = X_LABEL[data.x_key] || data.x_key;
        const traces = [];
        data.series.forEach((s, i) => {
          const color = Plot.COLORS[i % Plot.COLORS.length];
          const xs = s.points.map(p => p.x);
          const name = single ? metricLabel(meta, data.metric) : s.name;
          if (s.points.some(p => p.n > 1)) {
            traces.push({ x: xs, y: s.points.map(p => p.ymax), type: "scatter", mode: "lines",
              line: { width: 0 }, showlegend: false, hoverinfo: "skip",
              legendgroup: "g" + i, name: name + "·max" });
            traces.push({ x: xs, y: s.points.map(p => p.ymin), type: "scatter", mode: "lines",
              line: { width: 0 }, fill: "tonexty", fillcolor: hexA(color, 0.12),
              showlegend: false, hoverinfo: "skip",
              legendgroup: "g" + i, name: name + "·min" });
          }
          traces.push({
            x: xs, y: s.points.map(p => p.y), type: "scatter", mode: "lines+markers",
            name, legendgroup: "g" + i,
            line: { color, width: 2.5 }, marker: { color, size: 8 },
            text: s.points.map(p =>
              `${xName} = ${p.x}<br>` +
              t("mean {v}<br>", { v: fmtM(meta, data.metric, p.y) }) +
              t("range {a} ~ {b}<br>", { a: fmtM(meta, data.metric, p.ymin), b: fmtM(meta, data.metric, p.ymax) }) +
              `n = ${p.n}`),
            hovertemplate: "%{text}<extra>" + name + "</extra>",
            customdata: s.points,
          });
        });

        chartBody.replaceChildren();
        const xaxis = xIsCat
          ? { title: xName, type: "category", categoryorder: "array", categoryarray: data.x }
          : { title: xName, type: data.x_log ? "log" : "linear" };
        const div = Plot.chart(chartBody, traces, {
          height: 430, showlegend: !single,
          xaxis, yaxis: { title: metricLabel(meta, data.metric) },
        });
        div.on("plotly_click", ev => {
          const pt = ev.points && ev.points[0];
          const d = pt && pt.customdata;
          if (d && d.ids && d.ids.length)
            showIds(d.ids, `${xName} = ${d.x} · ${pt.data.name}`);
        });
      }

      reload();
    },
  });
})();
