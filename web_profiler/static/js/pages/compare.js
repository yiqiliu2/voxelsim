/* Compare — 配置对比页:勾选 2~12 条结果并排比较。
   选择器:GET /api/results(复用过滤与 q 搜索)
   对比:GET /api/compare?ids=a,b,c
   响应:{items:[{id, config:{...,impl_label,topo_label}, metrics:{...,time_ms}}]} */

(function () {
  const ZH_METRIC = {
    total_time: t("Execution time"), time_ms: t("Latency"), total_energy_mj: t("Total energy"),
    dynamic_energy_mj: t("Dynamic energy"), static_energy_mj: t("Static energy"),
    avg_power_w: t("Average power"), dynamic_power_w: t("Dynamic power"), static_power_w: t("Static power"),
    sa_util: t("SA utilization"), vu_util: t("VU utilization"), noc_util: t("NoC utilization"),
    overall_util: t("Overall utilization"), dram_r_util: t("DRAM read utilization"), dram_w_util: t("DRAM write utilization"),
    mm_gflops: t("Matrix throughput"), vu_gflops: t("Vector throughput"),
  };

  /* 分组柱状图选用的关键指标(按此顺序,缺数据的自动跳过)。 */
  const KEY_METRICS = ["time_ms", "total_energy_mj", "avg_power_w",
                       "mm_gflops", "sa_util", "dram_r_util"];

  /* 明细表"配置"小节的行:[key, label, render?] */
  const CFG_ROWS = [
    ["model", t("Model")], ["mode", t("Mode"), v => UI.modeChip(v)],
    ["impl", t("Impl"), v => UI.implChip(v)], ["root", t("Root")],
    ["batch_size", "Batch Size"], ["num_cores", t("Core count")],
    ["sa_size", t("SA array size")], ["sram_kb", t("SRAM capacity (KB)")],
    ["dram_bw", t("DRAM bandwidth (GB/s)")],
    ["noc_topo", t("NoC Topology"), (v, c) => c.topo_label || v],
    ["noc_bw", t("NoC bandwidth (B/cyc)")], ["core_group", "Core Group"],
    ["row", "DRAM Row"], ["seq_length", t("Sequence length")], ["trp", "tRP"], ["trcd", "tRCD"],
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

  const cfgShort = c =>
    `c${c.num_cores}·sa${c.sa_size}·sram${c.sram_kb}K·bw${c.dram_bw}·cg${c.core_group}`;
  const cfgLong = c =>
    `${c.model}/${c.mode} ${c.impl} bs${c.batch_size} ${cfgShort(c)}`;

  App.route("compare", {
    title: t("Compare"),
    async render(el, params) {
      el.append(UI.loading(t("Loading definitions…")));
      let meta, filters;
      try {
        [meta, filters] = await Promise.all([App.metricsMeta(), App.filters()]);
      } catch (e) {
        el.replaceChildren(UI.empty("⚠", t("Failed to load definitions: {msg}", { msg: e.message })));
        return;
      }
      el.replaceChildren();

      const state = { ids: [], q: "", model: "", mode: "", impl: "" };
      if (params.ids)
        state.ids = params.ids.split(",").map(s => s.trim()).filter(Boolean).slice(0, 12);
      const idInfo = {};   // id -> config(来自选择列表,用于 chip 标签)

      /* ---------------- 选择器 ---------------- */
      const chipsRow = h("div", { style: { display: "flex", flexWrap: "wrap", gap: "6px" } });
      const actionsRow = h("div.row", { style: { marginTop: "10px" } });
      const listWrap = h("div");
      const pickerCard = h("div.card.mb", {},
        UI.sectionTitle(t("Select results"), t("Check 2~12 to compare")),
        chipsRow, actionsRow, listWrap);
      const outEl = h("div.section-gap");
      el.append(pickerCard, outEl);

      let searchTimer = null;
      const mkFilter = (label, key, values) => {
        const s = h("select", { onchange: () => { state[key] = s.value; loadList(); } },
          [h("option", { value: "" }, t("All"))].concat(
            (values || []).map(v => h("option", { value: String(v) }, String(v)))));
        return h("label.field", {}, label, s);
      };
      const filterRow = h("div.row", { style: { margin: "12px 0" } },
        mkFilter(t("Model"), "model", filters.model),
        mkFilter(t("Mode"), "mode", filters.mode),
        mkFilter(t("Impl"), "impl", filters.impl),
        h("label.field", {}, t("Search (ID substring)"),
          h("input", {
            type: "search", placeholder: t("e.g. llama2-13/bs_32 …"), style: { width: "200px" },
            oninput: e => {
              clearTimeout(searchTimer);
              searchTimer = setTimeout(() => { state.q = e.target.value.trim(); loadList(); }, 300);
            },
          })));
      listWrap.append(filterRow);

      const tbl = UI.table([
        { key: "__sel", label: t("Sel"), sortable: false,
          render: (v, r) => state.ids.indexOf(r.id) >= 0 ? UI.chip("✓", "accent") : "" },
        { key: "model", label: t("Model") },
        { key: "mode", label: t("Mode"), render: v => UI.modeChip(v) },
        { key: "batch_size", label: "bs", num: true },
        { key: "num_cores", label: t("Cores"), num: true },
        { key: "sa_size", label: "SA", num: true },
        { key: "dram_bw", label: "BW", num: true },
        { key: "core_group", label: "CG", num: true },
        { key: "impl", label: t("Impl"), render: v => UI.implChip(v) },
        { key: "id", label: t("Result ID"), sortable: false,
          render: v => h("span.muted", { title: v }, "…/" + v.split("/").slice(-2).join("/")) },
      ], [], { onRow: toggleId });
      listWrap.append(tbl.el);

      function toggleId(row) {
        const i = state.ids.indexOf(row.id);
        if (i >= 0) state.ids.splice(i, 1);
        else {
          if (state.ids.length >= 12) { UI.toast(t("Select at most 12 results"), "error"); return; }
          state.ids.push(row.id);
        }
        paintChips();
        tbl.setRows(lastRows);
      }

      function paintChips() {
        chipsRow.replaceChildren(...(state.ids.length
          ? state.ids.map(id => {
              const c = idInfo[id];
              const label = c ? cfgShort(c) : "…/" + id.split("/").slice(-2).join("/");
              return h("span.chip.accent", { title: id }, label,
                h("b", {
                  style: { cursor: "pointer", marginLeft: "2px" },
                  onclick: () => {
                    state.ids.splice(state.ids.indexOf(id), 1);
                    paintChips(); tbl.setRows(lastRows);
                  },
                }, " ×"));
            })
          : [h("span.small.muted", {}, t("Nothing selected yet — click rows in the list below to check"))]));
        const url = location.origin + location.pathname +
          "#/compare?ids=" + state.ids.map(encodeURIComponent).join(",");
        actionsRow.replaceChildren(
          h("span", { class: "grow" }),
          h("span.small.muted", {}, t("{n}/12 selected", { n: state.ids.length })),
          UI.copyBtn(url, t("Compare link")),
          h("button.btn.primary", {
            disabled: state.ids.length < 2 ? "" : null,
            onclick: runCompare,
          }, t("⚖ Run compare")));
      }

      let lastRows = [];
      async function loadList() {
        tbl.setRows([]);
        const qs = new URLSearchParams({ page_size: "60" });
        if (state.q) qs.set("q", state.q);
        for (const k of ["model", "mode", "impl"]) if (state[k]) qs.set(k, state[k]);
        try {
          const d = await App.api("/api/results?" + qs.toString());
          lastRows = d.rows || [];
          lastRows.forEach(r => { idInfo[r.id] = r; });
          tbl.setRows(lastRows);
        } catch (e) {
          UI.toast(t("Failed to load result list: {msg}", { msg: e.message }), "error");
        }
      }

      /* ---------------- 对比 ---------------- */
      async function runCompare() {
        if (state.ids.length < 2) { UI.toast(t("Select at least 2 results"), "error"); return; }
        outEl.replaceChildren(UI.loading(t("Computing comparison…")));
        let d;
        try {
          d = await App.api("/api/compare?ids=" + state.ids.map(encodeURIComponent).join(","));
        } catch (e) {
          outEl.replaceChildren(UI.empty("⚠", t("Comparison failed: {msg}", { msg: e.message })));
          return;
        }
        d.items.forEach(it => { idInfo[it.id] = it.config; });
        paintChips();
        paintCompare(d.items);
      }

      function paintCompare(items) {
        outEl.replaceChildren();
        const labels = items.map((it, i) => "#" + (i + 1));

        /* 配置速览(图例) */
        const legendCard = h("div.card.mb", {},
          UI.sectionTitle(t("Compared configs"), t("{n} items · numbers match the x-axis below", { n: items.length })),
          h("div", { style: { display: "flex", flexDirection: "column", gap: "7px" } },
            items.map((it, i) => h("div.row", { style: { gap: "9px" } },
              h("b", { style: { color: Plot.COLORS[i % Plot.COLORS.length], fontFamily: "var(--mono)" } },
                "#" + (i + 1)),
              h("span.mono", { style: { fontSize: "11.5px" } }, cfgLong(it.config)),
              UI.copyBtn(it.id, "ID")))));
        outEl.append(legendCard);

        /* ① 关键指标分组柱状图(每个指标一张小图,各配置并排) */
        const present = KEY_METRICS.filter(k => items.some(it => it.metrics[k] != null));
        if (present.length) {
          const grid = h("div.grid.cols-3.mb");
          present.forEach((k, gi) => {
            const card = h("div.card.hoverable", { style: { animationDelay: gi * 60 + "ms" } },
              UI.sectionTitle(metricLabel(meta, k),
                (meta[k] || {}).better === "higher" ? t("higher is better") : t("lower is better")));
            const vals = items.map(it => it.metrics[k]);
            Plot.chart(card, [{
              type: "bar", x: labels, y: vals,
              marker: {
                color: items.map((_, i) => Plot.COLORS[i % Plot.COLORS.length]),
                opacity: 0.85,
              },
              text: vals.map(v => fmtM(meta, k, v)),
              hovertemplate: "%{x}: %{text}<extra></extra>",
            }], {
              height: 220, showlegend: false,
              margin: { l: 54, r: 12, t: 8, b: 34 },
              yaxis: { title: (meta[k] || {}).unit || "" },
            });
            grid.append(card);
          });
          outEl.append(grid);
        }

        /* ② 明细表:每列一个配置,每行一个指标;best 绿 / worst 红 */
        const tblCard = h("div.card", { style: { animationDelay: "200ms" } },
          UI.sectionTitle(t("Metric details"), t("green = best · red = worst (per metrics_meta.better)")));
        const sectionRow = text => h("tr", {},
          h("td", {
            colspan: items.length + 1,
            style: {
              color: "var(--text-2)", fontWeight: "700", fontSize: "11px",
              letterSpacing: "1.5px", background: "#0e1524", padding: "6px 12px",
            },
          }, text));
        const thead = h("thead", {}, h("tr", {},
          [h("th", {}, t("Metric"))].concat(
            items.map((it, i) => h("th", { class: "num", title: it.id }, "#" + (i + 1))))));
        const tbody = h("tbody");
        tbody.append(sectionRow(t("Config")));
        for (const [key, label, render] of CFG_ROWS) {
          if (items.every(it => it.config[key] == null)) continue;
          tbody.append(h("tr", {},
            h("td", { style: { color: "var(--text-2)" } }, label),
            items.map(it => h("td", {},
              render ? render(it.config[key], it.config)
                     : (it.config[key] == null ? "—" : String(it.config[key]))))));
        }
        tbody.append(sectionRow(t("Metric")));
        for (const k of Object.keys(meta)) {
          const vals = items.map(it => it.metrics[k]);
          if (vals.every(v => v == null)) continue;
          const nums = vals.filter(v => typeof v === "number");
          let best = -1, worst = -1;
          if (nums.length >= 2) {
            const mn = Math.min(...nums), mx = Math.max(...nums);
            if (mn !== mx) {
              const lowerBetter = (meta[k] || {}).better !== "higher";
              vals.forEach((v, i) => {
                if (v === mn) { if (lowerBetter) best = i; else worst = i; }
                if (v === mx) { if (lowerBetter) worst = i; else best = i; }
              });
            }
          }
          tbody.append(h("tr", {},
            h("td", {}, metricLabel(meta, k)),
            items.map((it, i) => h("td", {
              class: "num" + (i === best ? " best" : i === worst ? " worst" : ""),
            }, fmtM(meta, k, vals[i])))));
        }
        tblCard.append(h("div.tbl-wrap", { style: { maxHeight: "560px" } },
          h("table.tbl", {}, thead, tbody)));
        outEl.append(tblCard);
      }

      paintChips();
      await loadList();
      if (state.ids.length >= 2) runCompare();   // URL 带 ids 时直接载入
    },
  });
})();
