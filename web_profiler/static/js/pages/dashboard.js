/* Dashboard — fleet-wide overview of all simulation results. */

App.route("dashboard", {
  title: t("Dashboard"),
  async render(el) {
    el.append(UI.loading(t("Summarizing 3,000+ simulation results…")));
    const [ov, meta] = await Promise.all([App.api("/api/overview"), App.metricsMeta()]);
    el.replaceChildren();

    const models = Object.keys(ov.by_model || {});
    const roots = Object.keys(ov.by_root || {});

    /* ---------- stat row ---------- */
    const stats = h("div.grid.cols-4.mb");
    stats.append(
      UI.stat(t("Total results"), ov.total, t("{n} data roots", { n: roots.length }), 0),
      UI.stat(t("Models"), models.length, models.join(" / "), 60),
      UI.stat("decode", ov.by_mode.decode || 0, t("decode runs"), 120),
      UI.stat("prefill", ov.by_mode.prefill || 0, t("prefill runs"), 180));
    el.append(stats);

    /* ---------- donut + bar charts ---------- */
    const grid1 = h("div.grid.cols-3.mb");
    el.append(grid1);

    const donut = (container, title, obj, hint) => {
      const card = h("div.card.hoverable", {}, UI.sectionTitle(title, hint));
      const labels = Object.keys(obj), vals = Object.values(obj);
      Plot.chart(card, [{
        type: "pie", labels, values: vals, hole: 0.58,
        textinfo: "label+percent", textfont: { size: 11 },
        marker: { line: { color: "#0b101b", width: 2 } },
        hovertemplate: t("{label}: {value} runs ({percent})",
          { label: "%{label}", value: "%{value}", percent: "%{percent}" }) + "<extra></extra>",
      }], { height: 260, margin: { l: 10, r: 10, t: 10, b: 10 }, showlegend: false });
      container.append(card);
    };

    donut(grid1, t("By model"), ov.by_model, t("{n} models", { n: models.length }));
    donut(grid1, t("By mode"), ov.by_mode, "");
    donut(grid1, t("By implementation"), ov.by_impl, t("impl variants"));

    const grid2 = h("div.grid.cols-2.mb");
    el.append(grid2);

    const coresCard = h("div.card.hoverable", {}, UI.sectionTitle(t("By core count"), "num_cores"));
    const coreKeys = Object.keys(ov.by_cores).sort((a, b) => +a - +b);
    Plot.chart(coresCard, [{
      type: "bar", x: coreKeys, y: coreKeys.map(k => ov.by_cores[k]),
      marker: {
        color: coreKeys.map((_, i) => Plot.COLORS[i % Plot.COLORS.length]),
        opacity: 0.85, line: { width: 0 },
      },
      hovertemplate: t("{x} cores: {y} runs", { x: "%{x}", y: "%{y}" }) + "<extra></extra>",
    }], { height: 240, xaxis: { title: "num_cores", type: "category" }, yaxis: { title: t("Result count") } });
    grid2.append(coresCard);

    const rootCard = h("div.card.hoverable", {}, UI.sectionTitle(t("By data root"), t("results root")));
    const rootKeys = Object.keys(ov.by_root).sort((a, b) => ov.by_root[b] - ov.by_root[a]);
    Plot.chart(rootCard, [{
      type: "bar", orientation: "h",
      y: rootKeys, x: rootKeys.map(k => ov.by_root[k]),
      marker: { color: "#a78bfa", opacity: 0.8 },
      hovertemplate: t("{y}: {x} runs", { y: "%{y}", x: "%{x}" }) + "<extra></extra>",
    }], { height: 240, margin: { l: 130, r: 20, t: 10, b: 40 }, xaxis: { title: t("Result count") } });
    grid2.append(rootCard);

    /* ---------- latest results ---------- */
    const latestCard = h("div.card", {},
      UI.sectionTitle(t("Latest results"), t("index built {time}", { time: fmt.time(ov.built_at) })));
    const tbl = UI.table([
      { key: "model", label: t("Model"), render: v => h("b", {}, v) },
      { key: "mode", label: t("Mode"), render: v => UI.modeChip(v) },
      { key: "batch_size", label: "bs", num: true },
      { key: "num_cores", label: t("Cores"), num: true },
      { key: "sa_size", label: "SA", num: true },
      { key: "sram_kb", label: "SRAM", num: true, render: v => v + "K" },
      { key: "dram_bw", label: "DRAM BW", num: true },
      { key: "noc_topo", label: t("Topology"), render: (v, r) => r.topo_label || v },
      { key: "noc_bw", label: "NoC BW", num: true },
      { key: "core_group", label: "CG", num: true },
      { key: "impl", label: t("Impl"), render: v => UI.implChip(v) },
      { key: "root", label: t("Root"), render: v => UI.chip(v) },
      { key: "mtime", label: t("Time"), num: true, render: v => fmt.ago(v) },
    ], ov.latest || [], {
      onRow: r => App.navigate("result", { id: r.id }),
    });
    latestCard.append(tbl.el);
    el.append(latestCard);
  },
});
