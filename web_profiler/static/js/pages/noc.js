/* NoC — 拓扑距离矩阵: topo/n chip pickers + Plotly heatmap.
   Tables larger than 256x256 arrive block-averaged (downsampled/block flags
   shown in the card hint); avg/max hops + diameter come from the response. */

App.route("noc", {
  title: t("NoC Topology"),
  async render(el) {
    el.append(UI.loading(t("Loading NoC distance tables…")));
    let tables = [];
    try { tables = (await App.api("/api/noc/tables")).tables || []; }
    catch (e) { el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }
    if (!tables.length) { el.replaceChildren(UI.empty("✦", t("No distance tables under noc_distance_tables/"))); return; }

    const TOPO_ORDER = ["MESH", "TORUS", "ALL"];
    const seen = {};
    const topos = [];
    tables.forEach(t => {
      if (!seen[t.topo]) { seen[t.topo] = true; topos.push(t.topo); }
    });
    topos.sort((a, b) => {
      const ia = TOPO_ORDER.indexOf(a), ib = TOPO_ORDER.indexOf(b);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    const nsFor = topo => tables.filter(x => x.topo === topo).map(x => x.n).sort((a, b) => a - b);

    const state = { topo: topos[0], n: nsFor(topos[0])[0], reqSeq: 0 };

    /* ============================ pickers ============================ */
    const topoRow = h("div.chip-row");
    const nRow = h("div.chip-row");

    function paintChips() {
      topoRow.replaceChildren(...topos.map(t =>
        h("span.chip-toggle", {
          dataset: { v: t },
          class: t === state.topo ? "chip-toggle on" : "chip-toggle",
          onclick: () => {
            if (state.topo === t) return;
            state.topo = t;
            state.n = nsFor(t)[0];
            paintChips();
            load();
          },
        }, t)));
      nRow.replaceChildren(...nsFor(state.topo).map(n =>
        h("span.chip-toggle", {
          dataset: { v: String(n) },
          class: n === state.n ? "chip-toggle on" : "chip-toggle",
          onclick: () => {
            if (state.n === n) return;
            state.n = n;
            paintChips();
            load();
          },
        }, String(n))));
    }

    /* ============================ heatmap ============================ */
    const statsRow = h("div.grid.cols-3.mb");
    const chartHint = h("span.hint");
    const chartBody = h("div");
    const chartCard = h("div.card", {},
      h("h3.card-title", {}, t("Core-to-core distance matrix"), chartHint), chartBody);

    const HEAT = [[0, "#0d1422"], [0.3, "#155e75"], [0.65, "#38bdf8"], [1, "#a78bfa"]];

    async function load() {
      const seq = ++state.reqSeq;
      chartBody.replaceChildren(UI.loading(t("Loading distance matrix…")));
      let d;
      try { d = await App.api("/api/noc/tables/" + state.topo + "/" + state.n); }
      catch (e) {
        if (seq === state.reqSeq) chartBody.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message })));
        return;
      }
      if (seq !== state.reqSeq) return;   // stale response after a fast switch

      const st = d.stats || {};
      const matN = (d.matrix || []).length;
      statsRow.replaceChildren(
        UI.stat(t("Avg hops"), st.avg_hops != null ? (+st.avg_hops).toFixed(2) : "—",
          d.topo + " · n=" + d.n, 0),
        UI.stat(t("Max hops"), st.max_hops != null ? st.max_hops : "—", "off-diagonal max", 60),
        UI.stat(t("Diameter"), st.diameter != null ? st.diameter : "—", "diameter (hops)", 120));

      chartHint.textContent = d.downsampled
        ? t("Downsampled: block={block} block-averaged → {mat}×{mat} (original {n}×{n})", { block: d.block, mat: matN, n: d.n })
        : t("{n}×{n} full resolution", { n: d.n });

      chartBody.replaceChildren();
      Plot.chart(chartBody, [{
        type: "heatmap",
        z: d.matrix,
        colorscale: HEAT,
        colorbar: {
          thickness: 12, outlinewidth: 0,
          tickfont: { color: "#9fb0cc", size: 10 },
          title: { text: "hops", font: { color: "#9fb0cc", size: 11 } },
        },
        hovertemplate: t("src {y} → dst {x}: {z} hops", { y: "%{y}", x: "%{x}", z: "%{z}" }) + "<extra></extra>",
      }], {
        height: 560,
        margin: { l: 64, r: 30, t: 8, b: 50 },
        xaxis: { title: "dst core", constrain: "domain" },
        yaxis: { title: "src core", autorange: "reversed", scaleanchor: "x", constrain: "domain" },
      });
    }

    /* ============================ page ============================ */
    el.replaceChildren(
      h("div.card.filterbar", { style: { alignItems: "center" } },
        h("label.field", {}, t("Topology"), topoRow),
        h("label.field", { style: { flex: "1" } }, t("Size n"), nRow)),
      statsRow,
      chartCard);
    paintChips();
    load();
  },
});
