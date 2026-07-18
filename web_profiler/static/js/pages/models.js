/* Models — 模型库: card grid (op stats / TExpr batches) + op browser
   with server-side paging, substring search (300 ms debounce) and
   exact type filter over GET /api/models/<name>/ops. */

App.route("models", {
  title: t("Model Library"),
  async render(el) {
    el.append(UI.loading(t("Loading model library…")));
    let models = [];
    try { models = (await App.api("/api/models")).models || []; }
    catch (e) { el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }
    if (!models.length) { el.replaceChildren(UI.empty("⬡", t("No models under models/"))); return; }

    const browser = h("div");   // op browser mounts here

    /* ============================ cards ============================ */
    function shapeStr(s) {
      return Object.entries(s || {})
        .map(([k, v]) => k + "[" + ((v && v.shape) || []).join("×") + "]")
        .join("  ");
    }

    function modelCard(m, i) {
      const top = Object.entries(m.op_types || {}).slice(0, 3);
      const maxV = top.length ? top[0][1] : 1;
      return h("div.card.hoverable", {
        style: { cursor: "pointer", animationDelay: (i % 9) * 45 + "ms" },
        onclick: () => openOps(m),
      },
        h("div", { style: { display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" } },
          h("b", { style: { fontSize: "15px" } }, m.name),
          h("span", { style: { flex: "1" } }),
          m.has_parsed ? UI.chip("parsed", "green") : UI.chip(t("unparsed"), ""),
          m.has_original ? UI.chip("original", "accent") : null),
        h("div.row", { style: { gap: "18px", marginTop: "10px" } },
          h("div", {},
            h("div.stat-label", {}, t("Op count")),
            h("div.mono", { style: { fontSize: "20px", fontWeight: 700, marginTop: "2px" } },
              m.op_count ? fmt.int(m.op_count) : "—"))),
        top.length
          ? h("div", { style: { marginTop: "10px" } }, top.map(([tp, n]) =>
              h("div", { style: { display: "flex", alignItems: "center", gap: "8px", marginTop: "4px" } },
                h("span.small.mono", {
                  title: tp,
                  style: { width: "108px", flex: "0 0 108px", color: "var(--text-1)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
                }, tp),
                h("div", { style: { flex: "1", height: "5px", borderRadius: "3px", background: "#1b2942", overflow: "hidden" } },
                  h("div", { style: { height: "100%", width: Math.round(n / maxV * 100) + "%", borderRadius: "3px", background: "var(--grad)" } })),
                h("span.small.mono.muted", {}, String(n)))))
          : h("div.small.muted", { style: { marginTop: "10px" } },
              m.parse_error ? t("⚠ parsed JSON parse failed") : (m.has_parsed ? t("No op stats") : t("Unparsed — op browser unavailable"))),
        (m.texpr_batches && m.texpr_batches.length)
          ? h("div.chip-row", { style: { marginTop: "10px" } },
              m.texpr_batches.map(b => UI.chip("TExpr b" + b, "violet")))
          : null);
    }

    /* ============================ op browser ============================ */
    function openOps(m) {
      if (!m.has_parsed) { UI.toast(t("Model {name} has no parsed JSON — op browser unavailable", { name: m.name }), "info"); return; }

      const st = { page: 1, pageSize: 50, q: "", type: "", total: 0, busy: false };
      const types = Object.keys(m.op_types || {});
      const pages = () => Math.max(1, Math.ceil(st.total / st.pageSize));

      const tbl = UI.table([
        { key: "id", label: "ID", num: true },
        { key: "type", label: t("Type"), render: v => UI.chip(v, "accent") },
        { key: "expr", label: t("Expression"), sortable: false,
          render: v => h("span.mono", {
            style: { whiteSpace: "normal", wordBreak: "break-all", display: "inline-block", maxWidth: "520px", fontSize: "11.5px" },
          }, v) },
        { key: "shapes", label: t("Shapes"), sortable: false,
          render: v => h("span.mono.small", {
            style: { whiteSpace: "normal", wordBreak: "break-all", display: "inline-block", maxWidth: "240px" },
          }, shapeStr(v)) },
        { key: "inputs", label: t("Inputs"), sortable: false,
          render: v => h("span.mono.small.muted", {}, (v || []).join(", ")) },
      ], []);

      const info = h("span.hint");
      const pageInfo = h("span.small.muted.mono");
      const prevBtn = h("button.btn.sm.ghost", { onclick: () => { if (st.page > 1) { st.page--; load(); } } }, "‹ " + t("Prev"));
      const nextBtn = h("button.btn.sm.ghost", { onclick: () => { if (st.page < pages()) { st.page++; load(); } } }, t("Next") + " ›");

      const search = h("input", {
        type: "search", placeholder: t("Search expressions (300ms debounce)…"), style: { width: "240px" },
      });
      let deb = null;
      search.addEventListener("input", () => {
        clearTimeout(deb);
        deb = setTimeout(() => { st.q = search.value.trim(); st.page = 1; load(); }, 300);
      });

      const typeSel = h("select", {}, [h("option", { value: "" }, t("All types"))]
        .concat(types.map(tp => h("option", { value: tp }, tp))));
      typeSel.addEventListener("change", () => { st.type = typeSel.value; st.page = 1; load(); });

      const body = h("div");

      async function load() {
        if (st.busy) return;
        st.busy = true;
        body.replaceChildren(UI.loading(t("Loading operators…")));
        try {
          let url = "/api/models/" + encodeURIComponent(m.name) +
            "/ops?page=" + st.page + "&page_size=" + st.pageSize;
          if (st.q) url += "&q=" + encodeURIComponent(st.q);
          if (st.type) url += "&type=" + encodeURIComponent(st.type);
          const d = await App.api(url);
          st.total = d.total || 0;
          tbl.setRows(d.rows || []);
          pageInfo.textContent = t("Page {page} / {pages} · {n} ops total", { page: d.page, pages: pages(), n: fmt.int(st.total) });
          prevBtn.disabled = d.page <= 1;
          nextBtn.disabled = d.page >= pages();
          body.replaceChildren(tbl.el);
        } catch (e) {
          body.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message })));
        } finally { st.busy = false; }
      }

      const card = h("div.card", {},
        h("h3.card-title", {}, t("Op browser — {name}", { name: m.name }), info),
        h("div.row.mb", { style: { gap: "10px", flexWrap: "wrap" } },
          search, typeSel,
          h("span", { style: { flex: "1" } }),
          prevBtn, pageInfo, nextBtn,
          h("button.btn.sm.ghost", { onclick: () => browser.replaceChildren() }, "✕ " + t("Close"))),
        body);
      browser.replaceChildren(card);
      info.textContent = t("{n} ops · {m} types", { n: fmt.int(m.op_count), m: types.length });
      load();
      card.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    /* ============================ page ============================ */
    const parsedN = models.filter(m => m.has_parsed).length;
    const texN = models.reduce((n, m) => n + ((m.texpr_batches || []).length), 0);
    const stats = h("div.grid.cols-3.mb");
    stats.append(
      UI.stat(t("Total models"), models.length, "original ∪ parsed", 0),
      UI.stat(t("Parsed models"), parsedN, t("Op browser available"), 60),
      UI.stat(t("TExpr batch files"), texN, "TExpr_<model>-b*.json", 120));

    const grid = h("div.grid.cols-3.mb");
    models.forEach((m, i) => grid.append(modelCard(m, i)));

    el.replaceChildren(stats, grid, browser);
  },
});
