/* Logs — 日志查看: category-grouped file list + windowed viewer.
   The viewer pages through the file by byte offset (head/tail jumps, "load
   more"), can tail-poll every 2 s for growing files, and highlights a
   keyword client-side (escaped HTML + <span> wrap). */

App.route("logs", {
  title: t("Logs"),
  async render(el) {
    el.append(UI.loading(t("Scanning log files…")));
    let logs = [];
    try { logs = (await App.api("/api/logs/list")).logs || []; }
    catch (e) { el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }

    const CATS = [["test_logs", t("Test logs")], ["root", t("Root directory")], ["dse", "DSE"], ["thermal", t("Thermal")]];
    const catLabel = c => {
      const f = CATS.find(x => x[0] === c);
      return f ? f[1] : c;
    };
    const catOrder = c => {
      const i = CATS.findIndex(x => x[0] === c);
      return i < 0 ? 99 : i;
    };
    const CHUNK = 128 * 1024;

    const groups = {};
    logs.forEach(l => { (groups[l.category] = groups[l.category] || []).push(l); });
    const cats = Object.keys(groups).sort((a, b) => catOrder(a) - catOrder(b));

    const state = {
      entry: null, text: "", start: 0, end: 0, total: 0,
      keyword: "", auto: false, busy: false,
    };

    /* ============================ file list ============================ */
    function paintSelection() {
      listCard.querySelectorAll("[data-path]").forEach(row => {
        const sel = state.entry && row.dataset.path === state.entry.path;
        row.style.border = sel ? "1px solid #38bdf866" : "1px solid transparent";
        row.style.background = sel ? "#38bdf812" : "transparent";
      });
    }

    function fileRow(entry) {
      return h("div", {
        dataset: { path: entry.path },
        style: {
          padding: "7px 9px", borderRadius: "8px", cursor: "pointer",
          marginBottom: "3px", border: "1px solid transparent",
          transition: "background .15s, border-color .15s",
        },
        onclick: () => open(entry),
        onmouseenter: e => { if (!state.entry || state.entry.path !== entry.path) e.currentTarget.style.background = "#16203266"; },
        onmouseleave: e => { if (!state.entry || state.entry.path !== entry.path) e.currentTarget.style.background = "transparent"; },
      },
        h("div", {
          title: entry.path,
          style: {
            fontSize: "12px", fontFamily: "var(--mono)", color: "var(--text-0)",
            overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
          },
        }, entry.name),
        h("div.small.muted", {}, fmt.bytes(entry.size) + " · " + fmt.ago(entry.mtime)));
    }

    const listCard = h("div.card", {
      style: { padding: "14px", maxHeight: "calc(100vh - 130px)", overflowY: "auto" },
    }, h("h3.card-title", {}, t("Log files"), h("span.hint", {}, t("{n} files", { n: logs.length }))));
    if (!logs.length) {
      listCard.append(UI.empty("≣", t("No log files found")));
    } else {
      cats.forEach(cat => {
        listCard.append(h("div", {
          style: {
            fontSize: "10.5px", color: "var(--text-2)", letterSpacing: "1.5px",
            textTransform: "uppercase", margin: "12px 4px 6px",
          },
        }, catLabel(cat) + " · " + groups[cat].length));
        groups[cat].forEach(entry => listCard.append(fileRow(entry)));
      });
    }

    /* ============================ viewer ============================ */
    const vName = h("b", { style: { fontSize: "13.5px", fontFamily: "var(--mono)" } }, t("No file selected"));
    const vMeta = h("span.hint");
    const vPos = h("span.small.muted.mono");
    const logView = h("div.log-view", { style: { height: "520px", maxHeight: "520px" } },
      t("← Select a log file from the left"));
    const moreBtn = h("button.btn.sm.ghost", { onclick: loadMore }, t("Load more ▾"));
    moreBtn.disabled = true;

    const searchInput = h("input", {
      type: "search", placeholder: t("Highlight keyword…"), style: { width: "180px" },
    });
    let deb = null;
    searchInput.addEventListener("input", () => {
      clearTimeout(deb);
      deb = setTimeout(() => {
        state.keyword = searchInput.value.trim();
        if (state.entry) renderText("keep");
      }, 250);
    });

    const autoChk = h("input", { type: "checkbox" });
    autoChk.addEventListener("change", () => {
      state.auto = autoChk.checked;
      if (state.auto && state.entry) tailRefresh();
    });

    const viewerCard = h("div.card", {},
      h("h3.card-title", {}, vName, vMeta),
      h("div.row.mb", { style: { gap: "8px", flexWrap: "wrap" } },
        h("button.btn.sm.ghost", { onclick: () => jump("head") }, t("⇤ Head")),
        h("button.btn.sm.ghost", { onclick: () => jump("tail") }, t("Tail ⇥")),
        moreBtn,
        searchInput,
        h("span", { style: { flex: "1" } }),
        vPos,
        h("span.small.muted", {}, t("Auto-refresh")),
        h("label.switch", {}, autoChk, h("span.track"))),
      logView);

    /* ---------- rendering (escaped text + keyword highlight) ---------- */
    const esc = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

    function renderText(scroll) {   // scroll: "top" | "bottom" | "keep"
      const prevTop = logView.scrollTop;
      let html = esc(state.text);
      const kw = state.keyword;
      if (kw) {
        const pat = esc(kw).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        try {
          html = html.replace(new RegExp(pat, "gi"),
            m => '<span style="background:#fbbf2438;color:#ffe9b0;border-radius:2px;padding:0 1px">' + m + "</span>");
        } catch (e) { /* fall back to plain text on regex trouble */ }
      }
      logView.innerHTML = html;
      if (scroll === "top") logView.scrollTop = 0;
      else if (scroll === "bottom") logView.scrollTop = logView.scrollHeight;
      else logView.scrollTop = prevTop;
      updatePos();
    }

    function updatePos() {
      vPos.textContent = state.entry
        ? t("Bytes {start}–{end} / {total}", { start: fmt.int(state.start), end: fmt.int(state.end), total: fmt.bytes(state.total) })
        : "";
      moreBtn.disabled = !state.entry || state.end >= state.total;
    }

    /* ---------- data ---------- */
    async function fetchWindow(qs, silent) {
      try {
        return await App.api("/api/logs/view?path=" + encodeURIComponent(state.entry.path) + "&" + qs);
      } catch (e) {
        if (!silent) logView.textContent = t("Failed to read: {msg}", { msg: e.message });
        return null;
      }
    }

    async function open(entry) {
      state.entry = entry;
      state.text = "";
      state.start = 0;
      state.end = 0;
      state.total = entry.size;
      paintSelection();
      vName.textContent = entry.name;
      vMeta.textContent = catLabel(entry.category) + " · " + fmt.bytes(entry.size) + " · " + fmt.time(entry.mtime);
      logView.textContent = t("Loading…");
      await jump("head");
    }

    async function jump(where) {
      if (!state.entry) return;
      const w = await fetchWindow(where === "tail" ? "tail=1&length=" + CHUNK : "offset=0&length=" + CHUNK);
      if (!w) return;
      state.text = w.text;
      state.start = w.offset;
      state.end = w.offset + w.length;
      state.total = w.total_size;
      renderText(where === "tail" ? "bottom" : "top");
    }

    async function loadMore() {
      if (!state.entry || state.busy || state.end >= state.total) return;
      state.busy = true;
      const w = await fetchWindow("offset=" + state.end + "&length=" + CHUNK, true);
      state.busy = false;
      if (!w || !w.length) { updatePos(); return; }
      // chunk boundary may split a UTF-8 char (server decodes with errors=replace)
      state.text += w.text;
      state.end = w.offset + w.length;
      state.total = w.total_size;
      renderText("keep");
    }

    async function tailRefresh() {
      if (!state.entry || state.busy) return;
      state.busy = true;
      const w = await fetchWindow("tail=1&length=" + CHUNK, true);
      state.busy = false;
      if (!w) return;
      if (w.total_size === state.total && w.text === state.text) return;   // unchanged
      const nearBottom = logView.scrollHeight - logView.scrollTop - logView.clientHeight < 80;
      state.text = w.text;
      state.start = w.offset;
      state.end = w.offset + w.length;
      state.total = w.total_size;
      renderText(nearBottom ? "bottom" : "keep");
    }

    /* ============================ page ============================ */
    el.replaceChildren(
      h("div", { style: { display: "flex", gap: "16px", alignItems: "flex-start" } },
        h("div", { style: { flex: "0 0 300px", minWidth: "0" } }, listCard),
        h("div.grow", { style: { minWidth: "0" } }, viewerCard)));

    App.every(2000, () => { if (state.auto && state.entry) tailRefresh(); });
  },
});
