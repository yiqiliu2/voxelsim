/* System — 系统监控: host stat cards (cpu / loadavg / mem / disk), result
   roots, and the RAM usage timeline.  Host data and the timeline both
   refresh every 10 s; the chart updates in place via Plotly.react. */

App.route("system", {
  title: t("Monitor"),
  async render(el) {
    el.append(UI.loading(t("Loading system status…")));
    let host, ram;
    try {
      const pair = await Promise.all([
        App.api("/api/system/host"),
        App.api("/api/system/ram_timeline?max_points=200"),
      ]);
      host = pair[0];
      ram = pair[1];
    } catch (e) {
      el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message })));
      return;
    }

    /* ============================ stat cards ============================ */
    // value/sub updated in place on refresh (no card re-animation flicker)
    const mkStat = (label, delay) => {
      const val = h("div.stat-value", {}, "—");
      const sub = h("div.stat-sub", {}, "");
      const card = h("div.card.stat.hoverable", { style: { animationDelay: delay + "ms" } },
        h("div.stat-label", {}, label), val, sub);
      return { card: card, set: (v, s) => { val.textContent = v; sub.textContent = s || ""; } };
    };
    const stCpu = mkStat(t("CPU cores"), 0);
    const stLoad = mkStat(t("Load average"), 60);
    const stMem = mkStat(t("Memory used"), 120);
    const stDisk = mkStat(t("Disk used"), 180);

    function paintHost(d) {
      const la = d.loadavg || [];
      stCpu.set(d.cpu_count != null ? String(d.cpu_count) : "—", t("Logical cores (os.cpu_count)"));
      stLoad.set(la.length ? la[0].toFixed(2) : "—",
        la.length >= 3 ? "5m " + la[1].toFixed(2) + " · 15m " + la[2].toFixed(2) : "");
      const m = d.mem || {};
      stMem.set(m.used_mb != null ? fmt.int(m.used_mb) + " MB" : "—",
        t("{avail} MB available · {total} MB total", { avail: fmt.int(m.available_mb), total: fmt.int(m.total_mb) }));
      const dk = d.disk || {};
      stDisk.set(dk.used_gb != null ? fmt.num(dk.used_gb) + " GB" : "—",
        t("{free} GB available · {total} GB total", { free: fmt.num(dk.free_gb), total: fmt.num(dk.total_gb) }));
    }

    /* ============================ result roots ============================ */
    const rootsBody = h("div");
    function paintRoots(d) {
      const roots = (d.results && d.results.roots) || [];
      rootsBody.replaceChildren(roots.length
        ? h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px" } },
            roots.map(r => h("div", {
              style: {
                padding: "8px 13px", borderRadius: "9px",
                background: "#0d1422", border: "1px solid var(--border)",
              },
            },
              h("div", { style: { fontSize: "12.5px", fontWeight: 650 } }, r.name),
              h("div.small.mono.muted", {}, r.path))))
        : UI.empty("▤", t("No result roots found")));
    }

    /* ============================ RAM timeline ============================ */
    const ramHint = h("span.hint");
    const ramBody = h("div");
    let ramDiv = null;

    function ramTraces(d) {
      return [
        {
          type: "scatter", mode: "lines", name: t("Used MB"),
          x: d.t, y: d.used_mb,
          fill: "tozeroy", fillcolor: "#38bdf822",
          line: { color: "#38bdf8", width: 2 },
        },
        {
          type: "scatter", mode: "lines", name: t("Available MB"),
          x: d.t, y: d.available_mb,
          line: { color: "#34d399", width: 1.5 },
        },
        {
          type: "scatter", mode: "lines", name: t("Total MB"),
          x: d.t, y: d.total_mb,
          line: { color: "#5d6f8f", width: 1, dash: "dot" },
        },
      ];
    }

    function paintRam(d) {
      const n = (d.t || []).length;
      ramHint.textContent = d.source
        ? t("Source {source} · {n} points · 10s auto-refresh", { source: d.source, n })
        : t("dram_usage.log / dram_monitor.log not found");
      if (!n) {
        ramBody.replaceChildren(UI.empty("📉", t("No RAM samples yet")));
        ramDiv = null;
        return;
      }
      if (ramDiv && ramDiv.isConnected) {
        Plotly.react(ramDiv, ramTraces(d), Plot.baseLayout({
          height: 300,
          xaxis: { type: "date", gridcolor: "#1e2a3f66", linecolor: "#1e2a3f" },
          yaxis: { title: "MB", gridcolor: "#1e2a3f66", linecolor: "#1e2a3f" },
        }));
      } else {
        ramBody.replaceChildren();
        ramDiv = Plot.chart(ramBody, ramTraces(d), {
          height: 300,
          xaxis: { type: "date" },
          yaxis: { title: "MB" },
        });
      }
    }

    /* ============================ page ============================ */
    el.replaceChildren(
      h("div.grid.cols-4.mb", {}, stCpu.card, stLoad.card, stMem.card, stDisk.card),
      h("div.card.mb", {},
        h("h3.card-title", {}, t("RAM usage timeline"), ramHint),
        ramBody),
      h("div.card", {},
        h("h3.card-title", {}, t("Result roots"), h("span.hint", {}, "discover_result_roots")),
        rootsBody));

    paintHost(host);
    paintRoots(host);
    paintRam(ram);

    App.every(10000, async () => {
      try {
        const d = await App.api("/api/system/host");
        paintHost(d);
        paintRoots(d);
      } catch (e) { /* keep last good values on transient errors */ }
    });
    App.every(10000, async () => {
      try { paintRam(await App.api("/api/system/ram_timeline?max_points=200")); }
      catch (e) { /* keep last good chart */ }
    });
  },
});
