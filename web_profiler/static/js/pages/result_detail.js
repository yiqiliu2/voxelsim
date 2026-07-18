/* Result detail — deep-dive view of a single simulation result.
   除 detail 外的每个 section 独立拉取、独立降级:任何端点 404/400
   只显示提示卡片,不影响整页。路由无侧边栏项,渲染时高亮「结果浏览」。 */

App.route("result", {
  title: t("Result Detail"),
  async render(el, params) {
    document.querySelectorAll(".nav-item").forEach(a =>
      a.classList.toggle("active", a.dataset.route === "results"));

    const id = params.id || "";
    if (!id) { el.append(UI.empty("⚠", t("Missing result id"))); return; }

    const CYC_MS = 1 / 1.5e6;   // cycles → ms,与后端 NPU_FREQ_MHZ_DEFAULT=1500 一致
    const api = suffix => "/api/result/" + suffix + "?id=" + encodeURIComponent(id);

    el.append(UI.loading(t("Loading result detail…")));
    await App.metricsMeta();
    let d;
    try {
      d = await App.api(api("detail"));
    } catch (e) {
      el.replaceChildren(UI.empty("⚠", t("Failed to load result: {msg}", { msg: e.message })));
      return;
    }
    el.replaceChildren();
    const m = d.metrics || {};

    /* ---------------- 通用小组件 ---------------- */
    const unavail = (icon, text, detail) => h("div", {},
      UI.empty(icon, text),
      detail ? h("div.small.muted", {
        style: { textAlign: "center", marginTop: "-44px", paddingBottom: "22px" }
      }, detail) : null);

    function section(title, hint, delay) {
      const body = h("div");
      el.append(h("div.card.mb", { style: { animationDelay: delay + "ms" } },
        UI.sectionTitle(title, hint), body));
      return body;
    }

    /** 先放 loading,异步填充;失败降级为提示卡,离开页面后静默丢弃。 */
    function fillLater(body, fn) {
      body.append(UI.loading());
      (async () => {
        let node;
        try {
          node = await fn();
        } catch (e) {
          if (body.isConnected)
            body.replaceChildren(unavail("▦", t("No such data for this result"), e.message));
          return;
        }
        if (body.isConnected) body.replaceChildren(node);
      })();
    }

    /* ================= 1. 头部卡 ================= */
    const reproBtn = h("button.btn.sm.primary", {
      onclick: async () => {
        try {
          const data = await App.api(api("reproduce"));
          UI.modal(t("Reproduce command"), h("div", {},
            h("div.log-view", {}, data.command),
            h("div.row", { style: { marginTop: "12px" } },
              UI.copyBtn(data.command, t("Copy command")))),
            [{ label: t("Close") }]);
        } catch (e) { UI.toast(t("Failed to fetch reproduce command: {msg}", { msg: e.message }), "error"); }
      }
    }, "↻ " + t("Reproduce command"));

    const chipsRow = h("div.row", { style: { gap: "6px", marginTop: "10px" } },
      UI.chip(d.model, "accent"), UI.modeChip(d.mode), UI.implChip(d.impl),
      UI.chip("bs " + d.batch_size),
      UI.chip(t("{n} cores", { n: d.num_cores })),
      UI.chip("SA " + d.sa_size + " / VU " + (d.vu_size == null ? "?" : d.vu_size)),
      UI.chip("SRAM " + d.sram_kb + " KB"),
      UI.chip("DRAM " + d.dram_bw + " GB/s"),
      UI.chip((d.topo_label || d.noc_topo) + " / NoC " + d.noc_bw),
      UI.chip("CG " + d.core_group),
      d.seq_length ? UI.chip("seq " + d.seq_length) : null,
      d.trcd ? UI.chip("tRCD " + d.trcd + " · tRP " + d.trp) : null,
      UI.chip("root: " + d.root));

    el.append(h("div.card.mb", {},
      h("div.row", { style: { alignItems: "flex-start", justifyContent: "space-between", gap: "18px" } },
        h("div.grow", {},
          h("div.row", {},
            h("button.btn.sm.ghost", { onclick: () => App.navigate("results") }, "← " + t("Results")),
            h("div", { style: { fontSize: "15px", fontWeight: 650 } }, UI.cfgLabel(d))),
          h("div.row", { style: { gap: "8px", marginTop: "6px", flexWrap: "nowrap" } },
            h("div.small.muted.mono.grow", { style: { wordBreak: "break-all" } }, d.id),
            UI.copyBtn(d.id, "id")),
          chipsRow,
          h("div.small.muted", { style: { marginTop: "10px" } },
            t("Modified {time} ({ago}) · log size {size}", { time: fmt.time(d.mtime), ago: fmt.ago(d.mtime), size: fmt.bytes(d.size) }) +
            (d.has_overlap ? t(" · with overlap") : "") + (d.has_pickle ? t(" · with pickle") : ""))),
        h("div", { style: { display: "flex", flexWrap: "wrap", gap: "8px", justifyContent: "flex-end", maxWidth: "320px" } },
          reproBtn,
          h("a.btn.sm.ghost", { href: api("export.csv") + "&kind=summary" }, "⬇ " + t("Summary CSV")),
          h("a.btn.sm.ghost", { href: api("export.csv") + "&kind=operators" }, "⬇ " + t("Operators CSV")),
          h("a.btn.sm.ghost", { href: api("trace.json") }, "⬇ trace.json")))));

    /* ================= 2. 指标 stat 行 ================= */
    if (d.metrics) {
      const tot = m.total_energy_mj || 0;
      const share = v => tot > 0 && v != null ? t("{pct}% of total", { pct: (v / tot * 100).toFixed(1) }) : "";
      const stats = h("div.grid.cols-4.mb");
      stats.append(
        UI.stat(t("Latency"), fmt.metric("time_ms", m.time_ms),
          m.total_time ? fmt.int(m.total_time) + " cycles" : "", 60),
        UI.stat(t("Total energy"), fmt.metric("total_energy_mj", m.total_energy_mj), "", 120),
        UI.stat(t("Static energy"), fmt.metric("static_energy_mj", m.static_energy_mj),
          share(m.static_energy_mj), 180),
        UI.stat(t("Dynamic energy"), fmt.metric("dynamic_energy_mj", m.dynamic_energy_mj),
          share(m.dynamic_energy_mj), 240),
        UI.stat(t("Average power"), fmt.metric("avg_power_w", m.avg_power_w),
          t("static {s} W · dynamic {d} W", { s: fmt.num(m.static_power_w), d: fmt.num(m.dynamic_power_w) }), 300),
        UI.stat(t("Overall utilization"),
          m.overall_util == null ? "—" : fmt.pct(m.overall_util * 100, 1), "", 360),
        UI.stat(t("Matrix throughput"), fmt.metric("mm_gflops", m.mm_gflops),
          "VU " + fmt.num(m.vu_gflops) + " GF", 420),
        UI.stat(t("DRAM read utilization"), fmt.pct(m.dram_r_util),
          t("write {v}", { v: fmt.pct(m.dram_w_util) }), 480));
      el.append(stats);
    } else {
      el.append(h("div.card.mb", {},
        UI.empty("📊", t("No summary metrics for this result (incomplete log header)"))));
    }

    /* ================= 3. 能耗分解(双 donut) ================= */
    const energyBody = section(t("Energy breakdown"), t("static / dynamic · by component (mJ)"), 200);
    fillLater(energyBody, async () => {
      const COMPS = [
        ["sa", "SA", "#38bdf8"], ["vu", "VU", "#a78bfa"], ["sram", "SRAM", "#34d399"],
        ["core", "Core", "#fbbf24"], ["dram", "DRAM", "#f87171"], ["tsv", "TSV", "#f472b6"],
        ["noc", "NoC", "#22d3ee"]];
      const grid = h("div.grid.cols-2");
      let any = false;
      for (const [prefix, title] of [["static", t("Static energy")], ["dynamic", t("Dynamic energy")]]) {
        const labels = [], vals = [], colors = [];
        for (const [k, label, color] of COMPS) {
          const v = m[prefix + "_" + k + "_mj"];
          if (v != null && v > 0) { labels.push(label); vals.push(v); colors.push(color); }
        }
        const sub = h("div", {},
          h("div.small.muted", { style: { textAlign: "center" } },
            title + (vals.length ? t(" · total {v} mJ", { v: fmt.num(vals.reduce((a, b) => a + b, 0)) }) : "")));
        if (!vals.length) {
          sub.append(UI.empty("📉", t("No data")));
        } else {
          any = true;
          Plot.chart(sub, [{
            type: "pie", labels, values: vals, hole: 0.55,
            marker: { colors, line: { color: "#0b101b", width: 2 } },
            textinfo: "percent", textfont: { size: 11 },
            hovertemplate: "%{label}: %{value:.2f} mJ (%{percent})<extra></extra>",
          }], {
            height: 250, margin: { l: 20, r: 20, t: 10, b: 10 },
            legend: { orientation: "h", y: -0.08, font: { size: 11 } },
          });
        }
        grid.append(sub);
      }
      if (!any) return unavail("📉", t("No energy breakdown data for this result"));
      return grid;
    });

    /* ================= 4. 算子时间线(甘特) ================= */
    const tlBody = section(t("Operator timeline"), t("5-stage pipeline gantt · color = stage"), 260);
    fillLater(tlBody, async () => {
      const data = await App.api(api("operators"));
      const ops = data.operators || [];
      if (!ops.length) return unavail("📉", t("No operators in the log"));

      const STAGES = [
        ["ld", "DRAM Load", "#38bdf8", o => o.start_ld],
        ["bcast", "NoC Bcast", "#22d3ee", o => o.start_bcast],
        ["comp", "Compute", "#a78bfa", o => o.start_comp],
        ["shift", "Shift", "#818cf8", o => o.start_comp + o.dur_comp],
        ["reduce", "NoC Reduce", "#fbbf24", o => o.start_reduce],
        ["store", "DRAM Store", "#f87171", o => o.start_store],
      ];
      const win = { start: 0, size: 100 };
      const rangeLabel = h("span.small.muted");
      const plotWrap = h("div");
      const prevBtn = h("button.btn.sm.ghost", {
        onclick: () => { win.start -= win.size; repaint(); }
      }, "‹ " + t("Prev window"));
      const nextBtn = h("button.btn.sm.ghost", {
        onclick: () => { win.start += win.size; repaint(); }
      }, t("Next window") + " ›");
      const sizeSel = h("select", {
        onchange: e => { win.size = +e.target.value; win.start = 0; repaint(); }
      }, [50, 100, 200, 400].map(n =>
        h("option", { value: n, selected: n === win.size ? "selected" : null }, t("{n} ops/window", { n }))));

      function repaint() {
        win.start = Math.max(0, Math.min(win.start, Math.max(0, ops.length - 1)));
        const i1 = Math.min(ops.length, win.start + win.size);
        const slice = ops.slice(win.start, i1);
        const t0 = Math.min.apply(null, slice.map(o => o.start_ld));
        const labels = slice.map(o => "op " + o.op_id);
        const traces = STAGES.map(([k, label, color, startFn]) => ({
          type: "bar", orientation: "h", name: label,
          y: labels,
          x: slice.map(o => o["dur_" + k] * CYC_MS),
          base: slice.map(o => (startFn(o) - t0) * CYC_MS),
          marker: { color },
          customdata: slice.map(o => [o.op_id, o.op_type || "?", o.category || ""]),
          hovertemplate: label + " · %{x:.3f} ms" +
            "<br>op #%{customdata[0]} · %{customdata[1]} · %{customdata[2]}<extra></extra>",
        }));
        rangeLabel.textContent =
          t("Operators {a}–{b} / {n}", { a: win.start + 1, b: i1, n: ops.length }) + (data.classified ? "" : t(" · unclassified"));
        prevBtn.disabled = win.start <= 0;
        nextBtn.disabled = i1 >= ops.length;
        plotWrap.replaceChildren();
        Plot.chart(plotWrap, traces, {
          barmode: "stack",
          height: Math.max(320, Math.min(680, slice.length * 6)),
          xaxis: { title: t("Relative time in window (ms)") },
          yaxis: { autorange: "reversed", automargin: true },
          margin: { l: 74, r: 20, t: 30, b: 46 },
        });
      }

      const wrap = h("div", {},
        h("div.row.mb", {},
          h("span.small.muted", {}, t("Window")), sizeSel, prevBtn, nextBtn,
          h("span.grow"), rangeLabel),
        plotWrap);
      repaint();
      return wrap;
    });

    /* ================= 5. 算子类型分解(Fig.20 风格) ================= */
    const bdBody = section(t("Operator type breakdown"), t("paper Fig.20 style · busy-interval attributed time"), 320);
    fillLater(bdBody, async () => {
      let bd;
      try {
        bd = await App.api(api("op_breakdown"));
      } catch (e) {
        return unavail("▦", t("No operator classification data for this result (compile cache missing or op count mismatch)"), e.message);
      }
      const cats = bd.categories || [];
      if (!cats.length) return unavail("▦", t("No operator classification data"));
      const wrap = h("div");
      Plot.chart(wrap, cats.map((c, i) => ({
        type: "bar", orientation: "h", name: c.label,
        y: [t("Time")], x: [c.time_ms],
        marker: { color: Plot.COLORS[i % Plot.COLORS.length] },
        customdata: [[c.fraction * 100, c.op_count]],
        hovertemplate: c.label + ": %{x:.2f} ms (%{customdata[0]:.1f}%)" +
          " · %{customdata[1]} ops<extra></extra>",
      })), {
        barmode: "stack", height: 170,
        legend: { orientation: "h", y: 1.4, font: { size: 11 } },
        xaxis: { title: t("Attributed time (ms)") },
        yaxis: { showticklabels: false },
        margin: { l: 40, r: 20, t: 10, b: 44 },
      });
      wrap.append(h("div.row", { style: { gap: "6px", marginTop: "4px" } },
        cats.map(c => UI.chip(
          `${c.label} · ${c.time_ms.toFixed(2)} ms · ${(c.fraction * 100).toFixed(1)}% · ${c.op_count} ops`))));
      if (bd.source)
        wrap.append(h("div.small.muted", { style: { marginTop: "8px" } }, t("Classification source: {s}", { s: bd.source })));
      return wrap;
    });

    /* ================= 6. 功耗 Top 区间 ================= */
    const tpBody = section(t("Top power intervals"), t("top_power log · top 20 by power"), 380);
    fillLater(tpBody, async () => {
      const data = await App.api(api("top_power"));
      const iv = (data.intervals || []).slice(0, 20);
      if (!iv.length) return unavail("⚡", t("top_power log is empty or missing"));
      const wrap = h("div");
      Plot.chart(wrap, [{
        type: "bar", orientation: "h",
        y: iv.map((v, i) => "#" + (i + 1)),
        x: iv.map(v => v.power_w),
        marker: { color: "#f87171", opacity: 0.85 },
        customdata: iv.map(v => [v.t_start * CYC_MS, v.t_end * CYC_MS, (v.units || []).join(", ")]),
        hovertemplate: t("power {x} W", { x: "%{x:.2f}" }) +
          "<br>[%{customdata[0]:.2f} → %{customdata[1]:.2f} ms]<br>%{customdata[2]}<extra></extra>",
      }], {
        height: 380,
        xaxis: { title: t("Dynamic power (W)") },
        yaxis: { autorange: "reversed" },
        margin: { l: 60, r: 24, t: 20, b: 46 },
      });
      return wrap;
    });

    /* ================= 7. overlap 功耗曲线 ================= */
    const ovBody = section(t("Overlap / power curve"), t("overlap log · dynamic power and unit activity"), 440);
    fillLater(ovBody, async () => {
      let data;
      try {
        data = await App.api(api("overlap"));
      } catch (e) {
        return unavail("∿", t("No overlap log for this result"), e.message);
      }
      const iv = data.intervals || [];
      if (!iv.length) return unavail("∿", t("overlap log is empty"));

      let peak = 0, wsum = 0, tsum = 0;
      const unitCount = {};
      for (const v of iv) {
        const dt = Math.max(1, v.t_end - v.t_start);
        wsum += v.power_w * dt; tsum += dt;
        if (v.power_w > peak) peak = v.power_w;
        for (const u of (v.units || [])) unitCount[u] = (unitCount[u] || 0) + 1;
      }
      const units = Object.keys(unitCount).sort((a, b) => unitCount[b] - unitCount[a]);

      const wrap = h("div", {},
        h("div.row.mb", { style: { gap: "6px" } },
          UI.chip(t("intervals {n}", { n: fmt.int(iv.length) }), "accent"),
          UI.chip(t("peak {v} W", { v: fmt.num(peak) }), "red"),
          UI.chip(t("time-weighted avg {v} W", { v: fmt.num(wsum / (tsum || 1)) }), "warn")));
      Plot.chart(wrap, [{
        type: "scatter", mode: "lines",
        x: iv.map(v => (v.t_start + v.t_end) / 2 * CYC_MS),
        y: iv.map(v => v.power_w),
        line: { color: "#38bdf8", width: 1.6, shape: "hv" },
        fill: "tozeroy", fillcolor: "rgba(56,189,248,0.10)",
        hovertemplate: "%{x:.2f} ms · %{y:.2f} W<extra></extra>",
      }], {
        height: 230,
        xaxis: { title: t("Time (ms)") }, yaxis: { title: t("Dynamic power (W)") },
      });
      wrap.append(h("div.small.muted", { style: { margin: "10px 0 4px" } },
        t("Hardware unit activity (share of intervals present)")));
      Plot.chart(wrap, [{
        type: "bar", orientation: "h",
        y: units, x: units.map(u => unitCount[u] / iv.length * 100),
        marker: { color: "#a78bfa", opacity: 0.85 },
        hovertemplate: t("{y}: {x}% of intervals", { y: "%{y}", x: "%{x:.1f}" }) + "<extra></extra>",
      }], {
        height: 210,
        xaxis: { title: t("% intervals") }, yaxis: { automargin: true },
        margin: { l: 120, r: 24, t: 10, b: 44 },
      });
      return wrap;
    });

    /* ================= 8. 复现命令 ================= */
    const rpBody = section(t("Reproduce command"), t("icbm_launch.py · re-run this simulation"), 500);
    fillLater(rpBody, async () => {
      const data = await App.api(api("reproduce"));
      return h("div", {},
        h("div.log-view", { style: { maxHeight: "180px" } }, data.command),
        h("div.row", { style: { marginTop: "10px" } },
          UI.copyBtn(data.command, t("Copy command"))));
    });
  },
});
