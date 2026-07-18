/* Jobs — 任务中心: all jobs list + detail + live log tail.
   The list polls every 3 s while any job is queued/running; a JSON signature
   comparison keeps unchanged data from repainting (no flicker).  The log view
   appends by byte offset and polls every 2 s while the job is running. */

App.route("jobs", {
  title: t("Jobs"),
  async render(el, params) {
    el.append(UI.loading(t("Loading jobs…")));

    const TERMINAL = ["done", "failed", "cancelled", "interrupted"];
    const ACTIVE = ["queued", "running"];
    const TYPE_KIND = { sim: "accent", batch: "violet", dse: "warn", thermal: "red" };
    const isActive = j => ACTIVE.indexOf(j.status) >= 0;
    const isTerminal = j => TERMINAL.indexOf(j.status) >= 0;
    const shorten = (s, n) => s.length > n ? s.slice(0, n) + "…" : s;

    const state = {
      jobs: [], listSig: "",
      selectedId: params.id || null,
      detail: null, detailSig: "",
      logOffset: 0, logTotal: 0, logBusy: false, follow: true,
    };

    /* ============================ skeleton ============================ */
    const listHint = h("span.hint");
    const listRows = h("div");
    const listCard = h("div.card", {
      style: { padding: "14px", maxHeight: "calc(100vh - 130px)", overflowY: "auto" },
    }, h("h3.card-title", {}, t("Job list"), listHint), listRows);

    const detailPane = h("div.grow", { style: { minWidth: "0" } },
      UI.empty("☰", t("Select a job on the left to view details")));
    const layout = h("div", { style: { display: "flex", gap: "16px", alignItems: "flex-start" } },
      h("div", { style: { flex: "0 0 390px", minWidth: "0" } }, listCard),
      detailPane);

    /* ============================ job list ============================ */
    function progressOf(job) {
      const p = job.progress || { total: (job.units || []).length || 1 };
      const fin = (p.done || 0) + (p.failed || 0) + (p.cancelled || 0);
      return { fin: fin, total: p.total || 1, pct: p.total ? Math.round(fin / p.total * 100) : 0 };
    }

    function jobRow(job) {
      const sel = job.id === state.selectedId;
      const pr = progressOf(job);
      return h("div", {
        style: {
          padding: "9px 11px", borderRadius: "10px", cursor: "pointer", marginBottom: "5px",
          border: sel ? "1px solid #38bdf866" : "1px solid transparent",
          background: sel ? "#38bdf812" : "transparent",
          transition: "background .15s, border-color .15s",
        },
        onclick: () => selectJob(job.id),
        onmouseenter: e => { if (job.id !== state.selectedId) e.currentTarget.style.background = "#16203266"; },
        onmouseleave: e => { if (job.id !== state.selectedId) e.currentTarget.style.background = "transparent"; },
      },
        h("div", { style: { display: "flex", alignItems: "center", gap: "8px" } },
          h("b.mono", { style: { fontSize: "12.5px" } }, "#" + job.id),
          UI.chip(job.type, TYPE_KIND[job.type] || ""),
          h("span", { style: { flex: "1" } }),
          UI.statusPill(job.status),
          h("span.small.muted", {}, fmt.ago(job.created_at))),
        h("div", {
          title: job.label || "",
          style: { fontSize: "12px", color: "var(--text-1)", marginTop: "3px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" },
        }, job.label || "—"),
        h("div.row", { style: { gap: "8px", marginTop: "6px", flexWrap: "nowrap" } },
          h("div.progress-track.grow", {}, h("div.progress-fill", { style: { width: pr.pct + "%" } })),
          h("span.small.muted.mono", {}, pr.fin + "/" + pr.total)));
    }

    function paintList() {
      const n = state.jobs.length;
      const nActive = state.jobs.filter(isActive).length;
      listHint.textContent = t("{n} total", { n }) + (nActive ? t(" · {n} active", { n: nActive }) : "");
      if (!n) { listRows.replaceChildren(UI.empty("☰", t("No jobs yet — submit one from the Run page"))); return; }
      listRows.replaceChildren(...state.jobs.map(jobRow));
    }

    async function refreshJobs(force) {
      let data;
      try { data = await App.api("/api/jobs?limit=100"); }
      catch (e) {
        if (force) listRows.replaceChildren(UI.empty("⚠", t("Failed to load job list: {msg}", { msg: e.message })));
        return;
      }
      const sig = JSON.stringify(data.jobs || []);
      if (!force && sig === state.listSig) return;   // 无变化不重绘,避免闪动
      state.listSig = sig;
      state.jobs = data.jobs || [];
      // 任务刚进入终态 → 重建结果索引(让 logs_web 新结果立即可见)
      const prev = state.prevStatus || {};
      let newlyDone = false;
      for (const j of state.jobs) {
        if (prev[j.id] && !isTerminal({ status: prev[j.id] }) && isTerminal(j)) newlyDone = true;
        prev[j.id] = j.status;
      }
      state.prevStatus = prev;
      if (newlyDone && !state.rebuilding) {
        state.rebuilding = true;
        App.api("/api/index/rebuild", { method: "POST" })
          .then(r => UI.toast(t("Result index updated: {n} entries", { n: r.total }), "success", 2600))
          .catch(() => {})
          .finally(() => { state.rebuilding = false; });
      }
      paintList();
      if (state.selectedId) {
        const j = state.jobs.find(x => x.id === state.selectedId);
        if (j) paintDetail(j);
      }
    }

    /* ============================ detail ============================ */
    let R = null;   // regions of the currently built detail view

    function buildDetail(job) {
      R = {
        idEl: h("b.mono", { style: { fontSize: "15px" } }),
        status: h("span", { style: { display: "inline-flex", gap: "10px", alignItems: "center" } }),
        label: h("div", { style: { fontSize: "12.5px", color: "var(--text-1)", marginTop: "5px", wordBreak: "break-all" } }),
        times: h("div.small.muted", { style: { marginTop: "8px" } }),
        err: h("div"),
        progress: h("div", { style: { marginTop: "12px" } }),
        actions: h("div.row", { style: { gap: "8px" } }),
        params: h("pre", {
          style: {
            margin: "0", fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "1.6",
            whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#b6c6e2",
            maxHeight: "260px", overflow: "auto",
          },
        }),
        unitsHint: h("span.hint"),
        units: h("div"),
        logMeta: h("span.hint"),
        logView: h("div.log-view", { style: { height: "380px", maxHeight: "380px" } }),
      };
      const paramsText = JSON.stringify(job.params, null, 2);

      const followChk = h("input", { type: "checkbox" });
      followChk.checked = state.follow;
      followChk.addEventListener("change", () => {
        state.follow = followChk.checked;
        if (state.follow) scrollLog();
      });

      const headerCard = h("div.card.mb", {},
        h("div", { style: { display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap" } },
          R.idEl, UI.copyBtn(job.id, "ID"), R.status,
          h("span", { style: { flex: "1" } }), R.actions),
        R.label, R.times, R.err, R.progress);

      const paramsCard = h("div.card.mb", {},
        h("h3.card-title", {}, t("Launch params"), h("span.hint", {}, UI.copyBtn(paramsText, "JSON"))),
        R.params);
      R.params.textContent = paramsText;

      const unitsCard = h("div.card.mb", {},
        h("h3.card-title", {}, t("Execution units"), R.unitsHint), R.units);

      const logCard = h("div.card", {},
        h("h3.card-title", {}, t("Live log"), R.logMeta),
        h("div.row.mb", { style: { gap: "8px", flexWrap: "wrap" } },
          h("button.btn.sm.ghost", { onclick: () => jumpLog(false) }, "⇤ " + t("Top")),
          h("button.btn.sm.ghost", { onclick: () => jumpLog(true) }, t("Bottom") + " ⇥"),
          h("button.btn.sm.ghost", { onclick: downloadLog }, "⭳ " + t("Download log")),
          h("span", { style: { flex: "1" } }),
          h("span.small.muted", {}, t("Follow")),
          h("label.switch", {}, followChk, h("span.track"))),
        R.logView);

      detailPane.replaceChildren(headerCard, paramsCard, unitsCard, logCard);
    }

    function paintDetail(job) {
      const prev = state.detail;
      state.detail = job;
      if (!R) return;
      const sig = JSON.stringify(job);
      if (sig === state.detailSig) return;
      state.detailSig = sig;

      R.idEl.textContent = "#" + job.id;
      R.status.replaceChildren(UI.statusPill(job.status), UI.chip(job.type, TYPE_KIND[job.type] || ""));
      R.label.textContent = job.label || "";
      R.label.title = job.label || "";

      const times = [t("created {time}", { time: fmt.time(job.created_at) }) + " (" + fmt.ago(job.created_at) + ")"];
      if (job.started_at) times.push(t("started {time}", { time: fmt.time(job.started_at) }));
      if (job.ended_at) times.push(t("ended {time}", { time: fmt.time(job.ended_at) }));
      if (job.started_at) times.push(t("elapsed {dur}", { dur: fmt.dur((job.ended_at || Date.now() / 1000) - job.started_at) }));
      R.times.textContent = times.join("   ·   ");

      R.err.replaceChildren(job.error
        ? h("div", {
            style: {
              marginTop: "8px", padding: "8px 12px", borderRadius: "8px",
              background: "#f871711a", border: "1px solid #f8717150",
              color: "var(--danger)", fontSize: "12px",
            },
          }, "⚠ " + job.error)
        : "");

      const pr = progressOf(job);
      const p = job.progress || {};
      R.progress.replaceChildren(
        h("div.row", { style: { gap: "6px", marginBottom: "8px" } },
          UI.chip(t("done {n}", { n: p.done || 0 }), "green"),
          p.failed ? UI.chip(t("failed {n}", { n: p.failed }), "red") : null,
          p.running ? UI.chip(t("running {n}", { n: p.running }), "accent") : null,
          p.cancelled ? UI.chip(t("cancelled {n}", { n: p.cancelled }), "warn") : null,
          UI.chip(t("queued {n}", { n: p.queued || 0 }))),
        h("div.progress-track", {}, h("div.progress-fill", { style: { width: pr.pct + "%" } })));

      R.actions.replaceChildren(
        isActive(job) ? h("button.btn.sm.danger", { onclick: () => confirmCancel(job) }, "■ " + t("Cancel job")) : null,
        isTerminal(job) ? h("button.btn.sm.danger", { onclick: () => confirmDelete(job) }, "🗑 " + t("Delete record")) : null);

      const units = job.units || [];
      R.unitsHint.textContent = t("{n} units", { n: units.length });
      if (!units.length) {
        R.units.replaceChildren(UI.empty("▦", t("No execution units")));
      } else {
        const t = UI.table([
          { key: "idx", label: "#", num: true },
          { key: "label", label: t("Label"), render: (v, row) => h("span", { title: row.display || "" }, v) },
          { key: "status", label: t("Status"), render: v => UI.statusPill(v) },
          { key: "returncode", label: t("Return code"), num: true, render: v => v == null ? "—" : String(v) },
          { key: "expected_output", label: t("Expected output"), sortable: false,
            render: v => v ? h("span.mono.small", { title: v }, shorten(v, 64)) : h("span.muted", {}, "—") },
        ], units, { maxHeight: "300px" });
        R.units.replaceChildren(t.el);
      }
      updateLogMeta();

      // running -> terminal: pull the remaining log output once
      if (prev && prev.id === job.id && prev.status === "running" && job.status !== "running") fetchLog(false);
    }

    async function selectJob(id) {
      state.selectedId = id;
      state.detail = null;
      state.detailSig = "";
      paintList();
      const job = state.jobs.find(x => x.id === id);
      if (job) {
        buildDetail(job);
        paintDetail(job);
      } else {
        detailPane.replaceChildren(UI.loading(t("Loading job details…")));
        try {
          const d = await App.api("/api/jobs/" + encodeURIComponent(id));
          if (state.selectedId !== id) return;
          buildDetail(d);
          paintDetail(d);
        } catch (e) {
          detailPane.replaceChildren(UI.empty("⚠", t("Failed to load job details: {msg}", { msg: e.message })));
          return;
        }
      }
      resetLog();
      fetchLog(false);
    }

    /* ============================ actions ============================ */
    function confirmCancel(job) {
      UI.modal(t("Cancel job"), h("div", {},
        h("p", { style: { margin: "0 0 6px" } }, t("Cancel job #{id}?", { id: job.id })),
        h("p.small.muted", { style: { margin: "0" } }, t("Running units will be SIGTERM'd (SIGKILL after a 3 s grace period); queued units are marked cancelled directly."))), [
        { label: t("Keep it"), kind: "ghost" },
        { label: t("Cancel job"), kind: "danger", onClick: () => { doCancel(job.id); } },
      ]);
    }

    async function doCancel(id) {
      try {
        await App.api("/api/jobs/" + encodeURIComponent(id) + "/cancel", { method: "POST" });
        UI.toast(t("Job #{id} cancelled", { id }), "success");
      } catch (e) { UI.toast(t("Cancel failed: {msg}", { msg: e.message }), "error", 4500); }
      refreshJobs(true);
    }

    function confirmDelete(job) {
      UI.modal(t("Delete job record"), h("div", {},
        h("p", { style: { margin: "0 0 6px" } }, t("Delete the record of job #{id}?", { id: job.id })),
        h("p.small.muted", { style: { margin: "0" } }, t("The persisted JSON and log files will be deleted too; this cannot be undone."))), [
        { label: t("Keep it"), kind: "ghost" },
        { label: t("Delete"), kind: "danger", onClick: () => { doDelete(job.id); } },
      ]);
    }

    async function doDelete(id) {
      try {
        await App.api("/api/jobs/" + encodeURIComponent(id) + "/delete", { method: "POST" });
        UI.toast(t("Job #{id} deleted", { id }), "success");
        state.selectedId = null;
        state.detail = null;
        state.detailSig = "";
        R = null;
        detailPane.replaceChildren(UI.empty("☰", t("Select a job on the left to view details")));
      } catch (e) { UI.toast(t("Delete failed: {msg}", { msg: e.message }), "error", 4500); }
      refreshJobs(true);
    }

    /* ============================ live log ============================ */
    function scrollLog() { if (R && R.logView) R.logView.scrollTop = R.logView.scrollHeight; }

    function updateLogMeta() {
      if (!R || !R.logMeta) return;
      const live = state.detail && state.detail.status === "running";
      R.logMeta.textContent = (live ? t("polling every 2s · ") : "") +
        fmt.bytes(state.logOffset) + " / " + fmt.bytes(state.logTotal);
    }

    function resetLog() {
      state.logOffset = 0;
      state.logTotal = 0;
      if (R && R.logView) R.logView.replaceChildren();
      updateLogMeta();
    }

    async function fetchLog(tail) {
      const id = state.selectedId;
      if (!id || state.logBusy || !R || !R.logView) return;
      state.logBusy = true;
      try {
        const url = tail
          ? "/api/jobs/" + encodeURIComponent(id) + "/log?tail=1&length=65536"
          : "/api/jobs/" + encodeURIComponent(id) + "/log?offset=" + state.logOffset + "&length=65536";
        const w = await App.api(url);
        if (state.selectedId !== id || !R || !R.logView.isConnected) return;
        if (tail) {
          R.logView.replaceChildren();
          state.logOffset = w.offset;
        }
        if (w.length > 0) {
          state.logOffset = w.offset + w.length;
          R.logView.append(document.createTextNode(w.text));
          if (state.follow) scrollLog();
        }
        state.logTotal = w.total_size;
        updateLogMeta();
      } catch (e) { /* transient (job deleted etc.): next poll self-heals */ }
      finally { state.logBusy = false; }
    }

    function jumpLog(tail) {
      resetLog();
      fetchLog(tail);
    }

    async function downloadLog() {
      const id = state.selectedId;
      if (!id) return;
      try {
        let offset = 0, total = 0;
        const parts = [];
        for (let i = 0; i < 16; i++) {           // 16 x 1 MB server window cap
          const w = await App.api("/api/jobs/" + encodeURIComponent(id) +
            "/log?offset=" + offset + "&length=1048576");
          parts.push(w.text);
          total = w.total_size;
          offset = w.offset + w.length;
          if (w.eof || w.length === 0) break;
        }
        const url = URL.createObjectURL(new Blob(parts, { type: "text/plain;charset=utf-8" }));
        const a = h("a", { href: url, download: "job_" + id + ".log" });
        document.body.append(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 5000);
        UI.toast(offset < total
          ? t("Large log; downloaded first {offset} (of {total})", { offset: fmt.bytes(offset), total: fmt.bytes(total) })
          : t("Log downloaded"), "success");
      } catch (e) { UI.toast(t("Download failed: {msg}", { msg: e.message }), "error"); }
    }

    /* ============================ boot ============================ */
    el.replaceChildren(layout);

    // fast poll only while something is active; slow poll catches external submits
    App.every(3000, () => {
      if (state.jobs.some(isActive) || (state.detail && isActive(state.detail))) refreshJobs(false);
    });
    App.every(15000, () => refreshJobs(false));
    App.every(2000, () => {
      if (state.detail && state.detail.status === "running") fetchLog(false);
    });

    await refreshJobs(true);
    if (state.selectedId) await selectJob(state.selectedId);
  },
});
