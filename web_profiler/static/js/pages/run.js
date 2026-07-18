/* Run — launch simulations / sweeps / DSE / thermal analyses from the web.
   The sim tab exposes every sweepable parameter as a multi-stop "tier"
   slider; picking several stops on several sliders expands to a batch. */

App.route("run", {
  title: t("Run"),
  async render(el) {
    el.append(UI.loading(t("Loading tier definitions…")));
    const defs = await App.jobsDefs();
    el.replaceChildren();

    const tabs = UI.tabs([
      { key: "sim", label: "▶ " + t("Sim / Sweep") },
      { key: "dse", label: "◭ DSE Pareto" },
      { key: "thermal", label: "♨ " + t("Thermal") },
    ], "sim", key => paint(key));
    el.append(tabs.el);
    const body = h("div");
    el.append(body);

    /* ============================================================
       shared bits
       ============================================================ */
    const chipRow = (items, initial, onChange) => {
      // items: [{value, label, hint}]  multi-select chip group
      let sel = new Set((initial || []).map(String));
      const row = h("div.chip-row");
      const paint = () => row.querySelectorAll(".chip-toggle").forEach(c =>
        c.classList.toggle("on", sel.has(c.dataset.v)));
      for (const it of items) {
        row.append(h("span.chip-toggle", {
          dataset: { v: String(it.value) }, title: it.hint || "",
          onclick: () => {
            const sv = String(it.value);
            if (sel.has(sv)) { if (sel.size > 1) sel.delete(sv); } else sel.add(sv);
            paint(); onChange && onChange([...sel]);
          }
        }, it.label));
      }
      paint();
      return { el: row, get: () => [...sel] };
    };

    const segToggle = (options, initial, onChange) => {
      let cur = initial;
      const el2 = h("div.seg");
      const paint = () => el2.querySelectorAll(".seg-opt").forEach(o =>
        o.classList.toggle("on", o.dataset.v === cur));
      for (const o of options) {
        el2.append(h("span.seg-opt", {
          dataset: { v: o.value },
          onclick: () => { cur = o.value; paint(); onChange && onChange(cur); }
        }, o.label));
      }
      paint();
      return { el: el2, get: () => cur };
    };

    const submitJob = async (type, params, btn) => {
      btn.disabled = true;
      try {
        const job = await App.api("/api/jobs", { json: { type, params } });
        UI.toast(t("Job {id} created ({label})", { id: job.id, label: job.label || type }), "success");
        App.navigate("jobs", { id: job.id });
      } catch (e) {
        UI.toast(t("Submit failed: {msg}", { msg: e.message }), "error", 5200);
        btn.disabled = false;
      }
    };

    /* ============================================================
       TAB 1 — sim / batch
       ============================================================ */
    function paintSim(container) {
      const d = defs.defaults;
      const state = {
        model: [d.model], mode: d.mode, impl: [d.impl], root: d.root,
      };
      const tiers = {};   // key -> [values]

      const layout = h("div.run-layout");
      container.append(layout);

      /* ---- left: basic settings ---- */
      const left = h("div.col");
      layout.append(left);

      const modelChip = chipRow(
        Object.entries(defs.models).map(([m, info]) => ({
          value: m, label: m, hint: t("{b}B params · {n} layers", { b: info.params_b, n: info.layers }),
        })), [d.model], v => { state.model = v; updateCombo(); });

      const implChip = chipRow(defs.impls.map(i => ({ value: i.value, label: i.label })),
        [d.impl], v => { state.impl = v; updateCombo(); });

      const bsTierHint = h("span.hint");
      const modeSeg = segToggle([
        { value: "decode", label: t("Decode") },
        { value: "prefill", label: t("Prefill") },
      ], d.mode, v => {
        state.mode = v;
        // prefill is always single-request
        if (v === "prefill") { tiers.batch_size.set([1]); }
        bsTierHint.textContent = v === "prefill" ? t("prefill forces batch=1") : "";
        updateCombo();
      });

      const basicCard = h("div.card", {},
        UI.sectionTitle(t("Basic settings")),
        h("label.field.mb", {}, t("Model (multi-select)"), modelChip.el),
        h("label.field.mb", {}, t("Implementation (multi-select)"), implChip.el),
        h("label.field.mb", {}, t("Mode"), modeSeg.el),
        h("label.field", {}, t("Results root directory (root)"),
          h("input", {
            type: "text", value: d.root, style: { fontFamily: "var(--mono)" },
            oninput: e => { state.root = e.target.value.trim() || "logs_web"; },
          })));
      left.append(basicCard);

      const infoCard = h("div.card", {},
        UI.sectionTitle(t("Notes")),
        h("div.small.muted", { html:
          t("· Each parameter can take <b>multiple stops</b>; all multi-selections expand via Cartesian product into a batch of simulations<br>") +
          t("· Single combo = 1 simulation; multiple combos = batch (max 64)<br>") +
          t("· Missing hw-config JSON files are auto-generated; existing files are never overwritten<br>") +
          t("· decode split_factor={dsf}, prefill={psf}<br>", { dsf: defs.notes.decode_split_factor, psf: defs.notes.prefill_split_factor }) +
          t("· Concurrent worker slots: {n}", { n: defs.max_workers }) }));
      left.append(infoCard);

      /* ---- right: tier sliders ---- */
      const TIER_DEFS = [
        ["sa_size", t("SA array size"), ""], ["sram_kb", t("SRAM capacity"), "KB"],
        ["dram_bw", t("DRAM bandwidth"), "GB/s"], ["num_cores", t("Core count"), ""],
        ["noc_bw", t("NoC bandwidth"), "B/cyc"], ["noc_topo", t("NoC Topology"), "1=Mesh 2=Torus 3=All"],
        ["core_group", "Core Group", ""], ["row", "DRAM Row", ""],
        ["batch_size", "Batch Size", ""], ["sequence_length", t("Sequence length"), "tok"],
      ];
      const tierCard = h("div.card.grow", {}, UI.sectionTitle(t("Parameter stops"), t("Click stops to multi-select; combo count updates live below")), bsTierHint);
      layout.append(tierCard);
      for (const [key, label, unit] of TIER_DEFS) {
        const t = UI.tierSelect(
          { key, label, unit, stops: defs.tiers[key] },
          [d[key]], () => updateCombo());
        tiers[key] = t;
        tierCard.append(t.el);
      }

      /* ---- sticky action bar ---- */
      const comboN = h("span.combo-pill", {}, h("span.n", {}, "1"), h("span.t", {}, t("combo → 1 simulation")));
      const preview = h("div.small.muted", { style: { marginTop: "8px" } });
      const launchBtn = h("button.btn.primary", { onclick: submit }, "▶ " + t("Launch simulation"));
      const bar = h("div.action-bar", {},
        h("div.grow", {}, comboN, preview),
        launchBtn);
      container.append(bar);

      function selections() {
        const s = { model: state.model, impl: state.impl };
        for (const [k, t] of Object.entries(tiers)) s[k] = t.get();
        return s;
      }
      function comboCount() {
        return Object.values(selections()).reduce((n, vals) => n * vals.length, 1);
      }
      function updateCombo() {
        const n = comboCount();
        comboN.querySelector(".n").textContent = n;
        comboN.querySelector(".t").textContent =
          n === 1 ? t("combo → 1 simulation") : t("combos → {n} simulations (batch)", { n }) +
          (n > 64 ? t("  ⚠ over limit 64") : "");
        launchBtn.disabled = n > 64;
        launchBtn.textContent = n === 1 ? "▶ " + t("Launch simulation") : "▶ " + t("Launch {n} simulations", { n });
        // preview first few combos
        const s = selections();
        const keys = Object.keys(s);
        const combos = [[]];
        for (const k of keys) {
          const vals = s[k];
          const next = [];
          for (const c of combos) for (const v of vals) next.push([...c, [k, v]]);
          combos.splice(0, combos.length, ...next);
        }
        const labels = combos.slice(0, 4).map(c => {
          const m = Object.fromEntries(c);
          return `${m.model} ${state.mode} bs${m.batch_size} c${m.num_cores} sa${m.sa_size} ` +
                 `sram${m.sram_kb} bw${m.dram_bw} topo${m.noc_topo}/noc${m.noc_bw} cg${m.core_group} ${m.impl}`;
        });
        preview.textContent = labels.join("    |    ") + (combos.length > 4 ? t("    … {n} combos total", { n: combos.length }) : "");
      }

      async function submit() {
        const s = selections();
        const n = comboCount();
        const params = { mode: state.mode, root: state.root };
        for (const [k, vals] of Object.entries(s)) {
          params[k] = vals.length === 1 ? (k === "model" || k === "impl" ? vals[0] : +vals[0])
                                        : (k === "model" || k === "impl" ? vals : vals.map(Number));
        }
        await submitJob(n === 1 ? "sim" : "batch", params, launchBtn);
      }
      updateCombo();
    }

    /* ============================================================
       TAB 2 — DSE
       ============================================================ */
    function paintDse(container) {
      const state = { mode: "decode", models: ["llama2-13"], num_sweeps: null, max_cycles: null, core_group: null, exhaustive: false };
      const card = h("div.card", { style: { maxWidth: "720px" } });
      container.append(card);

      const modeSeg = segToggle([{ value: "decode", label: "Decode" }, { value: "prefill", label: "Prefill" }],
        "decode", v => state.mode = v);
      const modelsChip = chipRow(Object.keys(defs.models).map(m => ({ value: m, label: m })),
        state.models, v => state.models = v);
      const num = (label, key, ph) => h("label.field", {}, label,
        h("input", { type: "number", min: 1, placeholder: ph, style: { width: "140px" },
          oninput: e => state[key] = e.target.value ? +e.target.value : null }));
      const exh = h("label.switch", {},
        h("input", { type: "checkbox", onchange: e => state.exhaustive = e.target.checked }),
        h("span.track"));

      const btn = h("button.btn.primary.mt", {
        onclick: () => {
          const params = { mode: state.mode, models: state.models };
          for (const k of ["num_sweeps", "max_cycles", "core_group"]) if (state[k]) params[k] = state[k];
          if (state.exhaustive) params.exhaustive = true;
          submitJob("dse", params, btn);
        }
      }, "◭ " + t("Launch DSE sweep"));

      card.append(
        UI.sectionTitle(t("DSE Pareto exploration"), t("Runs dse_pareto.py")),
        h("div.row.mb", {}, h("label.field", {}, t("Mode"), modeSeg.el),
          num(t("Sweeps"), "num_sweeps", t("Default")),
          num(t("Max cycles"), "max_cycles", t("Default")),
          num("core group", "core_group", t("Default"))),
        h("label.field.mb", {}, t("Model set"), modelsChip.el),
        h("div.row", {}, exh, h("span.small", {}, t("Exhaustive mode (exhaustive)"))),
        btn);
    }

    /* ============================================================
       TAB 3 — thermal
       ============================================================ */
    function paintThermal(container) {
      const state = {
        models: ["llama2-13"], modes: ["decode"], impls: ["best"], backends: ["simple"],
        results_dir: "results/logs", out_dir: "results/thermal_validation_web",
        dram_bws: "12288", rows: "8192", core_groups: "8", dram_layers: null,
      };
      const card = h("div.card", { style: { maxWidth: "760px" } });
      container.append(card);
      const text = (label, key, w = 200) => h("label.field", {}, label,
        h("input", { type: "text", value: state[key], style: { width: w + "px", fontFamily: "var(--mono)" },
          oninput: e => state[key] = e.target.value.trim() }));
      const btn = h("button.btn.primary.mt", {
        onclick: () => {
          const params = {
            models: state.models.join(","), modes: state.modes.join(","),
            impls: state.impls.join(","), backends: state.backends.join(","),
            results_dir: state.results_dir, out_dir: state.out_dir,
            dram_bws: state.dram_bws, rows: state.rows, core_groups: state.core_groups,
          };
          if (state.dram_layers) params.dram_layers = +state.dram_layers;
          submitJob("thermal", params, btn);
        }
      }, "♨ " + t("Launch thermal analysis"));

      card.append(
        UI.sectionTitle(t("Thermal analysis pipeline"), t("Runs tsim_thermal.cli")),
        h("div.grid.cols-2", {},
          h("label.field", {}, t("Model"), chipRow(Object.keys(defs.models).concat("dit-xl").filter((v, i, a) => a.indexOf(v) === i).map(m => ({ value: m, label: m })), state.models, v => state.models = v).el),
          h("label.field", {}, t("Mode"), chipRow([{ value: "decode", label: "decode" }, { value: "prefill", label: "prefill" }], state.modes, v => state.modes = v).el),
          h("label.field", {}, t("Impl"), chipRow(defs.impls.map(i => ({ value: i.value, label: i.value })), state.impls, v => state.impls = v).el),
          h("label.field", {}, t("Thermal backend"), chipRow(["simple", "hotspot", "threedice", "adaptive_grid"].map(b => ({ value: b, label: b })), state.backends, v => state.backends = v).el)),
        h("div.row.mt", {},
          text(t("Results directory"), "results_dir", 220), text(t("Output directory"), "out_dir", 260)),
        h("div.row.mt", {},
          text(t("DRAM BW (comma-separated)"), "dram_bws", 160), text("Row", "rows", 120),
          text("Core Group", "core_groups", 120),
          h("label.field", {}, t("DRAM layers"), h("input", { type: "number", min: 1, max: 64, placeholder: t("Default"), style: { width: "90px" }, oninput: e => state.dram_layers = e.target.value }))),
        btn);
    }

    function paint(key) {
      body.replaceChildren();
      if (key === "sim") paintSim(body);
      else if (key === "dse") paintDse(body);
      else paintThermal(body);
    }
    paint("sim");
  },
});
