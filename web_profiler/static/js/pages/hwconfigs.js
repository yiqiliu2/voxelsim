/* HwConfigs — 硬件配置库: search / inspect / create.
   Create mirrors POST /api/hwconfigs: {name, config} with compute/noc/dram
   sections; existing files are never overwritten (server returns 409). */

App.route("hwconfigs", {
  title: t("HW Configs"),
  async render(el) {
    el.append(UI.loading(t("Loading hardware configs…")));
    let configs = [];
    try { configs = (await App.api("/api/hwconfigs")).configs || []; }
    catch (e) { el.replaceChildren(UI.empty("⚠", t("Failed to load: {msg}", { msg: e.message }))); return; }

    const TOPO = { 1: "Mesh", 2: "Torus", 3: "All" };
    const state = { q: "" };

    /* ============================ list ============================ */
    const countHint = h("span.hint");
    const tbl = UI.table([
      { key: "name", label: t("Name"), render: (v, r) => h("span", {},
          h("b", { style: { fontWeight: 650 } }, v),
          r.parse_error ? [" ", UI.chip(t("Parse failed"), "red")] : null) },
      { key: "sa", label: "SA", num: true, render: v => v == null ? "—" : v },
      { key: "dram_bw", label: "DRAM BW", num: true, render: v => v == null ? "—" : fmt.int(v) },
      { key: "noc_topo", label: t("Topology"), render: v => v == null ? "—" : UI.chip(v + "·" + (TOPO[v] || "?"), "accent") },
      { key: "noc_bw", label: "NoC BW", num: true, render: v => v == null ? "—" : v },
      { key: "row", label: "Row", num: true, render: v => v == null ? "—" : fmt.int(v) },
      { key: "freq_mhz", label: "Freq", num: true, render: v => v == null ? "—" : fmt.int(v) + " MHz" },
      { key: "trp", label: "tRP", num: true, render: v => v == null ? "—" : v },
      { key: "mtime", label: t("Modified"), num: true, render: v => fmt.ago(v) },
    ], configs, { onRow: r => openDetail(r.name) });

    function applyFilter() {
      const rows = state.q
        ? configs.filter(c => c.name.toLowerCase().indexOf(state.q) >= 0)
        : configs;
      tbl.setRows(rows);
      countHint.textContent = t("{shown} / {total} configs", { shown: rows.length, total: configs.length });
    }

    async function reload() {
      try {
        configs = (await App.api("/api/hwconfigs")).configs || [];
        applyFilter();
      } catch (e) { UI.toast(t("Refresh failed: {msg}", { msg: e.message }), "error"); }
    }

    /* ============================ detail modal ============================ */
    async function openDetail(name) {
      const box = h("div", {}, UI.loading(t("Loading config…")));
      UI.modal(t("HW config · {name}", { name }), box, [{ label: t("Close"), kind: "ghost" }]);
      try {
        const cfg = await App.api("/api/hwconfigs/" + encodeURIComponent(name));
        const text = JSON.stringify(cfg, null, 2);
        box.replaceChildren(
          h("div.row.mb", { style: { justifyContent: "flex-end" } }, UI.copyBtn(text, t("Copy JSON"))),
          h("pre", {
            style: {
              margin: "0", fontFamily: "var(--mono)", fontSize: "12px", lineHeight: "1.6",
              whiteSpace: "pre-wrap", wordBreak: "break-all", color: "#b6c6e2",
              maxHeight: "55vh", overflow: "auto",
            },
          }, text));
      } catch (e) {
        box.replaceChildren(UI.empty("⚠", t("Failed to read: {msg}", { msg: e.message })));
      }
    }

    /* ============================ create modal ============================ */
    function openCreate() {
      const F = {};   // "sec.key" -> input element

      const numField = (sec, key, label, def, isInt) => {
        const inp = h("input", {
          type: "number", value: def, step: isInt ? "1" : "any",
          style: { width: "100%" },
        });
        F[sec + "." + key] = inp;
        return h("label.field", {}, label, inp);
      };
      const textField = (sec, key, label, def, hint) => {
        const inp = h("input", {
          type: "text", value: def, placeholder: hint || "",
          style: { width: "100%", fontFamily: "var(--mono)" },
        });
        F[sec + "." + key] = inp;
        return h("label.field", {}, label, inp);
      };
      const boolField = (sec, key, label, def) => {
        const cb = h("input", { type: "checkbox" });
        cb.checked = def;
        const txt = h("span.small.muted", {}, def ? "true" : "false");
        cb.addEventListener("change", () => { txt.textContent = cb.checked ? "true" : "false"; });
        F[sec + "." + key] = cb;
        return h("label.field", {}, label,
          h("div.row", { style: { gap: "8px", height: "34px", flexWrap: "nowrap" } },
            h("label.switch", {}, cb, h("span.track")), txt));
      };

      const nameInput = h("input", {
        type: "text", placeholder: t("e.g. sa_128_vu_128_drambw_12288_noc_1_16"),
        style: { width: "100%", fontFamily: "var(--mono)" },
      });

      const topoSel = h("select", { style: { width: "100%" } },
        h("option", { value: "1" }, "1 · Mesh"),
        h("option", { value: "2" }, "2 · Torus"),
        h("option", { value: "3" }, "3 · All-to-All"));
      F["noc.topology"] = topoSel;

      const secTitle = txt => h("div", {
        style: {
          fontSize: "11px", fontWeight: 700, color: "var(--accent)", letterSpacing: "1.5px",
          textTransform: "uppercase", margin: "18px 0 10px",
        },
      }, txt);
      const grid3 = kids => h("div", {
        style: { display: "grid", gridTemplateColumns: "repeat(3, minmax(0,1fr))", gap: "10px 12px" },
      }, kids);

      const errBox = h("div", {
        style: {
          display: "none", marginTop: "14px", padding: "9px 13px", borderRadius: "8px",
          background: "#f871711a", border: "1px solid #f8717150", color: "var(--danger)", fontSize: "12.5px",
        },
      });
      const showErr = msg => { errBox.style.display = ""; errBox.textContent = "⚠ " + msg; };

      const form = h("div", {},
        h("label.field", {}, t("Config name (letters / digits / _ / -)"), nameInput),
        secTitle("compute"),
        grid3([
          numField("compute", "s0", "mm_pad_shape[0]", 128, true),
          numField("compute", "s1", "mm_pad_shape[1]", 128, true),
          numField("compute", "s2", "mm_pad_shape[2]", 128, true),
          numField("compute", "ew_pad_len", "ew_pad_len", 128),
          numField("compute", "ew_reuse_num", "ew_reuse_num", 128),
          numField("compute", "ew_flopc", "ew_flopc", 128),
          numField("compute", "mm_flopc", "mm_flopc", 32768),
          numField("compute", "load_store_bw_bytepc", "load_store_bw_bytepc", 256),
          numField("compute", "byte_per_elem", "byte_per_elem", 2),
          numField("compute", "mm_init_cycle", "mm_init_cycle", 128),
          textField("compute", "mm_reuse_list", t("mm_reuse_list (comma-separated)"), "128, 16384, 128"),
          boolField("compute", "ew_mm_overlap", "ew_mm_overlap", true),
        ]),
        secTitle("noc"),
        grid3([
          numField("noc", "bandwidth_bytepc", "bandwidth_bytepc", 16),
          h("label.field", {}, "topology", topoSel),
          boolField("noc", "default_noc", "default_noc", true),
        ]),
        secTitle("dram"),
        grid3([
          numField("dram", "CL", "CL", 14),
          numField("dram", "tRCD", "tRCD", 14),
          numField("dram", "tRP", "tRP", 14),
          numField("dram", "bandwidth_GBps", "bandwidth_GBps", 12288),
          numField("dram", "num_access_per_row", "num_access_per_row", 8192),
          numField("dram", "npu_freq_MHz", "npu_freq_MHz", 1500),
        ]),
        errBox);

      function collect() {
        const numOf = (k, label) => {
          const v = parseFloat(F[k].value);
          if (!Number.isFinite(v)) throw new Error(t("{label}: please enter a number", { label }));
          return v;
        };
        const shape = [0, 1, 2].map(i => {
          const v = numOf("compute.s" + i, "mm_pad_shape[" + i + "]");
          if (!Number.isInteger(v) || v <= 0) throw new Error(t("mm_pad_shape[{i}]: must be a positive integer", { i }));
          return v;
        });
        const reuse = F["compute.mm_reuse_list"].value.split(",")
          .map(s => s.trim()).filter(Boolean)
          .map(s => {
            const v = parseFloat(s);
            if (!Number.isFinite(v)) throw new Error(t("mm_reuse_list: non-numeric item {s}", { s }));
            return v;
          });
        if (!reuse.length) throw new Error(t("mm_reuse_list: need at least one number"));
        return {
          compute: {
            mm_pad_shape: shape,
            mm_reuse_list: reuse,
            ew_pad_len: numOf("compute.ew_pad_len", "ew_pad_len"),
            ew_reuse_num: numOf("compute.ew_reuse_num", "ew_reuse_num"),
            ew_flopc: numOf("compute.ew_flopc", "ew_flopc"),
            mm_flopc: numOf("compute.mm_flopc", "mm_flopc"),
            load_store_bw_bytepc: numOf("compute.load_store_bw_bytepc", "load_store_bw_bytepc"),
            byte_per_elem: numOf("compute.byte_per_elem", "byte_per_elem"),
            mm_init_cycle: numOf("compute.mm_init_cycle", "mm_init_cycle"),
            ew_mm_overlap: F["compute.ew_mm_overlap"].checked,
          },
          noc: {
            bandwidth_bytepc: numOf("noc.bandwidth_bytepc", "bandwidth_bytepc"),
            topology: parseInt(topoSel.value, 10),
            default_noc: F["noc.default_noc"].checked,
          },
          dram: {
            CL: numOf("dram.CL", "CL"),
            tRCD: numOf("dram.tRCD", "tRCD"),
            tRP: numOf("dram.tRP", "tRP"),
            bandwidth_GBps: numOf("dram.bandwidth_GBps", "bandwidth_GBps"),
            num_access_per_row: numOf("dram.num_access_per_row", "num_access_per_row"),
            npu_freq_MHz: numOf("dram.npu_freq_MHz", "npu_freq_MHz"),
          },
        };
      }

      const submitBtn = h("button.btn.primary", {}, "＋ " + t("Create config"));
      form.append(h("div", { style: { display: "flex", justifyContent: "flex-end", marginTop: "16px" } }, submitBtn));
      submitBtn.addEventListener("click", async () => {
        errBox.style.display = "none";
        const name = nameInput.value.trim();
        if (!/^[A-Za-z0-9_\-]+$/.test(name)) { showErr(t("Name may only contain letters, digits, _ and -")); return; }
        let cfg;
        try { cfg = collect(); } catch (e) { showErr(e.message); return; }
        submitBtn.disabled = true;
        try {
          await App.api("/api/hwconfigs", { json: { name: name, config: cfg } });
          UI.toast(t("Config {name} created", { name }), "success");
          close();
          reload();
        } catch (e) {
          showErr(e.message);           // 400 校验错误 / 409 重名
          submitBtn.disabled = false;
        }
      });

      const close = UI.modal(t("New HW config"), form, [
        { label: t("Cancel"), kind: "ghost" },
      ]);
    }

    /* ============================ page ============================ */
    const search = h("input", {
      type: "search", placeholder: t("Search configs…"), style: { width: "260px" },
      oninput: () => { state.q = search.value.trim().toLowerCase(); applyFilter(); },
    });

    el.replaceChildren(
      h("div.card.filterbar", {},
        h("label.field", {}, t("Search"), search),
        h("span.spacer"),
        h("button.btn.primary", { onclick: openCreate }, "＋ " + t("New config"))),
      h("div.card", {}, h("h3.card-title", {}, t("Config list"), countHint), tbl.el));
    applyFilter();
  },
});
