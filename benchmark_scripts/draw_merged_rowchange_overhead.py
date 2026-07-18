import os
import math
import matplotlib.pyplot as plt
from fig_common import *
from run_all_tests import (default_params, NPU_FREQ_MHZ, TRCD, TRP,
                            uniform_map_vs_bw_pairs, trp_sweep_list,
                            TRACE_OUT, batch_sizes, seq_lengths)
from parse_exposed import parse_log_file, get_exposed_row_conflict_time, get_total_time

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=20)
plt.rc('hatch', color='white')

BAR_WIDTH   = 1.
BAR_SPACING = 0.1
GROUP_SPACING = 1.5


def draw_merged_figure():
    model = "llama2-13"

    impl = {
        "uniform_dram": "Uniform",
        "best":    "Software-aware",
    }
    impl_keys = ["uniform_dram", "best"]
    color_idx = {"uniform_dram": 0, "best": 1}
    local_colors      = colors[:2]
    local_dark_colors = dark_colors[:2]
    cycles_to_ms = 1.0 / (NPU_FREQ_MHZ * 1e3)

    # ── Left panel data (MODE 8: uniform_map vs BW) ───────────────────────────────
    core_group_list = uniform_map_vs_bw_pairs["core_group"]
    dram_bw_list    = uniform_map_vs_bw_pairs["dram_bw"]
    bw_labels = [f"{bw / 1024:.0f}" for bw in dram_bw_list]

    cfg_l = {"model": model, "op_type": "decode", "bs": 32}
    for k, v in default_params.items():
        cfg_l[k] = v

    exe_times_l   = {k: [] for k in impl_keys}
    base_times_l  = {k: [] for k in impl_keys}

    for i in range(len(core_group_list)):
        cfg_l['core_group'] = core_group_list[i]
        cfg_l['dram_bw']    = dram_bw_list[i]
        for best_str in impl_keys:
            prefix = os.path.join(TRACE_OUT, model, f"bs_{cfg_l['bs']}",
                                  f"core_{cfg_l['num_cores']}", cfg_l['op_type'],
                                  f"sa_{cfg_l['sa']}-vu_{cfg_l['sa']}",
                                  f"sram_{cfg_l['sram_kb']}-drambw_{cfg_l['dram_bw']}_PLACEHOLDER",
                                  f"topo_{cfg_l['noc_topo']}-nocbw{cfg_l['noc_bw']}", best_str)
            logfile = os.path.join(prefix,
                                   f"output_cg_{cfg_l['core_group']}_row_{cfg_l['row']}.log")
            if not os.path.exists(logfile):
                print(f"WARNING: Missing {logfile}")
                exe_time = remaining = 0.0
            else:
                with open(logfile) as f:
                    exe_time = float(f.readline().split(', ')[1].split(':')[1].split(',')[0])
                with open(logfile) as f:
                    op_logs = parse_log_file(f.read(), dram_bw_GBps=cfg_l['dram_bw'],
                                             npu_freq_MHz=NPU_FREQ_MHZ)
                bd = get_exposed_row_conflict_time(op_logs, cfg_l['dram_bw'], NPU_FREQ_MHZ)
                lt = get_total_time(op_logs)
                if lt:
                    bd *= exe_time / lt
                remaining = exe_time - bd
            exe_times_l[best_str].append(exe_time   * cycles_to_ms)
            base_times_l[best_str].append(remaining  * cycles_to_ms)

    # ── Right panel data (tRP sweep) ─────────────────────────────────────────
    trp_list    = list(trp_sweep_list)
    trp_ns_list = [int(round(trp * 1000 / NPU_FREQ_MHZ)) for trp in trp_list]
    trp_labels  = [f"{ns}" for ns in trp_ns_list]

    cfg_r = {"model": model, "op_type": "decode",
             "bs": batch_sizes[0]}
    for k, v in default_params.items():
        cfg_r[k] = v

    exe_times_r      = {k: [] for k in impl_keys}
    base_times_r     = {k: [] for k in impl_keys}
    overhead_times_r = {k: [] for k in impl_keys}

    for trp in trp_list:
        for best_str in impl_keys:
            prefix = os.path.join(TRACE_OUT, model, f"bs_{cfg_r['bs']}",
                                  f"core_{cfg_r['num_cores']}",
                                  cfg_r['op_type'],
                                  f"sa_{cfg_r['sa']}-vu_{cfg_r['sa']}",
                                  f"sram_{cfg_r['sram_kb']}-drambw_{cfg_r['dram_bw']}_PLACEHOLDER",
                                  f"topo_{cfg_r['noc_topo']}-nocbw{cfg_r['noc_bw']}", best_str)
            logfile = os.path.join(prefix,
                                   f"output_cg_{cfg_r['core_group']}_row_{cfg_r['row']}"
                                   f"_trcd_{TRCD}_trp_{trp}.log")
            if not os.path.exists(logfile):
                print(f"WARNING: Missing {logfile}")
                exe_time = breakdown_time = 0.0
            else:
                with open(logfile) as f:
                    exe_time = float(f.readline().split(', ')[1].split(':')[1].split(',')[0])
                with open(logfile) as f:
                    op_logs = parse_log_file(f.read(), dram_bw_GBps=cfg_r['dram_bw'],
                                             npu_freq_MHz=NPU_FREQ_MHZ)
                breakdown_time = get_exposed_row_conflict_time(op_logs, cfg_r['dram_bw'],
                                                               NPU_FREQ_MHZ)
                lt = get_total_time(op_logs)
                if lt:
                    breakdown_time *= exe_time / lt
            exe_times_r[best_str].append(exe_time * cycles_to_ms)
            base_times_r[best_str].append(max(0.0, exe_time - breakdown_time) * cycles_to_ms)
            overhead_times_r[best_str].append(breakdown_time * cycles_to_ms)

    # ── Figure layout ──────────────────────────────────────────────────────────
    group_width = BAR_WIDTH * 2 + BAR_SPACING * 1 + GROUP_SPACING
    margin = (BAR_WIDTH + GROUP_SPACING) / 2
    num_l = len(core_group_list)
    num_r = len(trp_list)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 2.5), sharey=True,
                                      gridspec_kw={'width_ratios': [num_l, num_r]})
    plt.subplots_adjust(top=0.85, bottom=0.25, left=0.06, right=0.99, wspace=0.04)

    # ── Left panel ────────────────────────────────────────────────────────────
    for i in range(num_l):
        offset = group_width * i
        for j, key in enumerate(impl_keys):
            bar_x = offset + j * (BAR_WIDTH + BAR_SPACING)
            ci = color_idx[key]
            base = base_times_l[key][i]
            rect = ax_l.bar(bar_x, base, width=BAR_WIDTH, color=local_colors[ci],
                            hatch='', edgecolor='black')
            rect[0]._hatch_color = mpl.colors.to_rgba('white')
            overhead = exe_times_l[key][i] - base
            rect_ov = ax_l.bar(bar_x, overhead, bottom=base, width=BAR_WIDTH,
                               color=local_dark_colors[ci], hatch='//', edgecolor='black')
            rect_ov[0]._hatch_color = mpl.colors.to_rgba('white')

    xticks_l = [group_width * i + (BAR_WIDTH + BAR_SPACING) / 2 for i in range(num_l)]
    ax_l.set_xticks(xticks_l)
    ax_l.set_xticklabels(bw_labels, rotation=0, fontsize=20, ha='center')
    ax_l.set_xlabel("(a) DRAM Bandwidth (TBps)", fontsize=23)
    ax_l.set_ylabel("Decode TBT (ms)", fontsize=20)
    ax_l.set_xlim([-margin, num_l * group_width - margin])
    ylim_l = ax_l.get_ylim()
    ax_l.set_ylim([0, ylim_l[1]])
    ax_l.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax_l.set_axisbelow(True)

    # Annotation arrow pointing at conflict overhead on uniform bar
    if exe_times_l["uniform_dram"] and any(e > 0 for e in exe_times_l["uniform_dram"]):
        arrow_idx = min(4, num_l - 1)  # 12 TBps or last config
        bad_bar_x = group_width * arrow_idx + BAR_WIDTH / 2
        arrow_y_bottom = base_times_l["uniform_dram"][arrow_idx]
        arrow_y_top    = exe_times_l["uniform_dram"][arrow_idx]
        arrow_y_mid    = (arrow_y_bottom + arrow_y_top) / 1.9
        ax_l.annotate('DRAM row-buffer\nconflict overhead',
                      xy=(bad_bar_x, arrow_y_mid),
                      xytext=(bad_bar_x - 2, arrow_y_mid + ylim_l[1] * 0.2 - 1),
                      fontsize=18, ha='left',
                      arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # ── Right panel ───────────────────────────────────────────────────────────
    for i in range(num_r):
        offset = group_width * i
        for j, key in enumerate(impl_keys):
            bar_x = offset + j * (BAR_WIDTH + BAR_SPACING)
            ci = color_idx[key]
            base = base_times_r[key][i]
            rect = ax_r.bar(bar_x, base, width=BAR_WIDTH, color=local_colors[ci],
                            hatch='', edgecolor='black')
            rect[0]._hatch_color = mpl.colors.to_rgba('white')
            overhead = overhead_times_r[key][i]
            rect_ov = ax_r.bar(bar_x, overhead, bottom=base, width=BAR_WIDTH,
                               color=local_dark_colors[ci], hatch='//', edgecolor='black')
            rect_ov[0]._hatch_color = mpl.colors.to_rgba('white')

    xticks_r = [group_width * i + (BAR_WIDTH + BAR_SPACING) / 2 for i in range(num_r)]
    ax_r.set_xticks(xticks_r)
    ax_r.set_xticklabels(trp_labels, rotation=0, fontsize=20, ha='center')
    ax_r.set_xlabel("(b) DRAM tRP (ns)", fontsize=23)
    ax_r.set_xlim([-margin, num_r * group_width - margin])
    ax_r.tick_params(axis='y', labelleft=False)
    ax_r.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)
    ax_r.set_axisbelow(True)

    # ── Shared legend ──────────────────────────────────────────────────────────
    handles = []
    legend_labels = []
    for key in impl_keys:
        ci = color_idx[key]
        p = plt.Rectangle((0, 0), 1, 1, facecolor=local_colors[ci], edgecolor='black', hatch='')
        p._hatch_color = mpl.colors.to_rgba('white')
        handles.append(p)
        legend_labels.append(impl[key])
    fig.legend(handles, legend_labels, fontsize=18, ncol=2, frameon=False,
               handlelength=1.2, handletextpad=0.3, columnspacing=0.8,
               loc='upper right', bbox_to_anchor=(1.0, 1.05))

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("figures", exist_ok=True)
    out_path = f"figures/eval_merged_rowchange_overhead_{model}.pdf"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved {out_path}")


if __name__ == "__main__":
    draw_merged_figure()
