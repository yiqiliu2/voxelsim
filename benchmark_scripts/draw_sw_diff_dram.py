import os
from typing import List, Dict
import matplotlib.pyplot as plt
from fig_common import *
from legend_helper import SplitPatch, SplitLegendBox
from run_all_tests import default_params, NPU_FREQ_MHZ, DF_FACTOR, DF_DRAM_DIV, DF_DRAM_MUL
from parse_exposed import parse_log_file, get_hw_units_exclusive_time, get_total_time

PREFILL = True
TRACE_OUT = f"results/logs"
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=20)
plt.rc('hatch', color='white')

models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

BAR_WIDTH = 1.
BAR_SPACING = 0.1
GROUP_SPACING = 1.5


def draw_one_figure(is_prefill:bool, impl:dict, title:str, breakdown:List[str]=None, ax=None, draw_legend=True):
    standalone = ax is None
    total_label_length = sum([len(impl[key]) for key in impl])
    if total_label_length > 20:
        bbox = 1.55
        ncol = 4
        if standalone:
            fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2))
            plt.subplots_adjust(top=0.78, bottom=0.33, left=0.17, right=0.95)
    else:
        bbox = 1.3
        ncol = 4
        if standalone:
            fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
            plt.subplots_adjust(top=0.85, bottom=0.29, left=0.16, right=0.97)
    cfg = {"model": "llama2-13", "op_type": "decode", "bs": 32}
    params = default_params.copy() if "ipu" in title else default_params.copy()
    for key, value in params.items():
        cfg[key] = value
    if is_prefill:
        cfg["op_type"] = "prefill"
        cfg["bs"] = 1   # prefill = real single-request batch 1
    exe_times = {model: {} for model in models}
    breakdown_times = {model: {} for model in models}
    for m, model in enumerate(models):
        cfg["model"] = model
        for best_str in impl.keys():
            if best_str == "spmd_compiler":
                cfg['sram_kb'] = default_params['sram_kb']//2
            else:
                cfg['sram_kb'] = default_params['sram_kb']
            if best_str == "dataflow":
                if not is_prefill and model in ["gemma2", "llama3-70"]:
                    DF_FACTOR = 2
                else:
                    DF_FACTOR = 4
                cfg['num_cores'] = default_params['num_cores']//DF_FACTOR
                cfg['dram_bw'] = default_params['dram_bw']//DF_DRAM_DIV*DF_DRAM_MUL
                bs = cfg['bs']//DF_FACTOR
                data_str = "dataflow"
            else:
                cfg['num_cores'] = default_params['num_cores']
                cfg['dram_bw'] = default_params['dram_bw']
                bs = cfg['bs']
                data_str = best_str
            prefix = os.path.join(TRACE_OUT, f"{cfg['model']}", f"bs_{bs}", f"core_{cfg['num_cores']}",
                                f"{cfg['op_type']}", f"sa_{cfg['sa']}-vu_{cfg['sa']}",
                                f"sram_{cfg['sram_kb']}-drambw_{cfg['dram_bw']}_PLACEHOLDER",
                                f"topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}", data_str)
            os.makedirs("figures", exist_ok=True)
            logfile = os.path.join(prefix, f"output_cg_{cfg['core_group']}_row_{cfg['row']}.log")
            assert os.path.exists(logfile), f"Log file {logfile} does not exist. Check the path and parameters."
            best_util = 0
            with open(logfile, "r") as log:
                full_content = log.read()
                overview = full_content.split('\n')[0]
                exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])
                oall_util = float(full_content.split("Overall Util: ")[1].split('\n')[0])
                if oall_util > best_util:
                    best_util = oall_util
                exe_times[model][best_str] = exe_time
            if breakdown is not None:
                with open(logfile, "r") as log:
                    log_content = log.read()
                    op_logs = parse_log_file(log_content, dram_bw_GBps=cfg['dram_bw'], npu_freq_MHz=NPU_FREQ_MHZ)
                    breakdown_time = get_hw_units_exclusive_time(op_logs,
                                        [breakdown_type.lower() for breakdown_type in breakdown])
                    layer_total_time = get_total_time(op_logs)
                    breakdown_time *= (exe_time/layer_total_time)
                    remaining_time = exe_time - breakdown_time
                    breakdown_times[model][best_str] = remaining_time


        exe_times_copy = {model: {impl[label]: exe_times[model][label] for label in impl}}
        exe_times[model] = {impl[label]+f" - {breakdown[0]} overhead": exe_times[model][label] * CYCLE_TO_MS for label in impl}
        breakdown_times[model] = {impl[label]: breakdown_times[model][label] * CYCLE_TO_MS for label in impl}
        print(f"---------- {model} {is_prefill} ----------")
        labels_copy = list(impl.values())
        for label in impl.values():
            print(f"{label}: {exe_times_copy[model][label]} cycles")
        print(f"diff: {1-exe_times_copy[model][labels_copy[1]] / exe_times_copy[model][labels_copy[0]]} ")

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        offset = group_width*m
        local_dark_colors = dark_colors[:len(impl)]
        local_colors = colors[:len(impl)]
        if len(impl) > 2:
            local_dark_colors[-1], local_dark_colors[1] = local_dark_colors[1], local_dark_colors[-1]
            local_colors[-1], local_colors[1] = local_colors[1], local_colors[-1]
        rects = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(exe_times[model]))],
                height=[exe_times[model][label] for label in exe_times[model]], color=local_dark_colors,
                hatch=bar_hatches[1], width=BAR_WIDTH, edgecolor='black')
        for bc in rects:
            bc._hatch_color = mpl.colors.to_rgba('white')
        if breakdown is not None:
            rects_breakdown = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(breakdown_times[model]))],
                    height=[breakdown_times[model][label] for label in breakdown_times[model]], color=local_colors,
                    hatch=bar_hatches[0], label=list(breakdown_times[model].keys()), width=BAR_WIDTH, edgecolor='black')
            for bc in rects_breakdown:
                bc._hatch_color = mpl.colors.to_rgba('white')

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 -1/2)+1.05 for m in range(len(models))]
    ax.set_xticks(xticks)
    ax.set_xticklabels([modelnames[model] for model in models], rotation=15, fontsize=22, ha='right', rotation_mode='anchor')

    handles, labels = ax.get_legend_handles_labels()
    split_patch = SplitPatch(colors=dark_colors[:len(impl)], hatches=bar_hatches[:len(impl)])
    handles.append(split_patch)
    labels.append(f"{breakdown[0]} overhead")
    by_label = dict(zip(labels, handles))
    by_label_values = list(by_label.values())
    by_label_keys = list(by_label.keys())
    if standalone:
        ax.legend(by_label_values, by_label_keys, fontsize=22, ncol=ncol, frameon=False, handlelength=1,
                handletextpad=0.25, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1.08, bbox),
                loc="upper right", columnspacing=0.55,
                handler_map={SplitPatch:SplitLegendBox()})

    if is_prefill:
        ax.set_ylabel("Prefill\nTTFT (ms)", fontsize=23)
    else:
        ax.set_ylabel("Decode\nTBT (ms)", fontsize=23)

    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    if standalone:
        if is_prefill:
            plt.savefig(f"figures/eval_sw_{title}_prefill.pdf")
        else:
            plt.savefig(f"figures/eval_sw_{title}_decode.pdf")

    return rects

def draw_figures(impl:Dict[str, str], title:str, breakdown:List[str]=None):
    total_label_length = sum([len(impl[key]) for key in impl])
    bbox = 1.49
    ncol = 4
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 2.9))
    plt.subplots_adjust(top=0.86, bottom=0.32, left=0.105, right=0.99)
    plt.subplots_adjust(wspace=0.3)

    decode = draw_one_figure(is_prefill=False, impl=impl, title=title, breakdown=breakdown, ax=ax[0], draw_legend=False)
    prefill = draw_one_figure(is_prefill=True, impl=impl, title=title, breakdown=breakdown, ax=ax[1], draw_legend=False)

    # Collect unique handles and labels
    handle_dict = {}
    for i, ax in enumerate(fig.axes):
        h, l = ax.get_legend_handles_labels()
        handle_dict.update(dict(zip(l, h)))
        if i == 0:
            ax.annotate('DRAM access\noverhead',
                        xy=(6.5, 2.5*10000000*CYCLE_TO_MS),
                        xytext=(2.6, 2*10000000*CYCLE_TO_MS),
                        fontsize=19, ha='center',
                        arrowprops=dict(arrowstyle='->', lw=2, color='black', shrinkA=0, shrinkB=0))

    local_dark_colors = dark_colors[:len(impl)]
    local_colors = colors[:len(impl)]
    if len(impl) > 2:
        local_dark_colors[-1], local_dark_colors[1] = local_dark_colors[1], local_dark_colors[-1]
        local_colors[-1], local_colors[1] = local_colors[1], local_colors[-1]
    split_patch = SplitPatch(colors=local_dark_colors, hatches=bar_hatches[:len(impl)])
    fig.legend(handle_dict.values(), handle_dict.keys(),
                loc='upper right',
                ncol=ncol, fontsize=22, frameon=False,
                bbox_to_anchor=(1., 1.09),
                columnspacing=1.3, handlelength=1.5, handletextpad=0.5)

    plt.savefig(f"figures/eval_sw_{title}.pdf", bbox_inches='tight', pad_inches=0.04)

if __name__ == "__main__":
    # Draw DRAM
    impl = {
        "uniform_dram": "Uniform placement",
        "best": "Software-aware placement",
    }
    title = "dram_map"
    breakdown = ["DRAM"]
    draw_figures(impl=impl, title=title, breakdown=breakdown)

    # Draw compiler
    # impl = {
    #     "spmd_compiler": "SPMD",
    #     "dataflow": "Dataflow",
    #     "best": "Compute-shift",
    # }
    # title = "compiler"
    # breakdown = ["NoC"]
    # draw_figures(impl=impl, title=title, breakdown=breakdown)


