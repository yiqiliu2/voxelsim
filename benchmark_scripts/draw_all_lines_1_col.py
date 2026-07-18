import os
import matplotlib.pyplot as plt
from fig_common import *

import argparse
import numpy as np
from matplotlib.ticker import FixedLocator, LogLocator, ScalarFormatter

from run_all_tests import default_params, sweep_lists, NPU_FREQ_MHZ

TRACE_OUT = f"results/logs"
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis
FROM_ZERO = True

fontsize = 17

prefill_models = ["llama2-13", "gemma2", "opt-30", "llama3-70", "dit-xl"]
decode_models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

labels = {
    "sa": "Systolic\nArray Size",
    "vu": "Vector Unit Width",
    "dram_bw": "Total DRAM\nBW (TBps)",
    "sram_kb": "Per-core\nSRAM (MB)",
    "core_group": "Core Group Size",
    "noc_topo": "NoC Topology",
    "noc_bw": "NoC Link BW\n(Byte/cycle)",
    "num_cores": "Number\nof Cores",
}

indices = ["a", "b", "c", "d", "e", "f", "g", "h"]
index = 0

def plot_bw(variable: str,
            use_log_x: bool, use_log_y: bool, 
            models: list[str], ax: plt.Axes, train: bool, idx: int):
    cfg = {
        "model": "llama2-13",
        "op_type" : "decode",
        "bs": 32,
    }
    variable_range = sweep_lists[variable]
    for key, value in default_params.items():
        cfg[key] = value
    # if variable == "sram_kb":
    #     cfg["dram_bw"] = 4096

    for model_idx, model in enumerate(models):
        exe_times = []
        for v in variable_range:
            assert(variable in cfg), f"Variable {variable} not in cfg -- it will not affect the file name for the sweep."
            cfg[variable] = v
            cfg["model"] = model
            if train and model!="dit-xl":
                cfg["op_type"] = "prefill"
                cfg["bs"] = 1     # prefill = real single-request batch 1
            else:
                cfg["op_type"] = "decode"
                cfg["bs"] = 32
            #eg. f"results/logs/gemma2/bs_32/core_736/decode/sa_{cfg['sa']}-vu_{cfg['vu']}/sram_{cfg['sram']}-drambw_{cfg['dram_bw']}_PLACEHOLDER/topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}/best"
            best_str = "best"

            prefix = os.path.join(TRACE_OUT, f"{cfg['model']}", f"bs_{cfg['bs']}", f"core_{cfg['num_cores']}", 
                                f"{cfg['op_type']}", f"sa_{cfg['sa']}-vu_{cfg['sa']}", 
                                f"sram_{cfg['sram_kb']}-drambw_{cfg['dram_bw']}_PLACEHOLDER", 
                                f"topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}", best_str)
            logfile = os.path.join(prefix, f"output_cg_{cfg['core_group']}_row_{cfg['row']}.log")
            best_util = 0
            try:
                with open(logfile, "r") as log:
                    full_content = log.read()
                    overview = full_content.split('\n')[0]
                    exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])
                    oall_util = float(full_content.split("Overall Util: ")[1].split('\n')[0])
                    if(oall_util > best_util):
                        best_util = oall_util
                    exe_times.append(exe_time)
            except FileNotFoundError:
                print(f"Not found: {logfile}")
                exe_times.append(0)

        exe_times = [e * CYCLE_TO_MS for e in exe_times]  # cycles -> ms

        label = modelnames[model]
        if model == "dit-xl":
            label += " (10 iterations)"
            exe_times = [e * 10 for e in exe_times]  # simulator runs 1 iter; normalize to 10 iters


        if variable == "sram_kb" and train:
            print(f"{modelnames[model]} {variable}: {exe_times}")

        line_color = "#d61ad6" if model == "dit-xl" else colors1[model_idx]  # vivid magenta-purple for DiT-XL
        ax.plot(variable_range, exe_times, lines1[model_idx],
                marker=markers1[model_idx], color=line_color,
                label=label, linewidth=2, markersize=8)
        # print("setxticks")
        # ax.set_xticks(variable_range, labels=[str(v) for v in variable_range]) 
    # ax.set_xlabel(f"({indices[idx]}) {labels[variable]}", fontsize=fontsize)
    # ax.set_xlabel(f"({idx+1}) {labels[variable]}", fontsize=22)

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    # x axis is log
    if use_log_x:
        # ax.set_xscale("log")
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(FixedLocator([]))
    if use_log_y:
        ax.set_yscale("log")
    # else:
    #     ax.set_xscale("linear")
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    # ax.set_ylim([10, 220])

    ax.yaxis.set_ticks_position('none') 
    ax.tick_params(axis='y', which='major', pad=-1)
    # ax.set_xticks()
    # if ylabel:
    #     if mesh:
    #         ax.set_ylabel("Mesh", fontsize=25)
    #     else:
    #         ax.set_ylabel("All-to-All", fontsize=25)
        
    # if model in ["llama2-13", "gemma2"]:
    #     ax.set_ylim([6, 27])
    # if FROM_ZERO:
    #     ylim = ax.get_ylim()
    #     ax.set_ylim([0, ylim[1]])
    
    

# if __name__=="__main__":

def draw_combined_figure(variables: list):
    """
    Draw combined decode and prefill figure in 2 rows x N columns layout
    """

    decode_models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]
    prefill_models = ["llama2-13", "gemma2", "opt-30", "llama3-70", "dit-xl"]

    num_vars = len(variables)
    fig, axes = plt.subplots(2, num_vars, figsize=(12, 5.3))
    plt.subplots_adjust(top=0.92, bottom=0.21, left=0.062, right=0.99, wspace=0.2, hspace=0.15)

    # Plot decode (top row)
    for i, (variable, use_log_x, use_log_y) in enumerate(variables):
        ax = axes[0, i]
        plot_bw(variable=variable, 
                use_log_x=use_log_x, use_log_y=use_log_y, 
                models=decode_models, ax=ax, train=False, idx=i)
        
        if use_log_y:
            # Fixed major ticks 5/10/20/40 on the decode row; all major (no
            # minor), each with a gridline. Labels kept only on column 0.
            ax.yaxis.set_major_locator(FixedLocator([5, 10, 20, 40]))
            ax.yaxis.set_minor_locator(FixedLocator([]))
            ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="grey", zorder=1)
            if i == 0:
                ax.yaxis.set_major_formatter(ScalarFormatter())

        ax.tick_params(axis='x', labelsize=fontsize-1)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='y', which='minor', labelsize=fontsize-2)

        # Set x-ticks but remove labels for top row
        variable_range = sweep_lists[variable]
        ax.set_xticks(variable_range)
        ax.set_xticklabels([])

        # Remove y-ticks for all but first column
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        
        # Remove x-label for top row, will add for bottom row only
        ax.set_xlabel("")

    # Plot prefill (bottom row)
    for i, (variable, use_log_x, use_log_y) in enumerate(variables):
        ax = axes[1, i]
        plot_bw(variable=variable, 
                use_log_x=use_log_x, use_log_y=use_log_y, 
                models=prefill_models, ax=ax, train=True, idx=i)
        
        if use_log_y:
            ylim = ax.get_ylim()
            ax.set_ylim(ylim)
        
        ax.tick_params(axis='x', labelsize=fontsize-1)
        ax.tick_params(axis='y', labelsize=fontsize)

        # Set x-tick labels
        variable_range = sweep_lists[variable]
        if variable == "dram_bw":
            ax.set_xticks(variable_range, labels=[str(v//1000) for v in variable_range], rotation=30) 
        elif variable == "sram_kb":
            ax.set_xticks(variable_range, labels=['0.5']+[str(v//1024) for v in variable_range[1:]], rotation=30)
        elif variable == "noc_bw":
            ax.set_xticks(variable_range, labels=[str(v) for v in variable_range], rotation=30)
        else:
            ax.set_xticks(variable_range, labels=[str(v) for v in variable_range], rotation=30)
        
        # Remove y-ticks for all but first column
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel("")
        
        # Don't set xlabel here, will use fig.text for consistent altitude

    # Sync y-limits across each row, data-driven (ms axis) with a little headroom.
    ylim_low_decode = min([axes[0, i].get_ylim()[0] for i in range(num_vars)])
    ylim_high_decode = max([axes[0, i].get_ylim()[1] for i in range(num_vars)])
    for i in range(num_vars):
        # Keep the fixed 5/10/20/40 major ticks all in view (5 near bottom,
        # 40 with headroom on top).
        axes[0, i].set_ylim([min(ylim_low_decode * 0.8, 4.4), max(ylim_high_decode * 1.25, 46)])

    # Sync y-limits for prefill row
    ylim_low_prefill = min([axes[1, i].get_ylim()[0] for i in range(num_vars)])
    ylim_high_prefill = max([axes[1, i].get_ylim()[1] for i in range(num_vars)])
    for i in range(num_vars):
        axes[1, i].set_ylim([ylim_low_prefill * 0.8, ylim_high_prefill * 1.25])

    # Add x-axis labels at consistent altitude using fig.text
    xlabel_y = 0.11  # Fixed y-position for all labels
    for i, (variable, _, _) in enumerate(variables):
        # Calculate x position for each subplot
        ax_pos = axes[1, i].get_position()
        xlabel_x = (ax_pos.x0 + ax_pos.x1) / 2
        fig.text(xlabel_x, xlabel_y, f"({indices[i]}) {labels[variable]}", 
                ha='center', va='top', fontsize=fontsize+1)

    # Add legend at the top center - get from prefill row to include dit-xl
    handles, labels_legend = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='upper center', fontsize=fontsize+1, ncol=5, frameon=False,
               handlelength=1.2, handletextpad=0.4, borderaxespad=0, labelspacing=0.3,
               bbox_to_anchor=(0.525, 1.01), columnspacing=0.5)

    # Add y-labels on the left
    fig.text(0.013, 0.78, 'Decode TBT (ms)', ha='center', va='center', fontsize=fontsize, rotation=90)
    fig.text(0.013, 0.33, 'Prefill TTFT (ms)', ha='center', va='center', fontsize=fontsize, rotation=90)

    # Save figure
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fname = "figures/eval_lines_all_combined.pdf"
    plt.savefig(fname)
    print(f"Saved {fname}")

if __name__ == "__main__":
    # var_name, use_log_x, use_log_y
    variables = [
        ("noc_bw", True, True),
        ("dram_bw", False, True),
        ("sa", True, True),
        ("num_cores", True, True),
        ("sram_kb", True, True),
    ]
    draw_combined_figure(variables)
