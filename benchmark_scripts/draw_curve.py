import os
import matplotlib.pyplot as plt
from fig_common import *

import argparse
import numpy as np
import math
from matplotlib.ticker import FixedLocator
from run_all_tests import default_params, NPU_FREQ_MHZ

PREFILL = False
TRACE_OUT = f"results/logs"
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis

cg_list = [1,2,4,8]
# cg_list = [1,2,4]

models = ["llama2-13", "gemma2", "opt-30", "llama3-70", "dit-xl"]
if PREFILL:
    models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]
# models = ["llama2-13", "llama3-70", "opt-30", "gemma2"]

variable = {}
use_log = False
bar = False

labels = {
    "sa": "Systolic Array Size",
    "vu": "Vector Unit Width",
    "dram_bw": "DRAM BW (GBps)",
    "sram_kb": "Per-core SRAM Size (KB)",
    "core_group": "Core Groups Size",
    "noc_topo": "NoC Topology",
    "noc_bw": "NoC Link Bandwidth (Byte/cycle)",
    "num_cores": "Number of Cores",
}

benefits = []

def plot_bw(model: str, ax: plt.Axes, ylabel: bool = False, mesh: bool = False):

    cfg = {
        "model": "llama2-13",
        "op_type" : "decode",
        "bs": 32,
    }
    for key, value in default_params.items():
        cfg[key] = value

    if PREFILL:
        cfg["op_type"] = "prefill"

    cg_data = {}
    for cg in cg_list:
        exe_times = []
        for i in range(len(next(iter(variable.values())))):
            for key in variable.keys():
                cfg[key] = variable[key][i]
            cfg["model"] = model
            cfg["core_group"] = cg
            #eg. f"results/logs/gemma2/bs_32/core_736/decode/sa_{cfg['sa']}-vu_{cfg['vu']}/sram_{cfg['sram']}-drambw_{cfg['dram_bw']}_PLACEHOLDER/topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}/best"
            prefix = os.path.join(TRACE_OUT, f"{cfg['model']}", f"bs_{cfg['bs']}", f"core_{cfg['num_cores']}",
                                f"{cfg['op_type']}", f"sa_{cfg['sa']}-vu_{cfg['sa']}", 
                                f"sram_{cfg['sram_kb']}-drambw_{cfg['dram_bw']}_PLACEHOLDER", 
                                f"topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}", "best")
            logfile = os.path.join(prefix, f"output_cg_{cfg['core_group']}_row_{cfg['row']}.log")
            best_util = 0
            with open(logfile, "r") as log:
                full_content = log.read()
                overview = full_content.split('\n')[0]
                exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])
                oall_util = float(full_content.split("Overall Util: ")[1].split('\n')[0])
                if(oall_util > best_util):
                    best_util = oall_util
                exe_times.append(exe_time)

        exe_times = [e * CYCLE_TO_MS for e in exe_times]  # cycles -> ms

        cg_idx = int(math.log2(cg))
        if cg == 1:
            label_name = "Group Size"
        else:
            label_name = "Group Size"
        if bar:
            ax.bar(variable['num_cores'], exe_times, width=0.5, color=colors1[cg_idx], marker=markers[cg_idx], label=f"{label_name} {cg}")
        else:
            ax.plot(variable['num_cores'], exe_times, lines1[cg_idx], marker=markers1[cg_idx], color=colors1[cg_idx], label=f"{label_name} {cg}")

        print(f"Core Group: {cg}, Execution Times: {exe_times[0]}")
        cg_data[cg] = exe_times[0]
    print("Benifit of cg:", (cg_data[1] - cg_data[8]) / cg_data[1])
    benefits.append((cg_data[1] - cg_data[8]) / cg_data[1])
    print(" ")



    ax.set_xlabel(f"{modelnames[model]}", fontsize=23)

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)

    # x axis is log
    if use_log:
        ax.set_xscale("log")
    # else:
    #     ax.set_xscale("linear")
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    # ax.set_ylim([10, 220])

    ax.yaxis.set_ticks_position('none') 
    ax.tick_params(axis='y', which='major', pad=-1)

    # if ylabel:
    #     if mesh:
    #         ax.set_ylabel("Mesh", fontsize=25)
    #     else:
    #         ax.set_ylabel("All-to-All", fontsize=25)
        
    # if model in ["llama2-13", "gemma2"]:
    #     ax.set_ylim([6, 27])
    ylim = ax.get_ylim()
    # y-axis is in ms now -> auto ticks, keep from-zero.
    ax.set_ylim([0, ylim[1]*1.08])

    ticks = reversed(variable['num_cores'])
    ticks = [int(tick) for tick in ticks]
    sa_ticks = reversed([16, 23, 32, 45, 64, 90])
    tick_names = [f"{tick} / {sa_tick}" for tick, sa_tick in zip(ticks, sa_ticks)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_names, fontsize=18, rotation=45, ha='right', rotation_mode='anchor')
    
    
    

# if __name__=="__main__":

def draw_one_pass():

    # hbm_bw = args.hbm_bw # this is PER CORE -> probably need to rerun expierements for this

    # hbm_bws = [5000, 7000, 10000, 15000]
    # hbm_bws = [3000, 5000, 11000, 13000]

    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(1, len(models), figsize=(13, 4.3))
    plt.subplots_adjust(top=0.88, bottom=0.45, left=0.09, right=0.99)
    # plt.subplots_adjust(wspace=0.05)
    plt.subplots_adjust(wspace=0.23, hspace=0.15)

    for i, model in enumerate(models):
        ylabel = (i == 0)
        plot_bw(model=model, ax=ax[i], ylabel=ylabel, mesh=False)
        ylim0 = ax[i].get_ylim()
        ylim1 = ax[i].get_ylim()
        ylim = (min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1]))
        ax[i].set_ylim(ylim)
        ax[i].set_ylim(ylim)

    print("ALL benefits:", benefits)
    print("Max benefits:", max(benefits))
    print("Avg benefits:", sum(benefits) / len(benefits))

    ax[-1].legend(loc="upper right", fontsize=20, ncol=6, frameon=False, columnspacing=1,
                  handlelength=1.3, handletextpad=0.35, borderaxespad=0, labelspacing=0.3, 
                  bbox_to_anchor=(1.05,1.25))
    fig.text(0.5375, 0.025, f"{labels['num_cores']} / {labels['sa']}", ha='center', fontsize=25)
    if PREFILL:
        fig.text(0.012, 0.07, 'Prefill TTFT (ms)', ha='center', fontsize=25, rotation=90)
    else:
        fig.text(0.03, 0.2, 'Decode TBT (ms)', ha='center', fontsize=24, rotation=90)

    # plt.legend(loc="upper right")
    # plt.savefig(f"corelines.png")
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fname = f"figures/eval_core_group"
    if PREFILL:
        fname += "_prefill_smile_curve"
    else:
        fname += "_decode_smile_curve"
    fname += ".pdf"
    plt.savefig(fname)

if __name__ == "__main__":
    variable = {"sa":[16, 23, 32, 45, 64, 90],
                "sram_kb":[512, 1024, 2048, 4096, 8192, 16384],
                "num_cores":[1024, 512, 256, 128, 64, 32],
                }
    # variable = {"sa":[11, 16, 23, 32, 45, 64],
    #             "sram_kb":[256, 512, 1024, 2048, 4096, 8192],
    #             "num_cores":[2048, 1024, 512, 256, 128, 64],
    #             }
    use_log = True
    bar = False
    draw_one_pass()

    
