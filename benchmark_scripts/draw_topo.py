import os
import argparse
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator

from fig_common import *
from run_all_tests import default_params, sweep_lists, NPU_FREQ_MHZ

CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis

TRACE_OUT = f"results/logs"
WIDTH = 0.35

prefill_models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]
decode_models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

variable = ""
use_log = False
bar = False
from_zero = False

labels = {
    "sa": "Systolic Array Size",
    "vu": "Vector Unit Width",
    "dram_bw": "DRAM BW (TBps)",
    "sram_kb": "Per-core SRAM Size (KB)",
    "core_group": "Core Groups Size",
    "noc_topo": "NoC Topology",
    "noc_bw": "NoC Link Bandwidth (Byte/cycle)",
    "num_cores": "Number of Cores",
}

best_noc_reductions = []
noc_performance_slowdowns = []

def plot_bw(
    model: str,
    ax: plt.Axes,
    train: bool,
    ylabel: bool = False,
    mesh: bool = False,
    annotate_x: bool = True,
):
    cfg = {
        "model": "llama2-13",
        "op_type" : "decode",
        "bs": 32,
    }
    variable_range = np.array(sweep_lists[variable])
    for key, value in default_params.items():
        cfg[key] = value
    if train and model!="dit-xl":
        cfg["op_type"] = "prefill"
        cfg["bs"] = 1   # prefill = real single-request batch 1

    best_exe_times = []
    bad_exe_times = []
    for v in variable_range:
        assert(variable in cfg), f"Variable {variable} not in cfg -- it will not affect the file name for the sweep."
        cfg[variable] = v
        cfg["model"] = model

        best_str = "best"
        bad_str = "seq_noc"
        best_prefix = os.path.join(TRACE_OUT, f"{cfg['model']}", f"bs_{cfg['bs']}", f"core_{cfg['num_cores']}",
                              f"{cfg['op_type']}", f"sa_{cfg['sa']}-vu_{cfg['sa']}",
                              f"sram_{cfg['sram_kb']}-drambw_{cfg['dram_bw']}_PLACEHOLDER",
                              f"topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}", best_str)
        bad_prefix = os.path.join(TRACE_OUT, f"{cfg['model']}", f"bs_{cfg['bs']}", f"core_{cfg['num_cores']}",
                              f"{cfg['op_type']}", f"sa_{cfg['sa']}-vu_{cfg['sa']}",
                              f"sram_{cfg['sram_kb']}-drambw_{cfg['dram_bw']}_PLACEHOLDER",
                              f"topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}", bad_str)
        best_logfile = os.path.join(best_prefix, f"output_cg_{cfg['core_group']}_row_{cfg['row']}.log")
        bad_logfile = os.path.join(bad_prefix, f"output_cg_{cfg['core_group']}_row_{cfg['row']}.log")
        try:
            with open(best_logfile, "r") as log:
                full_content = log.read()
                overview = full_content.split('\n')[0]
                exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])
                best_exe_times.append(exe_time)
        except Exception as e:
            print(f"Failed to read {best_logfile}")
            print(f"Error: {e}")
            best_exe_times.append(0)
        try:
            with open(bad_logfile, "r") as log:
                full_content = log.read()
                overview = full_content.split('\n')[0]
                exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])
                bad_exe_times.append(exe_time)
        except Exception as e:
            print(f"Failed to read {bad_logfile}")
            print(f"Error: {e}")
            bad_exe_times.append(0)

    for i in range(len(variable_range)):
        print(f"{variable_range[i]}: best: {best_exe_times[i]}, bad: {bad_exe_times[i]}, reduction: {1-best_exe_times[i]/bad_exe_times[i] if bad_exe_times[i] else 0}")
    print(" ")

    bad_exe_arr = np.array(bad_exe_times, dtype=float) * CYCLE_TO_MS
    best_exe_arr = np.array(best_exe_times, dtype=float) * CYCLE_TO_MS

    left_positions = variable_range - WIDTH / 2
    right_positions = variable_range + WIDTH / 2
    best_noc_reductions.append(max([(bad - best) / bad for bad, best in zip(bad_exe_arr, best_exe_arr)]))
    noc_performance_slowdowns.append([(bad - best) / best for bad, best in zip(bad_exe_arr, best_exe_arr)])

    ax.bar(
        left_positions,
        bad_exe_arr,
        width=WIDTH,
        color=colors1[1],
        edgecolor='black',
        label="Sequential mapping",
    )
    ax.bar(
        right_positions,
        best_exe_arr,
        width=WIDTH,
        color=colors1[0],
        edgecolor='black',
        label="Dimension-ordered mapping",
    )

    # print("setxticks")
    # ax.set_xticks(variable_range, labels=[str(v) for v in variable_range])
    if annotate_x:
        ax.set_xlabel(f"{modelnames[model]}", fontsize=25)
    else:
        # omit any kind of annotations on x axis entirely
        # ax.set_xlabel("")
        ax.tick_params(axis='x', labelbottom=False)

    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5, color="grey", zorder=1)
    ax.set_axisbelow(True)

    # x axis is log
    if use_log:
        # ax.set_xscale("log")
        ax.set_xscale("log")
    # else:
    #     ax.set_xscale("linear")
    # ax.set_yticklabels(np.array(ax.get_yticks()).astype(int), position=(0.03, 0))
    # ax.set_xticklabels(np.array(ax.get_xticks()).astype(int), position=(0, 0.03))
    # ax.set_ylim([10, 220])

    ax.yaxis.set_ticks_position('none')
    ax.tick_params(axis='y', which='major', pad=-1)
    ax.yaxis.get_offset_text().set_fontsize(23)
    # ax.set_xticks()
    # if ylabel:
    #     if mesh:
    #         ax.set_ylabel("Mesh", fontsize=25)
    #     else:
    #         ax.set_ylabel("All-to-All", fontsize=25)

    # if model in ["llama2-13", "gemma2"]:
    #     ax.set_ylim([6, 27])
    if from_zero:
        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1] * 1.15])





# if __name__=="__main__":

def draw_one_pass(
    train: bool,
    legend: bool = True,
    fig: Optional[plt.Figure] = None,
    axes: Optional[Sequence[plt.Axes]] = None,
    row_label: Optional[str] = None,
    row_label_pos: Optional[Tuple[float, float]] = None,
    row_label_rotation: float = 90.0,
    save: bool = True,
):
    variable_range = sweep_lists[variable]

    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=25)

    models = prefill_models if train else decode_models

    created_fig = False
    if axes is None:
        fig, axes = plt.subplots(1, len(models), figsize=(12, 6))
        plt.subplots_adjust(top=0.83, bottom=0.28, left=0.12, right=0.98, wspace=0.52, hspace=0.32)
        created_fig = True

    if fig is None:
        raise ValueError("Figure reference is required when providing axes")

    axes_array = np.asarray(axes)
    if axes_array.ndim == 0:
        axes_array = axes_array[np.newaxis]

    for i, (ax_obj, model) in enumerate(zip(axes_array, models)):
        ylabel_flag = (i == 0)
        plot_bw(
            model=model,
            ax=ax_obj,
            ylabel=ylabel_flag,
            mesh=False,
            train=train,
            annotate_x=train
        )
        ylim = ax_obj.get_ylim()
        ax_obj.set_ylim([0, ylim[1] * 1.15])
        if train:
            ax_obj.set_xticks(np.array(variable_range) + 0.5*WIDTH, labels=["Mesh", "Torus", "All-to-all"], rotation=25, fontsize=23, ha='right', rotation_mode='anchor')
        ax_obj.tick_params(axis='y', labelsize=25)

    handles, label_names = axes_array[-1].get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []
    for handle, label_name in zip(handles, label_names):
        if label_name and label_name not in seen:
            seen.add(label_name)
            unique_handles.append(handle)
            unique_labels.append(label_name)

    if legend and fig is not None and unique_handles:
        fig.legend(
            unique_handles,
            unique_labels,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            fontsize=25,
            ncol=len(unique_handles),
            frameon=False,
            handlelength=1.3,
            handletextpad=0.4,
            borderaxespad=0.2,
            columnspacing=1.2,
        )

    if row_label is not None and row_label_pos is not None:
        fig.text(*row_label_pos, row_label, ha='center', fontsize=24, rotation=row_label_rotation)

    # if row_label is not None and axes_array.size > 0:
    #     first_ax = axes_array.flat[0]
    #     first_ax.set_ylabel(
    #         row_label,
    #         fontsize=24,  # Use the same fontsize you had
    #         rotation=row_label_rotation
    #     )

    if created_fig and save:
        if not os.path.exists("figures"):
            os.makedirs("figures")
        fname = f"figures/eval_{variable}"
        fname += "_prefill" if train else "_decode"
        fname += ".pdf"
        fig.savefig(fname)

def draw_two_passes():
    fig, axes = plt.subplots(2, len(prefill_models), figsize=(16, 6.8))
    fig.subplots_adjust(top=0.88, bottom=0.27, left=0.11, right=0.98, wspace=0.25, hspace=0.2)

    try:
        draw_one_pass(
            train=False,
            legend=True,
            fig=fig,
            axes=axes[0],
            row_label="Decode\nTBT (ms)",
            row_label_pos=(0.045, 0.6),
            save=False,
        )
    except Exception as e:
        print(f"Error in decode row of {variable}: {e}")

    try:
        draw_one_pass(
            train=True,
            legend=False,
            fig=fig,
            axes=axes[1],
            row_label="Prefill\nTTFT (ms)",
            row_label_pos=(0.045, 0.25),
            save=False,
        )
    except Exception as e:
        print(f"Error in prefill row of {variable}: {e}")
    # fig.text(0.54, 0.56, "(a) LLM Per-token Decode Latency", ha='center', fontsize=22)
    # fig.text(0.54, 0.02, "(b) LLM Prefill Latency", ha='center', fontsize=22)
    print("Best runtime reduction percent: ", max(best_noc_reductions))
    print(max([max(sample) for sample in noc_performance_slowdowns]))

    if not os.path.exists("figures"):
        os.makedirs("figures")
    fname = f"figures/eval_{variable}_combined.pdf"
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.04)

if __name__ == "__main__":

    variable = "noc_topo"
    use_log = False
    bar = True
    from_zero = False
    draw_two_passes()

