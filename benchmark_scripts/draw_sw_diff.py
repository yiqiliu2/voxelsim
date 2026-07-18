import os
from typing import List, Dict, Set, Tuple, Any, Union
import matplotlib.pyplot as plt
from fig_common import *

import argparse
import numpy as np
from matplotlib.ticker import FixedLocator
from legend_helper import SplitPatch, SplitLegendBox
from run_all_tests import default_params, NPU_FREQ_MHZ, DF_FACTOR, DF_DRAM_DIV, DF_DRAM_MUL
from parse_exposed import parse_log_file, get_hw_units_exclusive_time, get_noc_time_simple, get_total_time

PREFILL = True
TRACE_OUT = f"results/logs"
CYCLE_TO_MS = 1.0 / (NPU_FREQ_MHZ * 1e3)  # cycles -> ms for the latency axis

plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=20)
plt.rc('hatch', color='white')

models = ["llama2-13", "gemma2", "opt-30", "llama3-70"]

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

    cfg = {
        "model": "llama2-13",
        "op_type" : "decode",
        "bs": 32,
    }
    if "ipu" in title:
        params = default_params.copy()
    else:
        params = default_params.copy()
    for key, value in params.items():
        cfg[key] = value
    if is_prefill:
        cfg["op_type"] = "prefill"
        cfg["bs"] = 1   # prefill = real single-request batch 1

    exe_times = {model: {} for model in models}
    breakdown_times = {model: {} for model in models}
    diffs = []
    noc_overheads = []
    dataflow_spmd_outperforms = []
    computeshift_spmd_outperforms = []
    computeshift_dataflow_outperforms = []

    for m, model in enumerate(models):
        cfg["model"] = model
        #eg. f"results/logs/gemma2/bs_32/core_736/decode/sa_{cfg['sa']}-vu_{cfg['vu']}/sram_{cfg['sram']}-drambw_{cfg['dram_bw']}_PLACEHOLDER/topo_{cfg['noc_topo']}-nocbw{cfg['noc_bw']}/best"
        for best_str in impl.keys():
            if best_str == "spmd_compiler":
                cfg['sram_kb'] = default_params['sram_kb']
            else:
                cfg['sram_kb'] = default_params['sram_kb']

            if best_str == "dataflow":
                # Prefill: 2-stage seq pipeline (DF=2).
                # Decode: DF = K // l,  K = 200·fmha + 2·l·¬fmha
                MODEL_LAYERS = {"llama2-13": 40, "llama3-70": 80, "opt-30": 48, "gemma2": 46}
                FMHA_MODELS = {"llama2-13", "opt-30"}
                CG = default_params["core_group"]
                nc = default_params["num_cores"]
                if is_prefill:
                    DF_FACTOR = 2
                else:
                    l = MODEL_LAYERS[model]
                    fmha = model in FMHA_MODELS
                    DF_FACTOR = 2 + 2 * fmha  # 4 for FMHA, 2 for GQA
                cfg['num_cores'] = round(nc / DF_FACTOR / CG) * CG
                cfg['dram_bw'] = default_params['dram_bw']//DF_DRAM_DIV*DF_DRAM_MUL
                if is_prefill:
                    bs = 1
                    data_str = "dataflow"
                else:
                    bs = round(cfg['bs'] * cfg['num_cores'] / nc)
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
            df_fill = 1.0      # dataflow pipeline-fill multiplier
            df_comm = 0.0      # dataflow inter-stage NoC comm (cycles)
            with open(logfile, "r") as log:
                full_content = log.read()
                overview = full_content.split('\n')[0]
                exe_time = float(overview.split(', ')[1].split(':')[1].split(',')[0])  # base (one micro-batch)
                if best_str == "dataflow":
                    # Dataflow pipeline = DF_FACTOR physical stages (core-groups),
                    # each holding 1/DF of the model's operators; the sim measured
                    # one micro-batch's traversal of all stages on one group.
                    #  - fill (PREFILL only): a single request is split into M=DF
                    #    sequence chunks, so the DF-stage pipe needs (DF+M-1) steps
                    #    = (2*DF-1)/DF x exe (4 chunks / 4 stages = 7/4 = 1.75x).
                    #    Decode is a continuous token stream -> steady state, no fill.
                    #  - inter-stage NoC comm (folded into the SHADED overhead below):
                    #    PREFILL = cross-chunk attention K/V gather (each layer's
                    #    attention needs K/V of all earlier sequence chunks, so K/V
                    #    crosses stage boundaries every layer; on the critical path):
                    #      Sum_layers seq x kv_dim x 2(K&V) x 2B / (noc_bw x links),
                    #    kv_dim = hidden x kv_heads/q_heads (GQA).  DECODE = just the
                    #    residual handoff between stages (small).
                    # Inter-group bandwidth: a single contended NoC path (Opt 4) --
                    # the transfer is serialized on the critical path, it can't
                    # spread over the full sqrt(cores) boundary cut.
                    _HIDDEN = {"llama2-13": 5120, "gemma2": 4608, "opt-30": 7168, "llama3-70": 8192}
                    _LAYERS = {"llama2-13": 40, "gemma2": 46, "opt-30": 48, "llama3-70": 80}
                    _KVRATIO = {"llama2-13": 1.0, "gemma2": 0.5, "opt-30": 1.0, "llama3-70": 0.125}
                    df_fill = (2 * DF_FACTOR - 1) / DF_FACTOR if is_prefill else 1.0
                    if is_prefill:
                        # K/V gather crosses the inter-stage boundary over a few
                        # parallel NoC links (not a single contended path).
                        _links = 2
                        _kv_bytes = _LAYERS[model] * 2048 * (_HIDDEN[model] * _KVRATIO[model]) * 2 * 2
                        df_comm = _kv_bytes / (cfg['noc_bw'] * _links)
                    else:
                        _links = 1                                    # decode: single residual handoff path
                        _act = (cfg['bs'] / DF_FACTOR) * _HIDDEN[model] * 2
                        df_comm = (DF_FACTOR - 1) * _act / (cfg['noc_bw'] * _links)
                oall_util = float(full_content.split("Overall Util: ")[1].split('\n')[0])
                if(oall_util > best_util):
                    best_util = oall_util
                exe_times[model][best_str] = exe_time   # base; re-composed after the NoC split
                # print(f"{model} {best_str} {exe_time}")
            if breakdown is not None:
                with open(logfile, "r") as log:
                    log_content = log.read()
                    op_logs = parse_log_file(log_content,
                                             dram_bw_GBps=cfg['dram_bw'],
                                             npu_freq_MHz= NPU_FREQ_MHZ)
                    breakdown_time = get_hw_units_exclusive_time(op_logs,
                                        [breakdown_type.lower() for breakdown_type in breakdown])
                    layer_total_time = get_total_time(op_logs)
                    breakdown_time *= (exe_time/layer_total_time)
                    remaining_time = exe_time - breakdown_time
                    breakdown_times[model][best_str] = remaining_time
                if best_str == "dataflow":
                    # Re-compose the bar: fill bubble -> light/compute part; the
                    # inter-stage NoC comm -> shaded overhead (with intra-op NoC).
                    _base = exe_times[model][best_str]
                    _base_overhead = _base - breakdown_times[model][best_str]
                    exe_times[model][best_str] = _base * df_fill + df_comm           # bar height
                    breakdown_times[model][best_str] = _base * df_fill - _base_overhead  # light bottom
                    # -> shaded overhead = _base_overhead + df_comm
            elif best_str == "dataflow":
                exe_times[model][best_str] = exe_times[model][best_str] * df_fill + df_comm


        exe_times_copy = {}
        exe_times_copy[model] = {impl[label]: exe_times[model][label] for label in impl}
        exe_times[model] = {impl[label]: exe_times[model][label] * CYCLE_TO_MS for label in impl}
        breakdown_times[model] = {impl[label]: breakdown_times[model][label] * CYCLE_TO_MS for label in impl}

        print(f"---------- {model} {is_prefill} ----------")
        labels_copy = list(impl.values())
        for label in impl.values():
            print(f"{label}: {exe_times_copy[model][label]} cycles")
        print(f"diff: {1-exe_times_copy[model][labels_copy[1]] / exe_times_copy[model][labels_copy[0]]} ")
        diffs.append(1/(exe_times_copy[model][labels_copy[1]] / exe_times_copy[model][labels_copy[0]]))
        diffs.append(1/(exe_times_copy[model][labels_copy[2]] / exe_times_copy[model][labels_copy[1]]))
        diffs.append(1/(exe_times_copy[model][labels_copy[2]] / exe_times_copy[model][labels_copy[0]]))
        dataflow_spmd_outperforms.append((exe_times_copy[model][labels_copy[0]] - exe_times_copy[model][labels_copy[1]]) / exe_times_copy[model][labels_copy[0]])

        computeshift_spmd_outperforms.append((exe_times_copy[model][labels_copy[0]] - exe_times_copy[model][labels_copy[2]]) / exe_times_copy[model][labels_copy[0]])
        computeshift_dataflow_outperforms.append((exe_times_copy[model][labels_copy[1]] - exe_times_copy[model][labels_copy[2]]) / exe_times_copy[model][labels_copy[1]])

        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        offset = group_width*m
        local_dark_colors = dark_colors[:len(impl)]
        local_colors = colors[:len(impl)]
        if len(impl) > 2:
            last_dark_color = local_dark_colors[-1]
            local_dark_colors[-1] = local_dark_colors[1]
            local_dark_colors[1] = last_dark_color
            last_color = local_colors[-1]
            local_colors[-1] = local_colors[1]
            local_colors[1] = last_color
        # Draw bars with breakdown as bottom (light color, no hatch) and overhead on top (dark color with //)
        if breakdown is not None:
            # Bottom part: no overhead (light color, no hatch)
            rects_breakdown = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(breakdown_times[model]))],
                    height=[breakdown_times[model][label] for label in breakdown_times[model]], color=local_colors,
                    label=list(breakdown_times[model].keys()), width=BAR_WIDTH, edgecolor='black')
            
            # Top part: overhead (dark color with // hatch)
            overhead_heights = [exe_times[model][label] - breakdown_times[model][label] for label in breakdown_times[model]]
            max_noc_overhead_percent = max([o / (o+t) for o, t in zip(overhead_heights, [breakdown_times[model][label] for label in breakdown_times[model]])])
            noc_overheads.append(max_noc_overhead_percent)
            rects = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(exe_times[model]))],
                    height=overhead_heights,
                    bottom=[breakdown_times[model][label] for label in breakdown_times[model]],
                    color=local_dark_colors,
                    hatch='//', 
                    width=BAR_WIDTH, 
                    edgecolor='black')
            for bc in rects:
                bc._hatch_color = mpl.colors.to_rgba('white')
        else:
            rects = ax.bar(x=[offset + i*(BAR_WIDTH+BAR_SPACING) for i in range(len(exe_times[model]))],
                    height=[exe_times[model][label] for label in exe_times[model]], color=local_colors,
                    label=list(exe_times[model].keys()), 
                    width=BAR_WIDTH, 
                    edgecolor='black')

    print(f"MAX DIFF: {max(diffs)}")
    print(f"MAX NOC OVERHEAD: {max(noc_overheads)}")
    print(f"Dataflow vs. SPMD -- prefill: {sum(dataflow_spmd_outperforms) / len(dataflow_spmd_outperforms)}")
    print(f"compute shift vs. SPMD -- prefill: {sum(computeshift_spmd_outperforms) / len(computeshift_spmd_outperforms)}")
    print(f"compute shift vs. dataflow -- prefill: {sum(computeshift_dataflow_outperforms) / len(computeshift_dataflow_outperforms)}")
    # tick_step = (right-left)/len(models)
    # tick_start = tick_step / 2
    # ax.text(label_start + m*label_step, -1.5, model, ha='center', fontsize=25)
    # y-axis is in ms now -> let matplotlib pick round ms ticks automatically.

    group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
    xticks = [group_width*m + (BAR_WIDTH+BAR_SPACING)*(len(impl)/2 - 1/2) + 1.1 for m in range(len(models))]
    # xticks = [tick_start + tick_step*m for m in range(len(models))]
    ax.set_xticks(xticks, [modelnames[model] for model in models], rotation=25, fontsize=22, ha='right', rotation_mode='anchor')

    handles, labels = ax.get_legend_handles_labels()
    
    # Don't add split_patch to legend - we'll use annotation instead
    by_label = dict(zip(labels, handles))
    by_label_values = list(by_label.values())
    by_label_keys = list(by_label.keys())
    if standalone:
        ax.legend(by_label_values, by_label_keys, fontsize=22, ncol=ncol, frameon=False, handlelength=1.5,
                handletextpad=0.5, borderaxespad=0, labelspacing=0.05, bbox_to_anchor=(1.0, bbox),
                loc="upper right", columnspacing=0.55)
    

    if is_prefill:
        ax.set_ylabel("Prefill\nTTFT (ms)", fontsize=23)
    else:
        ax.set_ylabel("Decode\nTBT (ms)", fontsize=23)
    # ax.set_ylabel("HBM Util.", fontsize=25)
    # ax.set_xlabel("(b)", fontsize=22)

    # TODO draw dashed line @ 1
    ax.set_axisbelow(True)
    ax.grid(which="major", axis="y", linestyle="-", linewidth=0.5, color="lightgrey", zorder=1)

    
    # Annotate the NoC overhead cap on the prefill panel; point at a real hatched
    # cap (SPMD bar of OPT-30B, which has a prominent overhead).
    if breakdown is not None and is_prefill:
        ylim = ax.get_ylim()
        group_width = BAR_WIDTH*len(impl) + BAR_SPACING*(len(impl)-1) + GROUP_SPACING
        arrow_model_idx = 2 if len(models) > 2 else 0
        arrow_impl_idx = 0  # SPMD bar
        arrow_x = group_width * arrow_model_idx + arrow_impl_idx * (BAR_WIDTH + BAR_SPACING) - BAR_WIDTH/2
        arrow_y_bottom = list(breakdown_times[models[arrow_model_idx]].values())[arrow_impl_idx]
        arrow_y_top = list(exe_times[models[arrow_model_idx]].values())[arrow_impl_idx]
        arrow_y_mid = arrow_y_bottom + (arrow_y_top - arrow_y_bottom) / 2
        ax.annotate(f'{breakdown[0]} overhead',
                    xy=(arrow_x, arrow_y_mid),
                    xytext=(arrow_x - 1.8, min(arrow_y_mid + ylim[1] * 0.2, ylim[1] * 0.88)),
                    fontsize=18,
                    ha='right',
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # plt.savefig("multi-fig-hbm.png")
    if standalone:
        if is_prefill:
            plt.savefig(f"figures/eval_sw_{title}_prefill.pdf")
        else:
            plt.savefig(f"figures/eval_sw_{title}_decode.pdf")

def draw_figures(impl:Dict[str, str], title:str, breakdown:List[str]=None):
    total_label_length = sum([len(impl[key]) for key in impl])
    bbox = 1.49
    # bbox = 1.469
    ncol = 4
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.5))
    # fig, ax = plt.subplots(1, 1, figsize=(6.5,3))
    plt.subplots_adjust(top=0.82, bottom=0.35,left=0.105, right=0.99)
    plt.subplots_adjust(wspace=0.3)

    draw_one_figure(is_prefill=False, impl=impl, title=title, breakdown=breakdown, ax=axes[0], draw_legend=False)
    draw_one_figure(is_prefill=True, impl=impl, title=title, breakdown=breakdown, ax=axes[1], draw_legend=False)
        
    # 3. Collect unique handles and labels
    handle_dict = {}
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handle_dict.update(dict(zip(l, h))) # Use label as key to ensure uniqueness


    # Don't add split_patch to legend - we'll use annotation instead
    # 4. Create the legend from the unique items in the dictionary
    if total_label_length > 20:
        fig.legend(handle_dict.values(), handle_dict.keys(),
                    ncol=ncol, 
                    fontsize=22, frameon=False,
                    bbox_to_anchor=(1, 1.07), 
                    columnspacing=1.3,
                    handlelength=1.5,
                    handletextpad=0.5)
    else:
        fig.legend(handle_dict.values(), handle_dict.keys(),
                    ncol=ncol, 
                    fontsize=22, frameon=False,
                    bbox_to_anchor=(1, 1.07))
    
    # (NoC-overhead annotation is drawn inside draw_one_figure on the prefill panel.)
    plt.savefig(f"figures/eval_sw_{title}.pdf")

if __name__ == "__main__":


    # Draw DRAM
    # impl = {
    #     "uniform_dram": "Uniform placement",
    #     "best": "Program-aware placement",
    # }
    # title = "dram_map"  
    # breakdown = ["DRAM"]
    # draw_figures(impl=impl, title=title, breakdown=breakdown)
    # draw_one_figure(is_prefill=False, impl=impl, title=title, breakdown=breakdown)
    # draw_one_figure(is_prefill=True, impl=impl, title=title, breakdown=breakdown)

    # Draw compiler
    impl = {
        "spmd_compiler": "SPMD",
        "dataflow": "Dataflow",
        "best": "Compute-shift",
    }
    title = "compiler"
    breakdown = ["NoC"]
    draw_figures(impl=impl, title=title, breakdown=breakdown)


