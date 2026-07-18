#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
out_location = "results/logs/"

import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif'})
mpl.rcParams['pdf.fonttype'] = 42

designs = [
    "Min Cold",
    "Max Cold",
    "Naive",
    "Baseline",
    "ICBM Ordered",
    "ICBM",
    "Ideal"
]

modelnames = {
    "llama2-13": "Llama2-13B",
    "llama3-70": "Llama3-70B",
    "gemma2": "Gemma2-27B",
    "opt-30": "OPT-30B",
    "dit-xl": "DiT-XL"
}

# colors = ["brown", "royalblue", "peru", "forestgreen", "gray", "black", "red"] # ['#ff796c', '#a4d46c', '#fca45c', '#95dbd0']
# colors = ["#5c95ff", "#ed9b40", "#393e41", "#548c2f", "#ba3b46"]
# colors = ["#ffb563", "#ba324f", "#175676", "#ff70a2", "#270722"]

# colors = ["#ea591f", "#126b91", "#9bc53d", "#252422", "#c45ab3"]
colors = ["#ec6632", "#126b91", "#93bc38", "#252422", "#c45ab3"]*2
colors1 = ["#126b91", "#ec6632", "#93bc38", "#252422", "#c45ab3"]*2
dark_colors = ["#95340e", "#072836", "#495e1c", "#6a6762", "#873179"]*2
dark_colors1 = ["#072836", "#95340e", "#495e1c", "#6a6762", "#873179"]*2
gradient_colors = ["#6ec7ed", "#49b9e9", "#25abe4", "#1993c8", "#126b91", "#0b435b"]
markers = ["o", "v", "^", "*", ""]
markers1 = ["v", "o", "^", "*", ""]

bar_hatches: list = ['', '\\\\', '..', '//', '+', 'x', 'o', 'O', '.', '*']
bar_hatches1: list = ['\\\\', '', '..', '//', '+', 'x', 'o', 'O', '.', '*']
lines: list = ['-', '-', '-', '-', '--']
lines1: list = ['-.', '--', '-', ':', '--']


def IPU_Mk2_cycle_to_ms(cycles):
    return cycles / 1.325e6