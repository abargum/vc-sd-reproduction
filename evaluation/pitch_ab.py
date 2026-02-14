import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import math


plt.rcParams.update({
    'font.size': 14,
    "font.family": "serif", 
    "font.serif": ['DejaVu Serif'],
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def proportion_ci_95(p, n):
    if n == 0:
        return 0
    se = math.sqrt(p * (1 - p) / n)
    return 1.96 * se * 100 

df = pd.read_csv("evaluation/pitch_ab_test.csv")

ab_columns = [c for c in df.columns if re.match(r"A_.+\|B_.+", c)]

answer_rows = df.iloc[1:]

algos = ["nn", "praat", "sox"]

wins = {a: defaultdict(int) for a in algos}
counts = defaultdict(int)

for col in ab_columns:
    parts = col.split("|")
    A_alg = parts[0].replace("A_", "").strip()
    B_alg = parts[1].replace("B_", "").strip()

    for answer in answer_rows[col]:
        if answer not in ("A", "B"):
            continue

        pair = tuple(sorted([A_alg, B_alg]))
        counts[pair] += 1

        if answer == "A":
            wins[A_alg][B_alg] += 1
        else:
            wins[B_alg][A_alg] += 1

shift_wins = defaultdict(lambda: {a: defaultdict(int) for a in algos})
shift_counts = defaultdict(lambda: defaultdict(int))

for col in ab_columns:
    parts = col.split("|")
    A_alg = parts[0].replace("A_", "").strip()
    B_alg = parts[1].replace("B_", "").strip()
    shift = parts[2].strip() if len(parts) > 2 else "unknown"

    for answer in answer_rows[col]:
        if answer not in ("A", "B"):
            continue

        pair = tuple(sorted([A_alg, B_alg]))
        shift_counts[shift][pair] += 1

        if answer == "A":
            shift_wins[shift][A_alg][B_alg] += 1
        else:
            shift_wins[shift][B_alg][A_alg] += 1

large_shifts = ['-8', '-7', '-6', '+6', '+7', '+8']
small_shifts = ['-5', '-4', '-3', '+3', '+4', '+5']

def aggregate_shifts(shift_list):
    agg_wins = {a: defaultdict(int) for a in algos}
    agg_counts = defaultdict(int)

    for shift in shift_list:
        for pair in shift_counts[shift]:
            agg_counts[pair] += shift_counts[shift][pair]
        for a in algos:
            for b in algos:
                agg_wins[a][b] += shift_wins[shift][a][b]

    return agg_wins, agg_counts

large_wins, large_counts = aggregate_shifts(large_shifts)
small_wins, small_counts = aggregate_shifts(small_shifts)

matchups = [
    ('nn', 'praat'),
    ('nn', 'sox'),
    ('praat', 'sox')
]

colors = {
    'nn': '#FFB3BA',      
    'praat': '#B4E7B4',   
    'sox': '#FFE5A3'      
}

hatches = {
    'nn': '///',
    'praat': '\\\\\\',
    'sox': 'xxx'
}

fig = plt.figure(figsize=(14, 10))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.35)
ax_overall = fig.add_subplot(gs[0, :])
ax_large = fig.add_subplot(gs[1, 0])
ax_small = fig.add_subplot(gs[1, 1])

bar_height = 0.65

def draw_matchups(ax, wins_data, counts_data, title, show_xlabel=True):

    for idx, (algo1, algo2) in enumerate(matchups):
        y_pos = len(matchups) - 1 - idx
        pair = tuple(sorted([algo1, algo2]))
        total = counts_data[pair]

        if total == 0:
            continue

        pct1 = 100 * wins_data[algo1][algo2] / total
        pct2 = 100 * wins_data[algo2][algo1] / total

        rect1 = Rectangle((0, y_pos - bar_height/2), pct1, bar_height,
                           facecolor=colors[algo1],
                           edgecolor='#666666',
                           linewidth=1.2,
                           hatch=hatches[algo1])
        ax.add_patch(rect1)

        if pct1 > 5:
            ax.text(pct1 / 2, y_pos,
                    f'{pct1:.0f}%',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='#333333')

        rect2 = Rectangle((pct1, y_pos - bar_height/2), pct2, bar_height,
                           facecolor=colors[algo2],
                           edgecolor='#666666',
                           linewidth=1.2,
                           hatch=hatches[algo2])
        ax.add_patch(rect2)

        if pct2 > 5:
            ax.text(pct1 + pct2 / 2, y_pos,
                    f'{pct2:.0f}%',
                    ha='center', va='center',
                    fontsize=14, fontweight='bold',
                    color='#333333')

       
        p = pct1 / 100.0
        ci = proportion_ci_95(p, total)
        boundary_x = pct1

        ax.errorbar(boundary_x, y_pos,
                    xerr=ci,
                    fmt='o',
                    ecolor='#333333',
                    elinewidth=1.5,
                    capsize=5,
                    markersize=4,
                    color='#333333')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(matchups)-0.5)
    ax.set_yticks(range(len(matchups)))
    ax.set_yticklabels([''] * len(matchups))

    if show_xlabel:
        ax.set_xlabel("Match Preference Score (%)", fontsize=16, fontweight='normal')

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'] if show_xlabel else [])

    ax.grid(axis='x', linestyle='--', alpha=0.25, color='#CCCCCC')
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=12, color='#333333')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')

draw_matchups(ax_overall, wins, counts,
              "Overall Results", show_xlabel=False)

draw_matchups(ax_large, large_wins, large_counts,
              "Large Pitch Shifts (±6, ±7, ±8)", show_xlabel=True)

draw_matchups(ax_small, small_wins, small_counts,
              "Small Pitch Shifts (±3, ±4, ±5)", show_xlabel=True)

legend_patches = []
label_map = {
    "nn": "VC-SD",
    "praat": "Praat-Parselmouth",
    "sox": "RubberBand"
}

for algo in algos:
    legend_patches.append(
        mpatches.Patch(facecolor=colors[algo],
                       edgecolor='#666666',
                       linewidth=1.2,
                       label=label_map[algo])
    )

ax_overall.legend(handles=legend_patches,
                  loc='upper center',
                  bbox_to_anchor=(0.5, 1.28),
                  ncol=3,
                  frameon=True,
                  fancybox=False,
                  edgecolor='#CCCCCC',
                  framealpha=0.95)

plt.savefig("plots/pitch_ab.png", dpi=300, bbox_inches='tight', facecolor='white')