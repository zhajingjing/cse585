"""
Reads classification_template.csv (filled with F / P / D) and produces a
stacked-bar plot: one panel per k value, 7 axis bars per panel, each bar
stacking F/P/D counts across the 3 base prompts.
"""
import csv, os
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV  = os.path.join(PROJECT_ROOT, "ab_out", "classification_template.csv")
OUT  = os.path.join(PROJECT_ROOT, "ab_out", "classification_plot.png")
AXES = ["subject", "setting", "color", "motion", "texture", "style", "time_of_day"]
CLS  = ["F", "P", "D"]
LABEL = {"F": "Full", "P": "Partial", "D": "Dead"}
COLOR = {"F": "#2ca02c", "P": "#f0c24a", "D": "#d62728"}

rows = [r for r in csv.DictReader(open(CSV))
        if r["classification"].strip().upper() in CLS]
if not rows:
    raise SystemExit(f"No F/P/D classifications found in {CSV}")

ks = sorted({int(r["k"]) for r in rows})
fig, axs = plt.subplots(1, len(ks), figsize=(4 * len(ks), 5),
                        sharey=True, squeeze=False)
axs = axs[0]

for ax, k in zip(axs, ks):
    counts = {a: {c: 0 for c in CLS} for a in AXES}
    for r in rows:
        if int(r["k"]) != k:
            continue
        if r["axis"] in counts:
            counts[r["axis"]][r["classification"].strip().upper()] += 1
    bottom = np.zeros(len(AXES))
    for c in CLS:
        vals = np.array([counts[a][c] for a in AXES])
        ax.bar(AXES, vals, bottom=bottom, color=COLOR[c],
               edgecolor="white", linewidth=0.5,
               label=LABEL[c] if ax is axs[0] else "")
        bottom += vals
    ax.set_title(f"k = {k}")
    ax.set_xticks(range(len(AXES)))
    ax.set_xticklabels(AXES, rotation=30, ha="right")
    ax.set_ylim(0, 3.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#eee", linewidth=0.8)

axs[0].set_ylabel("count of bases (0–3)")
axs[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=3, frameon=False)
fig.suptitle("B feature survival by axis and skip step (manual classification)", y=1.03)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
