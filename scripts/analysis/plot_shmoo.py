"""
Shmoo-style plot: 2D grids of (axis × k), cell colored by classification.
One panel per base prompt, so you can see which (axis, k) combos preserve B.
"""
import csv, os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV   = os.path.join(PROJECT_ROOT, "ab_out", "classification_template.csv")
OUT   = os.path.join(PROJECT_ROOT, "ab_out", "classification_shmoo.png")
BASES = ["dog", "car", "dancer"]
AXES  = ["subject", "setting", "color", "motion", "texture", "style", "time_of_day"]
KS    = [2, 5, 7, 10]

# F=2 (green), P=1 (yellow), D=0 (red), blank=NaN (grey)
CODE = {"F": 2, "P": 1, "D": 0}
LABEL_OF_CODE = {2: "F", 1: "P", 0: "D"}

grids = {b: np.full((len(AXES), len(KS)), np.nan) for b in BASES}
for r in csv.DictReader(open(CSV)):
    c = r["classification"].strip().upper()
    if c not in CODE:
        continue
    b = r["base"]; axis = r["axis"]; k = int(r["k"])
    if b in grids and axis in AXES and k in KS:
        grids[b][AXES.index(axis), KS.index(k)] = CODE[c]

cmap = mcolors.ListedColormap(["#d62728", "#f0c24a", "#2ca02c"])
cmap.set_bad(color="#dddddd")
norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

fig, axs = plt.subplots(1, len(BASES), figsize=(4.2 * len(BASES), 5),
                        sharey=True, squeeze=False)
axs = axs[0]

for ax, b in zip(axs, BASES):
    g = np.ma.masked_invalid(grids[b])
    ax.imshow(g, cmap=cmap, norm=norm, aspect="auto", origin="upper")
    ax.set_title(f"base = {b}")
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f"k={k}" for k in KS])
    ax.set_yticks(range(len(AXES)))
    ax.set_yticklabels(AXES)
    for (i, j), v in np.ndenumerate(grids[b]):
        text = "—" if np.isnan(v) else LABEL_OF_CODE[int(v)]
        color = "black" if np.isnan(v) or v == 1 else "white"
        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    for x in np.arange(len(KS) + 1) - 0.5:
        ax.axvline(x, color="white", lw=1)
    for y in np.arange(len(AXES) + 1) - 0.5:
        ax.axhline(y, color="white", lw=1)
    ax.set_xlim(-0.5, len(KS) - 0.5)
    ax.set_ylim(len(AXES) - 0.5, -0.5)

# legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="#2ca02c", label="F  B fully survived"),
    Patch(color="#f0c24a", label="P  B partially survived"),
    Patch(color="#d62728", label="D  B dead (A dominates)"),
    Patch(color="#dddddd", label="—  not yet classified"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Shmoo: does B survive when resuming from A's step-k latent?", y=1.02)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(OUT)
