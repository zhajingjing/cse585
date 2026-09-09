import csv, glob, json, os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROOTS  = [os.path.join(PROJECT_ROOT, "ab_out"),
          os.path.join(PROJECT_ROOT, "ab_out_k27")]
WK     = os.path.join(PROJECT_ROOT, "eval", "teacache", "vbench", "workload_ab_axes_{b}.json")
BASES  = ["dog", "car", "dancer"]
AXES   = ["subject", "setting", "color", "motion", "texture", "style", "time_of_day"]
KS     = [2, 5, 7, 10]

prompt_axis = {}
for b in BASES:
    with open(WK.format(b=b)) as f:
        for e in json.load(f):
            prompt_axis[e["prompt_en"]] = (b, e["axis"])

margin  = np.full((len(AXES), len(BASES) * len(KS)), np.nan)
survive = np.full_like(margin, np.nan)
col_labels = []
for bi, b in enumerate(BASES):
    for ki, k in enumerate(KS):
        col_labels.append(f"{b}\nk={k}")
# Scan per-pair metrics.csv files (works even for partial runs)
for b in BASES:
    for root in ROOTS:
        for mpath in sorted(glob.glob(os.path.join(root, b, "pair_*", "metrics.csv"))):
            with open(mpath) as f:
                for r in csv.DictReader(f):
                    axis = prompt_axis.get(r["prompt_b"], (None, None))[1]
                    if axis not in AXES:
                        continue
                    k = int(r["variant"].replace("B_from_step", ""))
                    if k not in KS:
                        continue
                    col = BASES.index(b) * len(KS) + KS.index(k)
                    row = AXES.index(axis)
                    margin[row, col]  = float(r["clip_semantic_margin_B_minus_A"])
                    survive[row, col] = float(r["b_survival_ratio_mse"])

fig, axs = plt.subplots(1, 2, figsize=(22, 6), gridspec_kw={"wspace": 0.25})

# Left: CLIP margin (diverging, centered at 0)
m_abs = np.nanmax(np.abs(margin))
im0 = axs[0].imshow(margin, cmap="RdBu", vmin=-m_abs, vmax=m_abs, aspect="auto")
axs[0].set_title("CLIP margin = CLIP(→B) − CLIP(→A)\n(red = A dominates semantics, blue = B survives)")
# Right: B_surv (sequential)
im1 = axs[1].imshow(survive, cmap="viridis", vmin=0, vmax=1, aspect="auto")
axs[1].set_title("B_surv = MSE(→A) / MSE(→B)\n(dark = A dominates pixels, bright = B survives)")

for ax, data, fmt in [(axs[0], margin, "{:+.3f}"), (axs[1], survive, "{:.2f}")]:
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(AXES)))
    ax.set_yticklabels(AXES)
    for (row, col), v in np.ndenumerate(data):
        if np.isnan(v):
            continue
        color = "black" if (ax is axs[0] and abs(v) < m_abs * 0.5) or (ax is axs[1] and v > 0.5) else "white"
        ax.text(col, row, fmt.format(v), ha="center", va="center", color=color, fontsize=8)
    # vertical separators between bases
    for bi in range(1, len(BASES)):
        ax.axvline(bi * len(KS) - 0.5, color="white", linewidth=1.5)

fig.colorbar(im0, ax=axs[0], shrink=0.8)
fig.colorbar(im1, ax=axs[1], shrink=0.8)
fig.suptitle("A/B latent cache survival — how much of prompt B survives when resuming from A's step-k latent",
             fontsize=12, y=1.02)

out = os.path.join(PROJECT_ROOT, "ab_out", "summary_heatmap.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(out)
