#!/usr/bin/env python3
"""fig_matrix.pdf — the big contest: 10 generators x 6 domains fidelity heatmap.
All values measured (scratchpad/matrix.json). Sequential single-hue (cividis:
print- and CVD-safe), AGORA row boxed, generators split into prior (no access to the
real graph) vs fit-to-real (handed the real degree/structure)."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ACM/TAPS forbids Type 3 fonts in figures; fonttype 42 emits embedded TrueType.
# Font + rcParams must match make_figures.py, or this script silently diverges
# (it already reintroduced Type 3 once by omitting fonttype).
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import use_paper_font

plt.rcParams.update({
    "font.family": use_paper_font(), "font.size": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})

# prior (blind) generators first, then fit-to-real (marked ‡)
rows = ["AGORA", "Kronecker", "R-MAT", "BA", "WS", "ER",
        "config‡", "Chung-Lu‡", "DC-SBM‡", "RDPG‡"]
n_prior = 6
doms = ["finance", "crypto", "cyber", "e-comm", "transport†", "health†"]
D = {
    "AGORA":     [.801, .843, .807, .861, .696, .751],
    "Kronecker":[.700, .621, .730, .829, .728, .642],
    "R-MAT":    [.747, .664, .777, .863, .771, .713],
    "BA":       [.617, .587, .682, .825, .684, .595],
    "WS":       [.481, .543, .583, .656, .607, .552],
    "ER":       [.520, .525, .586, .757, .601, .488],
    "config‡":  [.844, .833, .832, .919, .835, .834],
    "Chung-Lu‡":[.604, .775, .854, .894, .850, .842],
    "DC-SBM‡":  [.829, .742, .826, .888, .823, .847],
    "RDPG‡":    [.835, .744, .831, .864, .811, .875],
}
data = np.array([D[r] for r in rows])

fig, ax = plt.subplots(figsize=(7.0, 4.2))
im = ax.imshow(data, cmap="cividis", vmin=0.45, vmax=0.95, aspect="auto")
for i in range(len(rows)):
    for j in range(len(doms)):
        v = data[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.76 else "black", fontsize=7.2)
ax.set_xticks(range(len(doms)))
ax.set_xticklabels(doms, fontsize=7.6, rotation=22, ha="right")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows, fontsize=7.6)
ax.tick_params(length=0)
# AGORA row boxed (Okabe-Ito vermillion, shared accent with make_figures.py)
ax.add_patch(Rectangle((-0.5, -0.5), len(doms), 1, fill=False,
                       edgecolor="#D55E00", lw=2.4, zorder=5))
# separator between prior and fit-to-real
ax.axhline(n_prior - 0.5, color="white", lw=3)
# group brackets on the far left
ax.annotate("", xy=(-1.35, -0.4), xytext=(-1.35, n_prior - 0.6),
            annotation_clip=False, arrowprops=dict(arrowstyle="-", color="#555", lw=1))
ax.text(-1.6, (n_prior - 1) / 2, "prior\n(blind)", rotation=90, va="center",
        ha="center", fontsize=6.6, color="#333")
ax.annotate("", xy=(-1.35, n_prior - 0.4), xytext=(-1.35, len(rows) - 0.6),
            annotation_clip=False, arrowprops=dict(arrowstyle="-", color="#555", lw=1))
ax.text(-1.6, n_prior + (len(rows) - n_prior - 1) / 2, "fit-to-\nreal ‡", rotation=90,
        va="center", ha="center", fontsize=6.6, color="#333")
cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cbar.set_label("fidelity vs real  (1 = identical distributions)", fontsize=6.8)
cbar.ax.tick_params(labelsize=6)
ax.set_title("Every generator vs AGORA on six domains", fontsize=9, pad=6)
fig.text(0.5, 0.005,
         "‡ handed the real degree/structure.  † no domain-matched public temporal "
         "graph; a generic interaction net is a structural proxy.",
         ha="center", fontsize=5.2, color="#666")
fig.savefig("./paper/figures/fig_matrix.pdf", bbox_inches="tight")
print("wrote fig_matrix.pdf")
