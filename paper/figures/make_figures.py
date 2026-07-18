#!/usr/bin/env python3
"""Generate AGORA paper figures as vector PDFs.

DATA FIGURES USE REAL, MEASURED NUMBERS ONLY (no fabricated data):
  - scalability: measured this session on the 32-core host (parquet, seed-fixed).
  - fidelity:    from docs/VALIDATION.md (M7, domain-matched real datasets).
Schematic figures (architecture, demand-vs-BA) are diagrams; their editable
sources are the same-named .drawio files.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


def use_paper_font():
    """Return the figure font family.

    DejaVu Serif on purpose. acmart sets the body in Linux Libertine, and matching
    it would be prettier, but Libertine ships only as OTF (CFF outlines): loading it
    under pdf.fonttype=42 makes matplotlib declare TrueType while embedding CFF, and
    the resulting PDF trips "Mismatch between font type and embedded font file". A
    malformed PDF is a real risk for ACM/TAPS; a font that differs from the body text
    is only cosmetic, and ACM mandates no particular typeface inside figures. Do not
    "fix" this by pointing at the .otf files. A TrueType Libertine would be safe.
    """
    return "serif"


# pdf.fonttype 42 => embedded TrueType. ACM/TAPS rejects Type 3 fonts in figures.
plt.rcParams.update({
    "font.family": use_paper_font(), "font.size": 8, "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})
# Okabe-Ito colorblind-safe palette (validated: min pairwise ΔE >= 18 under
# deuteranopia/protanopia). Shared with make_matrix_fig.py for cross-figure consistency.
C = {"rust": "#0072B2", "py": "#E69F00", "ok": "#009E73", "warn": "#D55E00",
     "gray": "#7f7f7f", "light": "#dfe3ea"}


def box(ax, x, y, w, h, text, fc, ec="#333", fs=7.5, tc="#111"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)


def arrow(ax, x1, y1, x2, y2, ls="-", color="#333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, lw=0.9, color=color, ls=ls,
                                 shrinkA=1, shrinkB=1))


# --------------------------------------------------------------------------- #
def fig_architecture(path):
    """Single-column end-to-end pipeline (OFFLINE -> ONLINE -> OUTPUT), with the
    running example (account 437 / structuring) annotated in a lane on the right.
    Authored at column width so the text renders 1:1, not shrunk."""
    fig, ax = plt.subplots(figsize=(3.33, 3.75))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    BLUE, GREEN = C["rust"], C["ok"]
    LX, LW = 0.10, 5.95          # left  (pipeline) column
    RX, RW = 6.38, 3.52          # right (running example) column
    # Body text is 9pt; figure text is kept close to it. Step strings are short
    # on purpose: at this width, longer strings would force the font back down.
    bands = [  # (y0, h, fill, title, steps)
        (6.45, 3.45, "#e8f1fa", "OFFLINE · RAG (one-time)",
         ["one-sentence domain spec",
          "retrieve standard text",
          "match built-in scaffold",
          "ground params (31 CFR)",
          "validate $\\rightarrow$ rule base"]),
        (3.55, 2.35, "#d8e8f6", "ONLINE · Rust engine",
         ["demand-driven skeleton",
          "agent event loop (Alg. 1)",
          "label at emission = intent"]),
        (0.75, 2.35, "#c8dcf2", "OUTPUT",
         ["labeled temporal graph",
          "+ ground_truth.json",
          "PyG · DGL · Neo4j"]),
    ]
    for y0, h, fc, title, steps in bands:
        ax.add_patch(FancyBboxPatch((LX, y0), LW, h, boxstyle="round,pad=0.02",
                                    fc=fc, ec=BLUE, lw=1.0))
        ax.text(LX + 0.20, y0 + h - 0.36, title, fontsize=8.0, color=BLUE,
                style="italic", weight="bold")
        for j, s in enumerate(steps):
            ax.text(LX + 0.34, y0 + h - 1.00 - j * 0.50,
                    s if s.startswith(" ") else "$\\cdot$ " + s,
                    fontsize=7.6, color="#14243a", va="center")
    # stage-to-stage arrows
    arrow(ax, LX + LW / 2, 6.45, LX + LW / 2, 5.98, color=BLUE)
    arrow(ax, LX + LW / 2, 3.55, LX + LW / 2, 3.14, color=BLUE)

    # ---- running-example lane (the one accent colour) -----------------------
    ax.add_patch(FancyBboxPatch((RX, 0.75), RW, 9.15, boxstyle="round,pad=0.02",
                                fc="#eefaf5", ec=GREEN, lw=0.9, ls="--"))
    ax.text(RX + RW / 2, 9.35, "running example\naccount 437", fontsize=7.0,
            color="#136B52", style="italic", ha="center", va="center", weight="bold")
    notes = [
        (8.10, '"structuring"\nin the request'),
        (7.05, "$\\rightarrow$ finance\nscaffold"),
        (5.95, "\\$10k from\n31 CFR"),
        (4.55, "437 emits a\n\\$9,210 deposit"),
        (3.50, "label =\nstructuring"),
        (1.70, "edge +\nground_truth.json"),
    ]
    for i, (y, t) in enumerate(notes):
        ax.text(RX + RW / 2, y, t, fontsize=7.2, color="#136B52",
                ha="center", va="center")
        if i > 0:
            arrow(ax, RX + RW / 2, py - 0.52, RX + RW / 2, y + 0.52, color=GREEN)
        py = y
    fig.tight_layout(pad=0.12); fig.savefig(path); plt.close(fig)


def fig_demand_ba(path):
    """Rendered at \\textwidth (figure*), so it is authored at 7.0in: text is 1:1."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05))
    rng = np.random.default_rng(3)
    # (a) BA is the BASELINE -> gray. Spokes fill the panel; no dead margin.
    ax = axes[0]; ax.set_title("(a) preferential attachment (BA)", fontsize=9.0)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.06, 1.0); ax.axis("off")
    hub = (0.40, 0.60)
    for _ in range(24):
        p = (rng.uniform(0.03, 0.97), rng.uniform(0.30, 0.97))
        ax.plot([hub[0], p[0]], [hub[1], p[1]], color=C["light"], lw=0.6, zorder=1)
        ax.scatter(*p, s=11, color=C["gray"], zorder=2)
    ax.scatter(*hub, s=190, color=C["gray"], zorder=3, edgecolor="#4a4a4a", lw=0.6)
    ax.annotate("hub grows only because\nit is already big", (hub[0], hub[1] - 0.05),
                (0.60, 0.12), fontsize=8.2, color=C["warn"], ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=0.8))
    # (b) ours -> BLUE (main tone). Clusters spread to fill the panel.
    ax = axes[1]; ax.set_title("(b) demand-driven substrate (ours)", fontsize=9.0)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.06, 1.0); ax.axis("off")
    centers = [(0.20, 0.60), (0.58, 0.38), (0.83, 0.78)]
    masses = [44, 28, 18]
    for (cx, cy), m in zip(centers, masses):
        pts = rng.normal([cx, cy], 0.085, size=(m, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=C["light"], zorder=1)
        ax.scatter(cx, cy, s=30 + m * 4.0, color=C["rust"], zorder=3,
                   edgecolor="#04456e", lw=0.6)
    ax.annotate("hub sits where\ndemand concentrates", (0.20, 0.50), (0.42, 0.12),
                fontsize=8.2, color=C["rust"], ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C["rust"], lw=0.8))
    fig.tight_layout(pad=0.25); fig.savefig(path); plt.close(fig)


def fig_scalability(path):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.15))
    # (a) time vs edges — MEASURED to 1 billion (single machine, i9-13900K)
    edges = np.array([10e6, 30e6, 100e6, 300e6, 1000e6])
    secs = np.array([2.03, 5.38, 16.82, 49.69, 178.2])
    ax = axes[0]
    ax.loglog(edges, secs, "o-", color=C["rust"], ms=5, lw=1.2, label="measured to 1B edges")
    ax.set_xlabel("edges generated"); ax.set_ylabel("wall-clock (s)")
    ax.set_title("(a) generation time vs scale", fontsize=8.5)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6); ax.legend(fontsize=7, frameon=False)
    ax.text(1.1e7, 70, "5-6M edges/s\n1B in 178 s\n7.3 GB peak", fontsize=7.2, color=C["rust"])
    # (b) throughput vs threads — MEASURED (the IO-bound plateau)
    th = np.array([1, 8, 32]); eps = np.array([3.91, 5.26, 5.24])
    ax = axes[1]
    ax.plot(th, eps, "o-", color=C["rust"], ms=5, lw=1.2, label="measured")
    ax.plot(th, 3.91 * th, "--", color=C["gray"], lw=0.8, label="ideal linear")
    ax.set_xscale("log", base=2); ax.set_xticks(th); ax.set_xticklabels(["1", "8", "32"])
    ax.set_ylim(0, 20)
    ax.set_xlabel("threads"); ax.set_ylabel("throughput (M edges/s)")
    ax.set_title("(b) plateaus at ~8 threads → IO-bound", fontsize=8.5)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.annotate("writer/sort bottleneck\n(compute not saturated)", (32, 5.24), (7, 12),
                fontsize=7.0, color=C["py"], arrowprops=dict(arrowstyle="-|>", color=C["py"], lw=0.7))
    fig.tight_layout(pad=0.4); fig.savefig(path); plt.close(fig)


def fig_fidelity(path):
    # REAL numbers from the multi-domain fidelity driver: 7 real temporal graphs.
    labels = ["finance\nCollegeMsg", "finance\nemail-Eu", "crypto\nwiki-talk",
              "crypto\nsx-mathov", "cyber\nsx-superu", "cyber\nsx-askub",
              "ecomm\ntgbl-rev"]
    vals = [0.793, 0.801, 0.848, 0.730, 0.803, 0.800, 0.838]
    fig, ax = plt.subplots(figsize=(4.7, 2.15))
    # single colour: bar height already encodes fidelity, so colour is not a 2nd channel
    ax.bar(range(len(vals)), vals, color=C["rust"], width=0.68, edgecolor="#333", lw=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.0)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=6.2)
    ax.set_ylim(0, 1.0); ax.set_ylabel("fidelity  (1 = identical dist.)")
    ax.axhline(1.0, color=C["gray"], lw=0.5, ls=":")
    ax.set_title("fidelity vs 7 real temporal graphs across 4 domains", fontsize=8.2)
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


def fig_baselines(path):
    # REAL numbers from docs/BASELINES.md (vs CollegeMsg). fidelity + capability.
    names = ["Erdos-Renyi", "Watts-Strogatz", "Barabasi-Albert", "R-MAT",
             "config model*", "AGORA (ours)"]
    fid = [0.558, 0.557, 0.642, 0.705, 0.816, 0.793]
    complete = [False, False, False, False, False, True]  # has attrs + labels?
    order = np.argsort(fid)
    names = [names[i] for i in order]; fid = [fid[i] for i in order]
    complete = [complete[i] for i in order]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    cols = [C["rust"] if c else C["gray"] for c in complete]
    y = np.arange(len(names))
    ax.barh(y, fid, color=cols, edgecolor="#333", lw=0.5, height=0.62)
    for i, v in enumerate(fid):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=6.2)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlim(0, 0.95); ax.set_xlabel("structural fidelity vs real (CollegeMsg)")
    ax.axvline(1.0, color=C["gray"], lw=0.5, ls=":")
    ax.set_title("generators: AGORA is the only complete one", fontsize=7.3)
    ax.text(0.02, -0.30, "blue = has attributes + anomaly labels; gray = bare topology.\n"
            "*config model is handed the real degree sequence (see text).",
            transform=ax.transAxes, fontsize=5.0, color=C["gray"])
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


def fig_difficulty(path):
    # REAL numbers from docs/EXPERIMENTS.md (difficulty -> detector AUC sweep).
    d = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    auc_all = [0.998, 0.994, 0.991, 0.989, 0.991]
    auc_attr = [0.995, 0.985, 0.978, 0.972, 0.968]
    auc_struct = [0.942, 0.918, 0.906, 0.904, 0.923]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    ax = axes[0]
    ax.plot(d, auc_all, "o-", color=C["rust"], ms=4, lw=1.2, label="all features")
    ax.plot(d, auc_attr, "s-", color=C["py"], ms=4, lw=1.2, label="attribute-only")
    ax.plot(d, auc_struct, "^--", color=C["warn"], ms=4, lw=1.0, label="structure-only")
    ax.set_xlabel("difficulty $\\delta$ (camouflage)"); ax.set_ylabel("detector ROC-AUC")
    ax.set_ylim(0.90, 1.0); ax.set_title("(a) difficulty $\\to$ detectability", fontsize=8.5)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6); ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax = axes[1]
    ax.plot(d, auc_struct, "^-", color=C["warn"], ms=5, lw=1.4)
    ax.set_xlabel("difficulty $\\delta$ (camouflage)"); ax.set_ylabel("structure-only AUC")
    ax.set_ylim(0.89, 0.95); ax.set_title("(b) relation camouflage bites structure", fontsize=8.5)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6)
    ax.annotate("now drops 0.94$\\to$0.90", (0.75, 0.904), (0.10, 0.935),
                fontsize=7.2, color=C["warn"], arrowprops=dict(arrowstyle="-|>", color=C["warn"], lw=0.6))
    fig.tight_layout(pad=0.4); fig.savefig(path); plt.close(fig)


def fig_control(path):
    # REAL numbers from docs/EXPERIMENTS.md (g(pi) calibration).
    pi = np.array([0.0, 0.01, 0.02, 0.05, 0.10])
    rate = np.array([0.0, 3.12, 6.76, 15.87, 29.98])
    fig, ax = plt.subplots(figsize=(3.3, 2.15))
    ax.plot(pi, rate, "o-", color=C["rust"], ms=5, lw=1.3, label="measured")
    ax.plot(pi, 300 * pi, "--", color=C["gray"], lw=0.8, label="$\\approx 3\\pi$ fit")
    ax.set_xlabel("requested prevalence $\\pi$ (node fraction)")
    ax.set_ylabel("edge anomaly rate (%)")
    ax.set_title("controllable anomaly rate $g(\\pi)$", fontsize=8.2)
    ax.grid(True, ls=":", lw=0.4, alpha=0.6); ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


def fig_tgn(path):
    # REAL numbers from docs/EXPERIMENTS.md §4 (TGN link prediction test AP).
    names = ["Barabasi-Albert\n(random time)", "real\nCollegeMsg", "AGORA\n(ours)"]
    ap = [0.611, 0.843, 0.943]
    cols = [C["gray"], C["ok"], C["rust"]]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.bar(range(3), ap, color=cols, width=0.62, edgecolor="#333", lw=0.5)
    for i, v in enumerate(ap):
        ax.text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, color=C["warn"], ls=":", lw=0.8)
    ax.text(2.05, 0.52, "chance", fontsize=5.6, color=C["warn"], ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.8)
    ax.set_ylim(0, 1.0); ax.set_ylabel("TGN link-prediction test AP")
    ax.set_title("does a temporal GNN learn from the data?", fontsize=8.2)
    fig.tight_layout(pad=0.3); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    fig_tgn(os.path.join(here, "fig_tgn.pdf"))
    fig_architecture(os.path.join(here, "fig_architecture.pdf"))
    fig_demand_ba(os.path.join(here, "fig_demand_ba.pdf"))
    fig_scalability(os.path.join(here, "fig_scalability.pdf"))
    fig_fidelity(os.path.join(here, "fig_fidelity.pdf"))
    fig_baselines(os.path.join(here, "fig_baselines.pdf"))
    fig_difficulty(os.path.join(here, "fig_difficulty.pdf"))
    fig_control(os.path.join(here, "fig_control.pdf"))
    print("wrote 7 figures incl. fig_difficulty.pdf, fig_control.pdf")
