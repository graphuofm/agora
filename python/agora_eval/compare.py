"""compare.py — fidelity scorecard between a real and a synthetic graph.

Computes the same GraphStats on both, then per-metric distances:
  - distributions (degree in/out/total, inter-event times): two-sample KS
    (scipy.stats.ks_2samp); lower = closer.
  - scalar fingerprints (alpha, B, M, clustering, reciprocity, …): absolute and
    relative deltas.
  - an overall FIDELITY SCORE in [0,1]: 1 − mean of the bounded per-metric
    distances (KS are already in [0,1]; scalar deltas are squashed). Honest and
    simple; reported alongside every component so nothing is hidden.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import numpy as np
from scipy import stats as ss

from .stats import GraphStats


def _ks(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 5 or b.size < 5:
        return float("nan")
    # subsample very large arrays for a stable, fast KS
    cap = 200_000
    if a.size > cap:
        a = np.random.default_rng(0).choice(a, cap, replace=False)
    if b.size > cap:
        b = np.random.default_rng(1).choice(b, cap, replace=False)
    return float(ss.ks_2samp(a, b).statistic)


def _rel(a, b) -> float:
    """Symmetric relative delta in [0,1]; nan-safe."""
    if a is None or b is None:
        return float("nan")
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    denom = abs(a) + abs(b)
    return abs(a - b) / denom if denom > 0 else 0.0


def compare(real: GraphStats, synth: GraphStats) -> Dict:
    ks = {
        "out_degree": _ks(real.out_deg.astype(float), synth.out_deg.astype(float)),
        "in_degree": _ks(real.in_deg.astype(float), synth.in_deg.astype(float)),
        "total_degree": _ks(real.tot_deg.astype(float), synth.tot_deg.astype(float)),
        "inter_event_time": _ks(real.inter_event, synth.inter_event),
    }
    scalars = {
        "powerlaw_alpha": (real.powerlaw_alpha, synth.powerlaw_alpha, _rel(real.powerlaw_alpha, synth.powerlaw_alpha)),
        "burstiness_b": (real.burstiness_b, synth.burstiness_b, _abs01(real.burstiness_b, synth.burstiness_b, 2.0)),
        "memory_m": (real.memory_m, synth.memory_m, _abs01(real.memory_m, synth.memory_m, 2.0)),
        "clustering": (real.clustering, synth.clustering, _abs01(real.clustering, synth.clustering, 1.0)),
        "reciprocity": (real.reciprocity, synth.reciprocity, _abs01(real.reciprocity, synth.reciprocity, 1.0)),
        "repeat_edge_ratio": (real.repeat_edge_ratio, synth.repeat_edge_ratio, _abs01(real.repeat_edge_ratio, synth.repeat_edge_ratio, 1.0)),
        "mean_degree": (real.mean_degree, synth.mean_degree, _rel(real.mean_degree, synth.mean_degree)),
    }
    # Overall fidelity: 1 − mean of all bounded distances actually available.
    dists = [v for v in ks.values() if not np.isnan(v)]
    dists += [t[2] for t in scalars.values() if not np.isnan(t[2])]
    fidelity = 1.0 - float(np.mean(dists)) if dists else float("nan")
    return {
        "fidelity_score": fidelity,
        "ks_distances": ks,
        "scalar_metrics": {k: {"real": v[0], "synth": v[1], "distance": v[2]} for k, v in scalars.items()},
        "real_summary": real.scalar_summary(),
        "synth_summary": synth.scalar_summary(),
        "sizes": {
            "real_edges": real.n_edges, "real_nodes": real.n_nodes,
            "synth_edges": synth.n_edges, "synth_nodes": synth.n_nodes,
        },
    }


def _abs01(a, b, scale) -> float:
    """Absolute delta normalized by `scale` into [0,1]; nan-safe."""
    if a is None or b is None:
        return float("nan")
    if (isinstance(a, float) and np.isnan(a)) or (isinstance(b, float) and np.isnan(b)):
        return float("nan")
    return min(1.0, abs(a - b) / scale)


def format_scorecard(result: Dict, real_name: str, synth_name: str) -> str:
    out = []
    out.append(f"=== fidelity scorecard: {synth_name}  vs  {real_name} (real) ===")
    s = result["sizes"]
    out.append(f"  size       real {s['real_edges']:,} edges / {s['real_nodes']:,} nodes   "
               f"synth {s['synth_edges']:,} / {s['synth_nodes']:,}")
    fs = result["fidelity_score"]
    out.append(f"  FIDELITY   {fs:.3f}   (1.0 = identical distributions; mean of all distances below)")
    out.append("  distribution distances (two-sample KS, lower=closer):")
    for k, v in result["ks_distances"].items():
        bar = _bar(v)
        out.append(f"    {k:<18} KS={_fmt(v)}  {bar}")
    out.append("  scalar metrics (real → synth, distance):")
    for k, v in result["scalar_metrics"].items():
        out.append(f"    {k:<18} {_fmt(v['real'])} → {_fmt(v['synth'])}   dist={_fmt(v['distance'])}")
    return "\n".join(out)


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and np.isnan(x):
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _bar(v: float) -> str:
    if np.isnan(v):
        return ""
    n = int(round((1.0 - min(1.0, v)) * 20))
    return "█" * n + "·" * (20 - n)
