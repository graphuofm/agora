"""metrics_advanced.py — SOTA fidelity metrics beyond the base scorecard.

Implements the higher layers of docs/EVAL.md so AGORA's evaluation out-rigors
LDBC (which stops at Layer-1 structural fidelity):

  Layer 2 (statistical rigor):  energy distance / 1-D Wasserstein (energy = MMD,
        Sejdinovic et al. 2013), and 2-sample Anderson-Darling (tail-sensitive,
        for heavy-tailed degrees where KS is blind).
  Layer 3 (temporal fidelity):  the 2-node δ-temporal-motif fingerprint
        (Paranjape, Benson & Leskovec, WSDM 2017) — a temporal signature no
        static generator can match.
  Layer 4 (extrinsic utility):  the discriminative score / Classifier Two-Sample
        Test (Lopez-Paz & Oquab, ICLR 2017; TimeGAN, Yoon et al. 2019) — train a
        classifier to tell real from synthetic on ID-AGNOSTIC edge features;
        report 2·|AUC−0.5| (0 = indistinguishable = best, 1 = trivially separable).

All functions take plain temporal edge arrays (src, dst, t) so any dataset with
those three columns works. Heavy inputs are subsampled for a stable, fast score.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Layer 2 — distributional distances (report effect sizes, not p-values)
# --------------------------------------------------------------------------- #
def _clean(a: np.ndarray, cap: int = 200_000, seed: int = 0) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size > cap:
        a = np.random.default_rng(seed).choice(a, cap, replace=False)
    return a


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1-D energy distance (= distance-based MMD; Sejdinovic et al. 2013).
    Units-bearing, triangle-inequality, stabler than KS. Lower = closer."""
    from scipy import stats as ss

    a, b = _clean(a, seed=0), _clean(b, seed=1)
    if a.size < 5 or b.size < 5:
        return float("nan")
    return float(ss.energy_distance(a, b))


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein-1 (earth-mover) distance. Lower = closer."""
    from scipy import stats as ss

    a, b = _clean(a, seed=0), _clean(b, seed=1)
    if a.size < 5 or b.size < 5:
        return float("nan")
    return float(ss.wasserstein_distance(a, b))


def anderson_darling(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """2-sample Anderson-Darling statistic (tail-weighted — the right test for
    heavy-tailed degree tails, where KS is insensitive). Higher = more different.
    Returns the normalized statistic; None if degenerate."""
    from scipy import stats as ss

    a, b = _clean(a, seed=0), _clean(b, seed=1)
    if a.size < 8 or b.size < 8:
        return None
    try:
        import warnings

        # anderson_ksamp warns on p-value capping (we only use the statistic).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = ss.anderson_ksamp([a, b])
        return float(res.statistic)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Layer 3 — 2-node temporal-motif fingerprint (Paranjape et al., WSDM 2017)
# --------------------------------------------------------------------------- #
def temporal_motifs_2node(
    src: np.ndarray, dst: np.ndarray, t: np.ndarray, delta_s: float
) -> np.ndarray:
    """Count the eight 2-node, 3-edge δ-temporal motifs and return a normalized
    fingerprint (probability 8-vector).

    For each unordered pair {u,v} (u<v), take its time-ordered event sequence
    (either direction) and slide a window of 3 consecutive events whose span
    t[i+2]-t[i] ≤ δ. Each event's direction relative to (u,v) is a bit
    (0: u→v, 1: v→u); the ordered triple of bits indexes one of 8 motifs.
    """
    src = np.asarray(src)
    dst = np.asarray(dst)
    t = np.asarray(t, dtype=float)
    n = src.size
    counts = np.zeros(8, dtype=np.int64)
    if n < 3:
        return counts.astype(float)

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)
    direction = (src != u).astype(np.int8)  # 0: u->v, 1: v->u
    # Canonical pair key; group events by pair, ordered by (pair, time).
    pair = u.astype(np.int64) * np.int64(0x100000000) + v.astype(np.int64)
    order = np.lexsort((t, pair))
    pk = pair[order]
    tt = t[order]
    dd = direction[order]

    # Iterate contiguous runs of equal pair key; within a run slide a 3-window.
    starts = np.flatnonzero(np.concatenate(([True], pk[1:] != pk[:-1])))
    ends = np.concatenate((starts[1:], [n]))
    for a, b in zip(starts, ends):
        L = b - a
        if L < 3:
            continue
        ts = tt[a:b]
        ds = dd[a:b]
        for i in range(L - 2):
            if ts[i + 2] - ts[i] <= delta_s:
                idx = (int(ds[i]) << 2) | (int(ds[i + 1]) << 1) | int(ds[i + 2])
                counts[idx] += 1
    tot = counts.sum()
    return counts.astype(float) / tot if tot > 0 else counts.astype(float)


def motif_l1(fa: np.ndarray, fb: np.ndarray) -> float:
    """L1 distance between two normalized motif fingerprints, in [0,2]."""
    if fa.sum() == 0 or fb.sum() == 0:
        return float("nan")
    return float(np.abs(fa - fb).sum())


# --------------------------------------------------------------------------- #
# Layer 4 — discriminative score / Classifier Two-Sample Test
# --------------------------------------------------------------------------- #
def _edge_features(src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> np.ndarray:
    """ID-AGNOSTIC per-edge features (never raw node IDs — those would trivially
    separate two graphs). Captures structure (degrees), recurrence, and timing."""
    n = src.size
    both = np.concatenate([src, dst])
    nodes, inv = np.unique(both, return_inverse=True)
    s = inv[:n]
    d = inv[n:]
    N = nodes.size
    outdeg = np.bincount(s, minlength=N).astype(float)
    indeg = np.bincount(d, minlength=N).astype(float)
    totdeg = outdeg + indeg

    # per-source inter-event gap (time since the source's previous event)
    order = np.lexsort((t, s))
    s_sorted = s[order]
    t_sorted = t[order].astype(float)
    same = np.zeros(n, dtype=bool)
    same[1:] = s_sorted[1:] == s_sorted[:-1]
    dt = np.empty(n)
    dt[0] = np.nan
    dt[1:] = t_sorted[1:] - t_sorted[:-1]
    gap = np.full(n, np.nan)
    gap[order] = np.where(same, dt, np.nan)
    has_gap = np.isfinite(gap).astype(float)
    gap_log = np.log1p(np.nan_to_num(gap, nan=0.0))

    # repeat indicator (has this (src,dst) pair occurred earlier in time)
    torder = np.argsort(t, kind="stable")
    seen: set = set()
    repeat = np.zeros(n)
    for i in torder:
        key = (int(s[i]), int(d[i]))
        if key in seen:
            repeat[i] = 1.0
        else:
            seen.add(key)

    tod = (t.astype(float) % 86400.0) / 3600.0
    dow = ((t.astype(float) // 86400.0) % 7.0)
    return np.column_stack([
        np.log1p(outdeg[s]), np.log1p(indeg[d]),
        np.log1p(totdeg[s]), np.log1p(totdeg[d]),
        gap_log, has_gap, repeat,
        np.sin(2 * np.pi * tod / 24.0), np.cos(2 * np.pi * tod / 24.0), dow,
    ])


def discriminative_score(
    real: Arrays, synth: Arrays, cap: int = 40_000, seed: int = 0
) -> Optional[Dict[str, float]]:
    """C2ST: train a classifier to separate real from synthetic edges on
    ID-agnostic features. Returns AUC, accuracy, and the discriminative score
    2·|AUC−0.5| ∈ [0,1] (0 = indistinguishable = best)."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:  # noqa: BLE001 — sklearn optional
        return None

    Xr = _edge_features(*real)
    Xs = _edge_features(*synth)
    rng = np.random.default_rng(seed)
    m = int(min(cap, Xr.shape[0], Xs.shape[0]))
    if m < 200:
        return None
    Xr = Xr[rng.choice(Xr.shape[0], m, replace=False)]
    Xs = Xs[rng.choice(Xs.shape[0], m, replace=False)]
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.ones(m), np.zeros(m)])
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=120)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    acc = float(accuracy_score(yte, (p > 0.5).astype(int)))
    return {"auc": auc, "accuracy": acc, "discriminative_score": 2.0 * abs(auc - 0.5)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def advanced_report(
    real: Arrays,
    synth: Arrays,
    real_stats,
    synth_stats,
    delta_s: Optional[float] = None,
) -> Dict:
    """Compute all advanced metrics for a real/synth pair. `*_stats` are the base
    GraphStats (for the degree/IET arrays); `real`/`synth` are (src,dst,t)."""
    # δ for temporal motifs: default to 1 hour, or the base activity window.
    if delta_s is None:
        span = max(1.0, real_stats.t_max - real_stats.t_min)
        delta_s = min(3600.0, span / 100.0)

    fr = temporal_motifs_2node(*real, delta_s=delta_s)
    fs = temporal_motifs_2node(*synth, delta_s=delta_s)

    disc = discriminative_score(real, synth)

    return {
        "layer2_distribution_distances": {
            "total_degree": {
                "energy": energy_distance(real_stats.tot_deg, synth_stats.tot_deg),
                "wasserstein": wasserstein(real_stats.tot_deg, synth_stats.tot_deg),
                "anderson_darling": anderson_darling(real_stats.tot_deg, synth_stats.tot_deg),
            },
            "inter_event": {
                "energy": energy_distance(real_stats.inter_event, synth_stats.inter_event),
                "wasserstein": wasserstein(real_stats.inter_event, synth_stats.inter_event),
                "anderson_darling": anderson_darling(real_stats.inter_event, synth_stats.inter_event),
            },
        },
        "layer3_temporal_motifs": {
            "delta_s": delta_s,
            "real_fingerprint": fr.tolist(),
            "synth_fingerprint": fs.tolist(),
            "l1_distance": motif_l1(fr, fs),
        },
        "layer4_discriminative": disc,
    }


def format_advanced(rep: Dict) -> str:
    out = ["  --- advanced metrics (docs/EVAL.md layers 2-4) ---"]
    l2 = rep["layer2_distribution_distances"]
    out.append("  L2 distribution distances (lower=closer):")
    for name, d in l2.items():
        ad = d["anderson_darling"]
        ad_s = f"{ad:.3f}" if isinstance(ad, float) else "n/a"
        out.append(f"    {name:<14} energy={_f(d['energy'])}  W1={_f(d['wasserstein'])}  AD={ad_s}")
    l3 = rep["layer3_temporal_motifs"]
    out.append(f"  L3 temporal-motif fingerprint L1={_f(l3['l1_distance'])} "
               f"(δ={l3['delta_s']:.0f}s; 0=identical dynamics, 2=disjoint)")
    disc = rep["layer4_discriminative"]
    if disc:
        out.append(f"  L4 discriminative score={_f(disc['discriminative_score'])} "
                   f"(AUC={_f(disc['auc'])}; 0=indistinguishable=best, 1=trivially separable)")
    else:
        out.append("  L4 discriminative: skipped (sklearn missing or too few edges)")
    return "\n".join(out)


def _f(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.3f}" if isinstance(x, float) else str(x)
