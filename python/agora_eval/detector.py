"""detector.py — a real anomaly detector over a AGORA output, for the L6
difficulty-vs-detectability experiment (docs/EVAL.md, the headline).

AGORA emits exact ground-truth labels, so we can train a supervised detector on a
generated graph and measure how detectable its anomalies are. Sweeping the
difficulty knob delta and re-measuring gives the difficulty->detector-AUC curve
that proves the difficulty axis is real, meaningful, and controllable --- which
no other generator can offer.

The detector is a standard gradient-boosted classifier over per-edge features
that combine STRUCTURE (endpoint degrees), ATTRIBUTES (amount, channel, type,
cross-border), and TIMING (hour, inter-event gap, recurrence). It uses no node
IDs (those would leak) and no label-derived features.
"""
from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _edges_path(agora_dir: str) -> str:
    hits = sorted(glob.glob(str(Path(agora_dir) / "edges*.csv")))
    if not hits:
        raise FileNotFoundError(f"no edges*.csv in {agora_dir}")
    return hits[0]


_STRUCT_FEATS = {"out_deg_src", "in_deg_dst", "tot_deg_src", "tot_deg_dst"}


def detect_auc(agora_dir: str, seed: int = 0,
               feature_set: str = "all") -> Optional[Dict[str, float]]:
    """Train a supervised detector on a AGORA output; return ROC-AUC and AP.
    feature_set: 'all' | 'attr' (drop structural degree feats) | 'struct'
    (degree feats only). Contrasting 'all' vs 'attr' isolates how much of the
    difficulty knob acts on attributes vs (un-camouflaged) structure."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.model_selection import train_test_split
    except Exception:  # noqa: BLE001
        return None

    df = pd.read_csv(_edges_path(agora_dir))
    if "label" not in df.columns:
        return None
    y = (df["label"].astype(str) != "normal").astype(int).to_numpy()
    n_anom = int(y.sum())
    if n_anom < 50 or n_anom == y.size:
        return {"roc_auc": float("nan"), "ap": float("nan"),
                "n_anom": n_anom, "n_total": int(y.size)}

    src = df["src"].to_numpy()
    dst = df["dst"].to_numpy()
    t = df["t"].to_numpy(dtype=float)
    both = np.concatenate([src, dst])
    nodes, inv = np.unique(both, return_inverse=True)
    s = inv[: src.size]
    d = inv[src.size:]
    N = nodes.size
    outdeg = np.bincount(s, minlength=N).astype(float)
    indeg = np.bincount(d, minlength=N).astype(float)
    totdeg = outdeg + indeg

    # per-source inter-event gap
    order = np.lexsort((t, s))
    s_sorted = s[order]
    t_sorted = t[order]
    same = np.zeros(s.size, dtype=bool)
    same[1:] = s_sorted[1:] == s_sorted[:-1]
    dt = np.empty(s.size)
    dt[0] = np.nan
    dt[1:] = t_sorted[1:] - t_sorted[:-1]
    gap = np.full(s.size, np.nan)
    gap[order] = np.where(same, dt, np.nan)

    tod = (t % 86400.0) / 3600.0
    feats = {
        "out_deg_src": np.log1p(outdeg[s]),
        "in_deg_dst": np.log1p(indeg[d]),
        "tot_deg_src": np.log1p(totdeg[s]),
        "tot_deg_dst": np.log1p(totdeg[d]),
        "gap_log": np.log1p(np.nan_to_num(gap, nan=0.0)),
        "has_gap": np.isfinite(gap).astype(float),
        "hour_sin": np.sin(2 * np.pi * tod / 24.0),
        "hour_cos": np.cos(2 * np.pi * tod / 24.0),
    }
    # numeric + categorical attribute columns present on the edge
    for col in df.columns:
        if col in ("src", "dst", "t", "label", "anomaly_id"):
            continue
        c = df[col]
        if pd.api.types.is_numeric_dtype(c):
            feats[col] = np.log1p(np.abs(c.to_numpy(dtype=float)))
        else:
            feats[col] = c.astype("category").cat.codes.to_numpy(dtype=float)
    if feature_set == "attr":
        feats = {k: v for k, v in feats.items() if k not in _STRUCT_FEATS}
    elif feature_set == "struct":
        feats = {k: v for k, v in feats.items() if k in _STRUCT_FEATS}
    X = np.column_stack(list(feats.values()))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(yte, p)),
        "ap": float(average_precision_score(yte, p)),
        "n_anom": n_anom, "n_total": int(y.size),
    }
