"""stats.py — distributional statistics on a temporal edge stream.

A `GraphStats` is computed identically on real and synthetic graphs so the two
are directly comparable. Everything works from a minimal temporal edge list
(src, dst, t) so any real dataset with those three columns can be compared.

Metrics (chosen to match the graph-fidelity literature, BACKGROUND.md §4):
  - in/out/total degree distributions (+ Clauset–Shalizi–Newman power-law fit)
  - inter-event-time distribution (per-source gaps) + burstiness B
  - clustering coefficient (sampled), reciprocity, density
  - temporal: events-per-window series, distinct nodes/edges, repeat-edge ratio
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GraphStats:
    n_edges: int
    n_nodes: int
    distinct_pairs: int
    repeat_edge_ratio: float
    density: float
    reciprocity: float
    mean_degree: float
    # degree arrays (for KS); kept as sorted numpy arrays
    out_deg: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    in_deg: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    tot_deg: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    # power-law exponent alpha (continuous MLE) for the total-degree tail
    powerlaw_alpha: Optional[float] = None
    powerlaw_xmin: Optional[float] = None
    # inter-event-time gaps (seconds) pooled across sources, and burstiness B
    inter_event: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    burstiness_b: Optional[float] = None
    memory_m: Optional[float] = None
    # clustering coefficient (avg local, sampled)
    clustering: Optional[float] = None
    # temporal span and per-window activity (for shape comparison)
    t_min: float = 0.0
    t_max: float = 0.0
    daily_events: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    def scalar_summary(self) -> Dict[str, float]:
        """The comparable scalar fingerprint (used in the scorecard + MMD)."""
        return {
            "n_edges": float(self.n_edges),
            "n_nodes": float(self.n_nodes),
            "density": self.density,
            "reciprocity": self.reciprocity,
            "mean_degree": self.mean_degree,
            "repeat_edge_ratio": self.repeat_edge_ratio,
            "powerlaw_alpha": self.powerlaw_alpha if self.powerlaw_alpha else float("nan"),
            "burstiness_b": self.burstiness_b if self.burstiness_b is not None else float("nan"),
            "memory_m": self.memory_m if self.memory_m is not None else float("nan"),
            "clustering": self.clustering if self.clustering is not None else float("nan"),
        }


def powerlaw_mle(x: np.ndarray, xmin: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
    """Continuous power-law exponent via MLE (Clauset, Shalizi & Newman 2009):
    alpha = 1 + n / Σ ln(x_i / xmin). If xmin is None, pick the xmin that
    minimizes the KS distance between the data tail and the fitted power law
    (the CSN procedure, simplified to a scan over candidate xmins)."""
    x = x[x > 0]
    if x.size < 50:
        return None, None
    if xmin is not None:
        tail = x[x >= xmin]
        if tail.size < 10:
            return None, None
        a = 1.0 + tail.size / np.sum(np.log(tail / xmin))
        return float(a), float(xmin)
    # Scan candidate xmins (unique values, capped) and minimize KS.
    cand = np.unique(x)
    if cand.size > 60:
        cand = cand[np.linspace(0, cand.size - 1, 60).astype(int)]
    best = (None, None, math.inf)
    for xm in cand[:-5]:  # leave a few points in the tail
        tail = x[x >= xm]
        if tail.size < 10:
            continue
        a = 1.0 + tail.size / np.sum(np.log(tail / xm))
        # KS between empirical tail CDF and the fitted power-law CDF.
        ts = np.sort(tail)
        emp = np.arange(1, ts.size + 1) / ts.size
        fit = 1.0 - (ts / xm) ** (1.0 - a)
        ks = np.max(np.abs(emp - fit))
        if ks < best[2]:
            best = (float(a), float(xm), ks)
    return best[0], best[1]


def burstiness(gaps: np.ndarray) -> Optional[float]:
    """B = (σ−μ)/(σ+μ) of inter-event times (Goh & Barabási 2008)."""
    g = gaps[np.isfinite(gaps)]
    if g.size < 2:
        return None
    mu, sigma = float(np.mean(g)), float(np.std(g, ddof=1))
    if sigma + mu <= 0:
        return None
    return (sigma - mu) / (sigma + mu)


def memory(prev_gap: np.ndarray, gap: np.ndarray) -> Optional[float]:
    """Lag-1 correlation of consecutive inter-event times."""
    if prev_gap.size < 2:
        return None
    if np.std(prev_gap) == 0 or np.std(gap) == 0:
        return None
    return float(np.corrcoef(prev_gap, gap)[0, 1])


def compute_stats(
    src: np.ndarray,
    dst: np.ndarray,
    t: np.ndarray,
    *,
    clustering_sample: int = 2000,
    window_s: Optional[float] = None,
    seed: int = 0,
) -> GraphStats:
    """Compute the full comparable statistic battery from a temporal edge list.
    `t` is in seconds (or any consistent unit); window_s buckets the activity
    timeline (defaults to 1/200th of the span)."""
    rng = np.random.default_rng(seed)
    n_edges = int(src.size)
    nodes = np.unique(np.concatenate([src, dst]))
    n_nodes = int(nodes.size)

    # Degrees.
    out_deg = np.bincount(_reindex(src, nodes), minlength=n_nodes)
    in_deg = np.bincount(_reindex(dst, nodes), minlength=n_nodes)
    tot_deg = out_deg + in_deg
    mean_degree = float(np.mean(tot_deg)) if n_nodes else 0.0

    # Distinct pairs + repeat ratio + reciprocity.
    pair_keys = src.astype(np.int64) * np.int64(n_nodes if n_nodes else 1) + dst.astype(np.int64)
    # use a hash of (src,dst) to bound memory; exact for moderate sizes
    distinct_pairs = int(np.unique(_pair_hash(src, dst)).size)
    repeat_edge_ratio = 1.0 - distinct_pairs / n_edges if n_edges else 0.0
    fwd = set(zip(src.tolist(), dst.tolist()))
    recip_pairs = sum(1 for (a, b) in fwd if (b, a) in fwd)
    reciprocity = recip_pairs / max(1, len(fwd))
    density = n_edges / max(1, n_nodes * (n_nodes - 1))

    # Power law on total degree.
    pl_a, pl_xmin = powerlaw_mle(tot_deg.astype(float))

    # Inter-event times per source (sort by (src, t)).
    order = np.lexsort((t, src))
    s_sorted, t_sorted = src[order], t[order]
    gaps, prev_for_mem, gap_for_mem = _per_source_gaps(s_sorted, t_sorted)
    b = burstiness(gaps)
    m = memory(prev_for_mem, gap_for_mem)

    # Clustering (sampled local clustering on the undirected simple graph).
    clustering = _sampled_clustering(src, dst, nodes, clustering_sample, rng)

    # Temporal activity timeline.
    t_min, t_max = float(t.min()) if n_edges else 0.0, float(t.max()) if n_edges else 0.0
    span = max(1e-9, t_max - t_min)
    w = window_s if window_s else span / 200.0
    bins = np.floor((t - t_min) / max(w, 1e-9)).astype(np.int64)
    daily = np.bincount(bins - bins.min()) if n_edges else np.array([])

    return GraphStats(
        n_edges=n_edges,
        n_nodes=n_nodes,
        distinct_pairs=distinct_pairs,
        repeat_edge_ratio=repeat_edge_ratio,
        density=density,
        reciprocity=reciprocity,
        mean_degree=mean_degree,
        out_deg=np.sort(out_deg),
        in_deg=np.sort(in_deg),
        tot_deg=np.sort(tot_deg),
        powerlaw_alpha=pl_a,
        powerlaw_xmin=pl_xmin,
        inter_event=gaps,
        burstiness_b=b,
        memory_m=m,
        clustering=clustering,
        t_min=t_min,
        t_max=t_max,
        daily_events=daily.astype(float),
    )


# --- helpers ----------------------------------------------------------------
def _reindex(ids: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    return np.searchsorted(nodes, ids)


def _pair_hash(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = src.astype(np.uint64)
    b = dst.astype(np.uint64)
    h = (a * np.uint64(0x9E3779B97F4A7C15)) ^ b
    return h


def _per_source_gaps(s_sorted: np.ndarray, t_sorted: np.ndarray):
    """Inter-event gaps per source, plus consecutive (prev,gap) pairs for M."""
    if s_sorted.size < 2:
        return np.array([]), np.array([]), np.array([])
    same = s_sorted[1:] == s_sorted[:-1]
    dt = (t_sorted[1:] - t_sorted[:-1]).astype(float)
    gaps = dt[same & (dt >= 0)]
    # memory: consecutive gaps within the same source
    # build per-source gap sequences via run boundaries
    prev_list: List[float] = []
    cur_list: List[float] = []
    # iterate runs (cheap enough at validation scale; downsample if huge)
    n = s_sorted.size
    i = 0
    cap = 5_000_000
    step = max(1, n // cap)
    while i < n - 1:
        j = i
        run_gaps = []
        while j < n - 1 and s_sorted[j + 1] == s_sorted[i]:
            d = float(t_sorted[j + 1] - t_sorted[j])
            if d >= 0:
                run_gaps.append(d)
            j += 1
        for k in range(1, len(run_gaps)):
            prev_list.append(run_gaps[k - 1])
            cur_list.append(run_gaps[k])
        i = j + 1
        if len(prev_list) > cap:
            break
    return gaps[::step] if gaps.size > cap else gaps, np.array(prev_list), np.array(cur_list)


def _sampled_clustering(src, dst, nodes, sample, rng) -> Optional[float]:
    """Average local clustering over `sample` random nodes (undirected)."""
    if nodes.size < 3:
        return None
    # adjacency as a dict of sets (sampled neighborhoods)
    import collections

    adj = collections.defaultdict(set)
    # cap edges scanned to keep this O(manageable)
    m = src.size
    cap = 3_000_000
    if m > cap:
        idx = rng.choice(m, cap, replace=False)
        s, d = src[idx], dst[idx]
    else:
        s, d = src, dst
    for a, b in zip(s.tolist(), d.tolist()):
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    cand = [n for n in rng.choice(nodes, min(sample, nodes.size), replace=False) if len(adj.get(int(n), ())) >= 2]
    if not cand:
        return 0.0
    total = 0.0
    for n in cand:
        nb = list(adj[int(n)])
        k = len(nb)
        links = 0
        nbset = adj[int(n)]
        for ii in range(k):
            ai = nb[ii]
            ain = adj.get(ai, ())
            for jj in range(ii + 1, k):
                if nb[jj] in ain:
                    links += 1
        total += 2.0 * links / (k * (k - 1))
    return total / len(cand)
