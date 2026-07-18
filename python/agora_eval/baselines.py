"""baselines.py — compare AGORA against classical graph generators.

For a full research paper we must show AGORA next to the generators it is
compared to (and accused of being): Erdos-Renyi, Barabasi-Albert (the "isn't
this just BA?" baseline), R-MAT, the configuration model (matches the real
degree sequence exactly), and Watts-Strogatz small-world.

None of these produce time, attributes, or labels. To evaluate them at all as
*temporal* graphs we give them the naive treatment a practitioner would: assign
uniform-random timestamps over the real span. That is exactly the point --- it
exposes how far bare topology (even topology that matches the degree
distribution) is from a realistic temporal graph, which AGORA closes.

Run:  python3 -m agora_eval baselines --real <path> --synth <agora_out_dir> [cols]
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from . import compare as cmp
from . import metrics_advanced as adv
from . import stats as st

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Classical generators -> (src, dst, t). Timestamps are uniform over the span.
# --------------------------------------------------------------------------- #
def _times(m: int, span: float, seed: int) -> np.ndarray:
    return np.sort(np.random.default_rng(seed).uniform(0.0, max(span, 1.0), m))


def _orient(u: np.ndarray, v: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    flip = np.random.default_rng(seed).random(u.size) < 0.5
    s = np.where(flip, v, u)
    d = np.where(flip, u, v)
    return s, d


def gen_er(n: int, m: int, span: float, seed: int) -> Arrays:
    import networkx as nx
    g = nx.gnm_random_graph(n, m, seed=seed, directed=True)
    e = np.array(list(g.edges()), dtype=np.int64)
    if e.size == 0:
        return np.array([]), np.array([]), np.array([])
    return e[:, 0], e[:, 1], _times(e.shape[0], span, seed)


def gen_ba(n: int, m_edges: int, span: float, seed: int) -> Arrays:
    import networkx as nx
    m = max(1, round(m_edges / max(1, n)))
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    e = np.array(list(g.edges()), dtype=np.int64)
    s, d = _orient(e[:, 0], e[:, 1], seed)
    return s, d, _times(s.size, span, seed)


def gen_ws(n: int, m_edges: int, span: float, seed: int) -> Arrays:
    import networkx as nx
    k = max(2, round(2 * m_edges / max(1, n)))
    k += k % 2  # must be even
    g = nx.watts_strogatz_graph(n, k, 0.1, seed=seed)
    e = np.array(list(g.edges()), dtype=np.int64)
    s, d = _orient(e[:, 0], e[:, 1], seed)
    return s, d, _times(s.size, span, seed)


def gen_rmat(n: int, m: int, span: float, seed: int,
             a: float = 0.45, b: float = 0.22, c: float = 0.22) -> Arrays:
    """Compact R-MAT: drop m directed edges into a 2^k x 2^k adjacency."""
    d = 1.0 - a - b - c
    k = int(np.ceil(np.log2(max(2, n))))
    rng = np.random.default_rng(seed)
    src = np.zeros(m, dtype=np.int64)
    dst = np.zeros(m, dtype=np.int64)
    probs = np.array([a, b, c, d])
    for level in range(k):
        r = rng.choice(4, size=m, p=probs)
        bit = 1 << level
        src |= np.where((r == 2) | (r == 3), bit, 0)
        dst |= np.where((r == 1) | (r == 3), bit, 0)
    N = 1 << k
    src %= N
    dst %= N
    return src, dst, _times(m, span, seed)


def gen_config(real: Arrays, span: float, seed: int) -> Arrays:
    """Directed configuration model on the REAL degree sequence: matches degree
    EXACTLY, destroys everything else (the sharpest 'degree isn't enough' baseline)."""
    import networkx as nx
    rs, rd, _ = real
    nodes, inv = np.unique(np.concatenate([rs, rd]), return_inverse=True)
    n = nodes.size
    s = inv[: rs.size]
    d = inv[rs.size:]
    outdeg = np.bincount(s, minlength=n)
    indeg = np.bincount(d, minlength=n)
    g = nx.directed_configuration_model(indeg.tolist(), outdeg.tolist(), seed=seed)
    e = np.array(list(g.edges()), dtype=np.int64)
    if e.size == 0:
        return np.array([]), np.array([]), np.array([])
    return e[:, 0], e[:, 1], _times(e.shape[0], span, seed)


def _empty() -> Arrays:
    z = np.array([], dtype=np.int64)
    return z, z, z.astype(float)


def gen_chunglu(real: Arrays, span: float, seed: int) -> Arrays:
    """Chung--Lu expected-degree model: edge prob p_ij ~ w_i w_j / 2m, using the
    real total-degree sequence as expected degrees. A soft configuration model."""
    import networkx as nx
    rs, rd, _ = real
    nodes, inv = np.unique(np.concatenate([rs, rd]), return_inverse=True)
    n = nodes.size
    s, d = inv[: rs.size], inv[rs.size:]
    deg = (np.bincount(s, minlength=n) + np.bincount(d, minlength=n)).astype(float)
    try:
        g = nx.expected_degree_graph(deg.tolist(), seed=seed, selfloops=False)
    except Exception:
        return _empty()
    e = np.array(list(g.edges()), dtype=np.int64)
    if e.size == 0:
        return _empty()
    su, du = _orient(e[:, 0], e[:, 1], seed)
    return su, du, _times(su.size, span, seed)


def gen_kronecker(n: int, m: int, span: float, seed: int,
                  theta=(0.99, 0.54, 0.54, 0.37)) -> Arrays:
    """Stochastic Kronecker graph (Leskovec et al., JMLR 2010) with the canonical
    fitted initiator matrix; m edges dropped by recursive quadrant descent."""
    a, b, c, d = theta
    k = int(np.ceil(np.log2(max(2, n))))
    rng = np.random.default_rng(seed)
    probs = np.array([a, b, c, d]) / (a + b + c + d)
    src = np.zeros(m, dtype=np.int64)
    dst = np.zeros(m, dtype=np.int64)
    for level in range(k):
        r = rng.choice(4, size=m, p=probs)
        bit = 1 << level
        src |= np.where((r == 2) | (r == 3), bit, 0)
        dst |= np.where((r == 1) | (r == 3), bit, 0)
    N = 1 << k
    return src % N, dst % N, _times(m, span, seed)


def gen_dcsbm(real: Arrays, span: float, seed: int, k_blocks: int = 4) -> Arrays:
    """Degree-corrected stochastic block model (Karrer--Newman, PRE 2011): random
    blocks, block-block edge counts and within-block degree propensity from the
    real graph, then m edges sampled from that generative model."""
    rs, rd, _ = real
    nodes, inv = np.unique(np.concatenate([rs, rd]), return_inverse=True)
    n = nodes.size
    s, d = inv[: rs.size], inv[rs.size:]
    m = int(s.size)
    if n < k_blocks:
        return _empty()
    deg = (np.bincount(s, minlength=n) + np.bincount(d, minlength=n)).astype(float)
    rng = np.random.default_rng(seed)
    block = rng.integers(0, k_blocks, n)
    omega = np.zeros((k_blocks, k_blocks))
    np.add.at(omega, (block[s], block[d]), 1.0)
    if omega.sum() == 0:
        return _empty()
    omega_flat = omega.ravel() / omega.sum()
    block_deg = np.zeros(k_blocks)
    np.add.at(block_deg, block, deg)
    theta = deg / np.maximum(block_deg[block], 1e-9)
    nodes_in = [np.where(block == b)[0] for b in range(k_blocks)]
    prob_in = []
    for b in range(k_blocks):
        w = theta[nodes_in[b]]
        prob_in.append(w / max(w.sum(), 1e-12) if w.size else w)
    pair = rng.choice(k_blocks * k_blocks, size=m, p=omega_flat)
    ab, bb = pair // k_blocks, pair % k_blocks
    src = np.zeros(m, dtype=np.int64)
    dst = np.zeros(m, dtype=np.int64)
    for b in range(k_blocks):
        if not nodes_in[b].size:
            continue
        ma = ab == b
        if ma.any():
            src[ma] = rng.choice(nodes_in[b], size=int(ma.sum()), p=prob_in[b])
        mb = bb == b
        if mb.any():
            dst[mb] = rng.choice(nodes_in[b], size=int(mb.sum()), p=prob_in[b])
    return src, dst, _times(m, span, seed)


def gen_rdpg(real: Arrays, span: float, seed: int, dim: int = 8) -> Arrays:
    """Random Dot Product Graph: adjacency spectral embedding of the real graph
    (latent x_i), then m edges sampled with dst prob ~ max(<x_src, x_j>, 0). A
    latent-space generator that captures geometry the degree sequence alone misses."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    rs, rd, _ = real
    nodes, inv = np.unique(np.concatenate([rs, rd]), return_inverse=True)
    n = nodes.size
    s, d = inv[: rs.size], inv[rs.size:]
    m = int(s.size)
    dim = int(min(dim, n - 2))
    if dim < 1:
        return _empty()
    A = sp.csr_matrix((np.ones(m), (s, d)), shape=(n, n))
    A = (A + A.T).astype(float)
    try:
        u, sv, _vt = spla.svds(A, k=dim)
    except Exception:
        return _empty()
    X = u * np.sqrt(np.maximum(sv, 0.0))  # n x dim latent positions
    Xp = np.maximum(X, 0.0)               # non-negative part -> mixed-membership
    rng = np.random.default_rng(seed)
    outdeg = np.bincount(s, minlength=n).astype(float)
    wsrc = outdeg / outdeg.sum() if outdeg.sum() else np.full(n, 1.0 / n)
    src = rng.choice(n, size=m, p=wsrc)
    # Scalable dst sampling: p(j|i) ~ <x_i,x_j> approximated as a low-rank mixture --
    # pick a latent dim k ~ Xp[i], then dst ~ that dim's node distribution. O(m*dim).
    col_sum = Xp.sum(0)
    valid = col_sum > 0
    dim_dst = np.zeros_like(Xp)
    dim_dst[:, valid] = Xp[:, valid] / col_sum[valid]
    row = Xp[src]
    rsum = row.sum(1, keepdims=True)
    rowp = np.where(rsum > 0, row / rsum, 1.0 / dim)
    k = (np.cumsum(rowp, 1) < rng.random((m, 1))).sum(1).clip(0, dim - 1)
    dst = np.empty(m, dtype=np.int64)
    for kk in range(dim):
        mask = k == kk
        if not mask.any():
            continue
        if valid[kk]:
            dst[mask] = rng.choice(n, size=int(mask.sum()), p=dim_dst[:, kk])
        else:
            dst[mask] = rng.integers(0, n, int(mask.sum()))
    return src.astype(np.int64), dst, _times(m, span, seed)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _row(name: str, arr: Arrays, real_stats, has_sem: bool,
         real_arr: Arrays) -> Dict:
    gs = st.compute_stats(*arr)
    res = cmp.compare(real_stats, gs)
    disc = adv.discriminative_score(real_arr, arr)
    fr = adv.temporal_motifs_2node(*real_arr, delta_s=3600.0)
    fm = adv.temporal_motifs_2node(*arr, delta_s=3600.0)
    return {
        "name": name, "edges": gs.n_edges, "nodes": gs.n_nodes,
        "mean_deg": gs.mean_degree, "clustering": gs.clustering,
        "reciprocity": gs.reciprocity, "burstiness": gs.burstiness_b,
        "alpha": gs.powerlaw_alpha, "fidelity": res["fidelity_score"],
        "motif_l1": adv.motif_l1(fr, fm),
        "disc": disc["discriminative_score"] if disc else float("nan"),
        "time": True, "attrs": has_sem, "labels": has_sem,
    }


def compare_baselines(real: Arrays, agora: Arrays, seed: int = 42) -> List[Dict]:
    rs, rd, rt = real
    n = int(np.unique(np.concatenate([rs, rd])).size)
    m = int(rs.size)
    span = float(rt.max() - rt.min()) if rt.size else 1.0
    real_stats = st.compute_stats(*real)

    rows = [{
        "name": "real (reference)", "edges": m, "nodes": n,
        "mean_deg": real_stats.mean_degree, "clustering": real_stats.clustering,
        "reciprocity": real_stats.reciprocity, "burstiness": real_stats.burstiness_b,
        "alpha": real_stats.powerlaw_alpha, "fidelity": 1.0, "motif_l1": 0.0,
        "disc": 0.0, "time": True, "attrs": True, "labels": True,
    }]
    rows.append(_row("AGORA (ours)", agora, real_stats, True, real))
    rows.append(_row("Erdos-Renyi", gen_er(n, m, span, seed), real_stats, False, real))
    rows.append(_row("Barabasi-Albert", gen_ba(n, m, span, seed), real_stats, False, real))
    rows.append(_row("R-MAT", gen_rmat(n, m, span, seed), real_stats, False, real))
    rows.append(_row("config model", gen_config(real, span, seed), real_stats, False, real))
    rows.append(_row("Watts-Strogatz", gen_ws(n, m, span, seed), real_stats, False, real))
    return rows


def format_table(rows: List[Dict], real_name: str) -> str:
    def f(x, p=3):
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "  n/a"
        return f"{x:.{p}f}" if isinstance(x, float) else str(x)

    out = [f"=== generator comparison vs {real_name} (real) ===",
           f"{'generator':<18}{'edges':>8}{'m.deg':>7}{'clust':>7}{'recip':>7}"
           f"{'burst':>7}{'alpha':>7}{'motifL1':>8}{'disc':>6}{'fidel':>7}  T A L"]
    for r in rows:
        caps = f"  {'Y' if r['time'] else '.'} {'Y' if r['attrs'] else '.'} {'Y' if r['labels'] else '.'}"
        out.append(
            f"{r['name']:<18}{r['edges']:>8}{f(r['mean_deg'],1):>7}{f(r['clustering']):>7}"
            f"{f(r['reciprocity']):>7}{f(r['burstiness']):>7}{f(r['alpha'],2):>7}"
            f"{f(r['motif_l1']):>8}{f(r['disc']):>6}{f(r['fidelity']):>7}{caps}")
    out.append("  T=temporal A=attributes L=anomaly-labels;  disc/motifL1/fidel are vs real")
    return "\n".join(out)
