"""deep_gen.py — a DEEP (learning-based) temporal-graph generative baseline.

This adds the missing "SOTA learned generator" column to the AGORA fidelity
matrix. Every other baseline in ``baselines.py`` is a formula (ER, BA, R-MAT,
config, WS); this one *trains a neural network on a real temporal edge list* and
then *samples* a new temporal graph of matched size.

Method — TIGGER-Lite (a faithful reimplementation of the generative core of
    Gupta, Manchanda, Ranu & Bagchi, "TIGGER: Scalable Generative Modelling for
    Temporal Interaction Graphs", AAAI 2022, arXiv:2203.03564).

Why a reimplementation rather than a `git clone`: the reference repo pins an old
torch + DGL stack and a bespoke data pipeline that does not run cleanly on a
CPU-only box with torch 2.x, and cloning it would make the result ir
reproducible for the SIGMOD matrix. So we reimplement the part that actually
*learns* — TIGGER's transductive temporal-random-walk language model:

  1. SAMPLE temporal random walks from the real graph. A walk is a time-
     respecting path e_1=(v0,v1,τ1), e_2=(v1,v2,τ2), … with τ1≤τ2≤…: from the
     current node we may only follow an out-edge whose timestamp is ≥ the time
     we arrived. This is exactly the temporal-walk object TIGGER models.
  2. TRAIN an LSTM to model  p(next_node, next_inter-edge-gap | walk-so-far)
     with learned per-node embeddings (the transductive setting) and a bucketed
     inter-edge-time head. The model genuinely learns the graph's transition
     structure and its timing — it is not a formula and sees no hand-coded
     degree/burstiness targets.
  3. GENERATE by autoregressively sampling fresh walks from the trained model
     and stitching their edges together into a temporal graph, until the
     requested edge budget is met. Walk start (node, time) pairs and walk
     lengths are drawn from the real empirical distributions; inter-edge gaps
     are sampled from the LSTM's predicted time-bucket and de-quantized against
     the real per-bucket gap pool, so the second-scale timing stays realistic.

CPU / runtime caveats (honesty first — see the returned report / smoke test):
  * Transductive TIGGER carries one embedding + one softmax row per node, so it
    targets small/medium graphs. Designed and validated for ≲ a few-thousand
    nodes on CPU (CollegeMsg: 1899 nodes, 59.8k edges).
  * To stay well under the ~10-minute CPU budget we cap the number of training
    walks (default 40k), max walk length (12 edges), and epochs (12). These caps
    are the only "shortcuts"; nothing about the objective is faked.
  * ``fit_generate`` inherits its node vocabulary from the TRAIN graph, so the
    generated node count ≈ the train node count (the matched-size case). If
    ``n_target`` differs it is used only to size the edge budget, not to resize
    the vocabulary.

Public API:
    fit_generate(train_src, train_dst, train_t, n_target, m_target, span, seed)
        -> (src: int64[m], dst: int64[m], t: int64[m])
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]

# Special input token for "the time-gap that led to this node is undefined"
# (used for the first two positions of every walk, whose absolute time is
# anchored from the empirical distribution rather than generated as a gap).
_START_GAP = 0  # bucket index 0 is reserved; real gap buckets are 1..NB


# --------------------------------------------------------------------------- #
# Temporal random-walk sampling
# --------------------------------------------------------------------------- #
class _TemporalGraph:
    """Out-adjacency sorted by (src, time) so a temporal walk step is a
    binary-search + uniform choice over the time-valid out-edge suffix."""

    def __init__(self, s: np.ndarray, d: np.ndarray, t: np.ndarray, num_nodes: int):
        order = np.lexsort((t, s))  # primary key src, secondary time
        self.s = s[order]
        self.d = d[order]
        self.t = t[order].astype(np.float64)
        self.num_nodes = num_nodes
        # per-source [lo, hi) ranges into the sorted arrays
        self.lo = np.searchsorted(self.s, np.arange(num_nodes), side="left")
        self.hi = np.searchsorted(self.s, np.arange(num_nodes), side="right")

    def sample_walk(self, start_idx: int, max_edges: int, rng: np.random.Generator):
        """Return (nodes, times) of a temporal walk anchored at real edge
        ``start_idx``. ``nodes`` has len = #edges+1; ``times`` has len = #edges."""
        u = int(self.s[start_idx])
        v = int(self.d[start_idx])
        tc = float(self.t[start_idx])
        nodes = [u, v]
        times = [tc]
        cur = v
        for _ in range(max_edges - 1):
            lo, hi = self.lo[cur], self.hi[cur]
            if hi <= lo:
                break
            block_t = self.t[lo:hi]
            j0 = np.searchsorted(block_t, tc, side="left")  # first out-edge with t >= tc
            if j0 >= (hi - lo):
                break
            pick = lo + int(rng.integers(j0, hi - lo))
            cur = int(self.d[pick])
            tc = float(self.t[pick])
            nodes.append(cur)
            times.append(tc)
        return nodes, times


def _sample_walks(g: _TemporalGraph, num_walks: int, max_edges: int,
                  rng: np.random.Generator):
    """Sample temporal walks starting from uniformly random real edges (so the
    start (node,time) distribution matches empirical). Returns the walks plus
    the pooled inter-edge gaps used to build time buckets."""
    m = g.s.size
    starts = rng.integers(0, m, size=num_walks)
    walks: List[Tuple[List[int], List[float]]] = []
    all_gaps: List[float] = []
    for si in starts:
        nodes, times = g.sample_walk(int(si), max_edges, rng)
        if len(nodes) < 2:
            continue
        walks.append((nodes, times))
        # inter-edge gaps τ_k - τ_{k-1} for k>=2 (the quantities the model predicts)
        for k in range(1, len(times)):
            gap = times[k] - times[k - 1]
            if gap >= 0:
                all_gaps.append(gap)
    return walks, np.asarray(all_gaps, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Inter-edge time bucketing (quantile buckets on log1p(gap), de-quantized by
# sampling from the real per-bucket value pool)
# --------------------------------------------------------------------------- #
class _TimeBuckets:
    def __init__(self, gaps: np.ndarray, nbuckets: int, rng: np.random.Generator):
        self.nbuckets = nbuckets
        gaps = gaps[np.isfinite(gaps) & (gaps >= 0)]
        if gaps.size < nbuckets * 4:
            # degenerate: single fallback bucket
            self.edges = np.array([0.0, np.inf])
            self.pools = [gaps if gaps.size else np.array([0.0])]
            self.nbuckets = 1
            return
        lg = np.log1p(gaps)
        qs = np.linspace(0.0, 1.0, nbuckets + 1)
        edges = np.unique(np.quantile(lg, qs))
        # bucketize
        idx = np.clip(np.searchsorted(edges, lg, side="right") - 1, 0, edges.size - 2)
        self.nbuckets = edges.size - 1
        self.edges = edges
        self.pools = []
        for b in range(self.nbuckets):
            pool = gaps[idx == b]
            self.pools.append(pool if pool.size else np.array([np.expm1(edges[b])]))

    def bucket_of(self, gaps: np.ndarray) -> np.ndarray:
        """Map raw gaps -> bucket index in [0, nbuckets)."""
        if self.nbuckets == 1:
            return np.zeros(gaps.shape, dtype=np.int64)
        lg = np.log1p(np.maximum(gaps, 0.0))
        return np.clip(np.searchsorted(self.edges, lg, side="right") - 1,
                       0, self.nbuckets - 1).astype(np.int64)

    def sample_value(self, buckets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """De-quantize bucket indices -> concrete non-negative gap seconds by
        drawing from that bucket's real value pool."""
        out = np.empty(buckets.shape, dtype=np.float64)
        for b in np.unique(buckets):
            mask = buckets == b
            pool = self.pools[int(b)]
            out[mask] = pool[rng.integers(0, pool.size, size=int(mask.sum()))]
        return out


# --------------------------------------------------------------------------- #
# The walk language model
# --------------------------------------------------------------------------- #
def _build_model(num_nodes: int, nbuckets: int, emb_dim: int, time_emb_dim: int,
                 hidden: int):
    import torch
    from torch import nn

    class WalkLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.node_emb = nn.Embedding(num_nodes, emb_dim)
            # +1 gap token: index 0 = START (undefined), 1..nbuckets = real buckets
            self.gap_emb = nn.Embedding(nbuckets + 1, time_emb_dim)
            self.lstm = nn.LSTM(emb_dim + time_emb_dim, hidden, batch_first=True)
            self.node_head = nn.Linear(hidden, num_nodes)
            self.gap_head = nn.Linear(hidden, nbuckets)

        def forward(self, node_ids, gap_toks, state=None):
            x = torch.cat([self.node_emb(node_ids), self.gap_emb(gap_toks)], dim=-1)
            h, state = self.lstm(x, state)
            return self.node_head(h), self.gap_head(h), state

    return WalkLSTM()


# --------------------------------------------------------------------------- #
# Training-tensor construction
# --------------------------------------------------------------------------- #
def _walks_to_tensors(walks, tb: _TimeBuckets, max_edges: int):
    """Pack walks into padded teacher-forcing tensors.

    Per walk with nodes [v0..v_ell] (ell edges) and edge times [τ1..τ_ell]
    stored 0-indexed as times[k-1]=τ_k:
      inputs at position j (0..ell-1): node v_j=nodes[j], gap-token gin[j]
        gin[0]=gin[1]=START; gin[j>=2]=bucket(τ_j-τ_{j-1})+1
                                     =bucket(times[j-1]-times[j-2])+1
      targets at position j: next node v_{j+1}=nodes[j+1]; next gap bucket
        gap target valid only for j in 1..ell-1 = bucket(τ_{j+1}-τ_j)
                                                =bucket(times[j]-times[j-1])
    """
    N = len(walks)
    T = max_edges  # number of unrolled positions (predicting v_1..v_T)
    in_node = np.zeros((N, T), dtype=np.int64)
    in_gap = np.zeros((N, T), dtype=np.int64)         # START==0 by default
    tgt_node = np.zeros((N, T), dtype=np.int64)
    tgt_gap = np.zeros((N, T), dtype=np.int64)
    node_mask = np.zeros((N, T), dtype=np.float32)
    gap_mask = np.zeros((N, T), dtype=np.float32)
    lengths = np.zeros(N, dtype=np.int64)

    for i, (nodes, times) in enumerate(walks):
        ell = len(nodes) - 1                # #edges
        ell = min(ell, T)
        lengths[i] = ell
        times = np.asarray(times, dtype=np.float64)
        # inter-edge gaps of edges reaching v_j: gapin[j] for j>=2
        for j in range(ell):
            in_node[i, j] = nodes[j]
            if j >= 2:
                in_gap[i, j] = tb.bucket_of(np.array([times[j - 1] - times[j - 2]]))[0] + 1
            else:
                in_gap[i, j] = _START_GAP  # 0
            tgt_node[i, j] = nodes[j + 1]
            node_mask[i, j] = 1.0
            if j >= 1:
                tgt_gap[i, j] = tb.bucket_of(np.array([times[j] - times[j - 1]]))[0]
                gap_mask[i, j] = 1.0
    return in_node, in_gap, tgt_node, tgt_gap, node_mask, gap_mask, lengths


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def fit_generate(
    train_src: np.ndarray,
    train_dst: np.ndarray,
    train_t: np.ndarray,
    n_target: int,
    m_target: int,
    span: float,
    seed: int = 0,
    *,
    num_walks: int = 40_000,
    max_edges: int = 12,
    epochs: int = 12,
    batch_size: int = 256,
    hidden: int = 256,
    emb_dim: int = 128,
    time_emb_dim: int = 16,
    nbuckets: int = 64,
    lr: float = 1e-3,
    temperature: float = 1.0,
    gen_batch: int = 1024,
    verbose: bool = True,
) -> Arrays:
    """Train TIGGER-Lite on (train_src, train_dst, train_t) and sample a
    synthetic temporal graph of ~n_target nodes / ~m_target edges over ~span
    seconds. Returns (src int64, dst int64, t int64) in the ORIGINAL node-id
    space of the training graph.

    All work is on CPU. See the module docstring for the caps that keep this
    under the ~10-minute budget.
    """
    import torch
    from torch import nn

    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    train_src = np.asarray(train_src)
    train_dst = np.asarray(train_dst)
    train_t = np.asarray(train_t, dtype=np.float64)

    # Reindex node ids to a contiguous 0..N-1 vocabulary; remember originals so
    # we can emit ids in the same universe as the real graph.
    nodes, inv = np.unique(np.concatenate([train_src, train_dst]), return_inverse=True)
    N = int(nodes.size)
    m = train_src.size
    s = inv[:m].astype(np.int64)
    d = inv[m:].astype(np.int64)
    t = train_t
    t_min = float(t.min())
    t_max = float(t.min() + span) if span > 0 else float(t.max())

    if verbose:
        print(f"[deep_gen] TIGGER-Lite: {N} nodes, {m} edges; "
              f"target {n_target} nodes / {m_target} edges over {span:.0f}s")

    # 1) sample temporal walks + build time buckets
    g = _TemporalGraph(s, d, t, N)
    walks, all_gaps = _sample_walks(g, num_walks, max_edges, rng)
    if len(walks) < 100:
        raise RuntimeError("too few temporal walks sampled; graph may be too sparse")
    tb = _TimeBuckets(all_gaps, nbuckets, rng)
    nb = tb.nbuckets
    walk_lengths_emp = np.array([len(w[0]) - 1 for w in walks], dtype=np.int64)
    if verbose:
        print(f"[deep_gen] sampled {len(walks)} walks "
              f"(mean len {walk_lengths_emp.mean():.1f} edges), "
              f"{nb} time buckets, {time.time()-t0:.1f}s")

    # 2) pack tensors
    in_node, in_gap, tgt_node, tgt_gap, node_mask, gap_mask, _ = _walks_to_tensors(
        walks, tb, max_edges)
    in_node_t = torch.from_numpy(in_node)
    in_gap_t = torch.from_numpy(in_gap)
    tgt_node_t = torch.from_numpy(tgt_node)
    tgt_gap_t = torch.from_numpy(tgt_gap)
    node_mask_t = torch.from_numpy(node_mask)
    gap_mask_t = torch.from_numpy(gap_mask)

    model = _build_model(N, nb, emb_dim, time_emb_dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(reduction="none")

    # 3) train
    n_seq = in_node_t.size(0)
    model.train()
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n_seq)
        tot_loss = 0.0
        for b in range(0, n_seq, batch_size):
            idx = perm[b:b + batch_size]
            opt.zero_grad()
            node_logits, gap_logits, _ = model(in_node_t[idx], in_gap_t[idx])
            B, T, _ = node_logits.shape
            nl = ce(node_logits.reshape(B * T, -1), tgt_node_t[idx].reshape(-1))
            nl = (nl * node_mask_t[idx].reshape(-1)).sum() / node_mask_t[idx].sum().clamp_min(1)
            gl = ce(gap_logits.reshape(B * T, -1), tgt_gap_t[idx].reshape(-1))
            gm = gap_mask_t[idx].reshape(-1)
            gl = (gl * gm).sum() / gm.sum().clamp_min(1)
            loss = nl + gl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.detach()) * idx.numel()
        if verbose:
            print(f"[deep_gen]   epoch {ep:2d}/{epochs}  loss={tot_loss/n_seq:.4f}  "
                  f"({time.time()-t0:.1f}s)")

    # 4) generate walks autoregressively and stitch edges
    model.eval()
    gen_s: List[np.ndarray] = []
    gen_d: List[np.ndarray] = []
    gen_t: List[np.ndarray] = []
    n_emitted = 0
    start_edge_pool = np.arange(m)

    with torch.no_grad():
        while n_emitted < m_target:
            B = min(gen_batch, max(64, (m_target - n_emitted) // max_edges + 64))
            # start (node, time) from real edges; target walk length per walk
            si = rng.choice(start_edge_pool, size=B)
            cur_node = torch.from_numpy(s[si].astype(np.int64))
            cur_time = t[si].astype(np.float64).copy()
            cur_gap_tok = torch.zeros(B, dtype=torch.long)     # START
            tgt_len = rng.choice(walk_lengths_emp, size=B)
            alive = np.ones(B, dtype=bool)
            state = None
            prev_node = cur_node.numpy().copy()

            for step in range(max_edges):
                node_logits, gap_logits, state = model(
                    cur_node[:, None], cur_gap_tok[:, None], state)
                node_logits = node_logits[:, 0, :] / max(temperature, 1e-6)
                gap_logits = gap_logits[:, 0, :]
                nxt = torch.multinomial(torch.softmax(node_logits, dim=-1), 1)[:, 0].numpy()
                gbuck = torch.multinomial(torch.softmax(gap_logits, dim=-1), 1)[:, 0].numpy()

                if step == 0:
                    # first edge: time is the anchored absolute start (no gap)
                    new_time = cur_time.copy()
                else:
                    dt = tb.sample_value(gbuck, rng)
                    new_time = cur_time + dt

                emit = alive & (new_time <= t_max) & (step < tgt_len)
                if emit.any():
                    gen_s.append(prev_node[emit].copy())
                    gen_d.append(nxt[emit].copy())
                    gen_t.append(new_time[emit].copy())
                    n_emitted += int(emit.sum())

                # advance state for still-alive walks
                alive = alive & (new_time <= t_max) & (step + 1 < tgt_len)
                cur_time = new_time
                prev_node = nxt.copy()
                cur_node = torch.from_numpy(nxt.astype(np.int64))
                # input gap token for next step: START for step0->1, else bucket+1
                cur_gap_tok = torch.from_numpy(
                    np.where(step == 0, 0, gbuck + 1).astype(np.int64))
                if not alive.any():
                    break

    if not gen_s:
        raise RuntimeError("generation produced no edges")
    out_s = np.concatenate(gen_s)
    out_d = np.concatenate(gen_d)
    out_t = np.concatenate(gen_t)

    # truncate / pad-by-resampling to exactly m_target edges, keep chronological
    if out_s.size > m_target:
        keep = rng.choice(out_s.size, m_target, replace=False)
        out_s, out_d, out_t = out_s[keep], out_d[keep], out_t[keep]
    order = np.argsort(out_t, kind="stable")
    out_s, out_d, out_t = out_s[order], out_d[order], out_t[order]

    # map contiguous ids back to original node universe
    src = nodes[out_s].astype(np.int64)
    dst = nodes[out_d].astype(np.int64)
    tt = np.clip(out_t, t_min, t_max).astype(np.int64)

    if verbose:
        print(f"[deep_gen] generated {src.size} edges / "
              f"{np.unique(np.concatenate([src,dst])).size} nodes in "
              f"{time.time()-t0:.1f}s total")
    return src, dst, tt


# --------------------------------------------------------------------------- #
# Smoke test: real fidelity on CollegeMsg (run as `python -m agora_eval.deep_gen`)
# --------------------------------------------------------------------------- #
def _smoke_test(path: Optional[str] = None, cap: int = 60_000, seed: int = 0):
    import time as _time

    try:
        from .load import load_real
        from .stats import compute_stats
        from .compare import compare, format_scorecard
    except ImportError:  # run as a plain script from inside the package dir
        from load import load_real
        from stats import compute_stats
        from compare import compare, format_scorecard

    path = path or "realdata/snap-collegemsg/CollegeMsg.txt"
    s, d, t = load_real(path)
    if s.size > cap:  # keep the earliest `cap` events chronologically
        order = np.argsort(t, kind="stable")[:cap]
        s, d, t = s[order], d[order], t[order]

    n = int(np.unique(np.concatenate([s, d])).size)
    m = int(s.size)
    span = float(t.max() - t.min())
    print(f"=== CollegeMsg (real): {m} edges, {n} nodes, span {span/86400:.1f} days ===")

    wall0 = _time.time()
    gs, gd, gt = fit_generate(s, d, t, n_target=n, m_target=m, span=span, seed=seed)
    train_wall = _time.time() - wall0

    real_stats = compute_stats(s, d, t)
    synth_stats = compute_stats(gs, gd, gt)
    res = compare(real_stats, synth_stats)

    print(format_scorecard(res, "CollegeMsg", "TIGGER-Lite (deep, learned)"))
    print(f"\n>>> REAL MEASURED fidelity_score = {res['fidelity_score']:.4f}")
    print(f">>> wall-clock train+generate     = {train_wall:.1f} s "
          f"({train_wall/60:.2f} min) on CPU")
    return res, train_wall


if __name__ == "__main__":
    _smoke_test()
