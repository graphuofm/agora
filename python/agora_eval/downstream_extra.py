"""downstream_extra.py — additional downstream link-prediction evaluators that
corroborate (or complicate) the TGN finding from ``tgn_eval.py``.

Same task/metric/splits as ``tgn_eval.run_tgn`` (future temporal link prediction,
chronological 70/15/15 split, 1 random-node negative per positive, per-batch
test AP + ROC-AUC averaged over batches, batch_size=200, msg = single zero
feature). We deliberately reuse PyG's ``TemporalData.train_val_test_split`` and
``TemporalDataLoader`` so the split boundaries, batch structure and negative
sampling are byte-for-byte the same convention as the TGN harness. This file
does NOT modify tgn_eval.py.

Two evaluators:

(A) EdgeBank (Poursafaei et al., NeurIPS 2022) — the standard non-learning
    memorization baseline. Predict (u,v) positive iff the directed pair was seen
    in history. Two variants:
      * EdgeBank_inf : unlimited memory (any pair ever seen so far).
      * EdgeBank_tw  : only pairs seen within a recent time window W = 0.15*span.
    Instant, no training. It streams predict-then-update (memory grows through
    val/test), mirroring TGN's memory.update_state during eval.

(B) GraphMixer (Cong et al., ICLR 2023) — a compact learned temporal model:
    a FIXED (non-trainable) cos time-encoding of a node's K most-recent 1-hop
    neighbor timestamps, an MLP-Mixer over those K tokens (link-encoder), plus a
    mean-pooled learnable node-id embedding (node-encoder). CPU-tractable: each
    graph is subsampled to ~40k edges and trained a few epochs. Because edges
    carry no attributes (matching TGN's zero message), the only learnable
    temporal signal is the recurrence/timing structure of neighbors — exactly
    what real / AGORA have and what BA (random timestamps) lacks.

Run:  PYTHONPATH=python python3 -m agora_eval.downstream_extra
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]


# --------------------------------------------------------------------------- #
# Shared data prep — identical convention to tgn_eval.run_tgn.
# --------------------------------------------------------------------------- #
def _build_temporal_data(src: np.ndarray, dst: np.ndarray, t: np.ndarray):
    """Contiguous node ids + chronological order + zero message, exactly as
    tgn_eval does, then PyG's 70/15/15 chronological split."""
    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n = src.size
    s = torch.tensor(inv[:n], dtype=torch.long)
    d = torch.tensor(inv[n:], dtype=torch.long)
    order = torch.tensor(np.argsort(t, kind="stable"))
    ts = torch.tensor(t[order.numpy()], dtype=torch.long)
    s, d = s[order], d[order]
    num_nodes = int(nodes.size)
    msg = torch.zeros((n, 1), dtype=torch.float)
    data = TemporalData(src=s, dst=d, t=ts, msg=msg)
    train, val, test = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)
    return data, train, val, test, num_nodes


def _subsample(src, dst, t, k, seed):
    """Keep k edges (chronological order preserved) for CPU-tractable training.
    Applied identically to every dataset for fairness."""
    if src.size <= k:
        return src, dst, t
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(src.size, size=k, replace=False))
    return src[idx], dst[idx], t[idx]


# --------------------------------------------------------------------------- #
# (A) EdgeBank
# --------------------------------------------------------------------------- #
def run_edgebank(src: np.ndarray, dst: np.ndarray, t: np.ndarray,
                 seed: int = 0, batch_size: int = 200,
                 window_ratio: float = 0.15) -> Dict[str, float]:
    torch.manual_seed(seed)
    data, train, val, test, num_nodes = _build_temporal_data(src, dst, t)

    span = float(data.t.max() - data.t.min())
    window = window_ratio * span

    # Seed memory from the full training history.
    tr_src = train.src.numpy(); tr_dst = train.dst.numpy(); tr_t = train.t.numpy()
    mem_inf = set(zip(tr_src.tolist(), tr_dst.tolist()))
    last_seen: Dict[Tuple[int, int], float] = {}
    for u, v, tt in zip(tr_src.tolist(), tr_dst.tolist(), tr_t.tolist()):
        last_seen[(u, v)] = float(tt)

    def _score_stream(loader):
        aps_i, aucs_i, aps_w, aucs_w = [], [], [], []
        for batch in TemporalDataLoader(loader, batch_size=batch_size):
            bsrc = batch.src.numpy(); bdst = batch.dst.numpy(); bt = batch.t.numpy()
            neg = torch.randint(0, num_nodes, (bsrc.shape[0],), dtype=torch.long).numpy()
            # scores under both memory modes
            pos_i = np.empty(bsrc.shape[0]); negp_i = np.empty(bsrc.shape[0])
            pos_w = np.empty(bsrc.shape[0]); negp_w = np.empty(bsrc.shape[0])
            for i in range(bsrc.shape[0]):
                u, v, nv, tt = int(bsrc[i]), int(bdst[i]), int(neg[i]), float(bt[i])
                pos_i[i] = 1.0 if (u, v) in mem_inf else 0.0
                negp_i[i] = 1.0 if (u, nv) in mem_inf else 0.0
                lp = last_seen.get((u, v)); ln = last_seen.get((u, nv))
                pos_w[i] = 1.0 if (lp is not None and lp >= tt - window) else 0.0
                negp_w[i] = 1.0 if (ln is not None and ln >= tt - window) else 0.0
            y = np.concatenate([np.ones(bsrc.shape[0]), np.zeros(bsrc.shape[0])])
            si = np.concatenate([pos_i, negp_i]); sw = np.concatenate([pos_w, negp_w])
            aps_i.append(average_precision_score(y, si)); aucs_i.append(roc_auc_score(y, si))
            aps_w.append(average_precision_score(y, sw)); aucs_w.append(roc_auc_score(y, sw))
            # update memory (predict-then-update)
            for u, v, tt in zip(bsrc.tolist(), bdst.tolist(), bt.tolist()):
                mem_inf.add((u, v))
                last_seen[(u, v)] = float(tt)
        return (float(np.mean(aps_i)), float(np.mean(aucs_i)),
                float(np.mean(aps_w)), float(np.mean(aucs_w)))

    _score_stream(val)  # stream val to grow memory (matches TGN eval update)
    ap_i, auc_i, ap_w, auc_w = _score_stream(test)
    return {
        "inf_test_ap": ap_i, "inf_test_auc": auc_i,
        "tw_test_ap": ap_w, "tw_test_auc": auc_w,
        "n_edges": int(src.size), "n_nodes": num_nodes,
    }


# --------------------------------------------------------------------------- #
# (B) GraphMixer
# --------------------------------------------------------------------------- #
class FixedTimeEncoder(nn.Module):
    """GraphMixer's fixed (non-trainable) cos time encoding: cos(w * dt) with
    geometrically-spaced frequencies w_k = 10^{-linspace(0,9,dim)}."""
    def __init__(self, dim: int):
        super().__init__()
        w = 1.0 / (10.0 ** np.linspace(0, 9, dim, dtype=np.float64))
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32).view(1, 1, dim))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:  # dt: (N, K)
        return torch.cos(dt.unsqueeze(-1) * self.w)       # (N, K, dim)


class MixerBlock(nn.Module):
    """One MLP-Mixer block: token-mixing (across the K neighbor slots) then
    channel-mixing (across features), each with LayerNorm + residual."""
    def __init__(self, num_tokens: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.token = nn.Sequential(
            nn.Linear(num_tokens, num_tokens * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(num_tokens * 2, num_tokens), nn.Dropout(dropout))
        self.ln2 = nn.LayerNorm(hidden)
        self.chan = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, K, hidden)
        y = self.ln1(x).transpose(1, 2)                  # (N, hidden, K)
        x = x + self.token(y).transpose(1, 2)
        x = x + self.chan(self.ln2(x))
        return x


class GraphMixer(nn.Module):
    def __init__(self, num_nodes: int, time_dim: int = 100, hidden: int = 100,
                 node_dim: int = 100, num_tokens: int = 20, blocks: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.K = num_tokens
        self.time_enc = FixedTimeEncoder(time_dim)
        self.proj = nn.Linear(time_dim, hidden)           # edges carry no features
        self.blocks = nn.ModuleList(
            [MixerBlock(num_tokens, hidden, dropout) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.node_emb = nn.Embedding(num_nodes, node_dim)
        nn.init.normal_(self.node_emb.weight, std=0.1)
        rep = hidden + node_dim
        self.link_mlp = nn.Sequential(
            nn.Linear(2 * rep, rep), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(rep, 1))

    def encode(self, node_ids, dt, nbr_ids, mask):
        """node_ids (N,), dt (N,K), nbr_ids (N,K), mask (N,K bool). -> (N, rep)"""
        m = mask.unsqueeze(-1).float()
        # link-encoder: fixed time-encoding of recent neighbor gaps -> Mixer
        tok = self.proj(self.time_enc(dt)) * m            # (N, K, hidden)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln(tok) * m
        link_rep = tok.sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1.0)
        # node-encoder: self embedding + mean of neighbor embeddings
        self_e = self.node_emb(node_ids)
        nbr_e = self.node_emb(nbr_ids.clamp(min=0)) * m
        node_rep = self_e + nbr_e.sum(1) / mask.float().sum(1, keepdim=True).clamp(min=1.0)
        return torch.cat([link_rep, node_rep], dim=-1)

    def score(self, rep_u, rep_v):
        return self.link_mlp(torch.cat([rep_u, rep_v], dim=-1)).squeeze(-1)


class NeighborStore:
    """Per-node ring buffer of the K most-recent 1-hop neighbors (symmetric,
    like PyG's LastNeighborLoader). gather() returns recency-ordered, padded
    slots so the Mixer's token positions are consistent."""
    def __init__(self, num_nodes: int, K: int):
        self.K = K
        self.nid = np.full((num_nodes, K), -1, dtype=np.int64)
        self.nt = np.full((num_nodes, K), -1.0, dtype=np.float64)
        self.ptr = np.zeros(num_nodes, dtype=np.int64)

    def reset(self):
        self.nid.fill(-1); self.nt.fill(-1.0); self.ptr.fill(0)

    def insert(self, s, d, t):
        for u, v, tt in zip(s, d, t):
            for a, b in ((u, v), (v, u)):
                p = self.ptr[a]
                self.nid[a, p] = b; self.nt[a, p] = tt
                self.ptr[a] = (p + 1) % self.K

    def gather(self, q, qt):
        """q (M,) query nodes, qt (M,) query times -> dt, nbr_ids, mask (M,K)."""
        nid = self.nid[q]; nt = self.nt[q]                # (M, K)
        valid = nt >= 0
        # recency order (largest time first); invalids (-1) sink to the end
        order = np.argsort(-np.where(valid, nt, -np.inf), axis=1)
        rows = np.arange(q.shape[0])[:, None]
        nid = nid[rows, order]; nt = nt[rows, order]; valid = valid[rows, order]
        dt = np.clip(qt[:, None] - nt, 0.0, None) * valid
        return dt, nid, valid


def run_graphmixer(src: np.ndarray, dst: np.ndarray, t: np.ndarray,
                   epochs: int = 10, seed: int = 0, batch_size: int = 200,
                   subsample: int = 40000, K: int = 20, lr: float = 1e-3
                   ) -> Dict[str, float]:
    torch.manual_seed(seed)
    src, dst, t = _subsample(src, dst, t, subsample, seed)
    data, train, val, test, num_nodes = _build_temporal_data(src, dst, t)

    model = GraphMixer(num_nodes, num_tokens=K)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    store = NeighborStore(num_nodes, K)

    def _feats(nodes_np, times_np):
        dt, nbr, mask = store.gather(nodes_np, times_np)
        return (torch.tensor(dt, dtype=torch.float32),
                torch.tensor(nbr, dtype=torch.long),
                torch.tensor(mask, dtype=torch.bool),
                torch.tensor(nodes_np, dtype=torch.long))

    def _batch_reps(bsrc, bdst, bneg, bt):
        # one gather over all role-nodes sharing the batch edge time
        q = np.concatenate([bsrc, bdst, bneg])
        qt = np.concatenate([bt, bt, bt]).astype(np.float64)
        dt, nbr, mask, nid = _feats(q, qt)
        rep = model.encode(nid, dt, nbr, mask)
        B = bsrc.shape[0]
        return rep[:B], rep[B:2 * B], rep[2 * B:]

    def _train_epoch():
        model.train(); store.reset()
        total = 0.0
        for batch in TemporalDataLoader(train, batch_size=batch_size):
            bsrc = batch.src.numpy(); bdst = batch.dst.numpy(); bt = batch.t.numpy()
            bneg = torch.randint(0, num_nodes, (bsrc.shape[0],), dtype=torch.long).numpy()
            opt.zero_grad()
            r_s, r_d, r_n = _batch_reps(bsrc, bdst, bneg, bt)
            pos = model.score(r_s, r_d); negp = model.score(r_s, r_n)
            loss = crit(pos, torch.ones_like(pos)) + crit(negp, torch.zeros_like(negp))
            loss.backward(); opt.step()
            store.insert(bsrc.tolist(), bdst.tolist(), bt.tolist())
            total += float(loss) * bsrc.shape[0]
        return total / train.num_events

    @torch.no_grad()
    def _eval(loader):
        model.eval()
        aps, aucs = [], []
        for batch in TemporalDataLoader(loader, batch_size=batch_size):
            bsrc = batch.src.numpy(); bdst = batch.dst.numpy(); bt = batch.t.numpy()
            bneg = torch.randint(0, num_nodes, (bsrc.shape[0],), dtype=torch.long).numpy()
            r_s, r_d, r_n = _batch_reps(bsrc, bdst, bneg, bt)
            pos = model.score(r_s, r_d).sigmoid(); negp = model.score(r_s, r_n).sigmoid()
            y = torch.cat([torch.ones(pos.size(0)), torch.zeros(negp.size(0))]).numpy()
            sc = torch.cat([pos, negp]).numpy()
            aps.append(average_precision_score(y, sc)); aucs.append(roc_auc_score(y, sc))
            store.insert(bsrc.tolist(), bdst.tolist(), bt.tolist())
        return float(np.mean(aps)), float(np.mean(aucs))

    best_val, best = -1.0, {"test_ap": float("nan"), "test_auc": float("nan")}
    for ep in range(1, epochs + 1):
        t0 = time.time()
        _train_epoch()                    # store now holds train history
        val_ap, val_auc = _eval(val)      # continues streaming (val -> test)
        test_ap, test_auc = _eval(test)
        if val_ap > best_val:
            best_val = val_ap
            best = {"test_ap": test_ap, "test_auc": test_auc, "val_ap": val_ap}
        print(f"    epoch {ep:2d}: val_AP={val_ap:.3f} test_AP={test_ap:.3f} "
              f"test_AUC={test_auc:.3f}  ({time.time()-t0:.1f}s)")
    best.update({"n_edges": int(src.size), "n_nodes": num_nodes})
    return best


# --------------------------------------------------------------------------- #
# Datasets + driver
# --------------------------------------------------------------------------- #
def _load_datasets(agora_dir: str, real_path: str):
    from . import load
    from . import baselines as bl
    rs, rd, rt = load.load_real(real_path)
    ss, sd, st = load.load_agora(agora_dir)
    span = float(rt.max() - rt.min())
    n = int(np.unique(np.concatenate([rs, rd])).size)
    m = int(rs.size)
    bs, bd, bt = bl.gen_ba(n, m, span, 42)
    return {"real": (rs, rd, rt), "AGORA": (ss, sd, st), "BA": (bs, bd, bt)}


def main():
    import os
    agora_dir = os.environ.get("AGORA_DIR", "/tmp/agoratgn_A")
    real_path = os.environ.get(
        "REAL_PATH", "realdata/snap-collegemsg/CollegeMsg.txt")
    ds = _load_datasets(agora_dir, real_path)

    eb: Dict[str, Dict] = {}
    gm: Dict[str, Dict] = {}
    for name, (s, d, t) in ds.items():
        print(f"\n### EdgeBank on {name} ({s.size} edges) ###")
        eb[name] = run_edgebank(s, d, t, seed=0)
        print(f"    inf: AP={eb[name]['inf_test_ap']:.3f} AUC={eb[name]['inf_test_auc']:.3f}"
              f"   tw:  AP={eb[name]['tw_test_ap']:.3f} AUC={eb[name]['tw_test_auc']:.3f}")
    for name, (s, d, t) in ds.items():
        print(f"\n### GraphMixer on {name} ###")
        gm[name] = run_graphmixer(s, d, t, seed=0)

    cols = ["real", "AGORA", "BA"]
    print("\n\n================ RESULTS: test AP / ROC-AUC ================")
    print(f"{'model':<22}" + "".join(f"{c:>18}" for c in cols))
    def line(label, getter):
        vals = "".join(f"{getter(c):>18}" for c in cols)
        print(f"{label:<22}{vals}")
    line("EdgeBank_inf", lambda c: f"{eb[c]['inf_test_ap']:.3f} / {eb[c]['inf_test_auc']:.3f}")
    line("EdgeBank_tw", lambda c: f"{eb[c]['tw_test_ap']:.3f} / {eb[c]['tw_test_auc']:.3f}")
    line("GraphMixer", lambda c: f"{gm[c]['test_ap']:.3f} / {gm[c]['test_auc']:.3f}")
    print("(TGN, prior run:    real 0.843/-  AGORA 0.943/-  BA 0.611/- )")


if __name__ == "__main__":
    main()
