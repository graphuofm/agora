"""tgn_eval.py — downstream utility via a real Temporal Graph Network (TGN).

The ultimate test of a temporal-graph benchmark is whether a temporal GNN trained
on it behaves as it would on real data. We train PyG's TGN on future temporal
link prediction and report test Average Precision (AP) and ROC-AUC. Run on the
same task for real / AGORA / BA:
  - real data and AGORA (which carry real recurrence, bursts, communities) yield
    HIGH AP: their temporal dynamics are learnable.
  - BA / configuration-model graphs given random timestamps yield near-chance AP:
    there is no temporal structure to learn.

Adapted from the canonical PyG TGN example. Messages are a single zero feature, so
the model learns purely from the temporal interaction STRUCTURE (memory + time
encoding), making the comparison about dynamics, not attributes. CPU-friendly.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage, LastAggregator, LastNeighborLoader,
)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        return self.lin_final(h.relu())


def run_tgn(src: np.ndarray, dst: np.ndarray, t: np.ndarray,
            epochs: int = 6, mem_dim: int = 64, seed: int = 0,
            batch_size: int = 200) -> Dict[str, float]:
    torch.manual_seed(seed)
    device = torch.device("cpu")

    # contiguous node ids; chronological order
    nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n = src.size
    s = torch.tensor(inv[:n], dtype=torch.long)
    d = torch.tensor(inv[n:], dtype=torch.long)
    order = torch.tensor(np.argsort(t, kind="stable"))
    ts = torch.tensor(t[order.numpy()], dtype=torch.long)
    s, d = s[order], d[order]
    num_nodes = int(nodes.size)
    msg = torch.zeros((n, 1), dtype=torch.float)

    data = TemporalData(src=s, dst=d, t=ts, msg=msg).to(device)
    train_data, val_data, test_data = data.train_val_test_split(
        val_ratio=0.15, test_ratio=0.15)

    nbr = LastNeighborLoader(num_nodes, size=10, device=device)
    memory = TGNMemory(
        num_nodes, data.msg.size(-1), mem_dim, mem_dim,
        message_module=IdentityMessage(data.msg.size(-1), mem_dim, mem_dim),
        aggregator_module=LastAggregator()).to(device)
    gnn = GraphAttentionEmbedding(mem_dim, mem_dim, data.msg.size(-1),
                                  memory.time_enc).to(device)
    link_pred = LinkPredictor(mem_dim).to(device)
    opt = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()
    assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def _epoch_train():
        memory.train(); gnn.train(); link_pred.train()
        memory.reset_state(); nbr.reset_state()
        total = 0.0
        for batch in TemporalDataLoader(train_data, batch_size=batch_size):
            opt.zero_grad()
            bsrc, bdst, bt, bmsg = batch.src, batch.dst, batch.t, batch.msg
            neg = torch.randint(0, num_nodes, (bsrc.size(0),), dtype=torch.long)
            nid = torch.cat([bsrc, bdst, neg]).unique()
            nid, edge_index, e_id = nbr(nid)
            assoc[nid] = torch.arange(nid.size(0), device=device)
            z, last_update = memory(nid)
            z = gnn(z, last_update, edge_index,
                    data.t[e_id].to(device), data.msg[e_id].to(device))
            pos = link_pred(z[assoc[bsrc]], z[assoc[bdst]])
            negp = link_pred(z[assoc[bsrc]], z[assoc[neg]])
            loss = crit(pos, torch.ones_like(pos)) + crit(negp, torch.zeros_like(negp))
            memory.update_state(bsrc, bdst, bt, bmsg)
            nbr.insert(bsrc, bdst)
            loss.backward(); opt.step(); memory.detach()
            total += float(loss) * bsrc.size(0)
        return total / train_data.num_events

    @torch.no_grad()
    def _eval(ev):
        memory.eval(); gnn.eval(); link_pred.eval()
        aps, aucs = [], []
        for batch in TemporalDataLoader(ev, batch_size=batch_size):
            bsrc, bdst, bt, bmsg = batch.src, batch.dst, batch.t, batch.msg
            neg = torch.randint(0, num_nodes, (bsrc.size(0),), dtype=torch.long)
            nid = torch.cat([bsrc, bdst, neg]).unique()
            nid, edge_index, e_id = nbr(nid)
            assoc[nid] = torch.arange(nid.size(0), device=device)
            z, last_update = memory(nid)
            z = gnn(z, last_update, edge_index,
                    data.t[e_id].to(device), data.msg[e_id].to(device))
            pos = link_pred(z[assoc[bsrc]], z[assoc[bdst]]).sigmoid()
            negp = link_pred(z[assoc[bsrc]], z[assoc[neg]]).sigmoid()
            y = torch.cat([torch.ones(pos.size(0)), torch.zeros(negp.size(0))])
            s_ = torch.cat([pos, negp]).cpu()
            aps.append(average_precision_score(y, s_))
            aucs.append(roc_auc_score(y, s_))
            memory.update_state(bsrc, bdst, bt, bmsg)
            nbr.insert(bsrc, bdst)
        return float(np.mean(aps)), float(np.mean(aucs))

    best_val = 0.0
    for ep in range(1, epochs + 1):
        _epoch_train()
        val_ap, _ = _eval(val_data)
        test_ap, test_auc = _eval(test_data)
        best_val = max(best_val, val_ap)
        print(f"    epoch {ep}: val_AP={val_ap:.3f} test_AP={test_ap:.3f} test_AUC={test_auc:.3f}")
    return {"test_ap": test_ap, "test_auc": test_auc, "val_ap": best_val,
            "n_edges": n, "n_nodes": num_nodes}
