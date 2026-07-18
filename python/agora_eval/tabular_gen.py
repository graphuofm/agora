"""tabular_gen.py — tabular deep-generative baselines (CTGAN, TVAE) for temporal
graphs.

Why these baselines?  A recurring reviewer question for a temporal-graph
generator is "why not just throw the edge list at a modern tabular synthesizer?"
CTGAN and TVAE (Xu et al., NeurIPS 2019) are the standard deep tabular models;
`ctgan` bundles both.  This module makes that comparison honest: it treats a
temporal edge stream as a flat TABLE with three columns (src, dst, t), learns
the joint distribution, samples a synthetic table of matched row count, and
decodes it back to (src, dst, t) arrays that plug straight into the `agora_eval`
fidelity harness (same Arrays convention as `baselines.py`).

--------------------------------------------------------------------------------
THE HARD PART — encoding high-cardinality node IDs for a tabular model
--------------------------------------------------------------------------------
Raw node IDs are categorical with (potentially) millions of levels and NO
ordinal meaning: id 5 is not "between" id 4 and id 6.  A tabular GAN cannot
one-hot millions of categories (memory blows up), and treating the integer id as
a continuous number is nonsense (it invents a fake ordering).  This is the known
reason tabular synthesizers are a poor fit for graphs — they model rows, not
topology.

Our choice: **degree-rank encoding** (scales to any cardinality).
  * From the training edges we compute each node's OUT-degree (as a source) and
    IN-degree (as a destination).
  * Nodes are ranked by degree; a node at rank position p out of N is encoded as
    the continuous value (p + u)/N in (0,1), with u~Uniform(0,1) jitter so the
    column is smooth rather than a comb of spikes.  Because a node appears in the
    `src` column exactly out-degree times, the column's marginal density carries
    the degree distribution as sampling frequency — giving the tabular model a
    fair shot at reproducing the degree distribution.
  * `src` is encoded by the OUT-degree ranking (over nodes that ever act as a
    source); `dst` by the IN-degree ranking (over nodes that ever act as a
    destination).  Time is shifted to start at 0 and modeled on its native scale
    (the model's mode-specific normalization handles the range).
  * DECODING is the exact inverse of the (p+u)/N map: a sampled value v is
    clipped to (0,1) and floored to a rank slot, which recovers a real node id.

HONEST LIMITATION.  This encoding is deliberately only degree-aware.  `src` and
`dst` are decoded independently, so the *pairing* — who connects to whom — is
reconstructed almost at random subject to the marginal degree bias.  That means
even in the best case this baseline behaves like a soft configuration model: it
can approximate the DEGREE distribution but it CANNOT reproduce topology
(reciprocity, clustering, repeat edges, motifs) or the per-source temporal
structure that depends on *which* pair fired.  We report this as a finding, not
a bug: it is exactly why bare tabular generators are the wrong tool for temporal
graphs, and it is what a graph-native generator must beat.

--------------------------------------------------------------------------------
Compute budget (CPU-only).  All three modeled columns are continuous, so there
is no giant one-hot; on CollegeMsg (~60k rows x 3 cols) CTGAN runs ~0.75 s/epoch
and TVAE ~0.9 s/epoch on 8 CPU threads.  Both default to 300 epochs (~4-5 min),
comfortably under the ~10-min-per-fit cap.  Lower `epochs` for larger inputs.
"""
from __future__ import annotations

import time
import warnings
from typing import Tuple

import numpy as np

Arrays = Tuple[np.ndarray, np.ndarray, np.ndarray]

# Default epoch caps (see module docstring for the CPU timing that motivates
# them). Tuned so a single fit on a ~60k-row table stays well under 10 min CPU.
DEFAULT_CTGAN_EPOCHS = 300
DEFAULT_TVAE_EPOCHS = 300


# --------------------------------------------------------------------------- #
# Degree-rank ID codec
# --------------------------------------------------------------------------- #
class _RankCodec:
    """Bijective-ish degree-rank codec for one node column.

    Fit on the column's node ids: nodes are sorted by their frequency (= degree
    in that role), and node at rank p (0-based, of N) maps to the open interval
    [p/N, (p+1)/N).  ``encode`` places each occurrence at (p+u)/N with uniform
    jitter u; ``decode`` inverts by flooring v*N back to the rank slot.
    """

    def __init__(self, ids: np.ndarray, seed: int):
        uniq, counts = np.unique(ids, return_counts=True)
        # ascending by degree; stable so equal-degree nodes keep id order.
        order = np.argsort(counts, kind="stable")
        self.nodes_by_rank = uniq[order].astype(np.int64)  # rank slot -> node id
        self.N = int(self.nodes_by_rank.size)
        # id -> rank slot, for encoding.
        self._rank_of = {int(nid): p for p, nid in enumerate(self.nodes_by_rank.tolist())}
        self._rng = np.random.default_rng(seed)

    def encode(self, ids: np.ndarray) -> np.ndarray:
        p = np.array([self._rank_of[int(i)] for i in ids], dtype=np.float64)
        u = self._rng.random(p.size)
        return (p + u) / self.N

    def decode(self, v: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0 - 1e-9)
        slot = np.clip(np.floor(v * self.N).astype(np.int64), 0, self.N - 1)
        return self.nodes_by_rank[slot]


def _encode_table(train_src, train_dst, train_t, seed):
    """Build the 3-column continuous training table + the codecs to decode with.

    Returns (DataFrame[src_enc,dst_enc,t_enc], src_codec, dst_codec, t_shift,
    train_span)."""
    import pandas as pd

    src_codec = _RankCodec(train_src, seed)
    dst_codec = _RankCodec(train_dst, seed + 1)
    t = np.asarray(train_t, dtype=np.float64)
    t_shift = float(t.min()) if t.size else 0.0
    t_enc = t - t_shift
    train_span = float(t_enc.max()) if t_enc.size else 0.0
    df = pd.DataFrame(
        {
            "src_enc": src_codec.encode(train_src),
            "dst_enc": dst_codec.encode(train_dst),
            "t_enc": t_enc,
        }
    )
    return df, src_codec, dst_codec, train_span


def _decode_table(samples, src_codec, dst_codec, train_span, span):
    """Decode a sampled DataFrame back to (src, dst, t) arrays.

    Node columns invert the rank codec; time is clipped to the learned range and
    linearly rescaled onto the requested `span`."""
    src = src_codec.decode(samples["src_enc"].to_numpy())
    dst = dst_codec.decode(samples["dst_enc"].to_numpy())
    t = np.asarray(samples["t_enc"].to_numpy(), dtype=np.float64)
    t = np.clip(t, 0.0, train_span if train_span > 0 else 1.0)
    if train_span > 0:
        t = t / train_span * float(max(span, 1.0))
    return src.astype(np.int64), dst.astype(np.int64), t.astype(np.float64)


def _seed_everything(seed: int):
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Public generators
# --------------------------------------------------------------------------- #
def ctgan_generate(train_src, train_dst, train_t, n_target, m_target, span, seed,
                   epochs: int = DEFAULT_CTGAN_EPOCHS) -> Arrays:
    """Learn a temporal edge list as a table with CTGAN and sample a synthetic
    graph of ~`m_target` edges.

    Signature matches the `agora_eval` baseline convention. `train_*` are the real
    (src, dst, t) arrays to learn from; `n_target`/`m_target`/`span` describe the
    target graph (node count is informational — the synthetic node set is the
    subset of training nodes that get decoded; edge count = `m_target`; sampled
    times are rescaled onto `span`). Returns (src, dst, t) numpy arrays.

    CPU budget: all columns are continuous, so `epochs` (default 300) keeps a
    ~60k-row fit near ~4 min on 8 threads — under the 10-min cap. Runs on CPU
    automatically when no CUDA device is available.
    """
    from ctgan import CTGAN

    _seed_everything(seed)
    df, src_codec, dst_codec, train_span = _encode_table(train_src, train_dst, train_t, seed)
    model = CTGAN(epochs=epochs, batch_size=500, enable_gpu=False, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)  # no discrete_columns: our encoding is fully continuous
        samples = model.sample(int(m_target))
    return _decode_table(samples, src_codec, dst_codec, train_span, span)


def tvae_generate(train_src, train_dst, train_t, n_target, m_target, span, seed,
                  epochs: int = DEFAULT_TVAE_EPOCHS) -> Arrays:
    """Learn a temporal edge list as a table with TVAE and sample a synthetic
    graph of ~`m_target` edges.

    Same convention, encoding, and decoding as `ctgan_generate` (see that
    docstring and the module header); only the synthesizer differs. TVAE is a
    variational autoencoder — usually faster and steadier than the GAN on CPU.
    Returns (src, dst, t) numpy arrays. `epochs` default 300 (~5 min on a ~60k
    row table, 8 threads), under the 10-min cap.
    """
    from ctgan import TVAE

    _seed_everything(seed)
    df, src_codec, dst_codec, train_span = _encode_table(train_src, train_dst, train_t, seed)
    model = TVAE(epochs=epochs, batch_size=500, enable_gpu=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(df)
        samples = model.sample(int(m_target))
    return _decode_table(samples, src_codec, dst_codec, train_span, span)


# --------------------------------------------------------------------------- #
# Smoke test / real-fidelity report
# --------------------------------------------------------------------------- #
def _smoke_test(path: str = "realdata/snap-collegemsg/CollegeMsg.txt",
                cap: int = 60_000, seed: int = 42):
    """Fit both models on a real graph and print measured fidelity + wall time.

    Run:  python3 -m agora_eval.tabular_gen   (from the `python/` dir)
    Honest reporting: prints whatever fidelity is measured, however low.
    """
    from . import compare as cmp
    from . import stats as st
    from .load import load_real

    src, dst, t = load_real(path)
    if src.size > cap:
        src, dst, t = src[:cap], dst[:cap], t[:cap]
    n = int(np.unique(np.concatenate([src, dst])).size)
    m = int(src.size)
    span = float(t.max() - t.min()) if t.size else 1.0
    print(f"real: {m:,} edges / {n:,} nodes / span {span/86400:.1f} days")
    real_stats = st.compute_stats(src, dst, t)

    for name, fn in [("CTGAN", ctgan_generate), ("TVAE", tvae_generate)]:
        t0 = time.time()
        gs, gd, gt = fn(src, dst, t, n, m, span, seed)
        fit_s = time.time() - t0
        syn_stats = st.compute_stats(gs, gd, gt)
        res = cmp.compare(real_stats, syn_stats)
        print(f"\n=== {name} ===")
        print(f"  fit+sample wall time : {fit_s:.1f}s")
        print(f"  synth: {gs.size:,} edges / "
              f"{np.unique(np.concatenate([gs, gd])).size:,} nodes")
        print(f"  FIDELITY SCORE       : {res['fidelity_score']:.3f}")
        print("  KS distances (lower=closer):")
        for k, v in res["ks_distances"].items():
            print(f"    {k:<18} {v:.3f}")
        print("  scalar metrics (real -> synth):")
        for k, v in res["scalar_metrics"].items():
            r, s2 = v["real"], v["synth"]
            rf = f"{r:.3f}" if isinstance(r, float) else str(r)
            sf = f"{s2:.3f}" if isinstance(s2, float) else str(s2)
            print(f"    {k:<18} {rf} -> {sf}   dist={v['distance']:.3f}")


if __name__ == "__main__":
    _smoke_test()
