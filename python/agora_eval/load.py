"""load.py — load temporal edge lists from AGORA output and real datasets into
the minimal (src, dst, t) arrays the fidelity stats need.

Real datasets vary wildly; `load_real` auto-detects a few canonical schemas
(Elliptic, SNAP temporal `src dst t`, generic CSV with named/indexed columns)
and falls back to a configurable column mapping.
"""
from __future__ import annotations

import glob
import gzip
import os
from typing import Optional, Tuple

import numpy as np


def filter_min_degree(src, dst, t, k: int):
    """Keep only edges whose BOTH endpoints have total degree ≥ k, iterated to a
    fixed point — mirrors the k-core-style node filtering that benchmarks like
    TGB apply (they drop low-activity nodes), so an unfiltered AGORA output can
    be compared fairly to a filtered real dataset."""
    while True:
        deg = {}
        for a in (src, dst):
            u, c = np.unique(a, return_counts=True)
            for node, cnt in zip(u.tolist(), c.tolist()):
                deg[node] = deg.get(node, 0) + cnt
        keep = np.array([deg.get(int(s), 0) >= k and deg.get(int(d), 0) >= k for s, d in zip(src, dst)])
        if keep.all() or not keep.any():
            return src[keep], dst[keep], t[keep]
        src, dst, t = src[keep], dst[keep], t[keep]


def load_agora(out_dir: str, event_type: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a AGORA output directory's edge stream (parquet or csv shards).
    `event_type` restricts to one relation (e.g. only WROTE_REVIEW) for a
    like-for-like comparison against a single-relation real dataset."""
    pq = sorted(glob.glob(os.path.join(out_dir, "edges_*.parquet")))
    if pq:
        import pyarrow.parquet as papq

        cols = ["src", "dst", "t"] + (["event_type"] if event_type else [])
        srcs, dsts, ts = [], [], []
        for f in pq:
            tbl = papq.read_table(f, columns=cols)
            s = tbl["src"].to_numpy()
            d = tbl["dst"].to_numpy()
            tt = tbl["t"].to_numpy()
            if event_type:
                et = tbl["event_type"].to_pylist()
                mask = np.array([x == event_type for x in et])
                s, d, tt = s[mask], d[mask], tt[mask]
            srcs.append(s)
            dsts.append(d)
            ts.append(tt)
        return (
            np.concatenate(srcs).astype(np.int64),
            np.concatenate(dsts).astype(np.int64),
            np.concatenate(ts).astype(np.float64),
        )
    csv = sorted(glob.glob(os.path.join(out_dir, "edges_*.csv")))
    if csv:
        return _load_csv_cols(csv, "src", "dst", "t")
    raise FileNotFoundError(f"no edge shards (edges_*.parquet/csv) in {out_dir}")


def _open(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def load_real(
    path: str,
    *,
    src_col=None,
    dst_col=None,
    t_col=None,
    delimiter: Optional[str] = None,
    has_header: Optional[bool] = None,
    t_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a real temporal edge list. `path` is a file (csv/txt[.gz]) or a
    directory (auto-finds an edgelist). Column selectors may be names or 0-based
    indices; if omitted, sensible defaults per detected format are used.
    `t_scale` multiplies timestamps into seconds (e.g. days->86400)."""
    if os.path.isdir(path):
        # A AGORA-style parquet output dir (or any edges_*.parquet) loads via the
        # columnar path; otherwise find a text edge list.
        if glob.glob(os.path.join(path, "edges_*.parquet")):
            return load_agora(path)
        path = _find_edgelist(path)

    # Detect Elliptic by filename.
    base = os.path.basename(path).lower()
    if "elliptic" in base and "edgelist" in base:
        return _load_elliptic(path)

    # Sniff delimiter + header.
    with _open(path) as fh:
        first = fh.readline()
    if delimiter is None:
        delimiter = "," if first.count(",") >= first.count("\t") and "," in first else (
            "\t" if "\t" in first else None  # None => any whitespace
        )
    if has_header is None:
        toks = first.replace(",", " ").split()
        has_header = any(not _isnum(tok) for tok in toks[:3])

    # Default columns: SNAP temporal nets are `src dst [weight] timestamp` or
    # `src dst timestamp`; pick first two as src,dst and the last numeric as t.
    return _load_generic(path, delimiter, has_header, src_col, dst_col, t_col, t_scale)


def _isnum(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _find_edgelist(d: str) -> str:
    pats = ["*edgelist*", "*edges*", "*temporal*", "*.txt*", "*.csv*"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(d, p)))
        hits = [h for h in hits if "classes" not in h.lower() and "features" not in h.lower()]
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no edge list found under {d}")


def _load_elliptic(edgelist_path: str):
    """Elliptic: edgelist (txId1,txId2) + features file carries the time step.
    We join the per-tx time step (features col index 1) onto edges as t (the
    edge time = the source tx's time step). Falls back to t=0 if no features."""
    import csv as _csv

    feat_path = None
    d = os.path.dirname(edgelist_path)
    for f in glob.glob(os.path.join(d, "*features*")):
        feat_path = f
        break
    tx_time = {}
    if feat_path:
        with _open(feat_path) as fh:
            for row in _csv.reader(fh):
                if not row:
                    continue
                try:
                    tx_time[row[0]] = float(row[1])  # col 1 = time step (1..49)
                except (ValueError, IndexError):
                    continue
    src, dst, t = [], [], []
    with _open(edgelist_path) as fh:
        r = _csv.reader(fh)
        header = next(r, None)
        if header and _isnum(header[0]):
            # no header; treat as data
            a, b = header[0], header[1]
            src.append(a); dst.append(b); t.append(tx_time.get(a, 0.0))
        for row in r:
            if len(row) < 2:
                continue
            src.append(row[0]); dst.append(row[1]); t.append(tx_time.get(row[0], 0.0))
    # map string txIds to ints
    ids = {v: i for i, v in enumerate(sorted(set(src) | set(dst)))}
    s = np.array([ids[x] for x in src], dtype=np.int64)
    dd = np.array([ids[x] for x in dst], dtype=np.int64)
    # time steps are 1..49; scale to seconds (1 step = 1 "day") for gap stats
    tt = np.array(t, dtype=np.float64) * 86400.0
    return s, dd, tt


def _load_generic(path, delimiter, has_header, src_col, dst_col, t_col, t_scale):
    import csv as _csv

    rows = []
    with _open(path) as fh:
        if delimiter is None:
            for line in fh:
                parts = line.split()
                if parts:
                    rows.append(parts)
        else:
            for parts in _csv.reader(fh, delimiter=delimiter):
                if parts:
                    rows.append(parts)
    if not rows:
        raise ValueError(f"{path} is empty")
    header = None
    if has_header:
        header = [h.strip() for h in rows[0]]
        rows = rows[1:]

    def col_idx(c, default):
        if c is None:
            return default
        if isinstance(c, int):
            return c
        if header and c in header:
            return header.index(c)
        return int(c)

    ncol = len(rows[0])
    si = col_idx(src_col, 0)
    di = col_idx(dst_col, 1)
    # `--t-col index` uses ROW ORDER as the temporal index (for datasets whose
    # timestamp is unusable, e.g. CICIDS truncated to MM:SS.s).
    row_order_time = isinstance(t_col, str) and t_col == "index"
    ti = (ncol - 1) if row_order_time else col_idx(t_col, ncol - 1)

    # map node labels to ints if non-numeric
    s_raw = [r[si] for r in rows if len(r) > max(si, di, ti)]
    d_raw = [r[di] for r in rows if len(r) > max(si, di, ti)]
    t_raw = [r[ti] for r in rows if len(r) > max(si, di, ti)]
    numeric_nodes = _isnum(s_raw[0]) and _isnum(d_raw[0])
    if numeric_nodes:
        s = np.array([int(float(x)) for x in s_raw], dtype=np.int64)
        dd = np.array([int(float(x)) for x in d_raw], dtype=np.int64)
    else:
        ids = {v: i for i, v in enumerate(sorted(set(s_raw) | set(d_raw)))}
        s = np.array([ids[x] for x in s_raw], dtype=np.int64)
        dd = np.array([ids[x] for x in d_raw], dtype=np.int64)
    if row_order_time:
        tt = np.arange(len(s_raw), dtype=np.float64) * t_scale
    else:
        tt = np.array([float(x) for x in t_raw], dtype=np.float64) * t_scale
    return s, dd, tt


def _load_csv_cols(files, src_name, dst_name, t_name):
    import csv as _csv

    srcs, dsts, ts = [], [], []
    for path in files:
        with open(path) as fh:
            r = _csv.DictReader(fh)
            for row in r:
                srcs.append(int(row[src_name]))
                dsts.append(int(row[dst_name]))
                ts.append(float(row[t_name]))
    return (
        np.array(srcs, dtype=np.int64),
        np.array(dsts, dtype=np.int64),
        np.array(ts, dtype=np.float64),
    )
