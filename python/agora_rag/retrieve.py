"""retrieve.py — hybrid retrieval (mustread.txt §9 step 3).

Combines DENSE retrieval (FAISS cosine over the embedding index) with a
hand-rolled BM25 lexical retriever (no ``rank_bm25`` dependency) via
Reciprocal-Rank Fusion (RRF, k=60). Hybrid retrieval gives precision on the
technical/regulatory terminology in the corpus while dense recall catches
paraphrase.

Public entry point::

    retrieve(query, corpus_dir, domain=None, k=12) -> List[RetrievedChunk]

Each result carries full provenance (id, domain, source_id, url, license_tier,
text) so every drafted rule can be traced back to a downloaded standard
(mustread.txt §9 step 5).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .embed import Embedder, index_dir, load_embed_info, load_meta

_TOKEN_RE = re.compile(r"[a-z0-9]+")
RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    """A retrieved corpus chunk with provenance and fusion score."""

    id: str
    domain: str
    source_id: str
    url: Optional[str]
    license_tier: Optional[str]
    text: str
    score: float  # RRF fused score (higher = better)
    row: int      # row index into the index/meta
    structured: bool = False
    section: Optional[str] = None       # eCFR section number (e.g. "1010.311")
    external_id: Optional[str] = None   # STIX/CAPEC id (e.g. "T1059")

    def provenance(self) -> Dict[str, Optional[str]]:
        return {"source_id": self.source_id, "url": self.url, "domain": self.domain}


# --------------------------------------------------------------------------- #
# BM25 (Okapi, ~40 lines of stdlib + numpy)
# --------------------------------------------------------------------------- #
class BM25:
    """Minimal Okapi BM25 over a fixed corpus of token lists."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        # term -> {doc_index: term_freq}
        self.postings: Dict[str, Dict[int, int]] = {}
        for i, doc in enumerate(docs):
            tf: Dict[str, int] = {}
            for tok in doc:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                self.postings.setdefault(tok, {})[i] = freq
        # idf per term (BM25 idf with +1 to stay non-negative)
        self.idf: Dict[str, float] = {}
        for tok, post in self.postings.items():
            df = len(post)
            self.idf[tok] = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for tok in set(query_tokens):
            post = self.postings.get(tok)
            if not post:
                continue
            idf = self.idf.get(tok, 0.0)
            for doc_i, freq in post.items():
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_i] / (self.avgdl or 1.0)
                )
                scores[doc_i] += idf * (freq * (self.k1 + 1.0)) / (denom or 1.0)
        return scores

    def top_k(self, query_tokens: List[str], k: int) -> List[int]:
        scores = self.scores(query_tokens)
        if not np.any(scores > 0):
            return []
        k = min(k, self.n)
        idx = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in idx[np.argsort(-scores[idx])] if scores[i] > 0]


# --------------------------------------------------------------------------- #
# Retriever: loads the index + meta once; caches BM25
# --------------------------------------------------------------------------- #
class Retriever:
    def __init__(self, corpus_dir: Path) -> None:
        import faiss  # local import: optional dep

        self.corpus_dir = corpus_dir
        info = load_embed_info(corpus_dir)
        if info is None:
            raise FileNotFoundError(
                f"no embedding index in {index_dir(corpus_dir)}; run "
                f"`python -m agora_rag index` first"
            )
        self.info = info
        self.meta = load_meta(corpus_dir)
        self.index = faiss.read_index(str(index_dir(corpus_dir) / "faiss.index"))
        # Reuse the SAME embedder family the index was built with (so the
        # hashing fallback is consistent across build + query).
        if info.embedder == "hashing-fallback":
            self.embedder = Embedder(dim=info.dim, force_hash=True)
        else:
            self.embedder = Embedder(model=info.model, dim=info.dim)
        self._bm25: Optional[BM25] = None
        self._bm25_rows: Optional[List[int]] = None  # row map when domain-filtered

    def _ensure_bm25(self, rows: Optional[List[int]]) -> BM25:
        # Build (and cache) BM25 over either the whole corpus or a domain subset.
        if rows is None:
            if self._bm25 is None:
                docs = [_tokenize(str(m.get("text", ""))) for m in self.meta]
                self._bm25 = BM25(docs)
                self._bm25_rows = None
            return self._bm25
        docs = [_tokenize(str(self.meta[r].get("text", ""))) for r in rows]
        return BM25(docs)

    def _dense_top(
        self, query: str, k: int, allowed: Optional[set]
    ) -> List[int]:
        qv = self.embedder.encode([query])
        qv = np.asarray(qv, dtype=np.float32)
        # over-fetch when filtering so we can keep k after the domain mask
        fetch = k if allowed is None else min(len(self.meta), max(k * 8, 200))
        _scores, idx = self.index.search(qv, fetch)
        out: List[int] = []
        for row in idx[0]:
            if row < 0:
                continue
            if allowed is not None and int(row) not in allowed:
                continue
            out.append(int(row))
            if len(out) >= k:
                break
        return out

    def retrieve(
        self, query: str, domain: Optional[str] = None, k: int = 12
    ) -> List[RetrievedChunk]:
        allowed: Optional[set] = None
        rows_subset: Optional[List[int]] = None
        if domain:
            rows_subset = [
                i for i, m in enumerate(self.meta) if m.get("domain") == domain
            ]
            allowed = set(rows_subset)
            if not rows_subset:
                allowed = None  # unknown/absent domain -> fall back to whole corpus

        # Dense ranking (list of global rows, best first).
        dense_rows = self._dense_top(query, max(k * 4, 40), allowed)

        # BM25 ranking.
        qtok = _tokenize(query)
        if allowed is not None and rows_subset:
            local = self._ensure_bm25(rows_subset).top_k(qtok, max(k * 4, 40))
            bm25_rows = [rows_subset[i] for i in local]
        else:
            bm25_rows = self._ensure_bm25(None).top_k(qtok, max(k * 4, 40))

        # Reciprocal-Rank Fusion.
        fused: Dict[int, float] = {}
        for rank, row in enumerate(dense_rows):
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, row in enumerate(bm25_rows):
            fused[row] = fused.get(row, 0.0) + 1.0 / (RRF_K + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._make_chunk(row, score) for row, score in ordered]

    def _make_chunk(self, row: int, score: float) -> RetrievedChunk:
        m = self.meta[row]
        section = m.get("section")
        external_id = m.get("external_id")
        return RetrievedChunk(
            id=str(m.get("id", "")),
            domain=str(m.get("domain", "")),
            source_id=str(m.get("source_id", "")),
            url=m.get("url"),
            license_tier=m.get("license_tier"),
            text=str(m.get("text", "")),
            score=float(score),
            row=row,
            structured=bool(m.get("structured", False)),
            section=str(section) if section else None,
            external_id=str(external_id) if external_id else None,
        )

    def retrieve_param(
        self,
        query: str,
        source_filter: Optional[List[str]] = None,
        domain: Optional[str] = None,
        section_hint: Optional[str] = None,
        k: int = 8,
    ) -> List[RetrievedChunk]:
        """Per-parameter retrieval for grounded extraction (extract.py).

        Over-fetch with the hybrid retriever, then re-rank deterministically so
        chunks from the cited source(s) and (if given) the cited section bubble
        to the top. This isolates, e.g., the ONE FinCEN section that defines the
        CTR threshold from the hundreds of unrelated "$10,000" mentions in the
        same title.
        """
        pool = self.retrieve(query, domain=domain, k=max(k * 6, 60))
        srcs = set(source_filter or [])

        def rank_key(c: RetrievedChunk) -> tuple:
            in_src = 1 if (srcs and c.source_id in srcs) else 0
            sec_hit = 1 if (section_hint and c.section == section_hint) else 0
            return (sec_hit, in_src, c.score)

        pool.sort(key=rank_key, reverse=True)
        return pool[:k]


def retrieve(
    query: str,
    corpus_dir: Path,
    domain: Optional[str] = None,
    k: int = 12,
) -> List[RetrievedChunk]:
    """Convenience one-shot hybrid retrieval (builds a Retriever each call)."""
    return Retriever(corpus_dir).retrieve(query, domain=domain, k=k)
