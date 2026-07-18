"""embed.py — embed the RAG corpus and build a FAISS index (mustread.txt §9 step 2-3).

Embeds ``corpus/chunks.jsonl`` with a SMALL open sentence-transformers model
(default ``BAAI/bge-small-en-v1.5``, fallback ``all-MiniLM-L6-v2``) and builds a
FAISS ``IndexFlatIP`` over L2-normalized vectors (so inner product == cosine).

CRITICAL offline-safety: if the model cannot be loaded (no network / not cached)
the pipeline falls back to a *deterministic* pure-numpy hashing embedder
(bag-of-hashed-character-3grams projected to 384 dims, L2-normalized). The
pipeline therefore ALWAYS runs and is reproducible. The embedder actually used
is recorded in the index metadata and printed.

Persisted layout under ``corpus/index/``::

    faiss.index        the FAISS IndexFlatIP
    meta.jsonl         one line per vector: {id, domain, source_id, url,
                       license_tier, text} (row-aligned with the index)
    embed_info.json    {embedder, model, dim, n_chunks, normalized}

Caching: re-embedding is skipped if the index exists and its chunk count
matches the corpus.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HASH_DIM = 384  # matches bge-small / MiniLM dims so the hashing fallback is drop-in
_META_FIELDS = (
    "id", "domain", "source_id", "url", "license_tier", "text",
    "structured", "section", "external_id",
)


@dataclass
class EmbedInfo:
    """Describes the embedder that produced an index."""

    embedder: str  # "sentence-transformers" | "hashing-fallback"
    model: str
    dim: int
    n_chunks: int
    normalized: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "embedder": self.embedder,
            "model": self.model,
            "dim": self.dim,
            "n_chunks": self.n_chunks,
            "normalized": self.normalized,
        }


# --------------------------------------------------------------------------- #
# Corpus IO
# --------------------------------------------------------------------------- #
def load_chunks(corpus_dir: Path) -> List[Dict[str, object]]:
    """Load ``corpus/chunks.jsonl`` into a list of dicts (insertion order)."""
    path = corpus_dir / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"corpus chunks not found at {path}; run `python -m agora_rag corpus` first"
        )
    chunks: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def count_chunks(corpus_dir: Path) -> int:
    path = corpus_dir / "chunks.jsonl"
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


# --------------------------------------------------------------------------- #
# Deterministic hashing embedder (offline fallback)
# --------------------------------------------------------------------------- #
def _hash_token(token: str, dim: int) -> Tuple[int, float]:
    """Map a token to a (bucket, sign) pair deterministically via blake2b."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    val = struct.unpack("<Q", digest)[0]
    bucket = val % dim
    sign = 1.0 if (val >> 63) & 1 else -1.0
    return bucket, sign


def hash_embed_one(text: str, dim: int = HASH_DIM) -> np.ndarray:
    """Embed one string as an L2-normalized bag-of-hashed-char-3grams vector.

    Deterministic and dependency-free (numpy + hashlib). Lowercased; pads with
    spaces so short strings still yield 3-grams.
    """
    vec = np.zeros(dim, dtype=np.float32)
    s = f"  {text.lower()}  "
    if len(s) >= 3:
        for i in range(len(s) - 2):
            gram = s[i : i + 3]
            bucket, sign = _hash_token(gram, dim)
            vec[bucket] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def hash_embed_batch(texts: List[str], dim: int = HASH_DIM) -> np.ndarray:
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        out[i] = hash_embed_one(t, dim)
    return out


# --------------------------------------------------------------------------- #
# Model loading with graceful offline fallback
# --------------------------------------------------------------------------- #
def _try_load_model(model_name: str):
    """Return a loaded SentenceTransformer, or raise on any failure."""
    from sentence_transformers import SentenceTransformer  # local import: optional dep

    return SentenceTransformer(model_name)


class Embedder:
    """Unified embed interface; either a real model or the hashing fallback."""

    def __init__(
        self, model: Optional[str] = None, dim: int = HASH_DIM, force_hash: bool = False
    ) -> None:
        self.dim = dim
        self._st_model = None
        if force_hash:
            self.embedder = "hashing-fallback"
            self.model = f"hashing-char3gram-{self.dim}d"
            print(
                f"[embed] hashing embedder (consistency with prebuilt index, dim={self.dim})"
            )
            return
        candidates = [m for m in (model, DEFAULT_MODEL, FALLBACK_MODEL) if m]
        # de-dup preserving order
        seen: set = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]
        for name in candidates:
            try:
                self._st_model = _try_load_model(name)
                self.embedder = "sentence-transformers"
                self.model = name
                self.dim = int(self._st_model.get_sentence_embedding_dimension())
                print(f"[embed] using sentence-transformers model: {name} (dim={self.dim})")
                return
            except Exception as exc:  # noqa: BLE001 — offline safety: any failure -> fallback
                print(f"[embed] could not load '{name}' ({type(exc).__name__}: {exc}); trying next")
        self.embedder = "hashing-fallback"
        self.model = f"hashing-char3gram-{self.dim}d"
        print(
            f"[embed] OFFLINE FALLBACK: deterministic hashing embedder "
            f"(char-3gram bag, dim={self.dim})"
        )

    def encode(self, texts: List[str], batch_size: int = 256) -> np.ndarray:
        if self._st_model is not None:
            arr = self._st_model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(arr, dtype=np.float32)
        return hash_embed_batch(texts, self.dim)


def _normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (mat / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
# Index build / load
# --------------------------------------------------------------------------- #
def index_dir(corpus_dir: Path) -> Path:
    return corpus_dir / "index"


def index_exists(corpus_dir: Path) -> bool:
    d = index_dir(corpus_dir)
    return (
        (d / "faiss.index").exists()
        and (d / "meta.jsonl").exists()
        and (d / "embed_info.json").exists()
    )


def load_embed_info(corpus_dir: Path) -> Optional[EmbedInfo]:
    path = index_dir(corpus_dir) / "embed_info.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return EmbedInfo(
        embedder=d["embedder"],
        model=d["model"],
        dim=int(d["dim"]),
        n_chunks=int(d["n_chunks"]),
        normalized=bool(d.get("normalized", True)),
    )


def build_index(
    corpus_dir: Path,
    model: Optional[str] = None,
    force: bool = False,
    batch_size: int = 256,
) -> EmbedInfo:
    """Embed the corpus and persist a FAISS index. Skips work if cache is valid."""
    import faiss  # local import: optional dep

    n = count_chunks(corpus_dir)
    if n == 0:
        raise FileNotFoundError(
            f"no chunks in {corpus_dir / 'chunks.jsonl'}; run the corpus pipeline first"
        )

    existing = load_embed_info(corpus_dir)
    if not force and index_exists(corpus_dir) and existing and existing.n_chunks == n:
        print(
            f"[embed] cache hit: index exists with {n:,} chunks "
            f"(embedder={existing.embedder}, model={existing.model}); skipping re-embed"
        )
        return existing

    print(f"[embed] embedding {n:,} chunks ...")
    chunks = load_chunks(corpus_dir)
    embedder = Embedder(model=model)
    texts = [str(c.get("text", "")) for c in chunks]

    vectors = embedder.encode(texts, batch_size=batch_size)
    vectors = _normalize(vectors)  # ensure unit norm regardless of embedder
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    d = index_dir(corpus_dir)
    d.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(d / "faiss.index"))

    with open(d / "meta.jsonl", "w", encoding="utf-8") as fh:
        for c in chunks:
            row = {k: c.get(k) for k in _META_FIELDS}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    info = EmbedInfo(
        embedder=embedder.embedder, model=embedder.model, dim=dim, n_chunks=n
    )
    (d / "embed_info.json").write_text(json.dumps(info.to_dict(), indent=2))
    print(
        f"[embed] wrote index: {n:,} vectors x {dim} dims -> {d} "
        f"(embedder={info.embedder})"
    )
    return info


def load_meta(corpus_dir: Path) -> List[Dict[str, object]]:
    """Load the row-aligned metadata for the index."""
    path = index_dir(corpus_dir) / "meta.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"index metadata not found at {path}; build the index first")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
