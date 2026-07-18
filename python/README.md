# agora_rag — RAG corpus fetcher (AGORA milestone M3a)

Acquires the authoritative-standards corpus described in `/CORPUS.md` for the
AGORA retrieval-to-rules pipeline (`mustread.txt` §9): fetch → parse → chunk.

## Usage

Run from this directory (`python/`). Requires Python ≥ 3.10, `requests`, and
`pypdf` (auto-installed by the parse stage if missing).

```bash
# full pipeline (fetch + parse + chunk) into ./corpus
python3 -m agora_rag corpus --dir corpus

# individual stages
python3 -m agora_rag corpus --fetch  --dir corpus
python3 -m agora_rag corpus --parse --chunk --dir corpus

# filters
python3 -m agora_rag corpus --fetch --domains finance,cyber --tier A

# per-source status table (+ chunk count)
python3 -m agora_rag corpus --status --dir corpus
```

Domains: `finance, crypto, cyber, transport, ecommerce, healthcare`.

## Output layout

```
corpus/
  raw/<domain>/<source_id>/        downloaded files + provenance.json
  text/<domain>/<source_id>.txt    plain-text conversion
  chunks.jsonl                     {id, domain, source_id, license_tier,
                                    url, seq, text} — ~1000 chars, 150 overlap
```

Each `provenance.json` records url, fetched_at, sha256, size, http status,
license tier, and parse status. Fetching is resumable (sources already `ok`
are skipped; `--force` refetches), polite (60 s timeout, 2 retries, UA
`agora-corpus-fetcher/0.1`, 200 MB per-download cap), and graceful — a failed
source is recorded, never fatal.

## License tiers (see CORPUS.md)

- **TIER A** — public domain / open license: **may be redistributed** with the
  released artifact (US federal, MITRE, EUR-Lex w/ attribution, CC0, arXiv).
- **TIER B** — free but copyright-reserved (FATF, BIS, Lockheed, Wolfsberg):
  **fetched locally only — never redistribute**; the artifact ships the URL +
  this fetcher, not the file.
- **TIER C** — reference-only / unclear rights (DeFiHackLabs, Etherscan
  labels, platform policies, paywalled HCM/CPT): **never fetched**; manifest
  entries exist with `fetch=False` for citation purposes only.

The `corpus/` directory is git-ignored data — do not commit fetched files.

## Offline RAG pipeline (M3b): description → validated rule base

The RAG pipeline (`mustread.txt` §9) turns a natural-language domain description
into a **validated, compiled rule-base YAML**, grounded in the M3a corpus. The
LLM is offline and one-time — never on the per-event hot path.

```bash
# 1) (re)build the FAISS embedding index over corpus/chunks.jsonl
python3 -m agora_rag index --corpus-dir corpus

# 2) compile a description into a rule base (embed-if-needed → retrieve →
#    draft → lint → Rust-validate → write)
python3 -m agora_rag build \
  --description "A ride-hailing marketplace where drivers and riders are matched; \
fraud includes fake GPS trips and collusive surge-inflating rings" \
  --domain-hint transport --out rulebase.yaml --corpus-dir corpus

# template drafter only (skip the Ollama probe)
python3 -m agora_rag build --description "..." --no-llm --out rb.yaml
```

### How it works (the 7 steps)

1. **Corpus** — the M3a `chunks.jsonl` (provenance per chunk).
2. **Embed** (`embed.py`) — a small open model (`BAAI/bge-small-en-v1.5`, fallback
   `all-MiniLM-L6-v2`) → a FAISS `IndexFlatIP` over L2-normalized vectors, with a
   row-aligned `meta.jsonl`. Cached: re-embedding is skipped when the index
   exists and the chunk count matches.
3. **Retrieve** (`retrieve.py`) — **hybrid**: dense FAISS cosine + a hand-rolled
   Okapi BM25 (no `rank_bm25` dep), fused by Reciprocal-Rank Fusion (k=60).
   Optional `--domain-hint` filter.
4. **Draft** (`draft.py`) — two modes behind one interface; output is **always**
   schema-valid typed config, never free text:
   - **LLM mode** — if an Ollama server answers at `http://localhost:11434`
     (2 s probe of `/api/tags`), a small open model (Qwen2.5-3B / Phi-3.5 /
     Llama-3.1-8B) drafts the structure under JSON-format decoding.
   - **Template/grounded mode** (always works) — maps the description to the
     closest of the 6 built-in domains and uses that gold YAML as a schema-valid
     scaffold, re-grounding `meta`/`provenance` to the retrieved standards.
   Every draft is tagged with the `source_id`/`url` of the grounding chunks.
5. **Validate + 7. Compile** (`validate_compile.py`) — a fast Python lint
   (primitives present, weights ≥ 0, names cross-referenced, `prevalence ≤ 0.05`),
   then serialize to YAML (serde external `!tags` preserved, see `yamltags.py`)
   and call the **Rust loader as the single source of truth**:
   `agora rules build --domain <yaml> --out <out>`. Non-zero exit surfaces its
   stderr. If an LLM draft fails Rust validation, the pipeline falls back to the
   always-valid template draft.
6. **Refine** — interactive slider/toggle editing is a demo-UI concern; the
   headless pipeline leaves it as a no-op hook.

### Offline-safety & reproducibility

- **No network is required.** If the embedding model can't be downloaded or
  loaded, `embed.py` falls back to a **deterministic** pure-numpy hashing
  embedder (bag-of-hashed-character-3grams → 384 dims, L2-normalized). The query
  side uses the *same* embedder family the index was built with. The embedder
  actually used (`sentence-transformers` vs `hashing-fallback`) is recorded in
  `corpus/index/embed_info.json` and printed.
- **No Ollama is required.** The drafter probes once and silently uses the
  template drafter if no LLM server is up.
- **Deterministic where possible**: the hashing embedder is content-addressed
  (blake2b); the template drafter is deterministic; LLM calls pin
  `temperature=0, seed=0`.
- **No new pip deps**: BM25 and the hashing embedder are implemented in-package.

### Index layout

```
corpus/index/
  faiss.index        IndexFlatIP over L2-normalized embeddings
  meta.jsonl         row-aligned {id, domain, source_id, url, license_tier, text}
  embed_info.json    {embedder, model, dim, n_chunks, normalized}
```
