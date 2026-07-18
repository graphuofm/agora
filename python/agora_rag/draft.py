"""draft.py — draft a rule base from retrieved context (mustread.txt §9 step 4).

Two modes behind one interface ``draft_rulebase(...) -> dict``; the result is
ALWAYS a schema-valid typed rule-base dict (never free text), per mustread §9
("output is ALWAYS valid typed config").

(a) LLM mode — if an Ollama server is reachable at http://localhost:11434, build
    a strict JSON-schema-instructed prompt from the retrieved chunks and ask a
    small open model (Qwen2.5 / Phi-3.5 / Llama-3.1) to emit the rule-base
    structure. Parsed and returned. Anything that goes wrong falls through to:

(b) TEMPLATE/grounded mode (always works) — pick the closest of the six built-in
    domains (by embedding similarity of the description to each domain's corpus
    chunks), load that domain's gold YAML as a *schema-valid scaffold*, and adapt
    its ``meta`` (id/name/description) and ``provenance`` from the retrieved
    chunks. Names/topology/processes are inherited from the proven scaffold so the
    output validates; provenance is re-grounded to the retrieved standards.

Every drafted rule's provenance is tagged with the source_id/url of the chunks
that grounded it (traceability, mustread §9 step 5). The mode that ran is
recorded under ``_draft_meta`` in the returned dict (stripped before YAML write).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .embed import Embedder, hash_embed_one, load_embed_info
from .retrieve import RetrievedChunk

BUILTIN_DOMAINS = ["finance", "crypto", "cyber", "transport", "ecommerce", "healthcare"]
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODELS = ["qwen2.5:3b", "qwen2.5:7b", "phi3.5", "llama3.1:8b"]


def _domains_dir() -> Path:
    # python/agora_rag/draft.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "crates" / "agora-rules" / "domains"


# --------------------------------------------------------------------------- #
# Domain selection by embedding similarity
# --------------------------------------------------------------------------- #
def _embed_query(description: str, embedder: Embedder) -> np.ndarray:
    v = embedder.encode([description])[0]
    return np.asarray(v, dtype=np.float32)


def closest_domain(
    description: str,
    retrieved: List[RetrievedChunk],
    corpus_dir: Path,
    domain_hint: Optional[str] = None,
) -> Tuple[str, Dict[str, float]]:
    """Map the description to the nearest built-in domain.

    Strategy: vote with the retrieved chunks' domains (lexically grounded), and
    break ties / sanity-check with embedding similarity of the description to a
    per-domain centroid of corpus-chunk embeddings. A valid ``domain_hint`` wins.
    Returns (domain, score_breakdown).
    """
    if domain_hint and domain_hint in BUILTIN_DOMAINS:
        return domain_hint, {domain_hint: 1.0}

    # Vote by retrieved-chunk domain membership, weighted by RRF score.
    votes: Dict[str, float] = Counter()
    for c in retrieved:
        if c.domain in BUILTIN_DOMAINS:
            votes[c.domain] += c.score
    if votes:
        best = max(votes.items(), key=lambda kv: kv[1])[0]
        return best, dict(votes)

    # Fallback: embedding similarity to per-domain centroids.
    info = load_embed_info(corpus_dir)
    force_hash = info is not None and info.embedder == "hashing-fallback"
    embedder = Embedder(
        model=(info.model if info and not force_hash else None),
        dim=(info.dim if info else 384),
        force_hash=force_hash,
    )
    qv = _embed_query(description, embedder)
    sims = _domain_centroid_sims(qv, corpus_dir, embedder)
    if sims:
        return max(sims.items(), key=lambda kv: kv[1])[0], sims
    return "finance", {"finance": 0.0}


def _domain_centroid_sims(
    qv: np.ndarray, corpus_dir: Path, embedder: Embedder
) -> Dict[str, float]:
    """Cosine similarity of the query to each domain's mean chunk embedding."""
    from .embed import load_meta

    try:
        meta = load_meta(corpus_dir)
    except FileNotFoundError:
        return {}
    by_domain: Dict[str, List[str]] = {}
    for m in meta:
        d = str(m.get("domain", ""))
        if d in BUILTIN_DOMAINS and len(by_domain.get(d, [])) < 64:
            by_domain.setdefault(d, []).append(str(m.get("text", "")))
    sims: Dict[str, float] = {}
    for d, texts in by_domain.items():
        cent = embedder.encode(texts).mean(axis=0)
        n = float(np.linalg.norm(cent))
        if n > 0:
            cent = cent / n
        sims[d] = float(np.dot(qv, cent))
    return sims


# --------------------------------------------------------------------------- #
# Provenance grounding from retrieved chunks
# --------------------------------------------------------------------------- #
def _provenance_from_chunks(
    retrieved: List[RetrievedChunk], limit: int = 6
) -> List[Dict[str, object]]:
    """Build a deduplicated provenance list (source_id/url/license) from chunks."""
    seen: set = set()
    prov: List[Dict[str, object]] = []
    for c in retrieved:
        key = (c.source_id, c.url)
        if key in seen:
            continue
        seen.add(key)
        entry: Dict[str, object] = {"source": c.source_id}
        if c.url:
            entry["url"] = c.url
        if c.license_tier:
            entry["license_tier"] = c.license_tier
        prov.append(entry)
        if len(prov) >= limit:
            break
    return prov


def _grounding_ids(retrieved: List[RetrievedChunk], limit: int = 8) -> List[str]:
    return [c.id for c in retrieved[:limit]]


def provenance_from_extractions(extracted: List[object]) -> List[Dict[str, object]]:
    """Build SCHEMA-VALID provenance entries from grounded parameter extractions.

    Each grounded factual parameter (e.g. the CTR threshold) yields a provenance
    line that names the source AND the parameter/section it was grounded in —
    replacing the old blanket re-stamp with traceable, parameter-level provenance.

    The Rust ``Provenance`` struct only accepts ``source``/``url``/``version``/
    ``license_tier``, so the parameter name + grounded value + section are folded
    into those string fields (the full quote + char span live in the
    ``rulebase.grounding.json`` sidecar). One entry per grounded parameter, plus
    one deduped source header.
    """
    prov: List[Dict[str, object]] = []
    seen: set = set()
    for ex in extracted:
        if not getattr(ex, "grounded", False) or getattr(ex, "value", None) is None:
            continue
        sid = getattr(ex, "source_id", None)
        section = getattr(ex, "section", None)
        name = getattr(ex, "name", None)
        value = getattr(ex, "value", None)
        key = (sid, section, name)
        if key in seen:
            continue
        seen.add(key)
        val = int(value) if float(value).is_integer() else value
        sec = f" §{section}" if section else ""
        entry: Dict[str, object] = {
            "source": f"{sid}{sec} — grounded {name}={val}",
            "version": "grounded-extraction",
        }
        url = getattr(ex, "url", None)
        if url:
            entry["url"] = url
        prov.append(entry)
    return prov


# --------------------------------------------------------------------------- #
# Template / grounded drafter (always works)
# --------------------------------------------------------------------------- #
def _load_scaffold(domain: str) -> Dict[str, object]:
    from . import yamltags

    path = _domains_dir() / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"built-in scaffold not found: {path}")
    return yamltags.load_file(str(path))


def draft_template(
    description: str,
    retrieved: List[RetrievedChunk],
    corpus_dir: Path,
    domain_hint: Optional[str] = None,
) -> Dict[str, object]:
    domain, breakdown = closest_domain(
        description, retrieved, corpus_dir, domain_hint
    )
    rb = _load_scaffold(domain)

    # Re-ground meta: keep the proven structure, adopt the user's description and
    # the retrieved standards' provenance (traceability).
    meta = dict(rb.get("meta", {}))
    meta["description"] = description.strip()
    chunk_prov = _provenance_from_chunks(retrieved)
    base_prov = list(meta.get("provenance", []) or [])
    # retrieved standards first (the grounding), then the scaffold's own cites
    meta["provenance"] = chunk_prov + base_prov
    rb["meta"] = meta

    rb["_draft_meta"] = {
        "mode": "template",
        "scaffold_domain": domain,
        "domain_selection": breakdown,
        "grounding_chunk_ids": _grounding_ids(retrieved),
        "grounding_sources": [p.get("source") for p in chunk_prov],
    }
    return rb


# --------------------------------------------------------------------------- #
# Ollama LLM drafter (optional)
# --------------------------------------------------------------------------- #
def _ollama_available(timeout: float = 2.0) -> Optional[str]:
    """Return an available Ollama model name, or None if unreachable."""
    try:
        import requests

        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return None
        tags = resp.json().get("models", [])
        names = {m.get("name", "") for m in tags}
        for preferred in OLLAMA_MODELS:
            for n in names:
                if n == preferred or n.startswith(preferred.split(":")[0]):
                    return n
        # any model is better than none
        return next(iter(names)) if names else None
    except Exception:  # noqa: BLE001 — probe must never raise
        return None


def _build_llm_prompt(
    description: str, retrieved: List[RetrievedChunk], domain: str
) -> str:
    ctx = "\n\n".join(
        f"[{c.source_id}] {c.text[:600]}" for c in retrieved[:8]
    )
    schema_hint = (
        "Emit a SINGLE JSON object for a AGORA rule base with keys: meta "
        "{id,name,description,schema_version,provenance:[{source,url,license_tier}]}, "
        "entity_types, relations, event_types, behaviors, constraints, adversaries, "
        "failures, control{prevalence(<=0.05),difficulty,type_mix,placement,cascade}. "
        "Every referenced name must be defined. Anomalies are rare. Output JSON only."
    )
    return (
        f"You are compiling a graph-anomaly simulation rule base for the '{domain}' "
        f"domain.\n\nDESCRIPTION:\n{description}\n\nGROUNDING STANDARDS:\n{ctx}\n\n"
        f"{schema_hint}\n"
    )


def _extract_json(text: str) -> Optional[Dict[str, object]]:
    # Grab the first balanced {...} block.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def draft_llm(
    description: str,
    retrieved: List[RetrievedChunk],
    corpus_dir: Path,
    domain_hint: Optional[str],
    model: str,
) -> Optional[Dict[str, object]]:
    try:
        import requests

        domain, _ = closest_domain(description, retrieved, corpus_dir, domain_hint)
        prompt = _build_llm_prompt(description, retrieved, domain)
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 0},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            return None
        rb = _extract_json(resp.json().get("response", ""))
        if not isinstance(rb, dict) or "meta" not in rb:
            return None
        # NOTE: provenance is NO LONGER silently re-stamped here. Grounding is now
        # derived from the parameter-extraction stage (extract.py): the pipeline
        # adds extraction-backed provenance + a rulebase.grounding.json sidecar so
        # each factual parameter traces to a specific source span, rather than the
        # LLM draft being blanket-tagged with the top retrieved chunks. We keep
        # only schema hygiene (a default schema_version) here.
        meta = dict(rb.get("meta", {}))
        meta.setdefault("schema_version", 1)
        if not meta.get("provenance"):
            # ensure the dict at least has the key; the pipeline fills real,
            # extraction-derived provenance after grounding.
            meta["provenance"] = []
        rb["meta"] = meta
        rb["_draft_meta"] = {
            "mode": "llm",
            "llm_model": model,
            "scaffold_domain": domain,
            "grounding_chunk_ids": _grounding_ids(retrieved),
        }
        return rb
    except Exception:  # noqa: BLE001 — any LLM failure -> template fallback
        return None


# --------------------------------------------------------------------------- #
# Unified interface
# --------------------------------------------------------------------------- #
def draft_rulebase(
    description: str,
    retrieved_chunks: List[RetrievedChunk],
    domain_hint: Optional[str] = None,
    corpus_dir: Optional[Path] = None,
    prefer_llm: bool = True,
) -> Dict[str, object]:
    """Draft a schema-valid rule-base dict. Tries LLM, falls back to template."""
    cdir = corpus_dir or Path("corpus")
    if prefer_llm:
        model = _ollama_available()
        if model:
            print(f"[draft] Ollama reachable; drafting with model '{model}'")
            rb = draft_llm(description, retrieved_chunks, cdir, domain_hint, model)
            if rb is not None:
                return rb
            print("[draft] LLM draft failed/invalid; falling back to template drafter")
        else:
            print("[draft] no Ollama server at localhost:11434; using template drafter")
    rb = draft_template(description, retrieved_chunks, cdir, domain_hint)
    sel = rb.get("_draft_meta", {}).get("scaffold_domain")
    print(f"[draft] template drafter: scaffolded from built-in domain '{sel}'")
    return rb
