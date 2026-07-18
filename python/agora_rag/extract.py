"""extract.py — grounded parameter extraction (mustread.txt §9, the scientific core).

The drafter (draft.py) reuses a proven scaffold as a STRUCTURAL prior. That is the
right default for topology / process shape — but factual numeric PARAMETERS (the
CTR reporting threshold, the SAR threshold, FATF wire de-minimis, ...) must be
GROUNDED in the downloaded standard, not copied from the scaffold. This module
closes that gap: for each extractable parameter it

  1. retrieves PER PARAMETER (source- and section-filtered), then
  2. pulls the value as a grounded span with a DETERMINISTIC regex pre-pass
     (no LLM): find ``$10,000`` / "10,000" / "ten thousand" near the parameter's
     trigger terms, in the cited section, and pick the best-supported value;
  3. only if the regex fails does it (optionally) ask Ollama for ONE parameter,
     constrained to {value, exact_quote}, and verifies the quote occurs in a
     retrieved chunk.

Each result is ``(value, source_id, char_span, exact_quote)``. The value is written
into the scaffold at the spec's ``yaml_path``; a parameter that cannot be grounded
keeps its scaffold default and is flagged ``ungrounded``. Determinism: the regex
pre-pass has no randomness; the optional LLM call uses temperature 0 / seed 0.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .retrieve import RetrievedChunk, Retriever

OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Parameter spec table (FINANCE first; a few high-value, verifiable facts)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParamSpec:
    """One extractable factual parameter.

    yaml_path     dotted path into the rule-base dict; list elements are selected
                  by name with ``key[name=value]`` and Tagged enums are descended
                  transparently (see _set_path / _get_path).
    query         per-parameter retrieval query.
    kind          extraction kind ("usd_amount").
    triggers      terms whose presence near a candidate number boosts it.
    source_filter source_ids whose chunks are preferred during retrieval.
    section_hint  eCFR section number to prefer (structure-aware chunking tag).
    expected      ground-truth value (used by the gold eval / wrong-scaffold test).
    prefer        "max"|"min"|"first" tie-break among equally-supported candidates.
    """

    name: str
    yaml_path: str
    query: str
    kind: str
    triggers: Tuple[str, ...]
    source_filter: Tuple[str, ...] = ()
    section_hint: Optional[str] = None
    expected: Optional[float] = None
    prefer: str = "first"
    domain: str = "finance"


# FinCEN 31 CFR is the authoritative source for CTR / SAR. The section_hint is
# the precise eCFR SECTION tag emitted by the structure-aware chunker, which lets
# per-parameter retrieval isolate the ONE defining section from the hundreds of
# unrelated "$10,000" mentions across Title 31.
PARAM_SPECS: Dict[str, List[ParamSpec]] = {
    "finance": [
        ParamSpec(
            name="ctr_threshold",
            yaml_path="constraints[name=ctr_structuring_window].check.threshold",
            query="currency transaction report filing obligation transaction in "
            "currency of more than dollars threshold",
            kind="usd_amount",
            triggers=(
                "transaction in currency", "currency transaction", "report",
                "filing", "more than", "exceeds", "reporting",
            ),
            source_filter=("fincen_31cfr_chx",),
            section_hint="1010.311",
            expected=10000.0,
            prefer="first",
        ),
        ParamSpec(
            name="sar_threshold",
            yaml_path="_grounding_only.sar_threshold",
            query="suspicious activity report banks reports of suspicious "
            "transactions involves or aggregates at least dollars in funds",
            kind="usd_amount",
            triggers=(
                "suspicious", "aggregates", "at least", "in funds", "report",
                "involves",
            ),
            source_filter=("fincen_31cfr_chx",),
            section_hint="1020.320",
            expected=5000.0,
            prefer="first",
        ),
        ParamSpec(
            name="structuring_upper",
            # the structuring near-threshold band upper bound mirrors the CTR
            # threshold (sub-CTR cash deposits); ground it from the same section.
            yaml_path="constraints[name=ctr_structuring_window].check.threshold",
            query="structuring breaking down a single sum of currency exceeding "
            "dollars into smaller sums at or below",
            kind="usd_amount",
            triggers=("structur", "breaking down", "exceeding", "at or below",
                      "evading", "reporting"),
            source_filter=("fincen_31cfr_chx",),
            section_hint="1010.311",
            expected=10000.0,
            prefer="first",
            # this is an alias of ctr_threshold for cross-checking; not written
            # separately (same yaml_path) — kept only in grounding sidecar.
        ),
    ],
}


def specs_for(domain: str) -> List[ParamSpec]:
    return PARAM_SPECS.get(domain, [])


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class Extracted:
    """The outcome of extracting one parameter."""

    name: str
    yaml_path: str
    value: Optional[float]
    grounded: bool
    source_id: Optional[str] = None
    chunk_id: Optional[str] = None
    char_span: Optional[Tuple[int, int]] = None
    exact_quote: Optional[str] = None
    method: str = "none"          # "regex" | "llm" | "none"
    expected: Optional[float] = None
    section: Optional[str] = None
    note: str = ""

    def to_grounding(self) -> Dict[str, Any]:
        return {
            "parameter": self.name,
            "yaml_path": self.yaml_path,
            "value": self.value,
            "grounded": self.grounded,
            "method": self.method,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "section": self.section,
            "char_span": list(self.char_span) if self.char_span else None,
            "exact_quote": self.exact_quote,
            "expected": self.expected,
            "note": self.note,
        }


# --------------------------------------------------------------------------- #
# Deterministic numeric pre-pass
# --------------------------------------------------------------------------- #
# A USD amount: optional $, digits with thousands separators, optional cents.
_USD_RE = re.compile(r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?|\$\s?\d+(?:\.\d+)?")
# Bare grouped numbers ("10,000") even without a $ sign.
_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")

# Spelled-out amounts ("ten thousand dollars").
_WORD_NUMS = {
    "thousand": 1000.0, "million": 1_000_000.0, "billion": 1_000_000_000.0,
}
_WORD_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "twenty-five": 25, "fifty": 50,
}
_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted(_WORD_UNITS, key=len, reverse=True)) + r")[\s-]+"
    r"(thousand|million|billion)\b",
    re.IGNORECASE,
)

# Sentinel amounts that are almost never a real regulatory threshold.
_IMPLAUSIBLE = {100.0, 100000.0, 1000000.0, 10000000.0}


def _parse_amount(token: str) -> Optional[float]:
    cleaned = token.replace("$", "").replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class _Candidate:
    value: float
    span: Tuple[int, int]      # char span in the chunk text
    quote: str                 # the surrounding sentence (the exact_quote)
    trigger_hits: int
    raw: str                   # the matched numeric token


def _sentence_around(text: str, start: int, end: int) -> Tuple[str, Tuple[int, int]]:
    """Return the sentence containing [start,end) and its char span in text."""
    left = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    s = left + 1 if left != -1 else 0
    nxt = text.find(". ", end)
    nl = text.find("\n", end)
    cands = [p for p in (nxt, nl) if p != -1]
    e = (min(cands) + 1) if cands else len(text)
    return text[s:e].strip(), (s, e)


def _numeric_candidates(text: str, triggers: Tuple[str, ...]) -> List[_Candidate]:
    low = text.lower()
    cands: List[_Candidate] = []
    seen_spans: set = set()

    def add(match_start: int, match_end: int, value: float, raw: str) -> None:
        if value is None or value <= 0:
            return
        key = (match_start, match_end)
        if key in seen_spans:
            return
        seen_spans.add(key)
        quote, _span = _sentence_around(text, match_start, match_end)
        win = low[max(0, match_start - 160):min(len(low), match_end + 160)]
        hits = sum(1 for t in triggers if t.lower() in win)
        cands.append(_Candidate(value, (match_start, match_end), quote, hits, raw))

    for m in _USD_RE.finditer(text):
        v = _parse_amount(m.group(0))
        if v is not None:
            add(m.start(), m.end(), v, m.group(0))
    for m in _NUM_RE.finditer(text):
        v = _parse_amount(m.group(0))
        if v is not None:
            add(m.start(), m.end(), v, m.group(0))
    for m in _WORD_RE.finditer(text):
        unit = _WORD_UNITS.get(m.group(1).lower(), 0)
        scale = _WORD_NUMS.get(m.group(2).lower(), 0.0)
        v = unit * scale
        if v > 0:
            add(m.start(), m.end(), v, m.group(0))
    return cands


def _score_candidate(c: _Candidate) -> Tuple[int, int, float]:
    """Higher is better: trigger support, then plausibility, then magnitude prior."""
    plausible = 0 if c.value in _IMPLAUSIBLE else 1
    return (c.trigger_hits, plausible, 0.0)


def _regex_extract(
    spec: ParamSpec, chunks: List[RetrievedChunk]
) -> Optional[Extracted]:
    """Deterministic regex/numeric pre-pass over the top retrieved chunks."""
    best: Optional[Tuple[Tuple[int, int, float], _Candidate, RetrievedChunk]] = None
    for chunk in chunks:
        for cand in _numeric_candidates(chunk.text, spec.triggers):
            if cand.trigger_hits == 0:
                continue  # require at least one trigger near the number
            score = _score_candidate(cand)
            if best is None:
                best = (score, cand, chunk)
                continue
            bscore, bcand, _ = best
            if score > bscore:
                best = (score, cand, chunk)
            elif score == bscore:
                # deterministic tie-break by spec.prefer, then earliest span
                if spec.prefer == "max" and cand.value > bcand.value:
                    best = (score, cand, chunk)
                elif spec.prefer == "min" and cand.value < bcand.value:
                    best = (score, cand, chunk)
    if best is None:
        return None
    _score, cand, chunk = best
    return Extracted(
        name=spec.name,
        yaml_path=spec.yaml_path,
        value=cand.value,
        grounded=True,
        source_id=chunk.source_id,
        chunk_id=chunk.id,
        char_span=cand.span,
        exact_quote=cand.quote,
        method="regex",
        expected=spec.expected,
        section=chunk.section,
    )


# --------------------------------------------------------------------------- #
# Optional one-parameter LLM fallback (constrained, verified)
# --------------------------------------------------------------------------- #
def _llm_extract_one(
    spec: ParamSpec, chunks: List[RetrievedChunk], model: str
) -> Optional[Extracted]:
    """Ask Ollama for ONE parameter as {value, exact_quote}; verify the quote."""
    try:
        import json as _json

        import requests

        ctx = "\n\n".join(f"[{c.source_id}#{c.id}] {c.text[:700]}" for c in chunks[:5])
        prompt = (
            f"Extract a single numeric regulatory parameter from the context.\n"
            f"PARAMETER: {spec.name}\nDESCRIPTION: {spec.query}\n\n"
            f"CONTEXT:\n{ctx}\n\n"
            f'Return ONLY JSON: {{"value": <number, no $ or commas>, '
            f'"exact_quote": "<verbatim sentence from the context containing the '
            f'number>"}}. If absent, return {{"value": null, "exact_quote": ""}}.'
        )
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 0},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return None
        obj = _json.loads(resp.json().get("response", "{}"))
        raw_val = obj.get("value")
        quote = (obj.get("exact_quote") or "").strip()
        if raw_val in (None, "") or not quote:
            return None
        value = _parse_amount(str(raw_val))
        if value is None or value <= 0:
            return None
        # Verify the quote actually occurs in a retrieved chunk (anti-hallucination).
        for c in chunks:
            idx = c.text.find(quote[:80]) if len(quote) >= 80 else c.text.find(quote)
            if idx != -1:
                return Extracted(
                    name=spec.name,
                    yaml_path=spec.yaml_path,
                    value=value,
                    grounded=True,
                    source_id=c.source_id,
                    chunk_id=c.id,
                    char_span=(idx, idx + len(quote)),
                    exact_quote=quote,
                    method="llm",
                    expected=spec.expected,
                    section=c.section,
                )
        return None  # quote not found in any chunk -> reject (hallucination guard)
    except Exception:  # noqa: BLE001 — any LLM failure -> ungrounded
        return None


def _ollama_model(timeout: float = 2.0) -> Optional[str]:
    try:
        import requests

        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        if resp.status_code != 200:
            return None
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        for pref in ("qwen2.5:7b", "qwen2.5:3b", "phi3.5", "llama3.1:8b"):
            for n in names:
                if n == pref or n.startswith(pref.split(":")[0]):
                    return n
        return names[0] if names else None
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_params(
    domain: str,
    corpus_dir: Path,
    retriever: Optional[Retriever] = None,
    allow_llm: bool = False,
    k: int = 8,
) -> List[Extracted]:
    """Extract every spec'd parameter for a domain (deterministic-first)."""
    specs = specs_for(domain)
    if not specs:
        return []
    retr = retriever or Retriever(corpus_dir)
    llm_model = _ollama_model() if allow_llm else None

    results: List[Extracted] = []
    for spec in specs:
        chunks = retr.retrieve_param(
            spec.query,
            source_filter=list(spec.source_filter) or None,
            domain=spec.domain,
            section_hint=spec.section_hint,
            k=k,
        )
        ex = _regex_extract(spec, chunks)
        if ex is None and llm_model:
            ex = _llm_extract_one(spec, chunks, llm_model)
        if ex is None:
            ex = Extracted(
                name=spec.name,
                yaml_path=spec.yaml_path,
                value=None,
                grounded=False,
                expected=spec.expected,
                method="none",
                note="no grounded value found in retrieved chunks",
            )
        results.append(ex)
    return results


# --------------------------------------------------------------------------- #
# yaml_path navigation (list-by-name + Tagged-transparent)
# --------------------------------------------------------------------------- #
def _step_into(node: Any, step: str) -> Tuple[Any, Any]:
    """Resolve one path step on ``node``; return (container, key) for set/get.

    ``container`` is the mapping/list to mutate; ``key`` indexes into it. Tagged
    enum wrappers (from yamltags) are descended into their ``.value`` mapping.
    """
    from .yamltags import Tagged

    # list selector: name[attr=value]
    m = re.match(r"^([A-Za-z0-9_]+)\[([A-Za-z0-9_]+)=([^\]]+)\]$", step)
    if m:
        listkey, attr, val = m.group(1), m.group(2), m.group(3)
        seq = node.get(listkey) if isinstance(node, dict) else None
        if isinstance(seq, list):
            for elem in seq:
                target = elem.value if isinstance(elem, Tagged) else elem
                if isinstance(target, dict) and str(target.get(attr)) == val:
                    return target, None  # caller continues from this dict
        raise KeyError(f"list selector '{step}' not found")
    # plain key on a dict (descend Tagged transparently)
    if isinstance(node, Tagged):
        node = node.value
    if isinstance(node, dict):
        return node, step
    raise KeyError(f"cannot resolve step '{step}' on {type(node).__name__}")


def _resolve(rb: Dict[str, Any], path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Walk ``path`` to the (container_dict, final_key) for get/set, or (None,None)."""
    from .yamltags import Tagged

    steps = path.split(".")
    node: Any = rb
    for i, step in enumerate(steps):
        last = i == len(steps) - 1
        try:
            container, key = _step_into(node, step)
        except KeyError:
            return None, None
        if key is None:
            # list selector resolved to a dict element; that dict IS the new node
            node = container
            if last:
                return None, None  # selector can't be a terminal value
            continue
        if last:
            return container, key
        nxt = container.get(key)
        node = nxt.value if isinstance(nxt, Tagged) else nxt
        if node is None:
            return None, None
    return None, None


def get_path(rb: Dict[str, Any], path: str) -> Any:
    container, key = _resolve(rb, path)
    if container is None or key is None:
        return None
    return container.get(key)


def set_path(rb: Dict[str, Any], path: str, value: Any) -> bool:
    """Set ``path`` to ``value`` in-place; return True on success."""
    if path.startswith("_grounding_only"):
        return False  # grounding-sidecar-only parameter (not written to YAML)
    container, key = _resolve(rb, path)
    if container is None or key is None:
        return False
    container[key] = value
    return True


# --------------------------------------------------------------------------- #
# Apply extracted values into a drafted rule-base dict
# --------------------------------------------------------------------------- #
@dataclass
class GroundingReport:
    extracted: List[Extracted] = field(default_factory=list)
    applied: List[str] = field(default_factory=list)        # yaml_paths written
    overridden: List[Dict[str, Any]] = field(default_factory=list)  # old->new
    ungrounded: List[str] = field(default_factory=list)     # param names

    def sidecar(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for ex in self.extracted:
            params[ex.name] = ex.to_grounding()
        return {
            "parameters": params,
            "applied_paths": self.applied,
            "overrides": self.overridden,
            "ungrounded": self.ungrounded,
        }


def apply_extractions(
    rb: Dict[str, Any], extracted: List[Extracted]
) -> GroundingReport:
    """Write grounded values into the scaffold; record provenance + overrides."""
    report = GroundingReport(extracted=extracted)
    for ex in extracted:
        if not ex.grounded or ex.value is None:
            report.ungrounded.append(ex.name)
            continue
        if ex.yaml_path.startswith("_grounding_only"):
            continue  # grounded for the record/eval, but not a YAML slot
        old = get_path(rb, ex.yaml_path)
        ok = set_path(rb, ex.yaml_path, float(ex.value))
        if not ok:
            ex.note = (ex.note + "; " if ex.note else "") + "yaml_path unresolved"
            report.ungrounded.append(ex.name)
            continue
        report.applied.append(ex.yaml_path)
        if old is not None and float(old) != float(ex.value):
            report.overridden.append(
                {
                    "parameter": ex.name,
                    "yaml_path": ex.yaml_path,
                    "scaffold_value": float(old),
                    "extracted_value": float(ex.value),
                    "source_id": ex.source_id,
                    "quote": ex.exact_quote,
                }
            )
    return report
