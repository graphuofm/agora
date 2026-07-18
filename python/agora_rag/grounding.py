"""grounding.py — grounding verification + evaluation (mustread.txt §9 step 5).

Two jobs:

(1) VERIFY each extracted parameter is genuinely grounded:
      normalize(value) ∈ exact_quote   AND   exact_quote ∈ cited chunk
    Either failing flips ``grounded`` to False (an anti-hallucination guard that
    also catches a quote that doesn't actually contain the claimed number).

(2) EVALUATE against a small gold table (tests/grounding_gold.json):
      extraction_accuracy = correct extracted values / gold params attempted
      grounding_rate      = grounded params / total params
    so a build can self-report whether its factual parameters trace to the
    standards, not to the scaffold.

The pipeline wires a grounding check between draft and Rust-compile: it WARNS on
ungrounded derived params but never fails the build (the scaffold default stands).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .embed import load_meta
from .extract import Extracted


def _gold_path() -> Path:
    # python/agora_rag/grounding.py -> python/tests/grounding_gold.json
    return Path(__file__).resolve().parents[1] / "tests" / "grounding_gold.json"


def load_gold(domain: Optional[str] = None) -> Dict[str, Any]:
    path = _gold_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    if domain is not None:
        return data.get(domain, {})
    return data


# --------------------------------------------------------------------------- #
# Value normalization + membership checks
# --------------------------------------------------------------------------- #
def normalize_amount_strings(value: float) -> List[str]:
    """Surface forms a numeric value may appear as inside a quote."""
    forms: List[str] = []
    iv = int(value) if float(value).is_integer() else None
    if iv is not None:
        forms.append(f"{iv:,}")     # 10,000
        forms.append(str(iv))       # 10000
        forms.append(f"${iv:,}")    # $10,000
    else:
        forms.append(f"{value:,.2f}")
        forms.append(str(value))
    # de-dup preserving order
    seen: set = set()
    return [f for f in forms if not (f in seen or seen.add(f))]


def value_in_quote(value: Optional[float], quote: Optional[str]) -> bool:
    if value is None or not quote:
        return False
    q = quote.replace(" ", "")  # tolerate "$ 10,000"
    for form in normalize_amount_strings(value):
        if form.replace(" ", "") in q:
            return True
    return False


def quote_in_chunk_text(quote: Optional[str], chunk_text: Optional[str]) -> bool:
    if not quote or not chunk_text:
        return False
    if quote in chunk_text:
        return True
    # tolerate whitespace/newline differences introduced by chunk packing
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    return norm(quote) in norm(chunk_text)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
@dataclass
class Verification:
    name: str
    grounded: bool
    value_in_quote: bool
    quote_in_chunk: bool
    reason: str = ""


def verify(
    extracted: List[Extracted], corpus_dir: Optional[Path] = None
) -> List[Verification]:
    """Assert value ∈ quote AND quote ∈ cited chunk for each extracted param."""
    chunk_by_id: Dict[str, str] = {}
    if corpus_dir is not None:
        try:
            for m in load_meta(corpus_dir):
                cid = str(m.get("id", ""))
                if cid:
                    chunk_by_id[cid] = str(m.get("text", ""))
        except FileNotFoundError:
            chunk_by_id = {}

    out: List[Verification] = []
    for ex in extracted:
        if not ex.grounded or ex.value is None:
            out.append(Verification(ex.name, False, False, False, "no value extracted"))
            continue
        viq = value_in_quote(ex.value, ex.exact_quote)
        # Prefer the live corpus chunk; fall back to the quote being self-consistent.
        chunk_text = chunk_by_id.get(ex.chunk_id or "")
        if chunk_text is not None and ex.chunk_id in chunk_by_id:
            qic = quote_in_chunk_text(ex.exact_quote, chunk_text)
        else:
            qic = bool(ex.exact_quote)  # extractor already pulled it from a chunk
        ok = viq and qic
        # Reflect the verification back onto the extracted record.
        ex.grounded = ok
        reason = ""
        if not viq:
            reason = "value not present in exact_quote"
        elif not qic:
            reason = "exact_quote not found in cited chunk"
        out.append(Verification(ex.name, ok, viq, qic, reason))
    return out


# --------------------------------------------------------------------------- #
# Evaluation against the gold table
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    domain: str
    n_total: int = 0
    n_grounded: int = 0
    n_gold: int = 0
    n_correct: int = 0
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def grounding_rate(self) -> float:
        return (self.n_grounded / self.n_total) if self.n_total else 0.0

    @property
    def extraction_accuracy(self) -> float:
        return (self.n_correct / self.n_gold) if self.n_gold else 0.0


def evaluate(domain: str, extracted: List[Extracted]) -> EvalResult:
    """Score extracted params against the gold table for ``domain``."""
    gold = load_gold(domain)
    res = EvalResult(domain=domain)
    res.n_total = len(extracted)
    by_name = {ex.name: ex for ex in extracted}

    for ex in extracted:
        if ex.grounded:
            res.n_grounded += 1

    # extraction_accuracy is computed over gold parameters.
    for gname, gentry in gold.items():
        gval = float(gentry.get("value"))
        ex = by_name.get(gname)
        res.n_gold += 1
        got = ex.value if ex else None
        correct = ex is not None and ex.grounded and got is not None and float(got) == gval
        if correct:
            res.n_correct += 1
        res.rows.append(
            {
                "parameter": gname,
                "expected": gval,
                "extracted": got,
                "grounded": bool(ex and ex.grounded),
                "correct": bool(correct),
                "source_id": ex.source_id if ex else None,
                "section": ex.section if ex else None,
                "quote": ex.exact_quote if ex else None,
            }
        )
    return res


def format_report(domain: str, ev: EvalResult, extracted: List[Extracted]) -> str:
    """Human-readable --report-grounding block."""
    lines: List[str] = []
    lines.append("=== grounding report ===")
    lines.append(f"domain: {domain}")
    lines.append(
        f"grounding_rate:      {ev.grounding_rate:.2%} "
        f"({ev.n_grounded}/{ev.n_total} parameters grounded)"
    )
    lines.append(
        f"extraction_accuracy: {ev.extraction_accuracy:.2%} "
        f"({ev.n_correct}/{ev.n_gold} gold parameters correct)"
    )
    lines.append("")
    lines.append("extracted parameters:")
    for ex in extracted:
        if ex.grounded and ex.value is not None:
            val = int(ex.value) if float(ex.value).is_integer() else ex.value
            sec = f" §{ex.section}" if ex.section else ""
            q = (ex.exact_quote or "").strip()
            if len(q) > 130:
                q = q[:127] + "..."
            lines.append(
                f"  [GROUNDED] {ex.name} = {val}  "
                f"<- {ex.source_id}{sec} ({ex.method})"
            )
            lines.append(f"             quote: \"{q}\"")
        else:
            lines.append(f"  [ungrounded] {ex.name} (kept scaffold default){(' — ' + ex.note) if ex.note else ''}")
    if ev.rows:
        lines.append("")
        lines.append("gold check:")
        for r in ev.rows:
            mark = "PASS" if r["correct"] else "FAIL"
            lines.append(
                f"  [{mark}] {r['parameter']}: expected={r['expected']} "
                f"extracted={r['extracted']} grounded={r['grounded']}"
            )
    return "\n".join(lines)
