"""pipeline.py — the end-to-end offline RAG build (mustread.txt §9).

Wires the 7-step pipeline: corpus (M3a) -> embed -> index/retrieve -> draft ->
validate -> (refine) -> compile. Refinement (step 6, interactive sliders) is a
demo-time UI concern and is left as a no-op hook here; the rest runs headless and
deterministically.

The build loop honours mustread §9's authority rule: the Rust loader is the
single source of truth. If an LLM draft fails Rust validation we fall back to the
always-valid template draft and re-validate.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import draft as draft_mod
from . import embed as embed_mod
from . import extract as extract_mod
from . import grounding as grounding_mod
from . import validate_compile as vc
from .retrieve import RetrievedChunk, Retriever


@dataclass
class BuildReport:
    embedder: str = ""
    embed_model: str = ""
    n_chunks: int = 0
    index_build_seconds: float = 0.0
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    drafter_mode: str = ""
    scaffold_domain: str = ""
    lint_errors: List[str] = field(default_factory=list)
    rust_ok: bool = False
    rust_stderr: str = ""
    out_path: Optional[str] = None
    written_yaml: Optional[str] = None
    # Grounded parameter extraction (mustread.txt §9).
    extracted: List[extract_mod.Extracted] = field(default_factory=list)
    grounding_overrides: List[Dict[str, Any]] = field(default_factory=list)
    ungrounded_params: List[str] = field(default_factory=list)
    grounding_sidecar_path: Optional[str] = None
    eval_result: Optional[grounding_mod.EvalResult] = None

    def summary(self) -> str:
        top = ", ".join(f"{c.source_id}({c.domain})" for c in self.retrieved[:5])
        n_grounded = sum(1 for e in self.extracted if e.grounded)
        return (
            f"embedder={self.embedder} model={self.embed_model} chunks={self.n_chunks}\n"
            f"drafter={self.drafter_mode} scaffold={self.scaffold_domain}\n"
            f"top retrieved: {top}\n"
            f"grounded params: {n_grounded}/{len(self.extracted)} "
            f"(overrides={len(self.grounding_overrides)})\n"
            f"rust_validated={self.rust_ok} out={self.out_path}"
        )


def ensure_index(corpus_dir: Path, model: Optional[str] = None) -> "embed_mod.EmbedInfo":
    """Build the embedding index if missing/stale; return its EmbedInfo."""
    return embed_mod.build_index(corpus_dir, model=model)


def build(
    description: str,
    out_path: Path,
    corpus_dir: Path,
    domain_hint: Optional[str] = None,
    k: int = 12,
    model: Optional[str] = None,
    prefer_llm: bool = True,
    report_grounding: bool = False,
) -> BuildReport:
    report = BuildReport(out_path=str(out_path))

    # Step 2-3: embed (if needed) + load retriever.
    t0 = time.time()
    info = ensure_index(corpus_dir, model=model)
    report.index_build_seconds = time.time() - t0
    report.embedder = info.embedder
    report.embed_model = info.model
    report.n_chunks = info.n_chunks

    retriever = Retriever(corpus_dir)
    report.retrieved = retriever.retrieve(description, domain=domain_hint, k=k)
    print(f"[retrieve] {len(report.retrieved)} chunks (domain filter={domain_hint or 'none'})")

    # Step 4: draft (LLM preferred, template fallback). Build a candidate list so
    # an LLM draft that fails Rust validation falls back to the template draft.
    candidates: List[Dict[str, object]] = []
    first = draft_mod.draft_rulebase(
        description,
        report.retrieved,
        domain_hint=domain_hint,
        corpus_dir=corpus_dir,
        prefer_llm=prefer_llm,
    )
    candidates.append(first)
    if first.get("_draft_meta", {}).get("mode") == "llm":
        # Always have the deterministic template draft ready as a fallback.
        candidates.append(
            draft_mod.draft_template(
                description, report.retrieved, corpus_dir, domain_hint
            )
        )

    # Step 5 + 7: lint -> GROUND -> serialize -> Rust-validate. Loop over candidates.
    last_err = ""
    for rb in candidates:
        dmeta = rb.get("_draft_meta", {})
        report.drafter_mode = str(dmeta.get("mode", ""))
        domain = str(dmeta.get("scaffold_domain", "") or domain_hint or "")
        report.scaffold_domain = domain

        # ---- Grounded parameter extraction (mustread.txt §9, the scientific core)
        # Re-ground factual numeric parameters from the standards. The scaffold
        # stays the STRUCTURAL prior; only spec'd factual params are re-grounded.
        extracted = extract_mod.extract_params(
            domain, corpus_dir, retriever=retriever, allow_llm=prefer_llm
        )
        if extracted:
            # Verify each grounded span: value ∈ quote AND quote ∈ cited chunk.
            grounding_mod.verify(extracted, corpus_dir=corpus_dir)
            gr = extract_mod.apply_extractions(rb, extracted)
            report.extracted = extracted
            report.grounding_overrides = gr.overridden
            report.ungrounded_params = gr.ungrounded
            # Replace silent provenance with extraction-derived, traceable lines.
            meta = dict(rb.get("meta", {}))
            ext_prov = draft_mod.provenance_from_extractions(extracted)
            meta["provenance"] = ext_prov + list(meta.get("provenance", []) or [])
            rb["meta"] = meta
            for ov in gr.overridden:
                print(f"[ground] OVERRIDE {ov['parameter']} @ {ov['yaml_path']}: "
                      f"scaffold={ov['scaffold_value']} -> EXTRACTED "
                      f"{ov['extracted_value']} (from {ov['source_id']})")
            for ex in extracted:
                if ex.grounded and ex.value is not None and ex.name not in {
                    o["parameter"] for o in gr.overridden
                }:
                    print(f"[ground] grounded {ex.name}={ex.value} from "
                          f"{ex.source_id}"
                          + (f" §{ex.section}" if ex.section else ""))
            if gr.ungrounded:
                print(f"[ground] WARNING: ungrounded derived params "
                      f"(kept scaffold default): {', '.join(gr.ungrounded)}")

        lint_res = vc.lint(rb)
        report.lint_errors = lint_res.errors
        if not lint_res.ok:
            last_err = "lint: " + "; ".join(lint_res.errors)
            print(f"[validate] python lint FAILED ({report.drafter_mode}): {last_err}")
            continue
        if lint_res.warnings:
            print(f"[validate] lint warnings: {'; '.join(lint_res.warnings)}")
        print(f"[validate] python lint passed ({report.drafter_mode} draft)")

        yaml_text = vc.to_yaml(rb)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml_text, encoding="utf-8")
        report.written_yaml = yaml_text

        # Rust loader = single source of truth. It re-serializes in place to out.
        rust = vc.run_rust_validator(out_path, out_path)
        if rust.ok:
            report.rust_ok = True
            report.rust_stderr = ""
            print(f"[compile] Rust validator ACCEPTED ({report.drafter_mode}): "
                  f"{rust.stdout.strip()}")
            _write_grounding_sidecar(report, out_path, domain)
            return report
        last_err = rust.stderr.strip()
        report.rust_stderr = last_err
        print(f"[compile] Rust validator REJECTED ({report.drafter_mode}): {last_err}")
        print("[compile] falling back to next candidate draft ...")

    report.rust_ok = False
    report.rust_stderr = last_err
    return report


def _write_grounding_sidecar(
    report: BuildReport, out_path: Path, domain: str
) -> None:
    """Write rulebase.grounding.json next to the YAML; compute the gold eval."""
    if not report.extracted:
        return
    ev = grounding_mod.evaluate(domain, report.extracted)
    report.eval_result = ev
    grr = extract_mod.GroundingReport(
        extracted=report.extracted,
        overridden=report.grounding_overrides,
        ungrounded=report.ungrounded_params,
        applied=[
            ex.yaml_path
            for ex in report.extracted
            if ex.grounded and ex.value is not None
            and not ex.yaml_path.startswith("_grounding_only")
        ],
    )
    sidecar = grr.sidecar()
    sidecar["domain"] = domain
    sidecar["eval"] = {
        "extraction_accuracy": ev.extraction_accuracy,
        "grounding_rate": ev.grounding_rate,
        "n_correct": ev.n_correct,
        "n_gold": ev.n_gold,
        "n_grounded": ev.n_grounded,
        "n_total": ev.n_total,
        "rows": ev.rows,
    }
    side_path = out_path.parent / (out_path.stem + ".grounding.json")
    side_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    report.grounding_sidecar_path = str(side_path)
    print(f"[ground] wrote grounding sidecar -> {side_path}")
