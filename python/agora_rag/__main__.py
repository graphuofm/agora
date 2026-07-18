"""CLI for the AGORA RAG corpus pipeline.

Examples:
    python3 -m agora_rag corpus --fetch --dir corpus
    python3 -m agora_rag corpus --fetch --domains finance,cyber --tier A
    python3 -m agora_rag corpus --parse --chunk
    python3 -m agora_rag corpus --status
    python3 -m agora_rag corpus            # fetch + parse + chunk (default)

    # offline RAG (M3b): (re)build the embedding index, then compile a rule base
    python3 -m agora_rag index --corpus-dir corpus
    python3 -m agora_rag build --description "..." --domain-hint finance \
        --out rulebase.yaml --corpus-dir corpus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from . import manifest
from .chunk import chunk_all
from .fetch import fetch_all
from .parse import parse_all


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _print_status(corpus_dir: Path) -> None:
    rows: List[Dict[str, str]] = []
    for src in manifest.SOURCES:
        prov_path = corpus_dir / "raw" / src.domain / src.id / "provenance.json"
        fetched, parsed, size = "-", "-", 0
        if not src.fetch:
            fetched = "ref-only"
        elif prov_path.exists():
            try:
                prov = json.loads(prov_path.read_text())
                fetched = prov.get("status", "?")
                size = sum(f.get("size", 0) for f in prov.get("files", []))
                parsed = prov.get("parse", {}).get("status", "-")
            except (json.JSONDecodeError, OSError):
                fetched = "corrupt-prov"
        rows.append(
            {
                "domain": src.domain,
                "id": src.id,
                "tier": src.license_tier,
                "fetched": fetched,
                "size": f"{size:,}" if size else "",
                "parsed": parsed,
            }
        )
    widths = {
        k: max(len(k), max(len(r[k]) for r in rows)) for k in rows[0]
    }
    header = "  ".join(k.upper().ljust(widths[k]) for k in widths)
    print(header)
    print("-" * len(header))
    last_domain = ""
    for r in rows:
        shown = dict(r)
        if r["domain"] == last_domain:
            shown["domain"] = ""
        last_domain = r["domain"]
        print("  ".join(shown[k].ljust(widths[k]) for k in widths))
    chunks = corpus_dir / "chunks.jsonl"
    if chunks.exists():
        with open(chunks, "r", encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
        print(f"\nchunks.jsonl: {n:,} chunks ({chunks.stat().st_size:,} bytes)")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="agora_rag")
    sub = parser.add_subparsers(dest="command", required=True)

    corpus = sub.add_parser("corpus", help="fetch/parse/chunk the RAG corpus")
    corpus.add_argument("--fetch", action="store_true", help="download TIER A/B sources")
    corpus.add_argument("--parse", action="store_true", help="convert raw files to text")
    corpus.add_argument("--chunk", action="store_true", help="write chunks.jsonl")
    corpus.add_argument("--status", action="store_true", help="print per-source status table")
    corpus.add_argument("--force", action="store_true", help="refetch even if already ok")
    corpus.add_argument("--dir", default="corpus", help="corpus directory (default: corpus)")
    corpus.add_argument(
        "--domains",
        help=f"comma-separated domain filter ({','.join(manifest.DOMAINS)})",
    )
    corpus.add_argument("--tier", help="comma-separated license tier filter (A,B)")

    # index: (re)build the embedding index over corpus/chunks.jsonl
    index_p = sub.add_parser("index", help="(re)build the FAISS embedding index")
    index_p.add_argument("--corpus-dir", default="corpus", help="corpus directory")
    index_p.add_argument("--model", default=None, help="embedding model name override")
    index_p.add_argument("--force", action="store_true", help="re-embed even if cached")

    # build: run the offline RAG (embed if needed -> retrieve -> draft -> validate)
    build_p = sub.add_parser("build", help="compile an NL description into a rule base")
    build_p.add_argument("--description", required=True, help="natural-language domain description")
    build_p.add_argument("--domain-hint", default=None, help="optional built-in domain hint")
    build_p.add_argument("--out", default="rulebase.yaml", help="output YAML path")
    build_p.add_argument("--corpus-dir", default="corpus", help="corpus directory")
    build_p.add_argument("-k", "--top-k", type=int, default=12, help="chunks to retrieve")
    build_p.add_argument("--model", default=None, help="embedding model name override")
    build_p.add_argument("--no-llm", action="store_true", help="skip Ollama; template drafter only")
    build_p.add_argument(
        "--report-grounding",
        action="store_true",
        help="print grounded-parameter extraction accuracy + grounding rate "
        "against the gold table (tests/grounding_gold.json)",
    )

    args = parser.parse_args(argv)

    if args.command == "index":
        from . import embed as embed_mod

        info = embed_mod.build_index(
            Path(args.corpus_dir).resolve(), model=args.model, force=args.force
        )
        print(
            f"[index] ready: {info.n_chunks:,} chunks, embedder={info.embedder}, "
            f"model={info.model}"
        )
        return 0

    if args.command == "build":
        from . import pipeline

        report = pipeline.build(
            description=args.description,
            out_path=Path(args.out).resolve(),
            corpus_dir=Path(args.corpus_dir).resolve(),
            domain_hint=args.domain_hint,
            k=args.top_k,
            model=args.model,
            prefer_llm=not args.no_llm,
            report_grounding=args.report_grounding,
        )
        print("\n=== build report ===")
        print(report.summary())
        print(f"index build: {report.index_build_seconds:.1f}s")
        if args.report_grounding:
            from . import grounding as grounding_mod

            domain = report.scaffold_domain or (args.domain_hint or "")
            ev = report.eval_result or grounding_mod.evaluate(domain, report.extracted)
            print()
            print(grounding_mod.format_report(domain, ev, report.extracted))
            if report.grounding_sidecar_path:
                print(f"\ngrounding sidecar: {report.grounding_sidecar_path}")
        if not report.rust_ok:
            print(f"\nERROR: Rust validation failed: {report.rust_stderr}")
            return 1
        return 0
    corpus_dir = Path(args.dir).resolve()
    domains = _split_csv(args.domains)
    tiers = [t.upper() for t in (_split_csv(args.tier) or [])] or None

    if domains:
        bad = [d for d in domains if d not in manifest.DOMAINS]
        if bad:
            parser.error(f"unknown domain(s) {bad}; valid: {manifest.DOMAINS}")
    if tiers and any(t not in ("A", "B") for t in tiers):
        parser.error("--tier accepts only A and/or B (TIER C is never fetched)")

    if args.status:
        _print_status(corpus_dir)
        return 0

    # default with no stage flags: run the full pipeline
    do_fetch = args.fetch or not (args.fetch or args.parse or args.chunk)
    do_parse = args.parse or not (args.fetch or args.parse or args.chunk)
    do_chunk = args.chunk or not (args.fetch or args.parse or args.chunk)

    if do_fetch:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        results = fetch_all(corpus_dir, domains=domains, tiers=tiers, force=args.force)
        ok = sum(1 for r in results if r["status"].startswith("ok"))
        print(f"[fetch] done: {ok}/{len(results)} sources ok")
    if do_parse:
        results = parse_all(corpus_dir, domains=domains)
        ok = sum(1 for r in results if r["status"] == "ok")
        print(f"[parse] done: {ok}/{len(results)} sources parsed")
    if do_chunk:
        chunk_all(corpus_dir, domains=domains)
    return 0


if __name__ == "__main__":
    sys.exit(main())
