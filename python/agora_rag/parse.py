"""Convert fetched raw files to plain text: corpus/text/<domain>/<source_id>.txt.

Handlers by format:
  pdf       — pypdf (auto `pip install pypdf` if missing; else skip+record)
  html      — stdlib html.parser, scripts/styles stripped
  xml       — text nodes via ElementTree (eCFR, CAPEC)
  stix      — ATT&CK STIX 2.1: external_id + name + description of
              attack-patterns and tactics
  json      — sensible flattening (NVD CVEs special-cased)
  csv       — header + rows as text lines, capped at 50,000 lines
  markdown  — passed through
  zip       — csv/txt/json/xml members extracted and handled as above

Parse status is recorded into each source's provenance.json under "parse".
"""
from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree

CSV_LINE_CAP = 50_000
_BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "table", "tr", "br", "section", "article",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "dt", "dd",
}

_pypdf_checked = False
_pypdf_ok = False


def _ensure_pypdf() -> bool:
    """Return True iff pypdf is importable (installing it if necessary)."""
    global _pypdf_checked, _pypdf_ok
    if _pypdf_checked:
        return _pypdf_ok
    _pypdf_checked = True
    show = subprocess.run(
        [sys.executable, "-m", "pip", "show", "pypdf"],
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        print("[parse] pypdf not installed; attempting pip install ...")
        inst = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--user", "pypdf"],
            capture_output=True,
            text=True,
        )
        if inst.returncode != 0:
            print("[parse] pip install pypdf FAILED; PDFs will be skipped")
            _pypdf_ok = False
            return False
    try:
        import pypdf  # noqa: F401

        _pypdf_ok = True
    except ImportError:
        _pypdf_ok = False
    return _pypdf_ok


class _TextExtractor(HTMLParser):
    """Strip tags/scripts/styles, keep visible text with block breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data)


def _clean(text: str) -> str:
    """Normalize whitespace but keep paragraph structure."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# ----------------------------------------------------------------------
# Per-format handlers: bytes -> plain text
# ----------------------------------------------------------------------

def parse_pdf(path: Path) -> str:
    from pypdf import PdfReader  # ensured by caller

    reader = PdfReader(str(path))
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")  # a single bad page must not kill the doc
    return "\n\n".join(pages)


def parse_html(data: bytes) -> str:
    extractor = _TextExtractor()
    extractor.feed(_read_text(data))
    return "".join(extractor.parts)


def parse_xml(data: bytes) -> str:
    root = ElementTree.fromstring(data)
    parts: List[str] = []

    def walk(elem: ElementTree.Element) -> None:
        if elem.text and elem.text.strip():
            parts.append(elem.text)
        for child in elem:
            walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail)
        tag = elem.tag.rsplit("}", 1)[-1].lower()
        if tag in ("p", "head", "div", "section", "description", "summary",
                   "entry", "attack_pattern", "extract", "fp-1", "fp-2"):
            parts.append("\n\n")
        else:
            parts.append(" ")

    walk(root)
    return "".join(parts)


def parse_stix(data: bytes) -> str:
    """ATT&CK STIX bundle -> 'EXT_ID | name (type)\\ndescription' blocks."""
    bundle = json.loads(data)
    blocks: List[str] = []
    for obj in bundle.get("objects", []):
        otype = obj.get("type")
        if otype not in ("attack-pattern", "x-mitre-tactic"):
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ext_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") in ("mitre-attack", "mitre-mobile-attack",
                                          "mitre-ics-attack"):
                ext_id = ref.get("external_id", "")
                break
        label = "tactic" if otype == "x-mitre-tactic" else "technique"
        name = obj.get("name", "")
        desc = (obj.get("description") or "").strip()
        blocks.append(f"{ext_id} | {name} ({label})\n{desc}")
    return "\n\n".join(blocks)


def _flatten_json(value: Any, prefix: str, out: List[str]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_json(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _flatten_json(v, f"{prefix}[{i}]", out)
    elif value is not None and value != "":
        out.append(f"{prefix}: {value}")


def parse_json(data: bytes) -> str:
    doc = json.loads(data)
    # NVD CVE API 2.0 special case: one paragraph per CVE
    if isinstance(doc, dict) and "vulnerabilities" in doc:
        blocks: List[str] = []
        for item in doc["vulnerabilities"]:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "?")
            desc = next(
                (d.get("value", "") for d in cve.get("descriptions", [])
                 if d.get("lang") == "en"),
                "",
            )
            cwes = [
                d.get("value", "")
                for w in cve.get("weaknesses", [])
                for d in w.get("description", [])
                if d.get("lang") == "en"
            ]
            sev = ""
            metrics = cve.get("metrics", {})
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                if metrics.get(key):
                    cd = metrics[key][0].get("cvssData", {})
                    sev = (f"CVSS {cd.get('baseScore', '?')} "
                           f"{cd.get('baseSeverity', '')}".strip())
                    break
            line = f"{cve_id}: {desc}"
            if cwes:
                line += f" [weaknesses: {', '.join(c for c in cwes if c)}]"
            if sev:
                line += f" [{sev}]"
            blocks.append(line)
        return "\n\n".join(blocks)
    lines: List[str] = []
    _flatten_json(doc, "", lines)
    return "\n".join(lines[:CSV_LINE_CAP])


def parse_csv(data: bytes) -> str:
    text = _read_text(data)
    reader = csv.reader(io.StringIO(text))
    lines: List[str] = []
    for i, row in enumerate(reader):
        if i >= CSV_LINE_CAP:
            lines.append(f"... [truncated at {CSV_LINE_CAP} lines]")
            break
        lines.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(lines)


def _parse_plain_lines(data: bytes) -> str:
    lines = _read_text(data).splitlines()
    if len(lines) > CSV_LINE_CAP:
        lines = lines[:CSV_LINE_CAP] + [f"... [truncated at {CSV_LINE_CAP} lines]"]
    return "\n".join(lines)


def parse_zip(path: Path) -> Tuple[str, List[str]]:
    """Extract text from csv/txt/json/xml members; return (text, skipped)."""
    texts: List[str] = []
    skipped: List[str] = []
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            ext = Path(member).suffix.lower()
            if ext == ".csv":
                texts.append(f"=== {member} ===\n" + parse_csv(zf.read(member)))
            elif ext == ".txt":
                texts.append(f"=== {member} ===\n" + _parse_plain_lines(zf.read(member)))
            elif ext == ".json":
                texts.append(f"=== {member} ===\n" + parse_json(zf.read(member)))
            elif ext == ".xml":
                texts.append(f"=== {member} ===\n" + parse_xml(zf.read(member)))
            else:
                skipped.append(member)
    return "\n\n".join(texts), skipped


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _iter_fetched(corpus_dir: Path) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    raw = corpus_dir / "raw"
    if not raw.is_dir():
        return
    for prov_path in sorted(raw.glob("*/*/provenance.json")):
        try:
            prov = json.loads(prov_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        yield prov_path, prov


def parse_source(src_dir: Path, prov: Dict[str, Any], text_dir: Path) -> Dict[str, Any]:
    """Parse all fetched files of one source into a single .txt."""
    fmt = prov.get("format", "")
    domain = prov.get("domain", "unknown")
    source_id = prov.get("source_id", src_dir.name)
    out_path = text_dir / domain / f"{source_id}.txt"
    notes: List[str] = []
    pieces: List[str] = []

    if fmt == "pdf" and not _ensure_pypdf():
        return {"status": "skipped: no pypdf"}

    for rec in prov.get("files", []):
        if rec.get("status") != "ok":
            continue
        fpath = src_dir / rec["file"]
        if not fpath.exists():
            notes.append(f"{rec['file']}: missing on disk")
            continue
        try:
            if fmt == "pdf":
                pieces.append(parse_pdf(fpath))
            elif fmt == "html":
                pieces.append(parse_html(fpath.read_bytes()))
            elif fmt == "xml":
                pieces.append(parse_xml(fpath.read_bytes()))
            elif fmt == "stix":
                pieces.append(parse_stix(fpath.read_bytes()))
            elif fmt == "json":
                pieces.append(parse_json(fpath.read_bytes()))
            elif fmt == "csv":
                pieces.append(parse_csv(fpath.read_bytes()))
            elif fmt == "markdown":
                pieces.append(_read_text(fpath.read_bytes()))
            elif fmt == "zip":
                text, skipped = parse_zip(fpath)
                pieces.append(text)
                if skipped:
                    notes.append(f"{rec['file']}: skipped members {skipped}")
            else:
                notes.append(f"{rec['file']}: unknown format {fmt!r}")
        except Exception as exc:
            notes.append(f"{rec['file']}: {type(exc).__name__}: {exc}")

    text = _clean("\n\n".join(p for p in pieces if p.strip()))
    if not text:
        return {
            "status": "failed",
            "error": "; ".join(notes) or "no parsable content",
        }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text + "\n", encoding="utf-8")
    result: Dict[str, Any] = {
        "status": "ok",
        "text_file": str(out_path.relative_to(text_dir.parent)),
        "chars": len(text),
    }
    if notes:
        result["notes"] = notes
    return result


def parse_all(corpus_dir: Path, domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Parse every fetched source; record status in provenance.json."""
    text_dir = corpus_dir / "text"
    results: List[Dict[str, Any]] = []
    for prov_path, prov in _iter_fetched(corpus_dir):
        if domains and prov.get("domain") not in domains:
            continue
        if prov.get("status", "").startswith("fail"):
            continue
        src_id = prov.get("source_id", prov_path.parent.name)
        print(f"[parse] {prov.get('domain')}/{src_id} ...", flush=True)
        info = parse_source(prov_path.parent, prov, text_dir)
        print(f"        -> {info['status']}"
              + (f" ({info.get('chars', 0):,} chars)" if info["status"] == "ok" else ""),
              flush=True)
        prov["parse"] = info
        prov_path.write_text(json.dumps(prov, indent=2))
        info = {**info, "source_id": src_id, "domain": prov.get("domain")}
        results.append(info)
    return results
