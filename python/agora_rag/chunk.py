"""Chunk corpus/text/*.txt into corpus/chunks.jsonl.

One JSON object per line::

  {id, domain, source_id, license_tier, url, seq, text,
   structured, section?, external_id?}

The corpus is already structured; naive fixed-size chunking destroys that
structure (mustread.txt §9 — "every rule traces to a standard"). So chunking is
now STRUCTURE-AWARE (additive — the CLI is unchanged):

  * STIX (MITRE ATT&CK / CAPEC):   one chunk per object (technique / tactic),
    tagged with its ``external_id`` (e.g. ``T1059``).
  * eCFR XML (FinCEN 31 CFR, FTC 16 CFR, 42 CFR, ...): split on SECTION (DIV8)
    boundaries; carry the section number (e.g. ``1010.311``) as metadata.
  * CSV (NCCI / LEIE / HCPCS / ICD): one chunk per small row-group, each
    prefixed with the header line, NOT one giant blob.
  * Prose PDFs / HTML / markdown (FATF, NIST, Basel, EUR-Lex, SUMO, ...): keep
    fixed-size paragraph packing — these are genuine prose.

Structured records are recovered from the RAW fetched file (XML/STIX/CSV), since
the cleaned ``.txt`` has already flattened the structure. When the raw file is
absent or unreadable we fall back to prose chunking of the ``.txt`` so the CLI
keeps working.

``section``/``external_id`` are only present on structured records; ``structured``
is always present (bool) so downstream extraction can filter precisely.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from xml.etree import ElementTree

CHUNK_CHARS = 1000
OVERLAP_CHARS = 150
MIN_TAIL = 200  # merge a tiny final fragment into the previous chunk

# Structured-record sizing.
CSV_ROWS_PER_CHUNK = 20      # rows per CSV row-group chunk (+ header line)
CSV_MAX_ROWS = 50_000        # cap rows ingested per CSV (matches parse.py)
SECTION_MAX_CHARS = 4000     # an oversized eCFR section is hard-split for embedding


def _split_long(paragraph: str, limit: int) -> List[str]:
    """Hard-split an oversized paragraph at whitespace near the limit."""
    out: List[str] = []
    text = paragraph
    while len(text) > limit:
        cut = text.rfind(" ", limit // 2, limit)
        if cut == -1:
            cut = limit
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def chunk_text(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """Greedy paragraph packing with sliding-window overlap (prose chunker)."""
    segments: List[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            segments.extend(_split_long(para, size))
        else:
            segments.append(para)

    # every segment is now <= size; pack greedily with overlap carry-over
    chunks: List[str] = []
    buf = ""
    for seg in segments:
        candidate = f"{buf}\n\n{seg}" if buf else seg
        if len(candidate) <= size:
            buf = candidate
            continue
        chunks.append(buf)
        tail = buf[-overlap:]
        cut = tail.find(" ")
        if 0 <= cut < overlap - 20:
            tail = tail[cut + 1:]  # start the overlap on a word boundary
        if tail and len(tail) + 2 + len(seg) <= size:
            buf = f"{tail}\n\n{seg}"
        else:
            buf = seg  # overlap would overflow: drop it for this chunk

    if buf:
        if chunks and len(buf) < MIN_TAIL and len(chunks[-1]) + len(buf) + 2 <= size + MIN_TAIL:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# A structured record: text + optional section / external_id metadata
# --------------------------------------------------------------------------- #
class Record:
    """One structured chunk-to-be, carrying its own provenance metadata."""

    __slots__ = ("text", "section", "external_id")

    def __init__(
        self,
        text: str,
        section: Optional[str] = None,
        external_id: Optional[str] = None,
    ) -> None:
        self.text = text
        self.section = section
        self.external_id = external_id


def _clean_ws(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _localname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _element_text(elem: ElementTree.Element) -> str:
    """All descendant text of an element, whitespace-normalized."""
    return _clean_ws(" ".join(elem.itertext()))


# --------------------------------------------------------------------------- #
# eCFR XML — split on DIV8 (TYPE="SECTION"); section number in @N
# --------------------------------------------------------------------------- #
def records_ecfr_xml(data: bytes) -> List[Record]:
    """One record per eCFR SECTION (DIV8), tagged with its section number."""
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return []
    records: List[Record] = []
    for div in root.iter():
        if _localname(div.tag) != "div8":
            continue
        if (div.get("TYPE") or div.get("type") or "").upper() != "SECTION":
            continue
        section = div.get("N") or div.get("n")
        text = _element_text(div)
        if not text:
            continue
        # An over-long section is hard-split but every piece keeps its section id.
        if len(text) > SECTION_MAX_CHARS:
            for piece in _split_long(text, CHUNK_CHARS):
                records.append(Record(piece, section=section))
        else:
            records.append(Record(text, section=section))
    return records


# --------------------------------------------------------------------------- #
# STIX 2.1 — one record per attack-pattern / tactic, tagged with external_id
# --------------------------------------------------------------------------- #
_STIX_KEEP = ("attack-pattern", "x-mitre-tactic")
_MITRE_SOURCES = ("mitre-attack", "mitre-mobile-attack", "mitre-ics-attack",
                  "capec")


def records_stix(data: bytes) -> List[Record]:
    """One record per ATT&CK technique/tactic, carrying its T-id / TAxxxx id."""
    try:
        bundle = json.loads(data)
    except (json.JSONDecodeError, ValueError):
        return []
    records: List[Record] = []
    for obj in bundle.get("objects", []):
        otype = obj.get("type")
        if otype not in _STIX_KEEP:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        ext_id = ""
        for ref in obj.get("external_references", []):
            if ref.get("source_name") in _MITRE_SOURCES and ref.get("external_id"):
                ext_id = ref.get("external_id", "")
                break
        label = "tactic" if otype == "x-mitre-tactic" else "technique"
        name = obj.get("name", "")
        desc = (obj.get("description") or "").strip()
        header = f"{ext_id} | {name} ({label})".strip(" |")
        text = _clean_ws(f"{header}\n{desc}" if desc else header)
        if text:
            records.append(Record(text, external_id=(ext_id or None)))
    return records


# --------------------------------------------------------------------------- #
# CAPEC XML — one record per <Attack_Pattern>, tagged with its CAPEC-<id>
# --------------------------------------------------------------------------- #
def records_capec_xml(data: bytes) -> List[Record]:
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return []
    records: List[Record] = []
    for elem in root.iter():
        if _localname(elem.tag) != "attack_pattern":
            continue
        ap_id = elem.get("ID") or elem.get("id")
        name = elem.get("Name") or elem.get("name") or ""
        ext_id = f"CAPEC-{ap_id}" if ap_id else None
        text = _clean_ws(f"{ext_id or ''} | {name}\n{_element_text(elem)}")
        if text:
            records.append(Record(text, external_id=ext_id))
    return records


# --------------------------------------------------------------------------- #
# CSV — header line + small row groups (one record per group, NOT one blob)
# --------------------------------------------------------------------------- #
def records_csv(data: bytes) -> List[Record]:
    text = _read_text(data)
    reader = csv.reader(io.StringIO(text))
    rows: List[List[str]] = []
    for i, row in enumerate(reader):
        if i >= CSV_MAX_ROWS:
            break
        rows.append(row)
    if not rows:
        return []
    header = " | ".join(c.strip() for c in rows[0])
    body = rows[1:]
    records: List[Record] = []
    for start in range(0, len(body), CSV_ROWS_PER_CHUNK):
        group = body[start:start + CSV_ROWS_PER_CHUNK]
        lines = [" | ".join(cell.strip() for cell in r) for r in group]
        block = header + "\n" + "\n".join(lines)
        records.append(Record(block.strip()))
    return records


def _read_text(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# --------------------------------------------------------------------------- #
# ZIP — structure-chunk csv/xml members; passthrough others to prose
# --------------------------------------------------------------------------- #
def records_zip(path: Path) -> Optional[List[Record]]:
    """Structured records from a zip's csv/xml members, or None if none apply."""
    records: List[Record] = []
    found = False
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError):
        return None
    with zf:
        for member in zf.namelist():
            ext = Path(member).suffix.lower()
            if ext == ".csv":
                found = True
                records.extend(records_csv(zf.read(member)))
            elif ext == ".xml":
                # generic eCFR-style xml inside a zip (rare); try section split
                found = True
                recs = records_ecfr_xml(zf.read(member))
                records.extend(recs)
    return records if found else None


# --------------------------------------------------------------------------- #
# Format dispatch: raw file -> structured records (or None to use prose)
# --------------------------------------------------------------------------- #
def structured_records(fmt: str, raw_paths: List[Path]) -> Optional[List[Record]]:
    """Return structured records for a source, or None if it's genuine prose.

    ``raw_paths`` are the ok-fetched raw files for the source (XML/STIX/CSV/zip).
    Prose formats (pdf/html/markdown) return None so the caller falls back to the
    fixed-size prose chunker over the cleaned ``.txt``.
    """
    fmt = (fmt or "").lower()
    if fmt in ("pdf", "html", "markdown", ""):
        return None  # genuine prose: keep fixed-size packing
    records: List[Record] = []
    any_structured = False
    for path in raw_paths:
        if not path.exists():
            continue
        try:
            if fmt == "stix":
                recs: Optional[List[Record]] = records_stix(path.read_bytes())
            elif fmt == "xml":
                data = path.read_bytes()
                recs = records_ecfr_xml(data)
                if not recs:  # not eCFR? try CAPEC attack-pattern layout
                    recs = records_capec_xml(data)
            elif fmt == "csv":
                recs = records_csv(path.read_bytes())
            elif fmt == "json":
                return None  # NVD/flattened json: prose-chunk the cleaned text
            elif fmt == "zip":
                recs = records_zip(path)
            else:
                recs = None
        except Exception:  # noqa: BLE001 — any structure failure -> prose fallback
            recs = None
        if recs:
            any_structured = True
            records.extend(recs)
    return records if any_structured else None


# --------------------------------------------------------------------------- #
# Provenance + raw-file discovery
# --------------------------------------------------------------------------- #
def _provenance_for(corpus_dir: Path, domain: str, source_id: str) -> Dict[str, Any]:
    prov_path = corpus_dir / "raw" / domain / source_id / "provenance.json"
    if prov_path.exists():
        try:
            return json.loads(prov_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _raw_files(corpus_dir: Path, domain: str, source_id: str,
               prov: Dict[str, Any]) -> List[Path]:
    """Absolute paths to the source's ok-fetched raw files."""
    src_dir = corpus_dir / "raw" / domain / source_id
    out: List[Path] = []
    for rec in prov.get("files", []):
        if rec.get("status") == "ok" and rec.get("file"):
            out.append(src_dir / rec["file"])
    return out


def _iter_text_files(corpus_dir: Path, domains: Optional[List[str]]) -> Iterator[Path]:
    text_dir = corpus_dir / "text"
    if not text_dir.is_dir():
        return
    for path in sorted(text_dir.glob("*/*.txt")):
        if domains and path.parent.name not in domains:
            continue
        yield path


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def chunk_all(corpus_dir: Path, domains: Optional[List[str]] = None) -> int:
    """Write corpus/chunks.jsonl (structure-aware); return the chunk count."""
    out_path = corpus_dir / "chunks.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in _iter_text_files(corpus_dir, domains):
            domain = path.parent.name
            source_id = path.stem
            prov = _provenance_for(corpus_dir, domain, source_id)
            url = next(
                (f.get("url", "") for f in prov.get("files", [])
                 if f.get("status") == "ok"),
                "",
            )
            tier = prov.get("license_tier", "?")
            fmt = prov.get("format", "")

            raw_paths = _raw_files(corpus_dir, domain, source_id, prov)
            records = structured_records(fmt, raw_paths)

            wrote = 0
            if records is not None:
                kind = "structured"
                for seq, rec in enumerate(records):
                    record = {
                        "id": f"{source_id}:{seq:05d}",
                        "domain": domain,
                        "source_id": source_id,
                        "license_tier": tier,
                        "url": url,
                        "seq": seq,
                        "text": rec.text,
                        "structured": True,
                    }
                    if rec.section:
                        record["section"] = rec.section
                    if rec.external_id:
                        record["external_id"] = rec.external_id
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    wrote += 1
            else:
                kind = "prose"
                text = path.read_text(encoding="utf-8")
                for seq, piece in enumerate(chunk_text(text)):
                    record = {
                        "id": f"{source_id}:{seq:05d}",
                        "domain": domain,
                        "source_id": source_id,
                        "license_tier": tier,
                        "url": url,
                        "seq": seq,
                        "text": piece,
                        "structured": False,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    wrote += 1

            n += wrote
            print(f"[chunk] {domain}/{source_id}: {wrote} {kind} chunks", flush=True)
    print(f"[chunk] total: {n} chunks -> {out_path}", flush=True)
    return n
