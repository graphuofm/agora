"""Polite, resumable downloader for TIER A/B corpus sources.

Layout:  <corpus>/raw/<domain>/<source_id>/<files>
         <corpus>/raw/<domain>/<source_id>/provenance.json

A source is skipped if its provenance.json already records status "ok"
(resume semantics). Failures are recorded, never fatal.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

import requests

from .manifest import MAX_DOWNLOAD_BYTES, USER_AGENT, Source, fetchable

TIMEOUT_S = 60
RETRIES = 2  # additional attempts after the first
CHUNK = 1 << 16

_EXT_BY_FORMAT: Dict[str, str] = {
    "pdf": ".pdf",
    "html": ".html",
    "xml": ".xml",
    "json": ".json",
    "csv": ".csv",
    "stix": ".json",
    "markdown": ".md",
    "zip": ".zip",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _filename_for(url: str, fmt: str, index: int) -> str:
    """Derive a stable local filename from the URL."""
    path = unquote(urlparse(url).path)
    name = Path(path).name or f"file_{index}"
    ext = _EXT_BY_FORMAT.get(fmt, "")
    if ext and not name.lower().endswith(ext):
        name += ext
    # keep it filesystem-safe
    name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return f"{index:02d}_{name}" if index > 0 else name


def _download_one(
    session: requests.Session, url: str, dest: Path, headers: Dict[str, str]
) -> Dict[str, Any]:
    """Download a single URL with retries and a size cap; return a record."""
    last_err: Optional[str] = None
    status_code: Optional[int] = None
    for attempt in range(1 + RETRIES):
        try:
            with session.get(
                url, headers=headers, timeout=TIMEOUT_S, stream=True
            ) as resp:
                status_code = resp.status_code
                if resp.status_code != 200:
                    last_err = f"HTTP {resp.status_code}"
                    if 400 <= resp.status_code < 500:
                        break  # client errors will not heal on retry
                    time.sleep(2 * (attempt + 1))
                    continue
                clen = resp.headers.get("Content-Length")
                if clen and int(clen) > MAX_DOWNLOAD_BYTES:
                    return {
                        "url": url,
                        "status": "skipped",
                        "error": f"content-length {clen} exceeds "
                        f"{MAX_DOWNLOAD_BYTES}-byte cap",
                        "http_status": resp.status_code,
                    }
                sha = hashlib.sha256()
                size = 0
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for block in resp.iter_content(CHUNK):
                        size += len(block)
                        if size > MAX_DOWNLOAD_BYTES:
                            fh.close()
                            tmp.unlink(missing_ok=True)
                            return {
                                "url": url,
                                "status": "skipped",
                                "error": f"download exceeded "
                                f"{MAX_DOWNLOAD_BYTES}-byte cap",
                                "http_status": resp.status_code,
                            }
                        sha.update(block)
                        fh.write(block)
                tmp.rename(dest)
                return {
                    "url": url,
                    "status": "ok",
                    "file": dest.name,
                    "http_status": resp.status_code,
                    "sha256": sha.hexdigest(),
                    "size": size,
                    "fetched_at": _now_iso(),
                }
        except requests.RequestException as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 * (attempt + 1))
    return {
        "url": url,
        "status": "failed",
        "error": last_err or "unknown error",
        "http_status": status_code,
    }


def fetch_source(source: Source, corpus_dir: Path, force: bool = False) -> Dict[str, Any]:
    """Fetch all URLs of one source; write provenance.json; return it."""
    out_dir = corpus_dir / "raw" / source.domain / source.id
    prov_path = out_dir / "provenance.json"
    if prov_path.exists() and not force:
        try:
            prev = json.loads(prov_path.read_text())
            if prev.get("status") == "ok":
                prev["status"] = "ok (cached)"
                return prev
        except (json.JSONDecodeError, OSError):
            pass  # corrupt provenance -> refetch

    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, **source.headers}

    files: List[Dict[str, Any]] = []
    for i, url in enumerate(source.urls):
        dest = out_dir / _filename_for(url, source.format, i)
        files.append(_download_one(session, url, dest, headers))

    n_ok = sum(1 for f in files if f["status"] == "ok")
    status = "ok" if n_ok == len(files) else ("partial" if n_ok else "failed")
    prov: Dict[str, Any] = {
        "source_id": source.id,
        "domain": source.domain,
        "name": source.name,
        "license_tier": source.license_tier,
        "redistributable": source.redistributable,
        "format": source.format,
        "status": status,
        "fetched_at": _now_iso(),
        "files": files,
    }
    if status == "failed":
        prov["error"] = "; ".join(
            str(f.get("error", "?")) for f in files if f["status"] != "ok"
        )
    prov_path.write_text(json.dumps(prov, indent=2))
    return prov


def fetch_all(
    corpus_dir: Path,
    domains: Optional[List[str]] = None,
    tiers: Optional[List[str]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Fetch every eligible source; never raise on a single failure."""
    results: List[Dict[str, Any]] = []
    for source in fetchable(domains, tiers):
        print(f"[fetch] {source.domain}/{source.id} ...", flush=True)
        try:
            prov = fetch_source(source, corpus_dir, force=force)
        except Exception as exc:  # truly unexpected: still record, continue
            prov = {
                "source_id": source.id,
                "domain": source.domain,
                "license_tier": source.license_tier,
                "redistributable": source.redistributable,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "fetched_at": _now_iso(),
            }
            out_dir = corpus_dir / "raw" / source.domain / source.id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
        sizes = sum(f.get("size", 0) for f in prov.get("files", []))
        print(
            f"        -> {prov['status']}"
            + (f" ({sizes:,} bytes)" if sizes else "")
            + (f" [{prov.get('error', '')}]" if prov["status"] == "failed" else ""),
            flush=True,
        )
        results.append(prov)
    return results
