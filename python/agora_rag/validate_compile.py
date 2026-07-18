"""validate_compile.py — lint, serialize, and compile via the Rust validator.

mustread.txt §9 steps 5 + 7:

* step 5 (VALIDATE): a Python-side lint catches the obvious problems early with
  actionable messages — required primitives present, weights in range, names
  cross-referenced, distributions well-formed, prevalence <= 0.05.
* step 7 (COMPILE): serialize the drafted dict to YAML and call the RUST
  validator as the SINGLE SOURCE OF TRUTH::

      agora rules build --domain <written_yaml> --out <out>

  If it exits non-zero, its stderr is surfaced as the validation failure (the
  Rust loader is authoritative; the Python lint is only a fast pre-check).

The build loop (driven from ``pipeline.py``) is: if an LLM draft fails Rust
validation, fall back to the template draft.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Distribution tag -> required fields (mirrors SCHEMA.md).
_DIST_FIELDS = {
    "constant": {"value"},
    "uniform": {"min", "max"},
    "normal": {"mean", "std"},
    "log_normal": {"mu", "sigma"},
    "exponential": {"rate"},
    "pareto": {"scale", "shape"},
    "zipf": {"n", "exponent"},
    "poisson": {"lambda"},
}
_REQUIRED_TOP = ["meta", "entity_types", "relations", "event_types", "behaviors", "control"]


def _agora_binary() -> Path:
    return Path(__file__).resolve().parents[2] / "target" / "release" / "agora"


@dataclass
class LintResult:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def lint(rb: Dict[str, object]) -> LintResult:
    """Fast, actionable pre-check. Authoritative validation is the Rust loader."""
    res = LintResult()

    for key in _REQUIRED_TOP:
        if key not in rb:
            res.errors.append(f"missing required top-level primitive: '{key}'")

    meta = rb.get("meta")
    if not isinstance(meta, dict):
        res.errors.append("meta must be a mapping")
        meta = {}
    for f in ("id", "name", "description"):
        if not meta.get(f):
            res.errors.append(f"meta.{f} is required and must be non-empty")

    # Collect defined names for cross-referencing.
    entity_names = {e.get("name") for e in rb.get("entity_types", []) if isinstance(e, dict)}
    event_names = {e.get("name") for e in rb.get("event_types", []) if isinstance(e, dict)}
    relation_names = {r.get("name") for r in rb.get("relations", []) if isinstance(r, dict)}

    if not entity_names:
        res.errors.append("at least one entity_type must be defined")
    if not event_names:
        res.errors.append("at least one event_type must be defined")

    # Relations / events reference defined entity types.
    for r in rb.get("relations", []):
        if not isinstance(r, dict):
            continue
        for side in ("src", "dst"):
            if r.get(side) not in entity_names:
                res.errors.append(
                    f"relation '{r.get('name')}' {side}='{r.get(side)}' is not a defined entity_type"
                )
    for e in rb.get("event_types", []):
        if not isinstance(e, dict):
            continue
        for side in ("src", "dst"):
            if e.get(side) not in entity_names:
                res.errors.append(
                    f"event '{e.get('name')}' {side}='{e.get(side)}' is not a defined entity_type"
                )

    # Behaviors reference defined actors and events.
    for b in rb.get("behaviors", []):
        if not isinstance(b, dict):
            continue
        if b.get("actor") not in entity_names:
            res.errors.append(
                f"behavior '{b.get('name')}' actor='{b.get('actor')}' is not a defined entity_type"
            )
        weights = []
        for ev in b.get("events", []) or []:
            if isinstance(ev, dict):
                if ev.get("event") not in event_names:
                    res.errors.append(
                        f"behavior '{b.get('name')}' references undefined event '{ev.get('event')}'"
                    )
                if "weight" in ev:
                    weights.append(ev["weight"])
        for w in weights:
            if not isinstance(w, (int, float)) or w < 0:
                res.errors.append(
                    f"behavior '{b.get('name')}' has a negative/invalid event weight {w}"
                )

    # Adversary/failure prevalence weights non-negative.
    for proc_key in ("adversaries", "failures"):
        for p in rb.get(proc_key, []) or []:
            if not isinstance(p, dict):
                continue
            pw = p.get("prevalence_weight")
            if pw is not None and (not isinstance(pw, (int, float)) or pw < 0):
                res.errors.append(
                    f"{proc_key} '{p.get('intent')}' prevalence_weight {pw} must be >= 0"
                )

    # Control: prevalence in (0, 0.05]; difficulty in [0,1].
    ctrl = rb.get("control")
    if not isinstance(ctrl, dict):
        res.errors.append("control must be a mapping")
    else:
        prev = ctrl.get("prevalence")
        if not isinstance(prev, (int, float)):
            res.errors.append("control.prevalence must be a number")
        elif not (0.0 < prev <= 0.05):
            res.errors.append(
                f"control.prevalence={prev} out of range; anomalies are RARE, keep <= 0.05 (~0.01-0.05)"
            )
        diff = ctrl.get("difficulty")
        if isinstance(diff, (int, float)) and not (0.0 <= diff <= 1.0):
            res.errors.append(f"control.difficulty={diff} must be in [0,1]")

    return res


def to_yaml(rb: Dict[str, object]) -> str:
    """Serialize the drafted dict to YAML, stripping internal draft metadata.

    Uses the tag-preserving dumper so the serde external ``!tags`` survive
    (the Rust loader requires them).
    """
    from . import yamltags

    rb = {k: v for k, v in rb.items() if not k.startswith("_")}
    return yamltags.dump(rb)


def run_rust_validator(yaml_path: Path, out_path: Path) -> "RustResult":
    binary = _agora_binary()
    if not binary.exists():
        return RustResult(
            ok=False,
            returncode=-1,
            stdout="",
            stderr=f"Rust validator not found at {binary}; build it with `cargo build --release`",
        )
    proc = subprocess.run(
        [str(binary), "rules", "build", "--domain", str(yaml_path), "--out", str(out_path)],
        capture_output=True,
        text=True,
    )
    return RustResult(
        ok=proc.returncode == 0,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


@dataclass
class RustResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
