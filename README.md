# AGORA — Synthetic Agentic Graph Architecture

**Generate large-scale, attributed, *temporal* graphs with ground-truth anomaly
labels — for any domain — fast.**

AGORA turns a natural-language description of a domain into a huge, realistic,
time-stamped graph in which **normal behavior runs and anomalies emerge under
control**, with **exact ground-truth labels** as a free byproduct. Point it at a
domain, get data that loads straight into PyG / DGL / Neo4j / igraph — no
preprocessing.

> Status: **engine working, pre-release.** The Rust core builds all six
> built-in domains end-to-end with ground-truth labels, streams to
> Parquet/CSV/GraphML, self-measures in a single pass, and runs deterministically
> across thread counts (100M edges in ~38s on a 32-core workstation). The
> offline RAG for zero-code domain migration is in place. Heading into the
> separate validation phase toward a SIGMOD submission. See
> [`mustread.txt`](mustread.txt) for the full blueprint.

## Quick start

### Python
```python
import agora
agora.doctor()                              # probe host
agora.generate("finance", nodes=1_000_000, edges=100_000_000,
              time_span_days=180, anomaly_rate=0.02, out="out")
# straight into PyG/DGL, no disk — returns numpy arrays + ground truth:
a = agora.generate_arrays("finance", nodes=100_000, edges=10_000_000, time_span_days=90)
```

### CLI
```bash
cargo build --release                      # or: maturin build (Python wheel)
./target/release/agora doctor               # probe host, recommend a feasible scale
./target/release/agora domains              # list the 6 built-in domains
./target/release/agora generate --domain finance --preset small --out out
./target/release/agora generate --domain finance --preset large --dry-run   # estimate before committing
./target/release/agora stats out            # single-pass introspection summary
./target/release/agora validate out         # consistency + Φ-constraint checks
```

Key `generate` flags: `--nodes --edges --time-span --anomaly-rate --difficulty
--type-mix --seed --format {parquet,csv,graphml} --threads --mem-budget
--shard-size --dry-run --json`. The five anomaly control axes
(prevalence/difficulty/type-mix/placement/cascade) are tunable knobs; anomalies
*emerge* from labeled adversary/failure processes, never injected templates.

## Why AGORA

Every prior generation of graph generators bought two of three properties and
sold the third:

| | scale | realism + semantics | controllable ground-truth |
|---|:---:|:---:|:---:|
| Fractal/recursive (R-MAT, Kronecker, TrillionG) | ✅ | ❌ | ❌ |
| Simulation (LDBC SNB Datagen) | ✅ | ✅ | ❌ |
| Curation + eval (TGB) | — (real data) | ✅ | ❌ |
| **AGORA** | ✅ | ✅ | ✅ |

AGORA is the first to hold all three **at once, on temporal graphs**: Era-2
simulation realism + Era-1 scale (a Rust/parallel engine, no per-edge LLM) +
**controllable, ground-truth anomalies** from parameterized adversary processes
in an omniscient simulator.

## How it works

```
 natural-language domain ─▶ RAG + small open LLM ─▶ refined rule base (static)
            (offline, one-time, off the hot path)            │
                                                             ▼
   Rust simulation engine ──▶ attributed temporal edges + ground-truth labels
   (agent-based / discrete-event, parallel)          + built-in streaming stats
                                                             │
                                                             ▼
                              Parquet / CSV / GraphML  +  reproducibility metadata
```

- **Rust hot path, Python harness.** The per-event simulation and the
  alignment/replay run in native Rust (parallel); Python only drives config, CLI
  and UI. The LLM is used **only offline** to author rules — never per edge.
- **Controllable anomalies, not injected.** Anomalies *emerge* from adversary
  agents whose actions are labeled by intent; difficulty, rate, type, placement
  and cascade are tunable knobs.
- **Introspection at scale.** Because AGORA generates the data and knows the
  ground truth, it self-measures (streaming statistics, sketches, system
  metrics) — so billion-edge datasets remain *measurable* and *evaluable*.

## Domains

Six built-in industries: **Finance/AML · Crypto/Blockchain · Cybersecurity
(IDS+APT) · Transportation · E-commerce review fraud · Healthcare/insurance
fraud**. Or describe any domain in natural language — zero code, zero retraining.

## Runs anywhere

`agora doctor` probes your machine (CPU / RAM / GPU / disk) and tells you what
scale is feasible; `agora generate --dry-run` estimates time, memory and disk
*before* it runs. The same tool adapts from a 16 GB laptop to an 80 GB H100 node.

## Documentation

- [`docs/INSTALL.md`](docs/INSTALL.md) — install three ways (pip / binary / source); platform support.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — crate map, determinism, the event loop, calibrated emergence.
- [`docs/OUTPUT.md`](docs/OUTPUT.md) — output format + copy-paste loaders (PyG/DGL/Neo4j/igraph/polars).
- [`crates/agora-rules/domains/SCHEMA.md`](crates/agora-rules/domains/SCHEMA.md) — rule-base authoring reference.
- [`mustread.txt`](mustread.txt) — full master blueprint (architecture, RAG, CLI, UX).
- [`DOMAINS.md`](DOMAINS.md) — rule dossiers for the 6 industries.
- [`CORPUS.md`](CORPUS.md) — the real industry-standard documents the RAG is built from.
- [`BACKGROUND.md`](BACKGROUND.md) — prerequisite knowledge primer.
- [`references.bib`](references.bib) — bibliography.

## Citation

A paper is forthcoming. Until then, please cite this repository.

## License

To be added.
