# Installing AGORA

AGORA is self-contained: a native Rust core with no runtime services. It runs
identically on a 16 GB laptop and an 80 GB H100 node — only the *feasible
scale* differs, and `agora doctor` tells you what that is (§13).

## Platforms

| OS | CLI binary | Python wheel | notes |
|---|:--:|:--:|---|
| Linux x86-64 | ✅ | ✅ | primary target; tested |
| macOS (Intel / Apple Silicon) | ✅ | ✅ | GPU probe returns none (no CUDA); CPU path |
| Windows x86-64 | ✅ | ✅ | NUMA reported as 1 node; everything else probed |

The host probe degrades gracefully off Linux: NUMA topology (read from sysfs)
falls back to a single node, and the GPU probe (which shells out to
`nvidia-smi`) reports no GPU when the driver isn't present. CPU, RAM and disk
come from the cross-platform `sysinfo` crate. Nothing platform-specific is on
the generation hot path.

## Option 1 — Python (`pip install`)

The Python harness wraps the same native engine via PyO3 (abi3 wheels, one
wheel per platform works across Python 3.9+):

```bash
pip install agora          # from a built wheel
# from source checkout:
pip install maturin
maturin build --release -m crates/agora-py/Cargo.toml
pip install target/wheels/agora-*.whl
```
```python
import agora
agora.doctor()                      # host capabilities
agora.generate("finance", nodes=1_000_000, edges=100_000_000,
              time_span_days=180, anomaly_rate=0.02, out="out")
# or get arrays straight into PyG/DGL, no disk:
a = agora.generate_arrays("finance", nodes=100_000, edges=10_000_000,
                         time_span_days=90)
```

## Option 2 — prebuilt CLI binary

A single static binary, no dependencies:

```bash
# download the release binary for your platform, then:
chmod +x agora && ./agora doctor
```

## Option 3 — build from source (Rust)

```bash
git clone https://anonymous.4open.science/r/agora && cd AGORA
cargo build --release            # binary at target/release/agora
./target/release/agora doctor
# or install onto PATH:
cargo install --path crates/agora-cli
```

Requires a recent stable Rust (1.80+). No CMake, no system libraries — `cargo`
fetches everything (arrow/parquet, rayon, etc.).

## First run

```bash
agora doctor                                  # what scale fits this machine
agora generate --domain finance --preset small --out out
agora stats out                               # introspection summary
```

`agora generate --dry-run` estimates time, RAM and disk *before* running so you
commit with eyes open; if a request can't fit the host, AGORA refuses with an
actionable message and a suggested feasible scale rather than OOM-ing.
