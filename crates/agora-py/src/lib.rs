//! PyO3 bindings for AGORA (blueprint §7, §13). Lets the Python harness/UI and
//! end users `pip install agora` and drive the Rust engine directly:
//!
//!   * [`domains`]         — list the built-in domains.
//!   * [`doctor`]          — probe the host (CPU/RAM/GPU/disk).
//!   * [`generate`]        — run the simulator, write files, return a summary.
//!   * [`generate_arrays`] — run in memory, return numpy columns (no disk).
//!
//! The engine, rule base, IO and host crates are all unmodified; this crate
//! only wraps their public APIs. The GIL is released for the duration of every
//! generation run so a AGORA call never blocks other Python threads.

use numpy::IntoPyArray;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use agora_core::world::union_attr_dictionaries;
use agora_core::{
    AnomalyRecord, EventBatch, EventSink, GenParams, GenSummary, NodeBatch,
};
use agora_host::HostProbe;
use agora_rules::{
    builtin_domains, load_builtin_rulebase, GenerationConfig, OutputFormat, RunMeta,
};

/// Map an `anyhow::Error` to a Python `RuntimeError`, preserving the engine's
/// already-actionable message (and its cause chain).
fn py_err(e: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{e:#}"))
}

// ---------------------------------------------------------------------------
// Resolved-parameter helper shared by `generate` and `generate_arrays`.
// ---------------------------------------------------------------------------

/// Everything the engine needs, plus the decode metadata the writers need.
struct Resolved {
    params: GenParams,
    config: GenerationConfig,
    event_type_names: Vec<String>,
    intent_names: Vec<String>,
    attr_dicts: Vec<(String, Vec<String>)>,
}

#[allow(clippy::too_many_arguments)]
fn resolve(
    domain: &str,
    nodes: u64,
    edges: u64,
    time_span_days: f64,
    seed: u64,
    anomaly_rate: Option<f64>,
    difficulty: Option<f64>,
    no_anomalies: bool,
    threads: Option<usize>,
    out: Option<&str>,
    fmt: &str,
    shard_index: u64,
    shard_count: u64,
) -> anyhow::Result<Resolved> {
    // Mirror the CLI: build a config so the same validation runs (actionable
    // range errors), then resolve the rule base (built-in id or path).
    let format = match fmt {
        "parquet" => OutputFormat::Parquet,
        "csv" => OutputFormat::Csv,
        "graphml" => OutputFormat::Graphml,
        other => {
            anyhow::bail!("unknown format `{other}`: expected parquet, csv or graphml")
        }
    };
    let shard_count = shard_count.max(1);
    if shard_count > 1 && shard_index >= shard_count {
        anyhow::bail!("shard_index {shard_index} must be < shard_count {shard_count}");
    }

    let mut cfg = GenerationConfig::scaffold(domain, agora_rules::Preset::Small);
    cfg.scale.nodes = nodes;
    cfg.scale.edges = edges;
    cfg.time.span_days = time_span_days;
    cfg.seed = seed;
    cfg.anomaly.rate = anomaly_rate;
    cfg.anomaly.difficulty = difficulty;
    cfg.anomaly.disabled = no_anomalies;
    cfg.runtime.threads = threads;
    cfg.output.format = format;
    if let Some(o) = out {
        cfg.output.path = std::path::PathBuf::from(o);
    }
    cfg.validate()?;

    let rulebase = load_builtin_rulebase(&cfg.domain)?;

    // Same thread default as the CLI: probe the host, fall back to cores - 2.
    let probe = HostProbe::probe(&cfg.output.path);
    let resolved_threads = cfg
        .runtime
        .threads
        .unwrap_or_else(|| probe.default_threads())
        .clamp(1, probe.cpu.logical_cores.max(1));

    let attr_dicts = union_attr_dictionaries(&rulebase);
    let event_type_names: Vec<String> =
        rulebase.event_types.iter().map(|e| e.name.clone()).collect();
    let intent_names: Vec<String> = {
        let mut v = vec!["normal".to_string()];
        v.extend(rulebase.adversaries.iter().map(|a| a.intent.clone()));
        v.extend(rulebase.failures.iter().map(|f| f.intent.clone()));
        v
    };

    let params = GenParams {
        rulebase,
        nodes: cfg.scale.nodes,
        target_edges: cfg.scale.edges,
        span_days: cfg.time.span_days,
        granularity_s: cfg.time.granularity_s,
        epoch_unix: cfg.time.epoch_unix,
        seed: cfg.seed,
        threads: resolved_threads,
        anomaly_rate: cfg.anomaly.rate,
        anomaly_difficulty: cfg.anomaly.difficulty,
        anomaly_type_mix: cfg.anomaly.type_mix.clone().unwrap_or_default(),
        anomaly_cascade: None,
        anomaly_communities: None,
        anomalies_disabled: cfg.anomaly.disabled,
        shard_index,
        shard_count,
    };

    Ok(Resolved {
        params,
        config: cfg,
        event_type_names,
        intent_names,
        attr_dicts,
    })
}

// ---------------------------------------------------------------------------
// Python conversions for the summary / ground truth.
// ---------------------------------------------------------------------------

fn anomaly_record_to_py<'py>(
    py: Python<'py>,
    r: &AnomalyRecord,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("id", r.id)?;
    d.set_item("intent", &r.intent)?;
    d.set_item("kind", &r.kind)?;
    d.set_item("camouflage", r.camouflage)?;
    d.set_item("members", r.members.clone())?;
    d.set_item("n_members", r.n_members)?;
    d.set_item("community", r.community)?;
    d.set_item("start_t", r.start_t)?;
    d.set_item("end_t", r.end_t)?;
    d.set_item("cascade", r.cascade)?;
    Ok(d)
}

fn summary_to_py<'py>(py: Python<'py>, s: &GenSummary) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("nodes", s.nodes)?;
    d.set_item("edges_written", s.edges_written)?;
    d.set_item("anomalous_edges", s.anomalous_edges)?;
    d.set_item("intent_names", s.intent_names.clone())?;
    d.set_item("event_type_names", s.event_type_names.clone())?;
    // edges_per_intent as {intent: count} for ergonomics.
    let epi = PyDict::new(py);
    for (name, count) in s.intent_names.iter().zip(s.edges_per_intent.iter()) {
        epi.set_item(name, *count)?;
    }
    d.set_item("edges_per_intent", epi)?;
    d.set_item("wall_time_s", s.wall_time_s)?;
    d.set_item("events_per_sec", s.events_per_sec)?;
    // attribute dictionaries: {attr: [values...]}
    let ad = PyDict::new(py);
    for (attr, values) in &s.attr_dictionaries {
        ad.set_item(attr, values.clone())?;
    }
    d.set_item("attr_dictionaries", ad)?;
    // ground truth: list of per-instance dicts.
    let gt = PyList::empty(py);
    for r in &s.ground_truth {
        gt.append(anomaly_record_to_py(py, r)?)?;
    }
    d.set_item("ground_truth", gt)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// domains()
// ---------------------------------------------------------------------------

/// Built-in domains: list of dicts with id, name, anomaly_source, summary.
#[pyfunction]
fn domains(py: Python<'_>) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for info in builtin_domains() {
        let d = PyDict::new(py);
        d.set_item("id", info.id)?;
        d.set_item("name", info.name)?;
        d.set_item("anomaly_source", info.anomaly_source)?;
        d.set_item("summary", info.summary)?;
        d.set_item("shipped", info.rulebase_yaml.is_some())?;
        list.append(d)?;
    }
    Ok(list.unbind())
}

// ---------------------------------------------------------------------------
// doctor()
// ---------------------------------------------------------------------------

/// Probe the host (CPU/RAM/GPU/disk) and return it as a dict. The probe is
/// serialized through serde_json so the dict mirrors `agora doctor` exactly.
#[pyfunction]
#[pyo3(signature = (out=None))]
fn doctor(py: Python<'_>, out: Option<&str>) -> PyResult<Py<PyAny>> {
    let path = std::path::PathBuf::from(out.unwrap_or("."));
    let probe = HostProbe::probe(&path);
    let value = serde_json::to_value(&probe).map_err(|e| py_err(e.into()))?;
    json_to_py(py, &value)
}

/// Recursively convert a `serde_json::Value` into native Python objects.
fn json_to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    use serde_json::Value;
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any().unbind()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any().unbind()
            } else {
                n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any().unbind()
            }
        }
        Value::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any().unbind()
        }
        Value::Object(map) => {
            let d = PyDict::new(py);
            for (k, val) in map {
                d.set_item(k, json_to_py(py, val)?)?;
            }
            d.into_any().unbind()
        }
    })
}

// ---------------------------------------------------------------------------
// generate(): the main file-writing entry point.
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (
    domain, *, nodes, edges, time_span_days, seed=42, anomaly_rate=None,
    difficulty=None, no_anomalies=false, threads=None, out=None,
    fmt="parquet", shard_index=0, shard_count=1, progress=false
))]
#[allow(clippy::too_many_arguments)]
fn generate(
    py: Python<'_>,
    domain: &str,
    nodes: u64,
    edges: u64,
    time_span_days: f64,
    seed: u64,
    anomaly_rate: Option<f64>,
    difficulty: Option<f64>,
    no_anomalies: bool,
    threads: Option<usize>,
    out: Option<&str>,
    fmt: &str,
    shard_index: u64,
    shard_count: u64,
    progress: bool,
) -> PyResult<Py<PyDict>> {
    let out = out.unwrap_or("./out");
    let resolved = resolve(
        domain, nodes, edges, time_span_days, seed, anomaly_rate, difficulty,
        no_anomalies, threads, Some(out), fmt, shard_index, shard_count,
    )
    .map_err(py_err)?;

    // Release the GIL: generation is pure Rust + IO, no Python touched inside.
    let summary = py
        .allow_threads(|| run_to_files(&resolved, progress))
        .map_err(py_err)?;

    summary_to_py(py, &summary).map(|d| d.unbind())
}

/// Build the file writer, run the engine, and write the side-car artifacts
/// (ground_truth.json + agora_meta.json) the same way the CLI does.
fn run_to_files(r: &Resolved, progress: bool) -> anyhow::Result<GenSummary> {
    let out_dir = r.config.output.path.clone();

    // Reproducibility record (config + seed + version + host probe), like CLI.
    let probe = HostProbe::probe(&out_dir);
    let mut meta = RunMeta::new(&r.config, serde_json::to_value(&probe)?);
    meta.write(&out_dir)?;

    let fmt_str = match r.config.output.format {
        OutputFormat::Parquet => "parquet",
        OutputFormat::Csv => "csv",
        OutputFormat::Graphml => "graphml",
    };
    let writer = agora_io::FormatSink::make(
        fmt_str,
        out_dir.clone(),
        r.config.output.shard_size_mb,
        r.event_type_names.clone(),
        r.intent_names.clone(),
        r.attr_dicts.clone(),
    )?;
    let mut sink = agora_io::ThreadedSink::new(writer, 3);

    let started = std::time::Instant::now();
    let mut last_print = std::time::Instant::now();
    let mut on_progress = move |edges: u64, frac: f64| {
        if !progress || last_print.elapsed().as_millis() < 500 {
            return;
        }
        last_print = std::time::Instant::now();
        let elapsed = started.elapsed().as_secs_f64();
        let rate = edges as f64 / elapsed.max(1e-9);
        eprint!(
            "\r  generating  {:>3.0}%  {edges} edges  {:.2}M ev/s   ",
            frac * 100.0,
            rate / 1e6
        );
    };
    let prog: Option<agora_core::ProgressFn<'_>> =
        if progress { Some(&mut on_progress) } else { None };

    let summary = agora_core::generate(&r.params, &mut sink, prog)?;
    if progress {
        eprintln!();
    }

    // Side-car artifacts: omniscient per-instance ground truth + node labels.
    std::fs::write(
        out_dir.join("ground_truth.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "intent_names": summary.intent_names,
            "edges_per_intent": summary.intent_names.iter().cloned()
                .zip(summary.edges_per_intent.iter().copied()).collect::<Vec<_>>(),
            "instances": summary.ground_truth,
        }))?,
    )?;
    {
        use std::io::Write as _;
        let path = out_dir.join("labels_nodes.csv");
        let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
        writeln!(w, "node_id,intent,kind,anomaly_id,camouflage,community")?;
        for inst in &summary.ground_truth {
            for &node in &inst.members {
                let cam = if inst.camouflage.is_nan() {
                    String::new()
                } else {
                    format!("{:.4}", inst.camouflage)
                };
                writeln!(
                    w,
                    "{node},{},{},{},{cam},{}",
                    inst.intent, inst.kind, inst.id, inst.community
                )?;
            }
        }
        w.flush()?;
    }

    // Embed the summary (sans the bulky per-instance records) in meta.
    let mut summary_for_meta = summary.clone();
    summary_for_meta.ground_truth = Vec::new();
    meta.result = Some(serde_json::to_value(&summary_for_meta)?);
    meta.write(&out_dir)?;

    Ok(summary)
}

// ---------------------------------------------------------------------------
// generate_arrays(): in-memory variant returning numpy columns, no disk IO.
// ---------------------------------------------------------------------------

/// An [`EventSink`] that accumulates the columnar stream in memory so it can be
/// handed to Python as numpy arrays without ever touching disk — the key
/// "graph straight into PyG/DGL" affordance (blueprint §7).
#[derive(Default)]
struct MemorySink {
    src: Vec<u64>,
    dst: Vec<u64>,
    t: Vec<i64>,
    event_type: Vec<u16>,
    label: Vec<u16>,
    anomaly_id: Vec<i64>,
    /// Attribute columns by union name; rows align to the edge index. Late-
    /// appearing columns are back-filled with NaN so every column stays the
    /// same length as the edge stream.
    attr_names: Vec<String>,
    attrs: Vec<Vec<f64>>,
}

impl MemorySink {
    fn column_index(&mut self, name: &str, rows_so_far: usize) -> usize {
        if let Some(i) = self.attr_names.iter().position(|n| n == name) {
            return i;
        }
        self.attr_names.push(name.to_string());
        self.attrs.push(vec![f64::NAN; rows_so_far]);
        self.attrs.len() - 1
    }
}

impl EventSink for MemorySink {
    fn write_batch(&mut self, batch: &EventBatch) -> anyhow::Result<()> {
        let base = self.src.len();
        self.src.extend_from_slice(&batch.src);
        self.dst.extend_from_slice(&batch.dst);
        self.t.extend_from_slice(&batch.t);
        self.event_type.extend_from_slice(&batch.event_type);
        self.label.extend_from_slice(&batch.label);
        self.anomaly_id.extend_from_slice(&batch.anomaly_id);

        let n = batch.len();
        // Map this batch's attr columns into our union columns by name.
        for (j, name) in batch.attr_names.iter().enumerate() {
            let idx = self.column_index(name, base);
            // Grow to the new full height with NaN, then fill this batch's rows.
            let col = &mut self.attrs[idx];
            col.resize(base, f64::NAN);
            col.extend_from_slice(&batch.attrs[j]);
        }
        // Any union column this batch didn't carry must still grow by `n` NaNs.
        let new_len = base + n;
        for col in &mut self.attrs {
            if col.len() < new_len {
                col.resize(new_len, f64::NAN);
            }
        }
        Ok(())
    }

    fn write_nodes(&mut self, _batch: &NodeBatch) -> anyhow::Result<()> {
        // Node tables are not surfaced by the array API (edges-only graph).
        Ok(())
    }
}

#[pyfunction]
#[pyo3(signature = (
    domain, *, nodes, edges, time_span_days, seed=42, anomaly_rate=None,
    difficulty=None, no_anomalies=false, threads=None, shard_index=0,
    shard_count=1
))]
#[allow(clippy::too_many_arguments)]
fn generate_arrays(
    py: Python<'_>,
    domain: &str,
    nodes: u64,
    edges: u64,
    time_span_days: f64,
    seed: u64,
    anomaly_rate: Option<f64>,
    difficulty: Option<f64>,
    no_anomalies: bool,
    threads: Option<usize>,
    shard_index: u64,
    shard_count: u64,
) -> PyResult<Py<PyDict>> {
    // fmt is irrelevant in-memory; pass parquet to satisfy validation.
    let resolved = resolve(
        domain, nodes, edges, time_span_days, seed, anomaly_rate, difficulty,
        no_anomalies, threads, None, "parquet", shard_index, shard_count,
    )
    .map_err(py_err)?;

    // Generate fully in memory with the GIL released.
    let (sink, summary) = py
        .allow_threads(|| -> anyhow::Result<(MemorySink, GenSummary)> {
            let mut sink = MemorySink::default();
            let summary = agora_core::generate(&resolved.params, &mut sink, None)?;
            Ok((sink, summary))
        })
        .map_err(py_err)?;

    let n = sink.src.len();
    let d = PyDict::new(py);
    // Zero-copy hand-off: the numpy crate takes ownership of each Vec.
    d.set_item("src", sink.src.into_pyarray(py))?;
    d.set_item("dst", sink.dst.into_pyarray(py))?;
    d.set_item("t", sink.t.into_pyarray(py))?;
    d.set_item("event_type", sink.event_type.into_pyarray(py))?;
    d.set_item("label_codes", sink.label.into_pyarray(py))?;
    d.set_item("anomaly_id", sink.anomaly_id.into_pyarray(py))?;

    // Attribute columns, each as its own numpy float array, keyed by name.
    let attrs = PyDict::new(py);
    for (name, col) in sink.attr_names.into_iter().zip(sink.attrs) {
        // Defensive: every column is the same height as the edge stream.
        let mut col = col;
        if col.len() < n {
            col.resize(n, f64::NAN);
        }
        attrs.set_item(name, col.into_pyarray(py))?;
    }
    d.set_item("attrs", attrs)?;

    d.set_item("intent_names", summary.intent_names.clone())?;
    d.set_item("event_type_names", summary.event_type_names.clone())?;
    d.set_item("n_edges", n)?;
    d.set_item("nodes", summary.nodes)?;
    d.set_item("anomalous_edges", summary.anomalous_edges)?;

    // attribute code dictionaries {attr: [values...]} so categorical columns
    // (stored as numeric codes) can be decoded.
    let ad = PyDict::new(py);
    for (attr, values) in &summary.attr_dictionaries {
        ad.set_item(attr, values.clone())?;
    }
    d.set_item("attr_dictionaries", ad)?;

    // ground truth instances, same shape as in generate().
    let gt = PyList::empty(py);
    for rec in &summary.ground_truth {
        gt.append(anomaly_record_to_py(py, rec)?)?;
    }
    d.set_item("ground_truth", gt)?;

    Ok(d.unbind())
}

// ---------------------------------------------------------------------------
// Module.
// ---------------------------------------------------------------------------

/// `import agora` — the Python entry point. Module name is `agora` so users
/// `import agora` after `pip install agora` (maturin maps the wheel's import
/// name to this `#[pymodule]`).
#[pymodule]
fn agora(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(domains, m)?)?;
    m.add_function(wrap_pyfunction!(doctor, m)?)?;
    m.add_function(wrap_pyfunction!(generate, m)?)?;
    m.add_function(wrap_pyfunction!(generate_arrays, m)?)?;
    Ok(())
}
