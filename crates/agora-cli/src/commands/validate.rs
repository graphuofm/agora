//! `agora validate` — consistency + constraint (Φ) checks on an output (§12).
//!
//! Two classes of result:
//!   HARD checks (fail = broken output): shards readable, time-sorted,
//!     endpoint ids in range, labels/event types in the declared sets.
//!   Φ candidates (informational): counts of constraint violations from the
//!     domain rule base. These are a detector's-eye view — violations may be
//!     legitimate tails OR injected anomalies; ground truth stays the label.

use std::collections::{HashMap, VecDeque};
use std::path::Path;

use owo_colors::OwoColorize;
use agora_rules::{load_builtin_rulebase, ConstraintCheck, RunMeta};

use super::Ctx;

pub fn run(ctx: &Ctx, output: &Path) -> anyhow::Result<()> {
    let meta_path = output.join("agora_meta.json");
    if !meta_path.exists() {
        anyhow::bail!(
            "`{}` is not a AGORA output directory (no agora_meta.json found)",
            output.display()
        );
    }
    let meta: RunMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path)?)?;
    let rb = load_builtin_rulebase(&meta.config.domain)?;
    let nodes = meta.config.scale.nodes;

    let mut event_types: Vec<String> = rb.event_types.iter().map(|e| e.name.clone()).collect();
    event_types.sort();
    let mut labels: Vec<String> = vec!["normal".into()];
    labels.extend(rb.adversaries.iter().map(|a| a.intent.clone()));
    labels.extend(rb.failures.iter().map(|f| f.intent.clone()));

    // Hard-check state.
    let mut total = 0u64;
    let mut last_t = i64::MIN;
    let mut unsorted = 0u64;
    let mut bad_ids = 0u64;
    let mut bad_enum = 0u64;

    // Φ state.
    let mut range_violations: HashMap<String, u64> = HashMap::new();
    struct WindowCheck {
        name: String,
        event: String,
        attr: Option<String>,
        floor: f64,
        threshold: f64,
        k: u32,
        window_s: u64,
        per_src: HashMap<u64, VecDeque<i64>>,
        violations: u64,
    }
    let mut window_checks: Vec<WindowCheck> = Vec::new();
    for c in &rb.constraints {
        match &c.check {
            ConstraintCheck::SubThresholdCount { event, attr, threshold, floor, k, window_s } => {
                window_checks.push(WindowCheck {
                    name: c.name.clone(),
                    event: event.clone(),
                    attr: Some(attr.clone()),
                    floor: *floor,
                    threshold: *threshold,
                    k: *k,
                    window_s: *window_s,
                    per_src: HashMap::new(),
                    violations: 0,
                });
            }
            ConstraintCheck::RateLimit { event, k, window_s } => {
                window_checks.push(WindowCheck {
                    name: c.name.clone(),
                    event: event.clone(),
                    attr: None,
                    floor: f64::NEG_INFINITY,
                    threshold: f64::INFINITY,
                    k: *k,
                    window_s: *window_s,
                    per_src: HashMap::new(),
                    violations: 0,
                });
            }
            _ => {}
        }
    }

    let shards = agora_io::read::edge_shards(output)?;
    for shard in &shards {
        agora_io::read::read_shard(shard, &mut |b| {
            for i in 0..b.src.len() {
                total += 1;
                let t = b.t[i];
                if t < last_t {
                    unsorted += 1;
                }
                last_t = t;
                if b.src[i] >= nodes || b.dst[i] >= nodes {
                    bad_ids += 1;
                }
                if event_types.binary_search(&b.event_type[i]).is_err()
                    || !labels.contains(&b.label[i])
                {
                    bad_enum += 1;
                }
                // AttrRange checks.
                for c in &rb.constraints {
                    if let ConstraintCheck::AttrRange { event, attr, min, max } = &c.check {
                        if &b.event_type[i] == event {
                            if let Some((_, vals)) =
                                b.numeric_attrs.iter().find(|(n, _)| n == attr)
                            {
                                let v = vals[i];
                                if !v.is_nan() && (v < *min || v > *max) {
                                    *range_violations.entry(c.name.clone()).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                }
                // Sliding-window checks.
                for wc in window_checks.iter_mut() {
                    if b.event_type[i] != wc.event {
                        continue;
                    }
                    if let Some(attr) = &wc.attr {
                        let v = b
                            .numeric_attrs
                            .iter()
                            .find(|(n, _)| n == attr)
                            .map(|(_, vals)| vals[i])
                            .unwrap_or(f64::NAN);
                        if !(v >= wc.floor && v < wc.threshold) {
                            continue; // only near-threshold events count
                        }
                    }
                    let q = wc.per_src.entry(b.src[i]).or_default();
                    let cutoff = t - wc.window_s as i64;
                    while q.front().is_some_and(|&ft| ft < cutoff) {
                        q.pop_front();
                    }
                    q.push_back(t);
                    if q.len() as u32 >= wc.k {
                        wc.violations += 1;
                        q.clear(); // count each run once
                    }
                }
            }
            Ok(())
        })?;
    }

    let hard_ok = unsorted == 0 && bad_ids == 0 && bad_enum == 0;
    if ctx.json {
        let phi: HashMap<String, u64> = range_violations
            .into_iter()
            .chain(window_checks.iter().map(|w| (w.name.clone(), w.violations)))
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "edges": total,
                "shards": shards.len(),
                "hard_checks": { "time_sorted": unsorted == 0, "ids_in_range": bad_ids == 0, "enums_valid": bad_enum == 0 },
                "phi_candidates": phi,
                "ok": hard_ok,
            }))?
        );
    } else {
        println!("{}", format!("agora validate — {}", output.display()).bold());
        println!("  edges       {total} across {} shard(s)", shards.len());
        let mark = |ok: bool| if ok { "PASS".green().to_string() } else { "FAIL".red().to_string() };
        println!("  [{}] time-sorted stream ({unsorted} out-of-order)", mark(unsorted == 0));
        println!("  [{}] endpoint ids in range ({bad_ids} bad)", mark(bad_ids == 0));
        println!("  [{}] event types & labels in declared sets ({bad_enum} bad)", mark(bad_enum == 0));
        println!("  Φ constraint candidates (informational, label is ground truth):");
        for (name, v) in &range_violations {
            println!("    {name:<28} {v}");
        }
        for wc in &window_checks {
            println!("    {:<28} {}", wc.name, wc.violations);
        }
        println!(
            "  verdict     {}",
            if hard_ok { "consistent".green().bold().to_string() } else { "BROKEN".red().bold().to_string() }
        );
    }
    anyhow::ensure!(hard_ok, "hard consistency checks failed");
    Ok(())
}
