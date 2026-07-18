//! `agora stats` — introspection summary of a generated output (§11b, §12).
//!
//! Stats are computed during generation (single pass) and stored as
//! agora_stats.json; this command renders them. If the file is missing (e.g.
//! the output was produced by an older build), it is recomputed by streaming
//! the shards.

use std::path::Path;

use owo_colors::OwoColorize;

use super::Ctx;

pub fn run(ctx: &Ctx, output: &Path) -> anyhow::Result<()> {
    let meta_path = output.join("agora_meta.json");
    if !meta_path.exists() {
        anyhow::bail!(
            "`{}` is not a AGORA output directory (no agora_meta.json found)",
            output.display()
        );
    }
    let stats_path = output.join("agora_stats.json");
    let report: serde_json::Value = if stats_path.exists() {
        serde_json::from_str(&std::fs::read_to_string(&stats_path)?)?
    } else {
        anyhow::bail!(
            "no agora_stats.json in `{}`; re-run generation (stats are computed \
             during generation in a single pass)",
            output.display()
        );
    };

    if ctx.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    let g = |k: &str| report.get(k).cloned().unwrap_or(serde_json::Value::Null);
    println!("{}", format!("agora stats — {}", output.display()).bold());
    println!(
        "  edges       {}   nodes {}   distinct pairs ~{}",
        g("total_edges"),
        g("nodes"),
        g("distinct_pairs")
    );
    println!("  time span   {} … {} (unix s)", g("t_min"), g("t_max"));

    println!("{}", "  event types".bold());
    if let Some(arr) = g("events_per_event_type").as_array() {
        for e in arr {
            println!("    {:<18} {}", e[0].as_str().unwrap_or("?"), e[1]);
        }
    }
    println!("{}", "  label introspection (exact, free: label = cause)".bold());
    if let Some(arr) = g("label_introspection").as_array() {
        for e in arr {
            println!("    {:<18} {}", e[0].as_str().unwrap_or("?"), e[1]);
        }
    }
    for key in ["out_degree", "in_degree"] {
        if let Some(d) = g(key).as_object() {
            println!(
                "  {key}  mean {:.2}  max {}  (log2 bins: {})",
                d["mean"].as_f64().unwrap_or(0.0),
                d["max"],
                d["log2_histogram"]
                    .as_array()
                    .map(|a| a.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","))
                    .unwrap_or_default()
            );
        }
    }
    if let Some(arr) = g("top_in_nodes").as_array() {
        let tops: Vec<String> = arr
            .iter()
            .take(5)
            .map(|e| format!("{}({})", e[0], e[1]))
            .collect();
        println!("  top hubs    {}", tops.join("  "));
    }
    if let Some(arr) = g("numeric_attrs").as_array() {
        println!("{}", "  numeric attrs".bold());
        for e in arr {
            let m = &e[1];
            println!(
                "    {:<12} mean {:.2}  p50 {:.2}  p99 {:.2}  max {:.2}",
                e[0].as_str().unwrap_or("?"),
                m["mean"].as_f64().unwrap_or(0.0),
                m["p50"].as_f64().unwrap_or(0.0),
                m["p99"].as_f64().unwrap_or(0.0),
                m["max"].as_f64().unwrap_or(0.0),
            );
        }
    }
    if let Some(arr) = g("daily_events").as_array() {
        let counts: Vec<u64> = arr.iter().filter_map(|v| v.as_u64()).collect();
        if !counts.is_empty() {
            println!("  temporal    {} days, {}", counts.len(), sparkline(&counts));
        }
    }
    // Temporal realism, measured (not asserted): burstiness B, memory M,
    // edge recurrence — comparable to the burstiness/TGB literature.
    let fmt_opt = |v: serde_json::Value| -> String {
        v.as_f64().map(|x| format!("{x:+.3}")).unwrap_or_else(|| "n/a".into())
    };
    println!(
        "  burstiness  B={} (0≈Poisson, →1 bursty)   memory M={}   repeat-edge {:.1}%",
        fmt_opt(g("burstiness_b")),
        fmt_opt(g("memory_m")),
        g("repeat_edge_ratio").as_f64().unwrap_or(0.0) * 100.0
    );
    Ok(())
}

/// Tiny unicode sparkline of daily event volume.
fn sparkline(v: &[u64]) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let max = *v.iter().max().unwrap_or(&1) as f64;
    // Downsample to at most 60 chars.
    let stride = (v.len() as f64 / 60.0).ceil() as usize;
    v.chunks(stride.max(1))
        .map(|c| {
            let avg = c.iter().sum::<u64>() as f64 / c.len() as f64;
            BARS[((avg / max) * 7.0) as usize]
        })
        .collect()
}
