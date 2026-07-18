//! `agora doctor` — host probe + recommended feasible scale (blueprint §11a).

use owo_colors::OwoColorize;
use agora_host::cost::{human_bytes, CostModel, Feasibility, RunRequest};
use agora_host::HostProbe;
use agora_rules::{OutputFormat, Preset};

use super::Ctx;

pub fn run(ctx: &Ctx) -> anyhow::Result<()> {
    let probe = HostProbe::probe(std::path::Path::new("."));

    if ctx.json {
        let rec = recommend(&probe);
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "host": probe,
                "recommended_preset": rec.map(|p| p.name()),
            }))?
        );
        return Ok(());
    }

    println!("{}", "agora doctor — host capability probe".bold());
    println!("  host      {} ({} kernel {})", probe.hostname, probe.os, probe.kernel);
    println!(
        "  cpu       {} — {} physical / {} logical cores, {} NUMA node(s)",
        probe.cpu.model, probe.cpu.physical_cores, probe.cpu.logical_cores, probe.numa_nodes
    );
    println!(
        "  memory    {} total, {} available",
        human_bytes(probe.mem.total_bytes),
        human_bytes(probe.mem.available_bytes)
    );
    if probe.gpus.is_empty() {
        println!("  gpu       none detected (CPU path will be used)");
    } else {
        for g in &probe.gpus {
            println!("  gpu       {} — {} MiB VRAM (driver {})", g.name, g.vram_mb, g.driver);
        }
    }
    match (probe.disk.free_bytes, probe.disk.total_bytes) {
        (Some(free), Some(total)) => println!(
            "  disk      {} free of {} on {}",
            human_bytes(free),
            human_bytes(total),
            probe.disk.location()
        ),
        // Never invent a figure for a filesystem we could not measure.
        _ => println!(
            "  disk      free space unknown on {} (capacity check will be skipped)",
            probe.disk.location()
        ),
    }
    println!("  threads   {} (auto: cores − 2)", probe.default_threads());
    println!();

    // Feasibility of each preset on this host.
    println!("{}", "preset feasibility on this host".bold());
    let model = CostModel::default();
    let mut best: Option<Preset> = None;
    for p in Preset::ALL {
        let (nodes, edges, _) = p.baseline();
        let req = RunRequest {
            nodes,
            edges,
            threads: None,
            bytes_per_edge_disk: OutputFormat::Parquet.bytes_per_edge(),
            mem_budget: None,
        };
        let est = model.estimate(&req, &probe);
        let verdict = match &est.feasibility {
            Feasibility::Ok => {
                best = Some(p);
                "ok".green().to_string()
            }
            Feasibility::Warn(_) => {
                best = Some(p);
                "ok (tight)".yellow().to_string()
            }
            Feasibility::Infeasible(_) => "too big".red().to_string(),
        };
        println!(
            "  {:<7} {:>12} nodes {:>14} edges  ~{:<9} ~{:<9} disk  {}",
            p.name(),
            nodes,
            edges,
            agora_host::cost::human_duration(est.est_wall_time_s),
            human_bytes(est.disk_bytes),
            verdict
        );
    }
    println!();
    match best {
        Some(p) => println!(
            "recommended start: {}",
            format!("agora generate --domain finance --preset {}", p.name()).bold()
        ),
        None => println!("this host is too constrained even for the tiny preset; free up disk space"),
    }
    Ok(())
}

fn recommend(probe: &HostProbe) -> Option<Preset> {
    let model = CostModel::default();
    Preset::ALL.into_iter().rfind(|p| {
        let (nodes, edges, _) = p.baseline();
        let req = RunRequest {
            nodes,
            edges,
            threads: None,
            bytes_per_edge_disk: OutputFormat::Parquet.bytes_per_edge(),
            mem_budget: None,
        };
        !matches!(model.estimate(&req, probe).feasibility, Feasibility::Infeasible(_))
    })
}
