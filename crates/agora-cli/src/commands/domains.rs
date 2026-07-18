//! `agora domains` — list built-in domains; `--show NAME` prints the dossier.

use owo_colors::OwoColorize;
use agora_rules::{builtin_domains, load_builtin_rulebase};

use super::Ctx;

pub fn run(ctx: &Ctx, show: Option<&str>) -> anyhow::Result<()> {
    match show {
        None => list(ctx),
        Some(id) => detail(ctx, id),
    }
}

fn list(ctx: &Ctx) -> anyhow::Result<()> {
    let domains = builtin_domains();
    if ctx.json {
        let v: Vec<_> = domains
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id,
                    "name": d.name,
                    "anomaly_source": d.anomaly_source,
                    "summary": d.summary,
                    "available": d.rulebase_yaml.is_some(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    println!("{}", "built-in domains".bold());
    for d in &domains {
        let status = if d.rulebase_yaml.is_some() {
            "available".green().to_string()
        } else {
            "coming (M4)".yellow().to_string()
        };
        println!("  {:<11} {:<38} [{}] {}", d.id.bold(), d.name, d.anomaly_source, status);
        println!("              {}", d.summary);
    }
    println!("\nuse `agora domains --show <id>` for full parameters, or describe your own domain to `agora rules build`.");
    Ok(())
}

fn detail(ctx: &Ctx, id: &str) -> anyhow::Result<()> {
    let rb = load_builtin_rulebase(id)?;
    if ctx.json {
        println!("{}", serde_json::to_string_pretty(&rb)?);
        return Ok(());
    }
    println!("{} — {}", rb.meta.id.bold(), rb.meta.name);
    println!("{}\n", rb.meta.description.trim());
    println!("{}", "entity types".bold());
    for e in &rb.entity_types {
        println!(
            "  {:<14} weight {:<6} {} attributes, {} state vars",
            e.name,
            e.population_weight,
            e.attributes.len(),
            e.state.len()
        );
    }
    println!("{}", "relations (skeleton)".bold());
    for r in &rb.relations {
        println!("  {:<16} {} -> {}  mean_degree {}", r.name, r.src, r.dst, r.mean_degree);
    }
    println!("{}", "event types".bold());
    for ev in &rb.event_types {
        println!("  {:<16} {} -> {}  ({} attrs)", ev.name, ev.src, ev.dst, ev.attributes.len());
    }
    println!("{}", "behaviors".bold());
    for b in &rb.behaviors {
        println!("  {:<18} actor {}  {:.2}/day", b.name, b.actor, b.timing.rate_per_day);
    }
    println!("{}", "anomaly processes (label = intent)".bold());
    for a in &rb.adversaries {
        println!(
            "  {:<16} adversarial  weight {:<5} camouflage {:<4} {} stage(s)",
            a.intent,
            a.prevalence_weight,
            a.camouflage,
            a.stages.len()
        );
    }
    for f in &rb.failures {
        println!("  {:<16} natural      weight {:<5}", f.intent, f.prevalence_weight);
    }
    println!("{}", "control defaults (the five axes)".bold());
    println!(
        "  prevalence {}  difficulty {}  cascade {}  placement {:?}",
        rb.control.prevalence, rb.control.difficulty, rb.control.cascade, rb.control.placement
    );
    println!("{}", "provenance".bold());
    for p in &rb.meta.provenance {
        println!(
            "  [{}] {}",
            p.license_tier.as_deref().unwrap_or("-"),
            p.source
        );
    }
    Ok(())
}
