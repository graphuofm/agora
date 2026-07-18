//! `agora init` — scaffold a generation config from a preset, host-checked.

use std::path::Path;

use agora_host::cost::{CostModel, Feasibility, RunRequest};
use agora_host::HostProbe;
use agora_rules::{builtin_domains, GenerationConfig, Preset};

use super::Ctx;

pub fn run(ctx: &Ctx, path: &Path, domain: &str, preset: &str, force: bool) -> anyhow::Result<()> {
    let preset = Preset::parse(preset).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown preset `{preset}`: expected one of {}",
            Preset::ALL.map(|p| p.name()).join(", ")
        )
    })?;
    if !builtin_domains().iter().any(|d| d.id == domain) && !Path::new(domain).exists() {
        anyhow::bail!(
            "unknown domain `{domain}`: expected one of [{}] or a rule-base path",
            builtin_domains().iter().map(|d| d.id).collect::<Vec<_>>().join(", ")
        );
    }
    if path.exists() && !force {
        anyhow::bail!("`{}` already exists (use --force to overwrite)", path.display());
    }

    let cfg = GenerationConfig::scaffold(domain, preset);

    // Host check: warn early if this preset won't fit (§11a ADVISE).
    let probe = HostProbe::probe(&cfg.output.path);
    let est = CostModel::default().estimate(
        &RunRequest {
            nodes: cfg.scale.nodes,
            edges: cfg.scale.edges,
            threads: None,
            bytes_per_edge_disk: cfg.output.format.bytes_per_edge(),
            mem_budget: None,
        },
        &probe,
    );

    cfg.save(path)?;
    ctx.say(format!("wrote {} (domain {}, preset {})", path.display(), domain, preset.name()));
    match est.feasibility {
        Feasibility::Infeasible(reason) => ctx.say(format!(
            "warning: this preset does NOT fit this host — {reason}\n         run `agora doctor` for a feasible scale"
        )),
        Feasibility::Warn(ws) => {
            for w in ws {
                ctx.say(format!("note: {w}"));
            }
        }
        Feasibility::Ok => {}
    }
    ctx.say(format!("next: agora generate --config {} --dry-run", path.display()));
    Ok(())
}
