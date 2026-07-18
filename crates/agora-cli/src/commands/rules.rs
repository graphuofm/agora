//! `agora rules` — offline RAG tooling (corpus fetch lands at M3a, build at M3b).
//!
//! Both subcommands delegate to the Python harness (`python/agora_rag`) since
//! the RAG is explicitly off the hot path (blueprint §9); this Rust side just
//! locates and invokes it with a clear error if the harness isn't installed.

use std::path::Path;

use super::Ctx;

pub fn build(_ctx: &Ctx, domain: &str, out: &Path) -> anyhow::Result<()> {
    // Built-in domains need no RAG: their rule bases ship compiled.
    if agora_rules::builtin_domains().iter().any(|d| d.id == domain) {
        let rb = agora_rules::load_builtin_rulebase(domain)?;
        std::fs::write(out, rb.to_yaml()?)?;
        println!("wrote compiled built-in rule base to {}", out.display());
        return Ok(());
    }
    // A path to a hand-authored rule base: load + validate + re-serialize.
    // (This is the authoring/validation path until the RAG lands at M3b.)
    if Path::new(domain).exists() {
        let rb = agora_rules::load_builtin_rulebase(domain)?;
        std::fs::write(out, rb.to_yaml()?)?;
        println!(
            "validated `{domain}` (id `{}`): {} entity types, {} events, {} adversaries, {} failures -> {}",
            rb.meta.id,
            rb.entity_types.len(),
            rb.event_types.len(),
            rb.adversaries.len(),
            rb.failures.len(),
            out.display()
        );
        return Ok(());
    }
    anyhow::bail!(
        "natural-language rule synthesis is the offline RAG pipeline (milestone M3b): \
         it will run via the Python harness `python -m agora_rag build`; \
         for now use a built-in domain id or hand-edit a rule-base YAML \
         (start from `agora rules build --domain finance --out my_domain.yaml`)"
    )
}

pub fn corpus(_ctx: &Ctx, fetch: bool, dir: &Path) -> anyhow::Result<()> {
    if !fetch {
        anyhow::bail!("nothing to do: pass --fetch to download the corpus (see CORPUS.md)");
    }
    anyhow::bail!(
        "the corpus fetcher lands at milestone M3a (`python -m agora_rag corpus --fetch --dir {}`); \
         the acquisition manifest is CORPUS.md",
        dir.display()
    )
}
