//! The `agora` CLI (blueprint §12). Subcommand implementations live in
//! `commands/`; this file only declares the interface and dispatches.
//!
//! UX contract (§13): errors are actionable one-liners, never stack traces;
//! `--json` makes every command machine-readable.

mod commands;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "agora",
    version,
    about = "AGORA — generate large-scale attributed temporal graphs with ground-truth anomaly labels",
    propagate_version = true
)]
struct Cli {
    /// Machine-readable JSON output.
    #[arg(long, global = true)]
    json: bool,
    /// Suppress non-essential output.
    #[arg(long, short, global = true)]
    quiet: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Probe this machine (CPU/RAM/GPU/disk) and recommend a feasible scale.
    Doctor,
    /// List the built-in domains, or show one in detail.
    Domains {
        /// Show full parameters of one domain.
        #[arg(long)]
        show: Option<String>,
    },
    /// Scaffold a generation config from a preset.
    Init {
        /// Path of the config file to create.
        config: PathBuf,
        #[arg(long, default_value = "finance")]
        domain: String,
        /// tiny | small | medium | large | huge (host-checked).
        #[arg(long, default_value = "small")]
        preset: String,
        #[arg(long)]
        force: bool,
    },
    /// Rule-base tooling: the offline RAG pipeline and corpus fetcher.
    Rules {
        #[command(subcommand)]
        cmd: RulesCmd,
    },
    /// Generate a dataset (the main command).
    Generate(Box<commands::generate::GenerateArgs>),
    /// Introspection summary of a generated output directory.
    Stats { output: PathBuf },
    /// Fidelity/consistency checks on a generated output directory.
    Validate { output: PathBuf },
}

#[derive(Subcommand)]
enum RulesCmd {
    /// Compile a domain description into a validated rule base (offline RAG).
    Build {
        /// Natural-language domain description, or a built-in domain id.
        #[arg(long)]
        domain: String,
        /// Output path for the compiled rule base.
        #[arg(long, default_value = "rulebase.yaml")]
        out: PathBuf,
    },
    /// Fetch the RAG knowledge corpus per CORPUS.md (license-tier aware).
    Corpus {
        #[arg(long)]
        fetch: bool,
        /// Corpus root directory.
        #[arg(long, default_value = "corpus")]
        dir: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();
    let ctx = commands::Ctx { json: cli.json, quiet: cli.quiet };
    let result = match cli.cmd {
        Cmd::Doctor => commands::doctor::run(&ctx),
        Cmd::Domains { show } => commands::domains::run(&ctx, show.as_deref()),
        Cmd::Init { config, domain, preset, force } => {
            commands::init::run(&ctx, &config, &domain, &preset, force)
        }
        Cmd::Rules { cmd } => match cmd {
            RulesCmd::Build { domain, out } => commands::rules::build(&ctx, &domain, &out),
            RulesCmd::Corpus { fetch, dir } => commands::rules::corpus(&ctx, fetch, &dir),
        },
        Cmd::Generate(args) => commands::generate::run(&ctx, *args),
        Cmd::Stats { output } => commands::stats::run(&ctx, &output),
        Cmd::Validate { output } => commands::validate::run(&ctx, &output),
    };
    if let Err(e) = result {
        // Actionable one-liner, no backtrace (§13).
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
