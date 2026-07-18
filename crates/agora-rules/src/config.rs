//! The user-facing generation config (`agora init` scaffolds it, `agora
//! generate` consumes it). YAML on disk; every field overridable by a CLI
//! flag. Validation errors are actionable, never stack traces (§13).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationConfig {
    /// Built-in domain id (finance, crypto, cyber, transport, ecommerce,
    /// healthcare) or a path to a compiled custom rule base YAML.
    pub domain: String,
    pub scale: ScaleConfig,
    pub time: TimeConfig,
    #[serde(default)]
    pub anomaly: AnomalyConfig,
    /// Master seed; same seed + config + version => identical output.
    pub seed: u64,
    pub output: OutputConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScaleConfig {
    pub nodes: u64,
    pub edges: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TimeConfig {
    /// Simulated span in days.
    pub span_days: f64,
    /// Timestamp resolution in seconds (1 = per-second timestamps).
    #[serde(default = "default_granularity")]
    pub granularity_s: u64,
    /// Epoch of t=0 as unix seconds (default 2025-01-01 UTC).
    #[serde(default = "default_epoch")]
    pub epoch_unix: i64,
}

fn default_granularity() -> u64 {
    1
}
fn default_epoch() -> i64 {
    1_735_689_600 // 2025-01-01T00:00:00Z
}

/// The five control axes (§3), user-facing. `None` = the domain's defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnomalyConfig {
    /// Axis 1: fraction of nodes participating in anomalous processes.
    #[serde(default)]
    pub rate: Option<f64>,
    /// Axis 2: difficulty/camouflage in [0,1].
    #[serde(default)]
    pub difficulty: Option<f64>,
    /// Axis 3: intent → weight overrides.
    #[serde(default)]
    pub type_mix: Option<Vec<(String, f64)>>,
    /// Axis 4/5 use the domain defaults unless overridden in the rule base.
    #[serde(default)]
    pub disabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputConfig {
    pub path: PathBuf,
    #[serde(default)]
    pub format: OutputFormat,
    /// Target shard size in MiB (streaming mode).
    #[serde(default = "default_shard_mb")]
    pub shard_size_mb: u64,
}

fn default_shard_mb() -> u64 {
    256
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Parquet,
    Csv,
    Graphml,
}

impl OutputFormat {
    /// Estimated on-disk bytes per edge (drives the dry-run cost model).
    pub fn bytes_per_edge(self) -> u64 {
        match self {
            OutputFormat::Parquet => 24, // columnar + compression
            OutputFormat::Csv => 64,
            OutputFormat::Graphml => 160,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    /// None = auto (cores - 2).
    #[serde(default)]
    pub threads: Option<usize>,
    /// None = auto-detect; Some(false) forces CPU.
    #[serde(default)]
    pub gpu: Option<bool>,
    /// RAM budget in GiB; None = auto (90% of available).
    #[serde(default)]
    pub mem_budget_gb: Option<f64>,
}

/// Host-relative size presets (§12). Concrete numbers are scaled to the
/// probed host by the CLI; these are the shape ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Preset {
    Tiny,
    Small,
    Medium,
    Large,
    Huge,
}

impl Preset {
    pub const ALL: [Preset; 5] = [
        Preset::Tiny,
        Preset::Small,
        Preset::Medium,
        Preset::Large,
        Preset::Huge,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Tiny => "tiny",
            Preset::Small => "small",
            Preset::Medium => "medium",
            Preset::Large => "large",
            Preset::Huge => "huge",
        }
    }

    pub fn parse(s: &str) -> Option<Preset> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }

    /// (nodes, edges, span_days) — absolute baseline; the CLI may cap to the
    /// host's feasible scale using the cost model.
    pub fn baseline(self) -> (u64, u64, f64) {
        match self {
            Preset::Tiny => (10_000, 1_000_000, 30.0),
            Preset::Small => (100_000, 10_000_000, 90.0),
            Preset::Medium => (1_000_000, 100_000_000, 180.0),
            Preset::Large => (10_000_000, 1_000_000_000, 365.0),
            Preset::Huge => (100_000_000, 10_000_000_000, 365.0),
        }
    }
}

impl GenerationConfig {
    pub fn load(path: &Path) -> anyhow::Result<GenerationConfig> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read config `{}`: {e}", path.display()))?;
        let cfg: GenerationConfig = serde_yaml::from_str(&text)
            .map_err(|e| anyhow::anyhow!("config `{}` is not valid: {e}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut s = String::from(
            "# AGORA generation config — edit freely; every field can also be set by a CLI flag.\n",
        );
        s.push_str(&serde_yaml::to_string(self)?);
        std::fs::write(path, s)
            .map_err(|e| anyhow::anyhow!("cannot write `{}`: {e}", path.display()))
    }

    /// Range validation with actionable messages (§13 ERRORS).
    pub fn validate(&self) -> anyhow::Result<()> {
        let bail = |m: String| Err(anyhow::anyhow!(m));
        if self.scale.nodes == 0 {
            return bail("scale.nodes must be ≥ 1".into());
        }
        if self.scale.edges == 0 {
            return bail("scale.edges must be ≥ 1".into());
        }
        if self.scale.edges < self.scale.nodes / 4 {
            return bail(format!(
                "scale.edges ({}) is implausibly small for {} nodes; expected at least nodes/4",
                self.scale.edges, self.scale.nodes
            ));
        }
        if self.time.span_days <= 0.0 {
            return bail("time.span_days must be > 0".into());
        }
        if self.time.granularity_s == 0 {
            return bail("time.granularity_s must be ≥ 1".into());
        }
        if let Some(r) = self.anomaly.rate {
            if !(0.0..=0.5).contains(&r) {
                return bail(format!(
                    "anomaly.rate {r} out of range: expected [0, 0.5] (anomalies are rare by definition)"
                ));
            }
        }
        if let Some(d) = self.anomaly.difficulty {
            if !(0.0..=1.0).contains(&d) {
                return bail(format!("anomaly.difficulty {d} out of range: expected [0, 1]"));
            }
        }
        if let Some(t) = self.runtime.threads {
            if t == 0 {
                return bail("runtime.threads must be ≥ 1 (or omit for auto)".into());
            }
        }
        if self.output.shard_size_mb < 16 {
            return bail("output.shard_size_mb must be ≥ 16".into());
        }
        Ok(())
    }

    /// A scaffold config for `agora init`.
    pub fn scaffold(domain: &str, preset: Preset) -> GenerationConfig {
        let (nodes, edges, span) = preset.baseline();
        GenerationConfig {
            domain: domain.to_string(),
            scale: ScaleConfig { nodes, edges },
            time: TimeConfig {
                span_days: span,
                granularity_s: default_granularity(),
                epoch_unix: default_epoch(),
            },
            anomaly: AnomalyConfig::default(),
            seed: 42,
            output: OutputConfig {
                path: PathBuf::from("./out"),
                format: OutputFormat::Parquet,
                shard_size_mb: default_shard_mb(),
            },
            runtime: RuntimeConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaffold_roundtrips_and_validates() {
        let cfg = GenerationConfig::scaffold("finance", Preset::Small);
        cfg.validate().unwrap();
        let y = serde_yaml::to_string(&cfg).unwrap();
        let back: GenerationConfig = serde_yaml::from_str(&y).unwrap();
        back.validate().unwrap();
        assert_eq!(back.domain, "finance");
    }

    #[test]
    fn bad_anomaly_rate_is_actionable() {
        let mut cfg = GenerationConfig::scaffold("finance", Preset::Tiny);
        cfg.anomaly.rate = Some(0.9);
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("out of range"), "got: {err}");
    }

    #[test]
    fn unknown_yaml_field_is_rejected() {
        let y = "domain: finance\nbogus_field: 1\n";
        assert!(serde_yaml::from_str::<GenerationConfig>(y).is_err());
    }
}
