//! `agora_meta.json` — full reproducibility record written for every run
//! (blueprint §12): config + seed + version + host probe + git revision.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::GenerationConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMeta {
    pub agora_version: String,
    /// Git commit the binary was built from, if known at build time.
    pub git_commit: Option<String>,
    pub created_utc: String,
    pub seed: u64,
    pub config: GenerationConfig,
    /// Host probe serialized as opaque JSON (avoids a agora-host dependency
    /// cycle; the probe crate owns the schema).
    pub host: serde_json::Value,
    /// Filled in after generation: edges written, wall time, output files…
    #[serde(default)]
    pub result: Option<serde_json::Value>,
}

impl RunMeta {
    pub fn new(config: &GenerationConfig, host: serde_json::Value) -> RunMeta {
        RunMeta {
            agora_version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("AGORA_GIT_COMMIT").map(str::to_string),
            created_utc: chrono::Utc::now().to_rfc3339(),
            seed: config.seed,
            config: config.clone(),
            host,
            result: None,
        }
    }

    pub fn write(&self, out_dir: &Path) -> anyhow::Result<std::path::PathBuf> {
        std::fs::create_dir_all(out_dir)?;
        let path = out_dir.join("agora_meta.json");
        std::fs::write(&path, serde_json::to_string_pretty(self)?)?;
        Ok(path)
    }
}
