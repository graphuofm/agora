//! agora-rules: the typed rule base (blueprint §5's seven primitives), the
//! user-facing generation config, presets, validation, and run metadata.
//!
//! Data contract position: `RuleBase -> World -> EventStream(+labels) -> Export`
//! — this crate owns the first link and the user-facing config that selects and
//! parameterizes it. Everything is strictly typed and serde-serializable so the
//! RAG compiler (M3), the CLI and the engine all speak the same schema.

pub mod config;
pub mod domains;
pub mod meta;
pub mod rulebase;

pub use config::{
    AnomalyConfig, GenerationConfig, OutputConfig, OutputFormat, Preset, RuntimeConfig,
    ScaleConfig, TimeConfig,
};
pub use domains::{builtin_domains, load_builtin_rulebase, DomainInfo};
pub use meta::RunMeta;
pub use rulebase::*;
