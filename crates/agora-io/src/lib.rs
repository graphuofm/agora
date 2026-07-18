//! agora-io: streaming shard writers + readers (blueprint §7).
//!
//! Writers consume the engine's columnar `EventBatch` and shard by size:
//! Parquet (primary), CSV, GraphML. Readers stream AGORA's own output back
//! for `agora stats`/`agora validate`.

pub mod csv;
pub mod graphml;
pub mod parquet;
pub mod read;
pub mod threaded;

pub use csv::CsvSink;
pub use graphml::GraphmlSink;
pub use parquet::ParquetSink;
pub use threaded::ThreadedSink;

use agora_core::{EventBatch, EventSink, NodeBatch};

/// Format-selected writer behind one concrete type.
///
/// The variants differ in size (Parquet carries Arrow builders) but exactly
/// one is constructed per run and lives for the whole run, so the size
/// disparity costs nothing — boxing would only add indirection on the write
/// path.
#[allow(clippy::large_enum_variant)]
pub enum FormatSink {
    Csv(CsvSink),
    Parquet(ParquetSink),
    Graphml(GraphmlSink),
}

impl FormatSink {
    /// `format`: "parquet" | "csv" | "graphml".
    pub fn make(
        format: &str,
        dir: std::path::PathBuf,
        shard_size_mb: u64,
        event_type_names: Vec<String>,
        intent_names: Vec<String>,
        attr_dicts: Vec<(String, Vec<String>)>,
    ) -> anyhow::Result<FormatSink> {
        Ok(match format {
            "parquet" => FormatSink::Parquet(ParquetSink::new(
                dir, shard_size_mb, event_type_names, intent_names, attr_dicts,
            )?),
            "csv" => FormatSink::Csv(CsvSink::new(
                dir, shard_size_mb, event_type_names, intent_names, attr_dicts,
            )?),
            "graphml" => FormatSink::Graphml(GraphmlSink::new(
                dir, event_type_names, intent_names, attr_dicts,
            )?),
            other => anyhow::bail!("unknown format `{other}`: expected parquet, csv or graphml"),
        })
    }
}

impl EventSink for FormatSink {
    fn write_batch(&mut self, batch: &EventBatch) -> anyhow::Result<()> {
        match self {
            FormatSink::Csv(s) => s.write_batch(batch),
            FormatSink::Parquet(s) => s.write_batch(batch),
            FormatSink::Graphml(s) => s.write_batch(batch),
        }
    }
    fn write_nodes(&mut self, batch: &NodeBatch) -> anyhow::Result<()> {
        match self {
            FormatSink::Csv(s) => s.write_nodes(batch),
            FormatSink::Parquet(s) => s.write_nodes(batch),
            FormatSink::Graphml(s) => s.write_nodes(batch),
        }
    }
    fn finish(&mut self) -> anyhow::Result<()> {
        match self {
            FormatSink::Csv(s) => s.finish(),
            FormatSink::Parquet(s) => s.finish(),
            FormatSink::Graphml(s) => s.finish(),
        }
    }
}
