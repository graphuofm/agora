//! agora-introspect: single-pass streaming statistics + sketches (blueprint
//! §11b). Generation and introspection are one source: the collector is a
//! sink tee'd with the file writer, so stats cost no second pass over data.

pub mod cm;
pub mod hll;
pub mod moments;
pub mod stats;

pub use stats::{StatsCollector, StatsReport};
