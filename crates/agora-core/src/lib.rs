//! agora-core: the discrete-event simulation engine (blueprint §7).
//!
//! Data contract position: `RuleBase -> World -> EventStream(+labels)`.
//! The engine emits columnar [`EventBatch`]es to an [`EventSink`] — columnar
//! from the start so the Arrow/Parquet writers (agora-io, M2) consume them
//! zero-copy. Determinism: master seed + splittable per-actor RNG streams =>
//! identical output for any thread count.

pub mod anomaly;
pub mod api;
pub mod sim;
pub mod world;

pub use api::*;
