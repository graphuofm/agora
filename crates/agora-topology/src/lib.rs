//! agora-topology: pluggable skeleton/evolution models (blueprint §8).
//!
//! The skeleton answers "who CAN interact": a static substrate the simulation
//! samples counterparties from in O(1). Each relation is materialized as CSR
//! adjacency (offsets + targets) over global node ids.
//!
//! Determinism: every model draws from `stream(seed, Topology, relation_idx,
//! src or 0)` — bit-identical across thread counts.

pub mod csr;
pub mod models;

pub use csr::{Csr, RelationGraph};
pub use models::build_relation;
