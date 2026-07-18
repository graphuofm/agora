//! agora-sample: seeded, splittable randomness + O(1) samplers (blueprint §7).
//!
//! Determinism contract: every random decision in AGORA draws from a stream
//! derived by [`stream`] from `(master_seed, purpose, ids…)`. Streams are
//! independent of thread assignment and iteration order, so any parallel
//! schedule produces bit-identical output for the same seed.

pub mod alias;
pub mod dist;
pub mod rng;

pub use alias::AliasTable;
pub use dist::DistSampler;
pub use rng::{stream, Rng64, StreamPurpose};
