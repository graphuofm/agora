//! Splittable stream derivation.
//!
//! A stream is identified by `(master_seed, purpose, a, b)` — e.g.
//! `(seed, ActorWindow, actor_id, window_idx)` — and hashed through
//! SplitMix64 into a PCG state. PCG64-Mcg is fast (one multiply + xor-shift
//! per draw) and statistically solid for simulation use.

use rand_pcg::Pcg64Mcg;

pub type Rng64 = Pcg64Mcg;

/// Namespaces for stream derivation. Adding a purpose never perturbs
/// existing streams (values are stable, append-only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub enum StreamPurpose {
    /// Node attribute/state initialization: (node_id, 0).
    NodeInit = 1,
    /// Skeleton topology build: (relation_idx, chunk).
    Topology = 2,
    /// Normal behavior of one actor in one time window: (actor_id, window).
    ActorWindow = 3,
    /// Anomaly campaign scheduling/recruitment: (process_idx, campaign_idx).
    Campaign = 4,
    /// Per-actor static draws (activity multiplier…): (actor_id, behavior_idx).
    ActorStatic = 5,
    /// Failure process draws: (entity_id, process_idx).
    Failure = 6,
}

#[inline]
fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Project tag baked into every stream derivation (versioned: changing this
/// changes all outputs, so it never changes within a major version).
const AGORA_TAG: u64 = 0x5A6A_0001_5161_0D00;

/// Derive an independent RNG stream. O(1), no allocation.
#[inline]
pub fn stream(master_seed: u64, purpose: StreamPurpose, a: u64, b: u64) -> Rng64 {
    // Chain SplitMix64 over the tuple: collision-free in practice and stable.
    let mut h = splitmix64(master_seed ^ AGORA_TAG);
    h = splitmix64(h ^ (purpose as u64));
    h = splitmix64(h ^ a);
    h = splitmix64(h ^ b);
    // 128-bit state from two further mixes.
    let lo = splitmix64(h) as u128;
    let hi = splitmix64(h ^ 0xDEAD_BEEF_CAFE_F00D) as u128;
    Pcg64Mcg::new((hi << 64) | lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn streams_are_deterministic() {
        let mut a = stream(42, StreamPurpose::ActorWindow, 7, 3);
        let mut b = stream(42, StreamPurpose::ActorWindow, 7, 3);
        for _ in 0..100 {
            assert_eq!(a.gen::<u64>(), b.gen::<u64>());
        }
    }

    #[test]
    fn streams_differ_across_ids_and_purposes() {
        let mut base = stream(42, StreamPurpose::ActorWindow, 7, 3);
        let mut other_actor = stream(42, StreamPurpose::ActorWindow, 8, 3);
        let mut other_purpose = stream(42, StreamPurpose::Campaign, 7, 3);
        let x = base.gen::<u64>();
        assert_ne!(x, other_actor.gen::<u64>());
        assert_ne!(x, other_purpose.gen::<u64>());
    }
}
