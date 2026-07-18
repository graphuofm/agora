//! Vose alias method: O(n) build, O(1) weighted sampling — the workhorse
//! behind categorical attributes and popularity-proportional counterparty
//! selection (blueprint §8: the per-event hot operation must be O(1)).

use rand::Rng;

use crate::rng::Rng64;

#[derive(Debug, Clone)]
pub struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    /// Build from non-negative weights (need not be normalized).
    /// Panics if all weights are zero or the table would exceed u32 slots.
    pub fn new(weights: &[f64]) -> AliasTable {
        assert!(!weights.is_empty(), "alias table needs at least one weight");
        assert!(weights.len() <= u32::MAX as usize, "alias table too large");
        let n = weights.len();
        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "alias table needs a positive total weight");

        // Scaled probabilities; classify into small/large worklists.
        let mut prob = vec![0.0f64; n];
        let mut alias = vec![0u32; n];
        let mut scaled: Vec<f64> = weights.iter().map(|w| w * n as f64 / total).collect();
        let mut small: Vec<u32> = Vec::with_capacity(n);
        let mut large: Vec<u32> = Vec::with_capacity(n);
        for (i, &s) in scaled.iter().enumerate() {
            if s < 1.0 {
                small.push(i as u32);
            } else {
                large.push(i as u32);
            }
        }
        while let (Some(&s), Some(&l)) = (small.last(), large.last()) {
            small.pop();
            prob[s as usize] = scaled[s as usize];
            alias[s as usize] = l;
            scaled[l as usize] -= 1.0 - scaled[s as usize];
            if scaled[l as usize] < 1.0 {
                large.pop();
                small.push(l);
            }
        }
        for &i in large.iter().chain(small.iter()) {
            prob[i as usize] = 1.0;
        }
        AliasTable { prob, alias }
    }

    pub fn len(&self) -> usize {
        self.prob.len()
    }

    pub fn is_empty(&self) -> bool {
        self.prob.is_empty()
    }

    /// O(1) draw of an index, distributed per the build weights.
    #[inline]
    pub fn sample(&self, rng: &mut Rng64) -> usize {
        let i = rng.gen_range(0..self.prob.len());
        if rng.gen::<f64>() < self.prob[i] {
            i
        } else {
            self.alias[i] as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::{stream, StreamPurpose};

    #[test]
    fn matches_weights_statistically() {
        let weights = [1.0, 2.0, 4.0, 8.0, 1.0];
        let table = AliasTable::new(&weights);
        let mut rng = stream(7, StreamPurpose::Topology, 0, 0);
        let mut counts = [0u64; 5];
        let n = 1_000_000;
        for _ in 0..n {
            counts[table.sample(&mut rng)] += 1;
        }
        let total: f64 = weights.iter().sum();
        for (i, &w) in weights.iter().enumerate() {
            let expected = w / total;
            let observed = counts[i] as f64 / n as f64;
            assert!(
                (observed - expected).abs() < 0.005,
                "slot {i}: expected {expected:.4}, observed {observed:.4}"
            );
        }
    }

    #[test]
    fn single_weight_always_zero() {
        let table = AliasTable::new(&[3.0]);
        let mut rng = stream(7, StreamPurpose::Topology, 0, 1);
        for _ in 0..100 {
            assert_eq!(table.sample(&mut rng), 0);
        }
    }
}
